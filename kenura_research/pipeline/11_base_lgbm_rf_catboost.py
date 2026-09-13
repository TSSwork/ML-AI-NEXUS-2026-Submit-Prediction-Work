"""
Cell 11 equivalent — matrix prep + LightGBM / RandomForest / CatBoost OOF.

Faithful port of basic_check.ipynb cell 11. Same params, same 5-fold
StratifiedKFold(seed=42), same scipy SLSQP blend-weight step (kept here
only for parity/logging with the original notebook — the honest,
non-leaky scoring lives in 22_honest_nested_stacking.py).

Saves oof_/test_preds_ for lgb, rf, cb, plus the prepped X/X_test/y/
cat_features/skf so downstream scripts (12, 18, 19, 22, 23, 24) can
reuse the exact same feature matrix without re-deriving it.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "train_fe.pkl", "rb") as f:
    train_fe = pickle.load(f)
with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)

target = 'readmitted_30d'
drop_cols = ['patient_id', target]

X = train_fe.drop(columns=drop_cols)
y = train_fe[target].values
X_test = test_fe.drop(columns=['patient_id'])

# Treat object columns as native LightGBM categories
cat_features = X.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
for c in cat_features:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 16,
    'max_depth': 4,
    'min_child_samples': 35,
    'feature_fraction': 0.80,
    'subsample': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 500
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train_fe))
test_preds_lgb = np.zeros(len(test_fe))
oof_rf = np.zeros(len(train_fe))
test_preds_rf = np.zeros(len(test_fe))
oof_cb = np.zeros(len(train_fe))
test_preds_cb = np.zeros(len(test_fe))
feature_importances = np.zeros(X.shape[1])

print("--> Training 5-Fold LightGBM & Random Forest Blend...")

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    base_lgb = lgb.LGBMClassifier(**lgb_params)
    base_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    feature_importances += base_lgb.feature_importances_ / 5

    oof_lgb[val_idx] = base_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += base_lgb.predict_proba(X_test)[:, 1] / 5

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    X_tr_rf = X_tr.select_dtypes(exclude=['category', 'object'])
    X_va_rf = X_va.select_dtypes(exclude=['category', 'object'])
    X_te_rf = X_test.select_dtypes(exclude=['category', 'object'])

    rf_model.fit(X_tr_rf, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_va_rf)[:, 1]
    test_preds_rf += rf_model.predict_proba(X_te_rf)[:, 1] / 5

    cb_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=4,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        verbose=False
    )
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for col in cat_features:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)

    cb_model.fit(X_tr_cb, y_tr, cat_features=cat_features, eval_set=(X_va_cb, y_va), early_stopping_rounds=30)
    oof_cb[val_idx] = cb_model.predict_proba(X_va_cb)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_te_cb)[:, 1] / 5

# Mathematically Optimal Blending via Scipy SLSQP (kept for parity w/ original
# notebook's logging — this weight-fit is scored on the same rows it's fit on,
# i.e. optimistic; see 22_honest_nested_stacking.py for the honest version)
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_rf + w3 * oof_cb
    return log_loss(y, np.clip(pred, 0.005, 0.995))


res = minimize(loss_func, [0.33, 0.33, 0.34], bounds=[(0, 1), (0, 1), (0, 1)],
                constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
w1, w2, w3 = res.x
print(f"\nOptimal Ensemble Weights -> LGBM: {w1:.3f} | RF: {w2:.3f} | CatBoost: {w3:.3f}")

oof_final = w1 * oof_lgb + w2 * oof_rf + w3 * oof_cb
test_final = w1 * test_preds_lgb + w2 * test_preds_rf + w3 * test_preds_cb

oof_preds = np.clip(oof_final, 0.005, 0.995)
test_preds = np.clip(test_final, 0.005, 0.995)

cv_logloss = log_loss(y, oof_preds)
cv_brier = brier_score_loss(y, oof_preds)
cv_auc = roc_auc_score(y, oof_preds)

print("\n" + "=" * 45)
print("             VALIDATION RESULTS             ")
print("=" * 45)
print(f"OOF Log Loss:    {cv_logloss:.5f}  (Baseline Logistic: 0.35261)")
print(f"OOF Brier Score: {cv_brier:.5f}  (Baseline Logistic: 0.10318)")
print(f"OOF ROC-AUC:     {cv_auc:.5f}  (Baseline Logistic: 0.68529)")
print("=" * 45)

os.makedirs(ROOT / "outputs", exist_ok=True)
sub = pd.DataFrame({
    'patient_id': test_fe['patient_id'],
    'readmitted_30d': test_preds
})
sub.to_csv(ROOT / "outputs" / "submission_04_catboost_lgbm_blend.csv", index=False)
print(f"Saved to {ROOT / 'outputs' / 'submission_04_catboost_lgbm_blend.csv'}")

print(f"LightGBM OOF Log Loss:      {log_loss(y, np.clip(oof_lgb, 0.005, 0.995)):.5f}")
print(f"Random Forest OOF Log Loss: {log_loss(y, np.clip(oof_rf, 0.005, 0.995)):.5f}")
print(f"CatBoost OOF Log Loss:      {log_loss(y, np.clip(oof_cb, 0.005, 0.995)):.5f}")

# --- persist artifacts for downstream numbered scripts ---
np.save(ARTIFACTS / "oof_lgb.npy", oof_lgb)
np.save(ARTIFACTS / "test_preds_lgb.npy", test_preds_lgb)
np.save(ARTIFACTS / "oof_rf.npy", oof_rf)
np.save(ARTIFACTS / "test_preds_rf.npy", test_preds_rf)
np.save(ARTIFACTS / "oof_cb.npy", oof_cb)
np.save(ARTIFACTS / "test_preds_cb.npy", test_preds_cb)
np.save(ARTIFACTS / "y.npy", y)

with open(ARTIFACTS / "X.pkl", "wb") as f:
    pickle.dump(X, f)
with open(ARTIFACTS / "X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open(ARTIFACTS / "cat_features.pkl", "wb") as f:
    pickle.dump(cat_features, f)

print(f"Saved LGBM/RF/CatBoost OOF + test preds + X/X_test/cat_features to {ARTIFACTS}")
