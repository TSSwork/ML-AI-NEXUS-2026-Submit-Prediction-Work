"""
NEW — seed-bagging for the 4 level-1 model families.

Problem: with 7000 rows / 881 positives, a single StratifiedKFold(seed=42)
split has real fold-to-fold variance -- both the reported OOF log loss and
the actual test predictions depend somewhat on which rows happened to land
in which fold. Cells 11/12/18 (and their ports in this pipeline, scripts
11/12/18) only ever look at one such split.

Fix: repeat each base learner's 5-fold CV under several different
StratifiedKFold seeds and average. Every row's final OOF prediction is the
mean of its (seed-only-out-of-fold) predictions across seeds; every test
row's prediction is the mean across all seed*fold models. This is standard
CV-seed bagging -- it reduces variance without touching the leakage
question (that's handled by the OOF-by-construction property + the honest
nested stacking in 22_honest_nested_stacking.py).

Seed 42 is reused from the already-computed artifacts of scripts
11 (LightGBM/CatBoost), 12 (EBM) and 18 (Spline GAM) instead of retraining
it, to avoid wasted compute. EXTRA_SEEDS below are trained fresh here.
"""
import pickle
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import lightgbm as lgb

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

EXTRA_SEEDS = [7, 123]  # + the already-computed seed 42 -> 3 seeds total
ALL_SEEDS = [42] + EXTRA_SEEDS

with open(ARTIFACTS / "X.pkl", "rb") as f:
    X = pickle.load(f)
with open(ARTIFACTS / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(ARTIFACTS / "cat_features.pkl", "rb") as f:
    cat_features = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

n_train, n_test = len(X), len(X_test)

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'min_child_samples': 35,
    'feature_fraction': 0.80, 'subsample': 0.85, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'verbose': -1, 'n_estimators': 500,
}

ebm_cols = [c for c in X.columns if not c.endswith('_isna')]

spline_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'length_of_stay_days'
]
nn_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
nn_cat_cols = X.select_dtypes(include=['category', 'object', 'string', 'str']).columns.tolist()
other_cols = [c for c in X.columns if c not in spline_cols and not c.endswith('_isna')]


def run_seed_lgb_cb(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(n_train)
    test_lgb = np.zeros(n_test)
    oof_cb = np.zeros(n_train)
    test_cb = np.zeros(n_test)

    for trn_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        m_lgb = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        oof_lgb[val_idx] = m_lgb.predict_proba(X_va)[:, 1]
        test_lgb += m_lgb.predict_proba(X_test)[:, 1] / 5

        X_tr_cb, X_va_cb, X_te_cb = X_tr.copy(), X_va.copy(), X_test.copy()
        for col in cat_features:
            X_tr_cb[col] = X_tr_cb[col].astype(str)
            X_va_cb[col] = X_va_cb[col].astype(str)
            X_te_cb[col] = X_te_cb[col].astype(str)
        m_cb = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4,
                                   loss_function='Logloss', eval_metric='Logloss',
                                   random_seed=seed, verbose=False)
        m_cb.fit(X_tr_cb, y_tr, cat_features=cat_features, eval_set=(X_va_cb, y_va),
                 early_stopping_rounds=30)
        oof_cb[val_idx] = m_cb.predict_proba(X_va_cb)[:, 1]
        test_cb += m_cb.predict_proba(X_te_cb)[:, 1] / 5

    return oof_lgb, test_lgb, oof_cb, test_cb


def run_seed_ebm(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_ebm = np.zeros(n_train)
    test_ebm = np.zeros(n_test)
    for trn_idx, val_idx in skf.split(X[ebm_cols], y):
        X_tr, y_tr = X[ebm_cols].iloc[trn_idx], y[trn_idx]
        X_va = X[ebm_cols].iloc[val_idx]
        ebm = ExplainableBoostingClassifier(max_bins=128, interactions=10,
                                             outer_bags=8, inner_bags=0, random_state=seed)
        ebm.fit(X_tr, y_tr)
        oof_ebm[val_idx] = ebm.predict_proba(X_va)[:, 1]
        test_ebm += ebm.predict_proba(X_test[ebm_cols])[:, 1] / 5
    return oof_ebm, test_ebm


def run_seed_spline(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_spline = np.zeros(n_train)
    test_spline = np.zeros(n_test)
    preprocessor = ColumnTransformer(
        transformers=[
            ('spline', make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), spline_cols),
            ('other_num', StandardScaler(), [c for c in other_cols if c in nn_num_cols]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [c for c in other_cols if c in nn_cat_cols]),
        ]
    )
    for trn_idx, val_idx in skf.split(X, y):
        X_tr_raw, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va_raw = X.iloc[val_idx]
        X_tr_sp = preprocessor.fit_transform(X_tr_raw)
        X_va_sp = preprocessor.transform(X_va_raw)
        X_te_sp = preprocessor.transform(X_test)
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        model.fit(X_tr_sp, y_tr)
        oof_spline[val_idx] = model.predict_proba(X_va_sp)[:, 1]
        test_spline += model.predict_proba(X_te_sp)[:, 1] / 5
    return oof_spline, test_spline


# seed 42 reused from existing per-model artifacts
oof_by_model = {
    'lgb': [np.load(ARTIFACTS / "oof_lgb.npy")],
    'cb': [np.load(ARTIFACTS / "oof_cb.npy")],
    'ebm': [np.load(ARTIFACTS / "oof_ebm.npy")],
    'spline': [np.load(ARTIFACTS / "oof_spline.npy")],
}
test_by_model = {
    'lgb': [np.load(ARTIFACTS / "test_preds_lgb.npy")],
    'cb': [np.load(ARTIFACTS / "test_preds_cb.npy")],
    'ebm': [np.load(ARTIFACTS / "test_preds_ebm.npy")],
    'spline': [np.load(ARTIFACTS / "test_preds_spline.npy")],
}

for seed in EXTRA_SEEDS:
    print(f"--> Seed {seed}: training LightGBM + CatBoost (5-fold)...")
    oof_lgb, test_lgb, oof_cb, test_cb = run_seed_lgb_cb(seed)
    oof_by_model['lgb'].append(oof_lgb)
    test_by_model['lgb'].append(test_lgb)
    oof_by_model['cb'].append(oof_cb)
    test_by_model['cb'].append(test_cb)

    print(f"--> Seed {seed}: training EBM (5-fold)...")
    oof_ebm, test_ebm = run_seed_ebm(seed)
    oof_by_model['ebm'].append(oof_ebm)
    test_by_model['ebm'].append(test_ebm)

    print(f"--> Seed {seed}: training Spline GAM (5-fold)...")
    oof_spline, test_spline = run_seed_spline(seed)
    oof_by_model['spline'].append(oof_spline)
    test_by_model['spline'].append(test_spline)

print("\n" + "=" * 58)
print(f"  SEED-BAGGED BASE LEARNERS  (seeds={ALL_SEEDS})")
print("=" * 58)

for name in ['lgb', 'cb', 'ebm', 'spline']:
    single_seed_loss = log_loss(y, np.clip(oof_by_model[name][0], 0.005, 0.995))
    bagged_oof = np.clip(np.mean(oof_by_model[name], axis=0), 0.005, 0.995)
    bagged_test = np.mean(test_by_model[name], axis=0)
    bagged_loss = log_loss(y, bagged_oof)
    bagged_auc = roc_auc_score(y, bagged_oof)

    print(f"{name:<8s} single-seed(42) LogLoss={single_seed_loss:.5f} | "
          f"{len(ALL_SEEDS)}-seed-bagged LogLoss={bagged_loss:.5f} | AUC={bagged_auc:.5f}")

    np.save(ARTIFACTS / f"oof_{name}_bagged.npy", bagged_oof)
    np.save(ARTIFACTS / f"test_preds_{name}_bagged.npy", bagged_test)

print("=" * 58)
print(f"Saved *_bagged.npy artifacts to {ARTIFACTS}")
