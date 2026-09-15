"""
NEW — verifies the 3 cells just appended to basic_check.ipynb (seed-bagged
Core-3: Spline GAM + EBM + CatBoost, no LightGBM, no phenotype features)
by running the identical logic against the same train_fe/test_fe
artifacts, before trusting the notebook cells for a real submission.
Expected honest OOF Log Loss: ~0.34855 (from 27_final_recommended_stack.py).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ARTIFACTS = Path(__file__).resolve().parents[2] / "pipeline" / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "train_fe.pkl", "rb") as f:
    train_fe = pickle.load(f)
with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)

target = 'readmitted_30d'
drop_cols = ['patient_id', target]
y = train_fe[target].values

PHENOTYPE_COLS = ['cluster_id', 'dist_phenotype_0', 'dist_phenotype_1',
                  'dist_phenotype_2', 'dist_phenotype_3', 'shock_index']
X_core3 = train_fe.drop(columns=[c for c in drop_cols + PHENOTYPE_COLS if c in train_fe.columns])
X_test_core3 = test_fe.drop(columns=[c for c in ['patient_id'] + PHENOTYPE_COLS if c in test_fe.columns])

cat_features_c3 = X_core3.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
for c in cat_features_c3:
    X_core3[c] = X_core3[c].astype('category')
    X_test_core3[c] = X_test_core3[c].astype('category')

print(f"X_core3 shape: {X_core3.shape} (should be 38 cols minus target/id -> 36 features)")

nn_num_cols_c3 = X_core3.select_dtypes(include=[np.number]).columns.tolist()
nn_cat_cols_c3 = X_core3.select_dtypes(include=['category', 'object', 'string', 'str']).columns.tolist()

spline_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'length_of_stay_days'
]
other_cols_c3 = [c for c in X_core3.columns if c not in spline_cols and not c.endswith('_isna')]
ebm_cols_c3 = [c for c in X_core3.columns if not c.endswith('_isna')]

SEEDS = [42, 7, 123]
n_tr, n_te = len(train_fe), len(test_fe)

oof_spline_runs, test_spline_runs = [], []
oof_ebm_runs, test_ebm_runs = [], []
oof_cb_runs, test_cb_runs = [], []

for seed in SEEDS:
    print(f"--> seed {seed}...")
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    spline_preprocessor_c3 = ColumnTransformer(
        transformers=[
            ('spline', make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), spline_cols),
            ('other_num', StandardScaler(), [c for c in other_cols_c3 if c in nn_num_cols_c3]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [c for c in other_cols_c3 if c in nn_cat_cols_c3])
        ]
    )
    oof_spline_s = np.zeros(n_tr)
    test_spline_s = np.zeros(n_te)
    oof_ebm_s = np.zeros(n_tr)
    test_ebm_s = np.zeros(n_te)
    oof_cb_s = np.zeros(n_tr)
    test_cb_s = np.zeros(n_te)

    for trn_idx, val_idx in skf_seed.split(X_core3, y):
        X_tr_raw, y_tr = X_core3.iloc[trn_idx], y[trn_idx]
        X_va_raw, y_va = X_core3.iloc[val_idx], y[val_idx]

        X_tr_sp = spline_preprocessor_c3.fit_transform(X_tr_raw)
        X_va_sp = spline_preprocessor_c3.transform(X_va_raw)
        X_te_sp = spline_preprocessor_c3.transform(X_test_core3)
        sm_ = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        sm_.fit(X_tr_sp, y_tr)
        oof_spline_s[val_idx] = sm_.predict_proba(X_va_sp)[:, 1]
        test_spline_s += sm_.predict_proba(X_te_sp)[:, 1] / 5

        ebm = ExplainableBoostingClassifier(max_bins=128, interactions=10, outer_bags=8, inner_bags=0, random_state=seed)
        ebm.fit(X_tr_raw[ebm_cols_c3], y_tr)
        oof_ebm_s[val_idx] = ebm.predict_proba(X_va_raw[ebm_cols_c3])[:, 1]
        test_ebm_s += ebm.predict_proba(X_test_core3[ebm_cols_c3])[:, 1] / 5

        X_tr_cb, X_va_cb, X_te_cb = X_tr_raw.copy(), X_va_raw.copy(), X_test_core3.copy()
        for col in cat_features_c3:
            X_tr_cb[col] = X_tr_cb[col].astype(str)
            X_va_cb[col] = X_va_cb[col].astype(str)
            X_te_cb[col] = X_te_cb[col].astype(str)
        cb = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, loss_function='Logloss',
                                 eval_metric='Logloss', random_seed=seed, verbose=False)
        cb.fit(X_tr_cb, y_tr, cat_features=cat_features_c3, eval_set=(X_va_cb, y_va), early_stopping_rounds=30)
        oof_cb_s[val_idx] = cb.predict_proba(X_va_cb)[:, 1]
        test_cb_s += cb.predict_proba(X_te_cb)[:, 1] / 5

    oof_spline_runs.append(oof_spline_s); test_spline_runs.append(test_spline_s)
    oof_ebm_runs.append(oof_ebm_s); test_ebm_runs.append(test_ebm_s)
    oof_cb_runs.append(oof_cb_s); test_cb_runs.append(test_cb_s)

oof_spline_c3 = np.clip(np.mean(oof_spline_runs, axis=0), 0.005, 0.995)
test_spline_c3 = np.mean(test_spline_runs, axis=0)
oof_ebm_c3 = np.clip(np.mean(oof_ebm_runs, axis=0), 0.005, 0.995)
test_ebm_c3 = np.mean(test_ebm_runs, axis=0)
oof_cb_c3 = np.clip(np.mean(oof_cb_runs, axis=0), 0.005, 0.995)
test_cb_c3 = np.mean(test_cb_runs, axis=0)

print(f"\nSpline GAM (3-seed bagged) OOF Log Loss: {log_loss(y, oof_spline_c3):.5f}  (expect ~0.34991)")
print(f"EBM (3-seed bagged) OOF Log Loss:        {log_loss(y, oof_ebm_c3):.5f}  (expect ~0.34928)")
print(f"CatBoost (3-seed bagged) OOF Log Loss:    {log_loss(y, oof_cb_c3):.5f}  (expect ~0.34965)")

S_train_core3 = np.column_stack([oof_spline_c3, oof_ebm_c3, oof_cb_c3])
S_test_core3 = np.column_stack([test_spline_c3, test_ebm_c3, test_cb_c3])


def fit_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))
    k = S.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=202)
honest_oof_core3 = np.zeros(len(y))
for trn_idx, val_idx in skf_meta.split(S_train_core3, y):
    w = fit_weights(S_train_core3[trn_idx], y[trn_idx])
    honest_oof_core3[val_idx] = np.dot(S_train_core3[val_idx], w)
honest_oof_core3 = np.clip(honest_oof_core3, 0.005, 0.995)

final_weights_core3 = fit_weights(S_train_core3, y)
test_preds_core3 = np.clip(np.dot(S_test_core3, final_weights_core3), 0.005, 0.995)

print("\n" + "=" * 55)
print("   CORE-3 HONEST STACK -- VERIFICATION RESULT   ")
print("=" * 55)
print(f"Weights -> Spline: {final_weights_core3[0]:.3f} | EBM: {final_weights_core3[1]:.3f} | CatBoost: {final_weights_core3[2]:.3f}")
print(f"Honest (nested-CV) OOF Log Loss: {log_loss(y, honest_oof_core3):.5f}  (target: 0.34855)")
print(f"ROC-AUC (honest):                {roc_auc_score(y, honest_oof_core3):.5f}")
print(f"Brier (honest):                  {brier_score_loss(y, honest_oof_core3):.5f}")
print("=" * 55)

sub = pd.DataFrame({'patient_id': test_fe['patient_id'], 'readmitted_30d': test_preds_core3})
out_path = ROOT / "legacy" / "submissions" / "submission_13_core3_honest.csv"
sub.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
