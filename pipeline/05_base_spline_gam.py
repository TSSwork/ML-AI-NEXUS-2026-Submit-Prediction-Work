"""
Cell 18 equivalent — Spline GAM (natural B-splines + logistic regression) OOF.

Faithful port of basic_check.ipynb cell 18. That cell reads `nn_num_cols` /
`nn_cat_cols`, which in the notebook are computed in cell 17 (the MLP cell)
as plain `X.select_dtypes(...)` calls — we recompute them here directly
instead of training the (unused-downstream) MLP, since they are just
column-type lists, not fitted state.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

with open(ARTIFACTS / "X.pkl", "rb") as f:
    X = pickle.load(f)
with open(ARTIFACTS / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# same column-type split cell 17 computed before cell 18 used it
nn_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
nn_cat_cols = X.select_dtypes(include=['category', 'object', 'string', 'str']).columns.tolist()

print("--> Training 5-Fold Spline GAM (Natural B-Splines + Logistic Regression)...")

oof_spline = np.zeros(len(X))
test_preds_spline = np.zeros(len(X_test))

spline_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'length_of_stay_days'
]
other_cols = [c for c in X.columns if c not in spline_cols and not c.endswith('_isna')]

spline_preprocessor = ColumnTransformer(
    transformers=[
        ('spline', make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), spline_cols),
        ('other_num', StandardScaler(), [c for c in other_cols if c in nn_num_cols]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [c for c in other_cols if c in nn_cat_cols])
    ]
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr_raw, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va_raw, y_va = X.iloc[val_idx], y[val_idx]

    X_tr_sp = spline_preprocessor.fit_transform(X_tr_raw)
    X_va_sp = spline_preprocessor.transform(X_va_raw)
    X_te_sp = spline_preprocessor.transform(X_test)

    spline_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    spline_model.fit(X_tr_sp, y_tr)

    oof_spline[val_idx] = spline_model.predict_proba(X_va_sp)[:, 1]
    test_preds_spline += spline_model.predict_proba(X_te_sp)[:, 1] / 5

oof_spline = np.clip(oof_spline, 0.005, 0.995)
test_preds_spline = np.clip(test_preds_spline, 0.005, 0.995)

spline_loss = log_loss(y, oof_spline)
spline_auc = roc_auc_score(y, oof_spline)

print("\n" + "=" * 45)
print(f"Spline GAM OOF Log Loss:  {spline_loss:.5f}  (Linear Logistic: 0.35261)")
print(f"Spline GAM ROC-AUC:       {spline_auc:.5f}  (Linear Logistic: 0.68529)")
print("=" * 45)

np.save(ARTIFACTS / "oof_spline.npy", oof_spline)
np.save(ARTIFACTS / "test_preds_spline.npy", test_preds_spline)
print(f"Saved Spline GAM OOF + test preds to {ARTIFACTS}")
