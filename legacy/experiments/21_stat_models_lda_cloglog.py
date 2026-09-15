"""
Cell 21 equivalent — Shrinkage LDA & Cloglog GLM, ported so their OOF
predictions are actually saved (the notebook trains them and prints their
solo scores but never feeds them into the final stack in cell 19).

The column-drop at the top of cell 21 ('unsupported_frail_home', etc.) is
a no-op against the current feature set (verified: none of those columns
exist in train_fe -- see the pipeline's chat history), so it's omitted
here; X is used as saved by 11_base_lgbm_rf_catboost.py.

These two are structurally the most different models from the tree/GBDT
family already in the stack: LDA assumes a linear discriminant boundary
under shared-covariance Gaussian class assumptions (Ledoit-Wolf shrinkage
handles the near-singular covariance from one-hot categoricals), and the
cloglog GLM is a linear-in-the-logit(-ish) hazard model, not a tree at
all. Testing whether either adds real complementary signal to the
EBM+CatBoost+Spline core.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ARTIFACTS = Path(__file__).resolve().parents[2] / "pipeline" / "artifacts"

with open(ARTIFACTS / "X.pkl", "rb") as f:
    X = pickle.load(f)
with open(ARTIFACTS / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stat_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
stat_cat_cols = [c for c in X.columns if c not in stat_num_cols]

stat_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), stat_num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), stat_cat_cols)
    ]
)

oof_lda = np.zeros(len(X))
test_lda = np.zeros(len(X_test))
oof_cloglog = np.zeros(len(X))
test_cloglog = np.zeros(len(X_test))

print("--> Training 5-Fold Shrinkage LDA & Cloglog GLM...")

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr_raw, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va_raw, y_va = X.iloc[val_idx], y[val_idx]

    X_tr_p = stat_preprocessor.fit_transform(X_tr_raw)
    X_va_p = stat_preprocessor.transform(X_va_raw)
    X_te_p = stat_preprocessor.transform(X_test)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr_p, y_tr)
    oof_lda[val_idx] = lda.predict_proba(X_va_p)[:, 1]
    test_lda += lda.predict_proba(X_te_p)[:, 1] / 5

    X_tr_const = sm.add_constant(X_tr_p, has_constant='add')
    X_va_const = sm.add_constant(X_va_p, has_constant='add')
    X_te_const = sm.add_constant(X_te_p, has_constant='add')

    glm_cloglog = sm.GLM(y_tr, X_tr_const, family=sm.families.Binomial(link=sm.families.links.CLogLog()))
    try:
        glm_res = glm_cloglog.fit_regularized(alpha=0.01, L1_wt=0.1, maxiter=200)
    except Exception:
        glm_res = glm_cloglog.fit()

    oof_cloglog[val_idx] = glm_res.predict(X_va_const)
    test_cloglog += glm_res.predict(X_te_const) / 5

oof_lda = np.clip(oof_lda, 0.005, 0.995)
oof_cloglog = np.clip(oof_cloglog, 0.005, 0.995)
test_lda = np.clip(test_lda, 0.005, 0.995)
test_cloglog = np.clip(test_cloglog, 0.005, 0.995)

print("\n" + "=" * 55)
print(f"1. Shrinkage LDA OOF Log Loss:  {log_loss(y, oof_lda):.5f} | AUC: {roc_auc_score(y, oof_lda):.5f}")
print(f"2. Cloglog GLM OOF Log Loss:    {log_loss(y, oof_cloglog):.5f} | AUC: {roc_auc_score(y, oof_cloglog):.5f}")
print("=" * 55)

np.save(ARTIFACTS / "oof_lda.npy", oof_lda)
np.save(ARTIFACTS / "test_preds_lda.npy", test_lda)
np.save(ARTIFACTS / "oof_cloglog.npy", oof_cloglog)
np.save(ARTIFACTS / "test_preds_cloglog.npy", test_cloglog)
print(f"Saved LDA/cloglog OOF + test preds to {ARTIFACTS}")
