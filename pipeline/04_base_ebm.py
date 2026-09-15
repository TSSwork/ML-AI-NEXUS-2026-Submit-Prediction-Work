"""
Cell 12 equivalent — Explainable Boosting Machine (EBM) OOF.

Faithful port of basic_check.ipynb cell 12. Reuses the exact X/X_test/y
persisted by 11_base_lgbm_rf_catboost.py so the feature matrix is
identical to what the notebook's cell 12 saw (it ran right after cell 11
in-kernel).
"""
import pickle
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

with open(ARTIFACTS / "X.pkl", "rb") as f:
    X = pickle.load(f)
with open(ARTIFACTS / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--> Training 5-Fold Explainable Boosting Machine (EBM)...")

oof_ebm = np.zeros(len(X))
test_preds_ebm = np.zeros(len(X_test))

# EBM handles categorical and numerical columns natively
ebm_cols = [c for c in X.columns if not c.endswith('_isna')]  # Drop raw flags to keep curves clean

for fold, (trn_idx, val_idx) in enumerate(skf.split(X[ebm_cols], y), 1):
    X_tr, y_tr = X[ebm_cols].iloc[trn_idx], y[trn_idx]
    X_va, y_va = X[ebm_cols].iloc[val_idx], y[val_idx]

    ebm = ExplainableBoostingClassifier(
        max_bins=128,
        interactions=10,
        outer_bags=8,
        inner_bags=0,
        random_state=42
    )
    ebm.fit(X_tr, y_tr)

    oof_ebm[val_idx] = ebm.predict_proba(X_va)[:, 1]
    test_preds_ebm += ebm.predict_proba(X_test[ebm_cols])[:, 1] / 5

oof_ebm = np.clip(oof_ebm, 0.005, 0.995)
test_preds_ebm = np.clip(test_preds_ebm, 0.005, 0.995)

ebm_loss = log_loss(y, oof_ebm)
ebm_auc = roc_auc_score(y, oof_ebm)

print("\n" + "=" * 45)
print(f"EBM OOF Log Loss:  {ebm_loss:.5f}")
print(f"EBM ROC-AUC:       {ebm_auc:.5f}")
print("=" * 45)

np.save(ARTIFACTS / "oof_ebm.npy", oof_ebm)
np.save(ARTIFACTS / "test_preds_ebm.npy", test_preds_ebm)
print(f"Saved EBM OOF + test preds to {ARTIFACTS}")
