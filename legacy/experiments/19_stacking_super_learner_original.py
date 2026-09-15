"""
Cell 19 equivalent — the ORIGINAL Level-2 stacking super-learner, reproduced
as-is for reference/comparison.

This intentionally keeps the same flaw as the notebook: the SLSQP weights
are fit on the full OOF matrix and then scored on those same OOF rows, so
the printed log loss below is optimistic (the meta-learner has effectively
seen its own "test" set once via the weight search). Kept here unchanged
so you can diff it against 22_honest_nested_stacking.py, which fixes this.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ARTIFACTS = Path(__file__).resolve().parents[2] / "pipeline" / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)

y = np.load(ARTIFACTS / "y.npy")
oof_spline = np.load(ARTIFACTS / "oof_spline.npy")
test_preds_spline = np.load(ARTIFACTS / "test_preds_spline.npy")
oof_ebm = np.load(ARTIFACTS / "oof_ebm.npy")
test_preds_ebm = np.load(ARTIFACTS / "test_preds_ebm.npy")
oof_cb = np.load(ARTIFACTS / "oof_cb.npy")
test_preds_cb = np.load(ARTIFACTS / "test_preds_cb.npy")
oof_lgb = np.load(ARTIFACTS / "oof_lgb.npy")
test_preds_lgb = np.load(ARTIFACTS / "test_preds_lgb.npy")

print("--> Optimizing Level-2 Stacking Super-Learner across 4 Model Families...")

S_train = np.column_stack([oof_spline, oof_ebm, oof_cb, oof_lgb])
S_test = np.column_stack([test_preds_spline, test_preds_ebm, test_preds_cb, test_preds_lgb])


def meta_loss(w):
    pred = np.dot(S_train, w)
    return log_loss(y, np.clip(pred, 0.005, 0.995))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

opt_res = minimize(meta_loss, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
w_spline, w_ebm, w_cb, w_lgb = opt_res.x

print("\n" + "=" * 50)
print("     SUPER-LEARNER META-WEIGHTS (ORIGINAL / LEAKY)  ")
print("=" * 50)
print(f"  1. Spline GAM (Parametric):        {w_spline:.3f}")
print(f"  2. EBM (Glass-Box GAM^2):          {w_ebm:.3f}")
print(f"  3. CatBoost (Symmetric Trees):     {w_cb:.3f}")
print(f"  4. LightGBM (Leaf-wise GBDT):      {w_lgb:.3f}")
print("=" * 50)

oof_super = np.clip(np.dot(S_train, opt_res.x), 0.005, 0.995)
test_super = np.clip(np.dot(S_test, opt_res.x), 0.005, 0.995)

super_loss = log_loss(y, oof_super)
super_auc = roc_auc_score(y, oof_super)
super_brier = brier_score_loss(y, oof_super)

print(f"--> Stacking Super-Learner OOF Log Loss (SAME-DATA, OPTIMISTIC): {super_loss:.5f}")
print(f"--> Stacking Super-Learner ROC-AUC:                              {super_auc:.5f}")
print(f"--> Stacking Super-Learner Brier Score:                          {super_brier:.5f}")
print("=" * 50)
print("NOTE: these weights were fit AND scored on the same OOF rows -- see")
print("22_honest_nested_stacking.py for a nested-CV score that isn't self-fit.")

sub_super = pd.DataFrame({
    'patient_id': test_fe['patient_id'],
    'readmitted_30d': test_super
})
sub_super.to_csv(ROOT / "legacy" / "submissions" / "submission_06_super_learner.csv", index=False)
print(f"Saved to {ROOT / 'outputs' / 'submission_06_super_learner.csv'}")
