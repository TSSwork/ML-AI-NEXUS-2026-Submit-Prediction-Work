"""
NEW — combines both fixes into the final submission:
  - level-1 predictions are seed-bagged (23_seed_bagged_base_learners.py)
  - level-2 meta-weights are scored with nested CV, not same-data (22_honest_nested_stacking.py)

This is the most trustworthy local estimate this pipeline can produce, and
the one most likely to track the real leaderboard score. Compares against
both the original notebook's single-seed/leaky number and the
honest-but-not-bagged number from script 22, so the size of each fix's
contribution is visible.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parents[2] / "pipeline" / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

MODEL_NAMES = ["Spline GAM", "EBM", "CatBoost", "LightGBM"]
oof_bagged = np.column_stack([
    np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    np.load(ARTIFACTS / "oof_cb_bagged.npy"),
    np.load(ARTIFACTS / "oof_lgb_bagged.npy"),
])
test_bagged = np.column_stack([
    np.load(ARTIFACTS / "test_preds_spline_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_ebm_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_cb_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_lgb_bagged.npy"),
])

N_META_FOLDS = 5
META_SEED = 202


def fit_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))

    init = np.full(S.shape[1], 1.0 / S.shape[1])
    bounds = [(0, 1)] * S.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


print(f"--> Nested {N_META_FOLDS}-fold CV over the SEED-BAGGED level-1 OOF matrix...")

skf_meta = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)
honest_oof = np.zeros(len(y))
fold_weights = []

for outer_trn_idx, outer_val_idx in skf_meta.split(oof_bagged, y):
    w = fit_weights(oof_bagged[outer_trn_idx], y[outer_trn_idx])
    fold_weights.append(w)
    honest_oof[outer_val_idx] = np.dot(oof_bagged[outer_val_idx], w)

honest_oof = np.clip(honest_oof, 0.005, 0.995)
fold_weights = np.array(fold_weights)

honest_loss = log_loss(y, honest_oof)
honest_auc = roc_auc_score(y, honest_oof)
honest_brier = brier_score_loss(y, honest_oof)

final_weights = fit_weights(oof_bagged, y)
test_super = np.clip(np.dot(test_bagged, final_weights), 0.005, 0.995)

print("\n" + "=" * 62)
print("     FINAL: SEED-BAGGED + HONEST NESTED-CV SUPER-LEARNER      ")
print("=" * 62)
print("Final production weights (fit on 100% of bagged OOF):")
for name, w in zip(MODEL_NAMES, final_weights):
    print(f"  {name:<12s} {w:.3f}")

print("\n" + "-" * 62)
single_seed_paths = {
    'Spline GAM': "oof_spline.npy", 'EBM': "oof_ebm.npy",
    'CatBoost': "oof_cb.npy", 'LightGBM': "oof_lgb.npy",
}
if all((ARTIFACTS / p).exists() for p in single_seed_paths.values()):
    oof_single_seed = np.column_stack([np.load(ARTIFACTS / p) for p in single_seed_paths.values()])
    w_single = fit_weights(oof_single_seed, y)
    original_leaky = log_loss(y, np.clip(np.dot(oof_single_seed, w_single), 0.005, 0.995))
    print(f"Original notebook style (1 seed, same-data weights) : {original_leaky:.5f}")

honest_single_seed_path = ARTIFACTS / "honest_oof_super.npy"
if honest_single_seed_path.exists():
    honest_single_seed = log_loss(y, np.load(honest_single_seed_path))
    print(f"Honest nested-CV, 1 seed  (script 22)                : {honest_single_seed:.5f}")

print(f"Honest nested-CV, seed-bagged (this script)         : {honest_loss:.5f}")
print("-" * 62)
print(f"Final Log Loss (honest):  {honest_loss:.5f}")
print(f"Final ROC-AUC:            {honest_auc:.5f}")
print(f"Final Brier Score:        {honest_brier:.5f}")
print("=" * 62)

sub_final = pd.DataFrame({
    'patient_id': test_fe['patient_id'],
    'readmitted_30d': test_super
})
out_path = ROOT / "legacy" / "submissions" / "submission_10_bagged_honest_super_learner.csv"
sub_final.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
