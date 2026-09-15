"""
NEW — fixes the score-leakage in cell 19 / 19_stacking_super_learner_original.py.

Problem: the original super-learner fits its 4 SLSQP blend weights on the
full 7000-row OOF matrix and then reports log loss on those SAME rows.
That number is optimistic -- the meta-learner has effectively "seen" every
row it's scored against once, via the weight search. It's a likely part of
why the local OOF score (0.349) sits meaningfully below the public
leaderboard (0.334-0.337): the local number is a bit rosier than reality.

Fix: nested CV over the level-1 OOF matrix. Split the OOF rows into K
outer folds; for each fold, fit the SLSQP weights on the OTHER folds only,
then score the held-out fold. Concatenating the held-out predictions
across folds gives an honest OOF score for the meta-learner itself, with
no row ever scored by weights that saw it.

The weights used for the actual test-set submission are still fit on 100%
of the OOF data (standard practice -- nested CV is for the honest SCORE
estimate; production weights should use every available row). We report
both numbers so the gap between them is visible rather than hidden.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parents[1] / "pipeline" / "artifacts"
ROOT = Path(__file__).resolve().parents[1]

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

MODEL_NAMES = ["Spline GAM", "EBM", "CatBoost", "LightGBM"]
S_train = np.column_stack([oof_spline, oof_ebm, oof_cb, oof_lgb])
S_test = np.column_stack([test_preds_spline, test_preds_ebm, test_preds_cb, test_preds_lgb])

N_META_FOLDS = 5
# Different seed than the base-learner CV (42) so the outer split isn't a
# trivial re-use of the same fold boundaries.
META_SEED = 202


def fit_weights(S, targets):
    def meta_loss(w):
        pred = np.dot(S, w)
        return log_loss(targets, np.clip(pred, 0.005, 0.995))

    init = np.full(S.shape[1], 1.0 / S.shape[1])
    bounds = [(0, 1)] * S.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    res = minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x


print(f"--> Nested {N_META_FOLDS}-fold CV over the level-1 OOF matrix (honest meta-score)...")

skf_meta = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)
honest_oof_super = np.zeros(len(y))
fold_weights = []

for fold, (outer_trn_idx, outer_val_idx) in enumerate(skf_meta.split(S_train, y), 1):
    w = fit_weights(S_train[outer_trn_idx], y[outer_trn_idx])
    fold_weights.append(w)
    honest_oof_super[outer_val_idx] = np.dot(S_train[outer_val_idx], w)

honest_oof_super = np.clip(honest_oof_super, 0.005, 0.995)
fold_weights = np.array(fold_weights)

honest_loss = log_loss(y, honest_oof_super)
honest_auc = roc_auc_score(y, honest_oof_super)
honest_brier = brier_score_loss(y, honest_oof_super)

# Weights actually shipped for the test set: fit on ALL OOF data (no
# leakage concern here -- this is just the final production fit).
final_weights = fit_weights(S_train, y)
leaky_oof_super = np.clip(np.dot(S_train, final_weights), 0.005, 0.995)
leaky_loss = log_loss(y, leaky_oof_super)

test_super = np.clip(np.dot(S_test, final_weights), 0.005, 0.995)

print("\n" + "=" * 58)
print("        HONEST (NESTED-CV) SUPER-LEARNER WEIGHTS          ")
print("=" * 58)
print("Per-fold meta-weights (fit on 4/5 of OOF, held out from its own score):")
for i, name in enumerate(MODEL_NAMES):
    fold_str = ", ".join(f"{w:.3f}" for w in fold_weights[:, i])
    print(f"  {name:<12s} folds: [{fold_str}]  mean={fold_weights[:, i].mean():.3f}")

print("\nFinal production weights (fit on 100% of OOF, used for test preds):")
for name, w in zip(MODEL_NAMES, final_weights):
    print(f"  {name:<12s} {w:.3f}")

print("\n" + "-" * 58)
print(f"Same-data (leaky) OOF Log Loss   [cell 19 style]: {leaky_loss:.5f}")
print(f"Nested-CV (honest) OOF Log Loss                 : {honest_loss:.5f}")
print(f"Optimism gap (leaky - honest)                   : {leaky_loss - honest_loss:+.5f}")
print("-" * 58)
print(f"Honest OOF ROC-AUC:      {honest_auc:.5f}")
print(f"Honest OOF Brier Score:  {honest_brier:.5f}")
print("=" * 58)

sub_super = pd.DataFrame({
    'patient_id': test_fe['patient_id'],
    'readmitted_30d': test_super
})
out_path = ROOT / "legacy" / "submissions" / "submission_09_honest_stacked_super_learner.csv"
sub_super.to_csv(out_path, index=False)
print(f"Saved to {out_path}")

np.save(ARTIFACTS / "honest_oof_super.npy", honest_oof_super)
np.save(ARTIFACTS / "final_weights.npy", final_weights)
