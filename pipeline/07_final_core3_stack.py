"""
NEW — the actual, verified best stack found by 25_ablation_table.py.

25_ablation_table.py's honest (nested-CV) leave-one-out pass showed that
dropping LightGBM from the 4-way stack IMPROVES the honest score
(0.34912 -> 0.34855, a real -0.00057), while dropping any of the other
three makes it worse. LightGBM is the weakest solo model (0.35124) and,
once the other three are already in the blend, it isn't adding
complementary signal -- the SLSQP optimizer is giving it a small positive
weight (~0.05-0.07) that's fitting fold-specific noise rather than a real
pattern, and nested CV catches that same-data fitting doesn't.

This drops LightGBM, keeps the seed-bagged {Spline GAM, EBM, CatBoost},
and produces the final submission from the honestly-best configuration
found so far in this pipeline.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ROOT = Path(__file__).resolve().parents[1]

with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

MODEL_NAMES = ["Spline GAM", "EBM", "CatBoost"]  # LightGBM dropped -- see docstring
oof = np.column_stack([
    np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    np.load(ARTIFACTS / "oof_cb_bagged.npy"),
])
test_p = np.column_stack([
    np.load(ARTIFACTS / "test_preds_spline_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_ebm_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_cb_bagged.npy"),
])

N_META_FOLDS = 5
META_SEED = 202


def fit_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))

    k = S.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


skf_meta = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)
honest_oof = np.zeros(len(y))
fold_weights = []
for trn_idx, val_idx in skf_meta.split(oof, y):
    w = fit_weights(oof[trn_idx], y[trn_idx])
    fold_weights.append(w)
    honest_oof[val_idx] = np.dot(oof[val_idx], w)

honest_oof = np.clip(honest_oof, 0.005, 0.995)
fold_weights = np.array(fold_weights)
honest_loss = log_loss(y, honest_oof)
honest_auc = roc_auc_score(y, honest_oof)
honest_brier = brier_score_loss(y, honest_oof)

final_weights = fit_weights(oof, y)
test_super = np.clip(np.dot(test_p, final_weights), 0.005, 0.995)

print("=" * 62)
print("  FINAL RECOMMENDED STACK: Spline GAM + EBM + CatBoost")
print("  (LightGBM dropped -- ablation showed it hurts the honest score)")
print("=" * 62)
print("Final production weights (fit on 100% of bagged OOF):")
for name, w in zip(MODEL_NAMES, final_weights):
    print(f"  {name:<12s} {w:.3f}")

print("\nPer-fold nested weights:")
for i, name in enumerate(MODEL_NAMES):
    fold_str = ", ".join(f"{w:.3f}" for w in fold_weights[:, i])
    print(f"  {name:<12s} folds: [{fold_str}]")

print("\n" + "-" * 62)
print(f"Honest OOF Log Loss:  {honest_loss:.5f}  (previous 4-model honest: 0.34912)")
print(f"Honest OOF ROC-AUC:   {honest_auc:.5f}")
print(f"Honest OOF Brier:     {honest_brier:.5f}")
print("-" * 62)

sub = pd.DataFrame({'patient_id': test_fe['patient_id'], 'readmitted_30d': test_super})
out_path = ROOT / "submissions" / "submission_13_core3_honest.csv"
sub.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
