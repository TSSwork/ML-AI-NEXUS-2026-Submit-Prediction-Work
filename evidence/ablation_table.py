"""
NEW — answers "did we check how the model improved on each treatment?"

The notebook never builds a controlled ablation: it reports each model's
own OOF loss and the final 4-way stack's OOF loss, but never asks "how
much does adding model N to the stack actually buy, once scored honestly
(nested CV, not same-data weights)?" This does exactly that, using the
seed-bagged level-1 predictions from 23_seed_bagged_base_learners.py.

Models are added one at a time, in order of standalone strength (best
solo model first), and at every step the honest (nested 5-fold CV, as in
22_honest_nested_stacking.py) log loss of the SLSQP convex blend over the
models included so far is reported, plus the marginal delta from adding
the newest model.
"""
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parents[1] / "pipeline" / "artifacts"

with open(ARTIFACTS / "train_fe.pkl", "rb") as f:
    train_fe = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

MODEL_NAMES = ["Spline GAM", "EBM", "CatBoost", "LightGBM"]
oof = {
    "Spline GAM": np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    "EBM": np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    "CatBoost": np.load(ARTIFACTS / "oof_cb_bagged.npy"),
    "LightGBM": np.load(ARTIFACTS / "oof_lgb_bagged.npy"),
}

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


def honest_cv_loss(names):
    S = np.column_stack([oof[n] for n in names])
    if S.shape[1] == 1:
        # nothing to weight-fit -- honest score is just the model's own OOF
        return log_loss(y, np.clip(S[:, 0], 0.005, 0.995))
    skf = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)
    honest = np.zeros(len(y))
    for trn_idx, val_idx in skf.split(S, y):
        w = fit_weights(S[trn_idx], y[trn_idx])
        honest[val_idx] = np.dot(S[val_idx], w)
    return log_loss(y, np.clip(honest, 0.005, 0.995))


print("=" * 62)
print("  STEP 1 -- standalone strength of each seed-bagged base model")
print("=" * 62)
solo_scores = {}
for name in MODEL_NAMES:
    p = np.clip(oof[name], 0.005, 0.995)
    ll = log_loss(y, p)
    solo_scores[name] = ll
    print(f"  {name:<12s} solo honest LogLoss = {ll:.5f} | AUC = {roc_auc_score(y, p):.5f}")

ranked = sorted(MODEL_NAMES, key=lambda n: solo_scores[n])
print(f"\nRanked best->worst solo: {ranked}")

print("\n" + "=" * 62)
print("  STEP 2 -- incremental stacking, honest nested-CV at each step")
print("=" * 62)
included = []
prev_loss = None
for name in ranked:
    included.append(name)
    ll = honest_cv_loss(included)
    delta = "" if prev_loss is None else f"  (delta = {ll - prev_loss:+.5f})"
    print(f"  + {name:<12s} -> honest stack of {included}: {ll:.5f}{delta}")
    prev_loss = ll

print("\n" + "=" * 62)
print("  STEP 3 -- is each of the 4 models pulling its weight, or is it")
print("            just noise the optimizer is fitting to?")
print("=" * 62)
full = MODEL_NAMES
full_loss = honest_cv_loss(full)
print(f"Full 4-model honest stack: {full_loss:.5f}")
for name in MODEL_NAMES:
    without = [n for n in MODEL_NAMES if n != name]
    without_loss = honest_cv_loss(without)
    print(f"  Drop {name:<12s} -> honest stack of {without}: {without_loss:.5f}  "
          f"(cost of removing it = {without_loss - full_loss:+.5f})")

best_solo_name = ranked[0]
best_solo_loss = solo_scores[best_solo_name]
print("\n" + "-" * 62)
print(f"Best single model alone ({best_solo_name}):        {best_solo_loss:.5f}")
print(f"Full 4-model stack (honest, seed-bagged):        {full_loss:.5f}")
print(f"What the ENTIRE stacking machinery buys over just")
print(f"using the single best model:                      {best_solo_loss - full_loss:+.5f}")
print("-" * 62)
