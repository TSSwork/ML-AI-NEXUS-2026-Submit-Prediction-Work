"""
NEW — tests the user's hypothesis: LightGBM and CatBoost are the same
MODEL CATEGORY (both gradient-boosted decision trees -- leaf-wise vs.
symmetric/oblivious growth is a detail, not a different inductive bias),
so swapping LightGBM's stack slot for a genuinely different kind of
model may do more than either keeping it (0.34912) or just dropping it
(0.34855, see 27_final_recommended_stack.py).

Candidate: RandomForestClassifier from cell 11 / 11_base_lgbm_rf_catboost.py
-- it was trained the whole time but never actually included in the final
4-model stack. Bagged (variance-reduction, decorrelated via bootstrap +
feature subsampling) rather than boosted (sequential bias-reduction) --
a different bias/variance regime than both LightGBM and CatBoost, even
though it's still tree-based.

NOTE: RF here is single-seed (42) only, unlike the other three which are
3-seed bagged (23_seed_bagged_base_learners.py) -- so this is a fair
"is it worth bagging RF too" screen, not yet an apples-to-apples final
number. If RF looks promising, seed-bag it the same way before trusting
the comparison fully.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

y = np.load(ARTIFACTS / "y.npy")

base = {
    "Spline GAM": np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    "EBM": np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    "CatBoost": np.load(ARTIFACTS / "oof_cb_bagged.npy"),
    "LightGBM": np.load(ARTIFACTS / "oof_lgb_bagged.npy"),
    "RandomForest (1 seed)": np.load(ARTIFACTS / "oof_rf.npy"),
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
    S = np.column_stack([base[n] for n in names])
    skf = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)
    honest = np.zeros(len(y))
    for trn_idx, val_idx in skf.split(S, y):
        w = fit_weights(S[trn_idx], y[trn_idx])
        honest[val_idx] = np.dot(S[val_idx], w)
    return log_loss(y, np.clip(honest, 0.005, 0.995))


print("=" * 66)
print("  Does RandomForest (bagging) beat LightGBM (boosting) as the 4th slot?")
print("=" * 66)
print(f"RandomForest solo (1 seed): {log_loss(y, np.clip(base['RandomForest (1 seed)'], 0.005, 0.995)):.5f}")
print(f"LightGBM solo (3-seed bag): {log_loss(y, np.clip(base['LightGBM'], 0.005, 0.995)):.5f}")
print()

core = ["Spline GAM", "EBM", "CatBoost"]
configs = {
    "Core 3 only (no 4th tree model)": core,
    "Core 3 + LightGBM (boosting, current)": core + ["LightGBM"],
    "Core 3 + RandomForest (bagging)": core + ["RandomForest (1 seed)"],
    "Core 3 + LightGBM + RandomForest (5-way)": core + ["LightGBM", "RandomForest (1 seed)"],
}

for label, names in configs.items():
    loss = honest_cv_loss(names)
    print(f"  {label:<42s} honest LogLoss = {loss:.5f}")
