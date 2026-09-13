"""
NEW — extends 28_swap_lgbm_for_diverse_model.py with the two genuinely
non-tree models available (21_stat_models_lda_cloglog.py): Shrinkage LDA
and a Cloglog GLM. Both are weaker solo than the core three, but a weak
solo model can still add value to a stack if its errors are
uncorrelated with the core's -- that's the actual test of "different
category" being useful, not just solo strength.
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
y = np.load(ARTIFACTS / "y.npy")

base = {
    "Spline GAM": np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    "EBM": np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    "CatBoost": np.load(ARTIFACTS / "oof_cb_bagged.npy"),
    "LightGBM": np.load(ARTIFACTS / "oof_lgb_bagged.npy"),
    "RandomForest": np.load(ARTIFACTS / "oof_rf.npy"),
    "LDA": np.load(ARTIFACTS / "oof_lda.npy"),
    "Cloglog GLM": np.load(ARTIFACTS / "oof_cloglog.npy"),
}

N_META_FOLDS, META_SEED = 5, 202


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


print("Solo scores:")
for name in ["LDA", "Cloglog GLM"]:
    print(f"  {name:<12s} {log_loss(y, np.clip(base[name], 0.005, 0.995)):.5f}")

core = ["Spline GAM", "EBM", "CatBoost"]
configs = {
    "Core 3 only (best so far)": core,
    "Core 3 + LDA": core + ["LDA"],
    "Core 3 + Cloglog GLM": core + ["Cloglog GLM"],
    "Core 3 + LDA + Cloglog GLM": core + ["LDA", "Cloglog GLM"],
    "Core 3 + LDA + Cloglog + RF": core + ["LDA", "Cloglog GLM", "RandomForest"],
    "Everything (7-way)": list(base.keys()),
}

print()
for label, names in configs.items():
    print(f"  {label:<32s} honest LogLoss = {honest_cv_loss(names):.5f}")
