"""
NEW — fixes a real bug found in cell 19 / 19_stacking_super_learner_original.py.

The original cell computes `S_train_clipped` / `S_test_clipped` with a
comment "Convert probabilities to log-odds (logits) for stable linear
meta-modeling" -- but then never actually applies a logit transform, and
`meta_loss` / the final blend both operate on the raw, un-transformed
`S_train` / `S_test`. The clipped arrays are computed and never read
again. So today's "super-learner" is a plain convex combination of raw
probabilities (a linear opinion pool), not the log-odds blend the comment
describes.

This matters beyond tidiness: linear pooling of several individually
well-calibrated probability forecasts is a known-suboptimal combination
rule for log loss -- averaging probabilities directly tends to produce an
UNDER-confident (too close to the base rate) pooled forecast, because the
pool's variance shrinks faster than its bias does. Combining in logit
space (or via a logistic-regression meta-learner over the logits) avoids
that shrinkage and is the textbook-correct way to stack probabilistic
classifiers. This script builds and honestly (nested 5-fold CV) scores
three variants so you can see, with real numbers, whether that shrinkage
is actually costing you anything here:

  A. Same-space convex blend on RAW probabilities (what cell 19 actually
     does today, despite the comment) -- baseline for this comparison.
  B. Convex blend in LOGIT space (what the comment in cell 19 claims to
     do): weights >=0, sum=1, applied to logit(p), blended back through
     sigmoid.
  C. Logistic-regression meta-learner over the 4 logits (intercept + 4
     coefficients, L2-regularized) -- lets the meta-learner also correct
     systematic over/under-confidence per base model, not just re-weight.

Uses the seed-bagged level-1 predictions from 23_seed_bagged_base_learners.py.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path(__file__).resolve().parents[2] / "pipeline" / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)
y = np.load(ARTIFACTS / "y.npy")

MODEL_NAMES = ["Spline GAM", "EBM", "CatBoost", "LightGBM"]
oof_p = np.column_stack([
    np.load(ARTIFACTS / "oof_spline_bagged.npy"),
    np.load(ARTIFACTS / "oof_ebm_bagged.npy"),
    np.load(ARTIFACTS / "oof_cb_bagged.npy"),
    np.load(ARTIFACTS / "oof_lgb_bagged.npy"),
])
test_p = np.column_stack([
    np.load(ARTIFACTS / "test_preds_spline_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_ebm_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_cb_bagged.npy"),
    np.load(ARTIFACTS / "test_preds_lgb_bagged.npy"),
])

EPS = 1e-6
oof_p = np.clip(oof_p, EPS, 1 - EPS)
test_p = np.clip(test_p, EPS, 1 - EPS)
oof_logit = np.log(oof_p / (1 - oof_p))
test_logit = np.log(test_p / (1 - test_p))

N_META_FOLDS = 5
META_SEED = 202
skf_meta = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=META_SEED)


def fit_convex_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))

    k = S.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


def honest_convex_raw_prob():
    """Variant A: today's actual behavior -- convex blend of raw probabilities."""
    honest = np.zeros(len(y))
    for trn_idx, val_idx in skf_meta.split(oof_p, y):
        w = fit_convex_weights(oof_p[trn_idx], y[trn_idx])
        honest[val_idx] = np.dot(oof_p[val_idx], w)
    return np.clip(honest, 0.005, 0.995)


def honest_convex_logit():
    """Variant B: convex blend applied in logit space, then sigmoid back."""
    honest = np.zeros(len(y))
    for trn_idx, val_idx in skf_meta.split(oof_logit, y):
        def meta_loss(w, S=oof_logit[trn_idx], t=y[trn_idx]):
            pred = 1 / (1 + np.exp(-np.dot(S, w)))
            return log_loss(t, np.clip(pred, 0.005, 0.995))

        k = oof_logit.shape[1]
        init = np.full(k, 1.0 / k)
        bounds = [(0, 1)] * k
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        w = minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x
        honest[val_idx] = 1 / (1 + np.exp(-np.dot(oof_logit[val_idx], w)))
    return np.clip(honest, 0.005, 0.995)


def honest_logreg_meta(C=1.0):
    """Variant C: logistic-regression meta-learner over the 4 logits."""
    honest = np.zeros(len(y))
    for trn_idx, val_idx in skf_meta.split(oof_logit, y):
        meta = LogisticRegression(C=C, max_iter=1000)
        meta.fit(oof_logit[trn_idx], y[trn_idx])
        honest[val_idx] = meta.predict_proba(oof_logit[val_idx])[:, 1]
    return np.clip(honest, 0.005, 0.995)


print("=" * 66)
print("  A vs B vs C -- does fixing the logit-space bug actually help?")
print("=" * 66)

honest_a = honest_convex_raw_prob()
loss_a = log_loss(y, honest_a)
print(f"A. Convex blend, RAW probability space (today's actual cell 19): {loss_a:.5f}")

honest_b = honest_convex_logit()
loss_b = log_loss(y, honest_b)
print(f"B. Convex blend, LOGIT space (what the comment claims to do):   {loss_b:.5f}  "
      f"(delta vs A = {loss_b - loss_a:+.5f})")

honest_c = honest_logreg_meta(C=1.0)
loss_c = log_loss(y, honest_c)
print(f"C. Logistic-regression meta-learner over logits (C=1.0):        {loss_c:.5f}  "
      f"(delta vs A = {loss_c - loss_a:+.5f})")

# quick calibration-slope check: fit y ~ logit(oof_super) and read off the
# slope -- a slope < 1 means the blend is under-confident (linear-pool
# shrinkage); a slope of ~1 means it's already calibrated in the aggregate.
for label, honest in [("A (raw-prob convex)", honest_a), ("C (logreg meta)", honest_c)]:
    logit_pred = np.log(np.clip(honest, EPS, 1 - EPS) / (1 - np.clip(honest, EPS, 1 - EPS)))
    slope_check = LogisticRegression(max_iter=1000)
    slope_check.fit(logit_pred.reshape(-1, 1), y)
    print(f"Calibration slope check for {label}: slope={slope_check.coef_[0][0]:.3f} "
          f"(1.0 = perfectly calibrated in the aggregate; <1 = under-confident)")

best_label, best_honest, best_loss = min(
    [("A", honest_a, loss_a), ("B", honest_b, loss_b), ("C", honest_c, loss_c)],
    key=lambda t: t[2]
)
print("\n" + "-" * 66)
print(f"Best of the three: variant {best_label} at honest LogLoss {best_loss:.5f}")
print(f"AUC: {roc_auc_score(y, best_honest):.5f} | Brier: {brier_score_loss(y, best_honest):.5f}")
print("-" * 66)

# Ship whichever variant wins, fit on 100% of OOF for the test-set weights.
if best_label == "A":
    final_w = fit_convex_weights(oof_p, y)
    test_super = np.clip(np.dot(test_p, final_w), 0.005, 0.995)
elif best_label == "B":
    k = oof_logit.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    def meta_loss(w):
        pred = 1 / (1 + np.exp(-np.dot(oof_logit, w)))
        return log_loss(y, np.clip(pred, 0.005, 0.995))

    final_w = minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x
    test_super = np.clip(1 / (1 + np.exp(-np.dot(test_logit, final_w))), 0.005, 0.995)
else:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_logit, y)
    test_super = np.clip(meta.predict_proba(test_logit)[:, 1], 0.005, 0.995)
    print("Meta-learner coefficients (per model, on logit scale):")
    for name, coef in zip(MODEL_NAMES, meta.coef_[0]):
        print(f"  {name:<12s} {coef:+.3f}")
    print(f"  intercept    {meta.intercept_[0]:+.3f}")

sub = pd.DataFrame({'patient_id': test_fe['patient_id'], 'readmitted_30d': test_super})
out_path = ROOT / "legacy" / "submissions" / "submission_11_logit_space_super_learner.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved best variant ({best_label}) to {out_path}")
