"""
NEW — the actionable half of "Transportability Weighting": 32_adversarial_
validation.py found a real (if mild) train/test shift (OOF AUC 0.555, driven
by age, length_of_stay_days, n_missing, care_pathway, hospital_type). This
script turns that into importance-sampling weights and retrains the honest
Core-3 (Spline GAM + EBM + CatBoost, no LightGBM, no phenotype features)
with them, to see whether up-weighting train rows that "look like test"
actually improves the honest score, or whether the shift -- while real --
is too mild to be worth correcting this way.

w_i = P(test | x_i) / P(train | x_i) = p_i / (1 - p_i), using the OOF
"looks-like-test" probability from the adversarial classifier (never the
in-sample prediction -- that would leak). Weights are clipped and
re-normalized to mean 1 to avoid a few extreme-density-ratio rows
dominating the loss (a standard, necessary stabilizer for importance
weighting with a finite sample).

NOTE: this reweights TRAINING rows only. Honest scoring here means
"nested-CV weights for the stacking blend," same as elsewhere in this
pipeline -- it does NOT mean this validates the reweighting itself
against real test labels (those don't exist locally). It answers a
narrower question: does up-weighting test-like train rows change what
the model learns enough to move the honest OOF number, in the direction
matching how the real test set is shifted.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ROOT = Path(__file__).resolve().parents[2]

with open(ARTIFACTS / "train_fe.pkl", "rb") as f:
    train_fe = pickle.load(f)
with open(ARTIFACTS / "test_fe.pkl", "rb") as f:
    test_fe = pickle.load(f)

target = 'readmitted_30d'
drop_cols = ['patient_id', target]
y = train_fe[target].values

X = train_fe.drop(columns=drop_cols)
X_test = test_fe.drop(columns=['patient_id'])
cat_features = X.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
for c in cat_features:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

# ---- build importance-sampling weights from the adversarial OOF probs ----
p_test = np.load(ARTIFACTS / "oof_adversarial_train.npy")
p_test = np.clip(p_test, 0.02, 0.98)  # avoid near-0/near-1 density-ratio blowups
raw_weights = p_test / (1 - p_test)
sample_weights = raw_weights / raw_weights.mean()  # normalize to mean 1

print(f"Sample weight stats: min={sample_weights.min():.2f} max={sample_weights.max():.2f} "
      f"p5={np.percentile(sample_weights, 5):.2f} p95={np.percentile(sample_weights, 95):.2f}")

spline_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'length_of_stay_days'
]
nn_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
nn_cat_cols = X.select_dtypes(include=['category', 'object', 'string', 'str']).columns.tolist()
other_cols = [c for c in X.columns if c not in spline_cols and not c.endswith('_isna')]
ebm_cols = [c for c in X.columns if not c.endswith('_isna')]

SEEDS = [42, 7, 123]
n_tr, n_te = len(train_fe), len(test_fe)

oof_spline_runs, test_spline_runs = [], []
oof_ebm_runs, test_ebm_runs = [], []
oof_cb_runs, test_cb_runs = [], []

for seed in SEEDS:
    print(f"--> seed {seed}...")
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    spline_preprocessor = ColumnTransformer(
        transformers=[
            ('spline', make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), spline_cols),
            ('other_num', StandardScaler(), [c for c in other_cols if c in nn_num_cols]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [c for c in other_cols if c in nn_cat_cols]),
        ]
    )
    oof_spline_s = np.zeros(n_tr); test_spline_s = np.zeros(n_te)
    oof_ebm_s = np.zeros(n_tr); test_ebm_s = np.zeros(n_te)
    oof_cb_s = np.zeros(n_tr); test_cb_s = np.zeros(n_te)

    for trn_idx, val_idx in skf_seed.split(X, y):
        X_tr_raw, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va_raw, y_va = X.iloc[val_idx], y[val_idx]
        w_tr = sample_weights[trn_idx]

        X_tr_sp = spline_preprocessor.fit_transform(X_tr_raw)
        X_va_sp = spline_preprocessor.transform(X_va_raw)
        X_te_sp = spline_preprocessor.transform(X_test)
        sm_ = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        sm_.fit(X_tr_sp, y_tr, sample_weight=w_tr)
        oof_spline_s[val_idx] = sm_.predict_proba(X_va_sp)[:, 1]
        test_spline_s += sm_.predict_proba(X_te_sp)[:, 1] / 5

        ebm = ExplainableBoostingClassifier(max_bins=128, interactions=10, outer_bags=8, inner_bags=0, random_state=seed)
        ebm.fit(X_tr_raw[ebm_cols], y_tr, sample_weight=w_tr)
        oof_ebm_s[val_idx] = ebm.predict_proba(X_va_raw[ebm_cols])[:, 1]
        test_ebm_s += ebm.predict_proba(X_test[ebm_cols])[:, 1] / 5

        X_tr_cb, X_va_cb, X_te_cb = X_tr_raw.copy(), X_va_raw.copy(), X_test.copy()
        for col in cat_features:
            X_tr_cb[col] = X_tr_cb[col].astype(str)
            X_va_cb[col] = X_va_cb[col].astype(str)
            X_te_cb[col] = X_te_cb[col].astype(str)
        cb = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, loss_function='Logloss',
                                 eval_metric='Logloss', random_seed=seed, verbose=False)
        cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, cat_features=cat_features,
               eval_set=(X_va_cb, y_va), early_stopping_rounds=30)
        oof_cb_s[val_idx] = cb.predict_proba(X_va_cb)[:, 1]
        test_cb_s += cb.predict_proba(X_te_cb)[:, 1] / 5

    oof_spline_runs.append(oof_spline_s); test_spline_runs.append(test_spline_s)
    oof_ebm_runs.append(oof_ebm_s); test_ebm_runs.append(test_ebm_s)
    oof_cb_runs.append(oof_cb_s); test_cb_runs.append(test_cb_s)

oof_spline = np.clip(np.mean(oof_spline_runs, axis=0), 0.005, 0.995)
test_spline = np.mean(test_spline_runs, axis=0)
oof_ebm = np.clip(np.mean(oof_ebm_runs, axis=0), 0.005, 0.995)
test_ebm = np.mean(test_ebm_runs, axis=0)
oof_cb = np.clip(np.mean(oof_cb_runs, axis=0), 0.005, 0.995)
test_cb = np.mean(test_cb_runs, axis=0)

# NOTE: OOF log loss here is measured against the (unweighted) TRUE train
# labels as always -- sample_weight only changes what the optimizer
# prioritizes during fitting, not how we score it afterward.
print(f"\nSpline GAM (shift-reweighted) OOF Log Loss: {log_loss(y, oof_spline):.5f}  (unweighted baseline: 0.34991)")
print(f"EBM (shift-reweighted) OOF Log Loss:        {log_loss(y, oof_ebm):.5f}  (unweighted baseline: 0.34928)")
print(f"CatBoost (shift-reweighted) OOF Log Loss:    {log_loss(y, oof_cb):.5f}  (unweighted baseline: 0.34965)")

S_train = np.column_stack([oof_spline, oof_ebm, oof_cb])
S_test = np.column_stack([test_spline, test_ebm, test_cb])


def fit_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))
    k = S.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=202)
honest_oof = np.zeros(len(y))
for trn_idx, val_idx in skf_meta.split(S_train, y):
    w = fit_weights(S_train[trn_idx], y[trn_idx])
    honest_oof[val_idx] = np.dot(S_train[val_idx], w)
honest_oof = np.clip(honest_oof, 0.005, 0.995)

final_weights = fit_weights(S_train, y)
test_preds = np.clip(np.dot(S_test, final_weights), 0.005, 0.995)
honest_loss = log_loss(y, honest_oof)

print("\n" + "=" * 62)
print("  CORE-3, SHIFT-REWEIGHTED (ADVERSARIAL-VALIDATION WEIGHTS)")
print("=" * 62)
print(f"Weights -> Spline: {final_weights[0]:.3f} | EBM: {final_weights[1]:.3f} | CatBoost: {final_weights[2]:.3f}")
print(f"Honest (nested-CV) OOF Log Loss: {honest_loss:.5f}")
print(f"Reference -- Core-3 unweighted (honest): 0.34855")
print(f"Delta: {honest_loss - 0.34855:+.5f}")
print(f"ROC-AUC (honest): {roc_auc_score(y, honest_oof):.5f}")
print(f"Brier (honest):   {brier_score_loss(y, honest_oof):.5f}")
print("=" * 62)

sub = pd.DataFrame({'patient_id': test_fe['patient_id'], 'readmitted_30d': test_preds})
out_path = ROOT / "outputs" / "submission_15_core3_shift_reweighted.csv"
sub.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
