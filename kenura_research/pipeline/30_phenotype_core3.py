"""
NEW — validates the K-Means clinical-phenotyping cells now appended to
basic_check.ipynb (K-means cluster_id + distance-to-centroid features,
then Core-3 = Spline GAM + EBM + CatBoost, LightGBM excluded per the
ablation finding in 25_ablation_table.py / 27_final_recommended_stack.py).

Mirrors the notebook cells exactly (same feature logic, same CV seed) so
a clean run here is strong evidence the notebook cells will run too, and
gives the honest number needed to decide whether to actually submit
before spending one of the limited daily submissions on it.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
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

# ---- Phenotyping (cell A) ----
print("--> Deriving Unsupervised Clinical Phenotypes via K-Means...")
for df in [train_fe, test_fe]:
    df['shock_index'] = df['heart_rate_bpm'] / df['systolic_bp_mmhg']

pheno_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'shock_index'
]

scaler_pheno = StandardScaler()
X_pheno_train = scaler_pheno.fit_transform(train_fe[pheno_cols])
X_pheno_test = scaler_pheno.transform(test_fe[pheno_cols])

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
train_fe['cluster_id'] = kmeans.fit_predict(X_pheno_train).astype(str)
test_fe['cluster_id'] = kmeans.predict(X_pheno_test).astype(str)

train_distances = kmeans.transform(X_pheno_train)
test_distances = kmeans.transform(X_pheno_test)
for k in range(4):
    train_fe[f'dist_phenotype_{k}'] = train_distances[:, k]
    test_fe[f'dist_phenotype_{k}'] = test_distances[:, k]

print("\n" + "=" * 55)
print("     DISCOVERED CLINICAL PHENOTYPES & RISK RATES     ")
print("=" * 55)
for c in range(4):
    grp = train_fe[train_fe['cluster_id'] == str(c)]
    print(f"Phenotype {c} (n={len(grp)}): Readmit Rate = {grp[target].mean():.1%} | "
          f"Age = {grp['age'].mean():.1f} | Creatinine = {grp['creatinine_mg_dl'].mean():.2f} | "
          f"Shock Index = {grp['shock_index'].mean():.2f}")
print("=" * 55)

X = train_fe.drop(columns=drop_cols)
X_test = test_fe.drop(columns=['patient_id'])
cat_features = X.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
for c in cat_features:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Core-3 training (cell B) ----
print("\n--> Training Core-3 (Spline GAM + EBM + CatBoost) on phenotype-enriched features...")

nn_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
nn_cat_cols = X.select_dtypes(include=['category', 'object', 'string', 'str']).columns.tolist()

spline_cols = [
    'age', 'creatinine_mg_dl', 'hemoglobin_g_dl', 'sodium_mmol_l',
    'heart_rate_bpm', 'systolic_bp_mmhg', 'prior_admissions_12m', 'length_of_stay_days'
]
other_cols = [c for c in X.columns if c not in spline_cols and not c.endswith('_isna')]

spline_preprocessor = ColumnTransformer(
    transformers=[
        ('spline', make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), spline_cols),
        ('other_num', StandardScaler(), [c for c in other_cols if c in nn_num_cols]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [c for c in other_cols if c in nn_cat_cols]),
    ]
)

oof_spline_pheno = np.zeros(len(train_fe))
test_spline_pheno = np.zeros(len(test_fe))
for trn_idx, val_idx in skf.split(X, y):
    X_tr_raw, y_tr = X.iloc[trn_idx], y[trn_idx]
    X_va_raw = X.iloc[val_idx]
    X_tr_sp = spline_preprocessor.fit_transform(X_tr_raw)
    X_va_sp = spline_preprocessor.transform(X_va_raw)
    X_te_sp = spline_preprocessor.transform(X_test)
    m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    m.fit(X_tr_sp, y_tr)
    oof_spline_pheno[val_idx] = m.predict_proba(X_va_sp)[:, 1]
    test_spline_pheno += m.predict_proba(X_te_sp)[:, 1] / 5
oof_spline_pheno = np.clip(oof_spline_pheno, 0.005, 0.995)
print(f"Spline GAM (phenotype) OOF Log Loss: {log_loss(y, oof_spline_pheno):.5f}  (no-phenotype: 0.35039)")

ebm_cols = [c for c in X.columns if not c.endswith('_isna')]
oof_ebm_pheno = np.zeros(len(train_fe))
test_ebm_pheno = np.zeros(len(test_fe))
for trn_idx, val_idx in skf.split(X[ebm_cols], y):
    X_tr, y_tr = X[ebm_cols].iloc[trn_idx], y[trn_idx]
    X_va = X[ebm_cols].iloc[val_idx]
    ebm = ExplainableBoostingClassifier(max_bins=128, interactions=10, outer_bags=8, inner_bags=0, random_state=42)
    ebm.fit(X_tr, y_tr)
    oof_ebm_pheno[val_idx] = ebm.predict_proba(X_va)[:, 1]
    test_ebm_pheno += ebm.predict_proba(X_test[ebm_cols])[:, 1] / 5
oof_ebm_pheno = np.clip(oof_ebm_pheno, 0.005, 0.995)
print(f"EBM (phenotype) OOF Log Loss:         {log_loss(y, oof_ebm_pheno):.5f}  (no-phenotype: 0.35034)")

cb_pheno_cat = X.select_dtypes(include=['category']).columns.tolist()
oof_cb_pheno = np.zeros(len(train_fe))
test_cb_pheno = np.zeros(len(test_fe))
for trn_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[trn_idx].copy(), y[trn_idx]
    X_va, y_va = X.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test.copy()
    for col in cb_pheno_cat:
        X_tr[col] = X_tr[col].astype(str)
        X_va[col] = X_va[col].astype(str)
        X_te[col] = X_te[col].astype(str)
    cb = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, loss_function='Logloss',
                             eval_metric='Logloss', random_seed=42, verbose=False)
    cb.fit(X_tr, y_tr, cat_features=cb_pheno_cat, eval_set=(X_va, y_va), early_stopping_rounds=30)
    oof_cb_pheno[val_idx] = cb.predict_proba(X_va)[:, 1]
    test_cb_pheno += cb.predict_proba(X_te)[:, 1] / 5
oof_cb_pheno = np.clip(oof_cb_pheno, 0.005, 0.995)
print(f"CatBoost (phenotype) OOF Log Loss:    {log_loss(y, oof_cb_pheno):.5f}  (no-phenotype: 0.35038)")

S_train_core3 = np.column_stack([oof_spline_pheno, oof_ebm_pheno, oof_cb_pheno])
S_test_core3 = np.column_stack([test_spline_pheno, test_ebm_pheno, test_cb_pheno])


def fit_weights(S, targets):
    def meta_loss(w):
        return log_loss(targets, np.clip(np.dot(S, w), 0.005, 0.995))
    k = S.shape[1]
    init = np.full(k, 1.0 / k)
    bounds = [(0, 1)] * k
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    return minimize(meta_loss, init, method='SLSQP', bounds=bounds, constraints=constraints).x


skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=202)
honest_oof_core3 = np.zeros(len(y))
for trn_idx, val_idx in skf_meta.split(S_train_core3, y):
    w = fit_weights(S_train_core3[trn_idx], y[trn_idx])
    honest_oof_core3[val_idx] = np.dot(S_train_core3[val_idx], w)
honest_oof_core3 = np.clip(honest_oof_core3, 0.005, 0.995)

final_weights_core3 = fit_weights(S_train_core3, y)
oof_core3 = np.clip(np.dot(S_train_core3, final_weights_core3), 0.005, 0.995)
test_preds_core3 = np.clip(np.dot(S_test_core3, final_weights_core3), 0.005, 0.995)

print("\n" + "=" * 55)
print("   CORE-3 + PHENOTYPE FEATURES -- FINAL RESULTS       ")
print("=" * 55)
print(f"Weights -> Spline: {final_weights_core3[0]:.3f} | EBM: {final_weights_core3[1]:.3f} | CatBoost: {final_weights_core3[2]:.3f}")
print(f"Honest (nested-CV) OOF Log Loss:  {log_loss(y, honest_oof_core3):.5f}")
print(f"Same-data OOF Log Loss:           {log_loss(y, oof_core3):.5f}")
print(f"ROC-AUC (honest):                 {roc_auc_score(y, honest_oof_core3):.5f}")
print(f"Reference -- Core-3 WITHOUT phenotype features (honest, seed-bagged): 0.34855")
print("=" * 55)

sub = pd.DataFrame({'patient_id': test_fe['patient_id'], 'readmitted_30d': test_preds_core3})
out_path = ROOT / "outputs" / "submission_10_core3_phenotype.csv"
sub.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
