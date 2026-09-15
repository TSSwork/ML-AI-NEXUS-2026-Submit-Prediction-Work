"""
ML & AI NEXUS 2026 - Kenura Research Pipeline
Focus: High predictive accuracy (target: < 0.336 Log Loss), probability calibration,
robustness under missingness, and automated Trust Card auditing.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb

warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
OUTPUT_DIR = 'legacy/submissions'
SUBMISSION_NAME = 'submission_03_calibrated_lgbm.csv'
RANDOM_SEED = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 2. Data Loading & Feature Engineering
# -------------------------------------------------------------
print("--> Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = 'readmitted_30d'
drop_cols = ['patient_id', target_col]

MISSING_COLS = [
    'creatinine_mg_dl', 'heart_rate_bpm', 'hemoglobin_g_dl',
    'sodium_mmol_l', 'systolic_bp_mmhg', 'followup_days'
]

def engineer_features(df):
    data = df.copy()
    
    # A. Missingness Indicators (critical since test has higher missingness)
    for col in MISSING_COLS:
        data[f'{col}_isna'] = data[col].isna().astype(int)
    data['n_missing'] = data[[f'{c}_isna' for c in MISSING_COLS]].sum(axis=1)
    
    # B. Clinical Threshold & Interaction Features
    # High risk flags based on standard medical cutoffs
    data['high_creatinine'] = (data['creatinine_mg_dl'] > 1.3).astype(float)
    data['low_hemoglobin'] = (data['hemoglobin_g_dl'] < 12.0).astype(float)
    data['abnormal_sodium'] = ((data['sodium_mmol_l'] < 135) | (data['sodium_mmol_l'] > 145)).astype(float)
    data['stage2_hypertension'] = (data['systolic_bp_mmhg'] >= 140).astype(float)
    
    # Chronic burden sum
    chronic_conds = ['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure']
    data['chronic_burden'] = data[chronic_conds].sum(axis=1)
    
    # Non-linear clinical interactions
    data['age_comorbidity_mult'] = data['age'] * (data['comorbidity_count'] + 1)
    data['utilization_risk'] = data['prior_admissions_12m'] * 2 + data['missed_appointments_12m']
    
    # Age groups (matches EDA binning for Trust Card)
    data['age_group'] = pd.cut(
        data['age'],
        bins=[-np.inf, 29, 49, 69, np.inf],
        labels=['<30', '30-49', '50-69', '70+']
    ).astype(str)
    
    return data

train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

y = train_feat[target_col].values
X_train_raw = train_feat.drop(columns=drop_cols)
X_test_raw = test_feat.drop(columns=['patient_id'])

# Define categorical and numerical columns
cat_cols = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway', 'age_group']
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

# -------------------------------------------------------------
# 3. Model Training & 5-Fold Stratified Cross-Validation
# -------------------------------------------------------------
print("--> Training 5-Fold Stratified LightGBM...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_lgb = np.zeros(len(train_feat))
oof_blend = np.zeros(len(train_feat))
test_preds_lgb = np.zeros(len(test_feat))
test_preds_blend = np.zeros(len(test_feat))

# Conservative LightGBM hyperparameters to prevent overfitting private test set
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 16,             # Restricted depth to generalize well
    'max_depth': 4,
    'min_child_samples': 40,
    'feature_fraction': 0.8,
    'subsample': 0.85,
    'reg_alpha': 0.5,             # L1 Regularization
    'reg_lambda': 1.0,            # L2 Regularization
    'random_state': RANDOM_SEED,
    'verbose': -1,
    'n_estimators': 400
}

# Preprocessing for linear baseline component (impute + scale + OHE)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y), 1):
    X_tr, y_tr = X_train_raw.iloc[train_idx].copy(), y[train_idx]
    X_va, y_va = X_train_raw.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test_raw.copy()
    
    # --- A. Fit LightGBM (native categorical handling) ---
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype('category')
        X_va[col] = X_va[col].astype('category')
        X_te[col] = X_te[col].astype('category')
        
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    val_pred_lgb = model_lgb.predict_proba(X_va)[:, 1]
    test_pred_lgb = model_lgb.predict_proba(X_te)[:, 1]
    
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += test_pred_lgb / N_SPLITS
    
    # --- B. Fit Logistic Regression component for blending ---
    X_tr_num = scaler.fit_transform(num_imputer.fit_transform(X_tr[num_cols]))
    X_va_num = scaler.transform(num_imputer.transform(X_va[num_cols]))
    X_te_num = scaler.transform(num_imputer.transform(X_te[num_cols]))
    
    X_tr_cat = ohe.fit_transform(cat_imputer.fit_transform(X_tr[cat_cols]))
    X_va_cat = ohe.transform(cat_imputer.transform(X_va[cat_cols]))
    X_te_cat = ohe.transform(cat_imputer.transform(X_te[cat_cols]))
    
    X_tr_lr = np.hstack([X_tr_num, X_tr_cat])
    X_va_lr = np.hstack([X_va_num, X_va_cat])
    X_te_lr = np.hstack([X_te_num, X_te_cat])
    
    model_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED)
    model_lr.fit(X_tr_lr, y_tr)
    
    val_pred_lr = model_lr.predict_proba(X_va_lr)[:, 1]
    test_pred_lr = model_lr.predict_proba(X_te_lr)[:, 1]
    
    # Blend: 80% Non-linear Tree + 20% Calibrated Linear Baseline
    val_blend = (0.80 * val_pred_lgb) + (0.20 * val_pred_lr)
    test_blend = (0.80 * test_pred_lgb) + (0.20 * test_pred_lr)
    
    oof_blend[val_idx] = val_blend
    test_preds_blend += test_blend / N_SPLITS

# -------------------------------------------------------------
# 4. Calibration & Metrics Summary
# -------------------------------------------------------------
print("\n" + "="*50)
print("             VALIDATION RESULTS SUMMARY            ")
print("="*50)

# Safety clip predictions away from exact 0 and 1
oof_blend = np.clip(oof_blend, 0.005, 0.995)
final_test_preds = np.clip(test_preds_blend, 0.005, 0.995)

lgb_loss = log_loss(y, oof_lgb)
blend_loss = log_loss(y, oof_blend)
blend_brier = brier_score_loss(y, oof_blend)
blend_auc = roc_auc_score(y, oof_blend)

print(f"-> LightGBM OOF Log Loss:     {lgb_loss:.5f}")
print(f"-> Blend OOF Log Loss:        {blend_loss:.5f}  (Previous Logistic: 0.3526)")
print(f"-> Blend Brier Score:         {blend_brier:.5f}")
print(f"-> Blend ROC-AUC:             {blend_auc:.5f}")

# Calibration Intercept & Slope check
prob_true, prob_pred = calibration_curve(y, oof_blend, n_bins=10)
calib_gap = np.mean(np.abs(prob_true - prob_pred))
print(f"-> Mean Calibration Error:    {calib_gap:.5f}")

# -------------------------------------------------------------
# 5. Automated Subgroup Audit (for Model Trust Card)
# -------------------------------------------------------------
print("\n" + "="*50)
print("       SUBGROUP RELIABILITY AUDIT (FOR TRUST CARD)  ")
print("="*50)

audit_df = train_df.copy()
audit_df['pred'] = oof_blend

for attr in ['sex', 'rurality', 'hospital_type']:
    print(f"\nAudit by [{attr}]:")
    for group, grp_data in audit_df.groupby(attr):
        grp_loss = log_loss(grp_data[target_col], grp_data['pred'])
        grp_prev = grp_data[target_col].mean()
        grp_pred_mean = grp_data['pred'].mean()
        print(f"  {group:12s} | n={len(grp_data):4d} | Actual: {grp_prev:.3f} | Pred: {grp_pred_mean:.3f} | LogLoss: {grp_loss:.4f}")

# -------------------------------------------------------------
# 6. Uncertainty & Human Referral Test (Section 7 of Card)
# -------------------------------------------------------------
eps = 1e-15
entropy = -(oof_blend * np.log(oof_blend + eps) + (1 - oof_blend) * np.log(1 - oof_blend + eps))
cutoff = np.percentile(entropy, 90)
uncertain_mask = entropy >= cutoff

retained_loss = log_loss(y[~uncertain_mask], oof_blend[~uncertain_mask])
caught_cases = y[uncertain_mask].sum()
total_cases = y.sum()

print("\n" + "="*50)
print("       UNCERTAINTY / ABSTENTION SIMULATION         ")
print("="*50)
print(f"Referring top 10% uncertain cases improves retained Log Loss to: {retained_loss:.5f}")
print(f"This 10% review queue captures {caught_cases} of {total_cases} true readmissions ({caught_cases/total_cases*100:.1f}%)")

# -------------------------------------------------------------
# 7. Generate Kaggle Submission File
# -------------------------------------------------------------
sub_df = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'readmitted_30d': final_test_preds
})

sub_path = os.path.join(OUTPUT_DIR, SUBMISSION_NAME)
sub_df.to_csv(sub_path, index=False)

print("\n" + "="*50)
print(f"--> Submission successfully written to: {sub_path}")
print(f"--> Predicted test mean: {final_test_preds.mean():.5f} (Train prevalence: {y.mean():.5f})")
print("="*50)