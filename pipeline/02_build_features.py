"""
Cell 10 equivalent — feature engineering.

`build_features` is copied verbatim from basic_check.ipynb cell 10
(STEP 2: Feature Engineering & Preprocessing). Original notebook is
untouched; this script just runs the same logic as a standalone step.
"""
import pickle
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

MISSING_COLS = [
    'creatinine_mg_dl', 'heart_rate_bpm', 'hemoglobin_g_dl',
    'sodium_mmol_l', 'systolic_bp_mmhg', 'followup_days'
]


def build_features(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()

    for df in [tr, te]:
        # 1. Missingness flags & total missing count
        for col in MISSING_COLS:
            df[f'{col}_isna'] = df[col].isna().astype(int)
        df['n_missing'] = df[[f'{c}_isna' for c in MISSING_COLS]].sum(axis=1)

        # 2. Key clinical interactions from our charts
        # Chart finding: heart_failure (27.3%) & CKD (20.3%) are the two highest risks
        df['severe_cardiorenal'] = (df['heart_failure'] == 1) & (df['chronic_kidney_disease'] == 1).astype(int)

        # Chart finding: prior_admissions is monotonic and steep
        df['high_utilizer'] = (df['prior_admissions_12m'] >= 3).astype(int)
        df['risk_load'] = df['prior_admissions_12m'] * 2 + df['comorbidity_count']

        # Chart finding: age right-shift interaction with comorbidities
        df['age_x_comorbidity'] = df['age'] * (df['comorbidity_count'] + 1)

        # 3. High/low lab abnormality flags (where KDE tails diverge)
        df['anemia_flag'] = (df['hemoglobin_g_dl'] < 11.5).astype(float)
        df['elevated_creatinine'] = (df['creatinine_mg_dl'] > 1.3).astype(float)

    # 4. Impute continuous labs using train medians (avoids test-to-train leakage)
    for col in MISSING_COLS:
        med = tr[col].median()
        tr[col] = tr[col].fillna(med)
        te[col] = te[col].fillna(med)

    return tr, te


if __name__ == "__main__":
    with open(ARTIFACTS / "train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(ARTIFACTS / "test.pkl", "rb") as f:
        test = pickle.load(f)

    train_fe, test_fe = build_features(train, test)

    print("Features created successfully!")
    print(f"Train columns before: {train.shape[1]} -> after: {train_fe.shape[1]}")

    with open(ARTIFACTS / "train_fe.pkl", "wb") as f:
        pickle.dump(train_fe, f)
    with open(ARTIFACTS / "test_fe.pkl", "wb") as f:
        pickle.dump(test_fe, f)

    print(f"Saved engineered train_fe/test_fe to {ARTIFACTS}")
