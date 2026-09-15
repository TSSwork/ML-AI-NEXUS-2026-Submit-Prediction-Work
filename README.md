# 30-Day Hospital Readmission — ML & AI NEXUS 2026

**Team: Data Lab**

This project predicts the probability of an unplanned hospital readmission in 30 days. The data
is a synthetic hospital dataset. The competition scores trustworthiness. It does not score only
the leaderboard rank. Trustworthiness includes calibration, robustness, subgroup fairness,
explainability and reproducibility.

[COMPETITION.md](COMPETITION.md) contains the competition brief, the rules and the data
dictionary.

**All of the work is in one notebook:
[`notebooks/final_model_report.ipynb`](notebooks/final_model_report.ipynb).** The notebook runs
from the first cell to the last cell. It reads the data from `data/`. It writes the final
submission to `submissions/`.

This document and the notebook use ASD Simplified Technical English (STE).

---

## Final model

The final model is the **Core-3 Stack**. The name gives the composition: three core model
families. These are a Spline GAM, an EBM and a CatBoost model. The notebook trains each model
with three fold seeds. The SLSQP optimizer calculates the blend weights.

The word "honest" in this report applies only to a score. It means that nested cross-validation
measured the score. The optimizer calculates the blend weights on four fifths of the rows. The
notebook measures the score on the other fifth. Thus the weights never see the rows of their own
measurement. The word is a property of the measurement. It is not a property of the model.

| Item | Value |
|---|---|
| **Honest (nested-CV) OOF Log Loss** | **0.34855** |
| ROC-AUC | 0.69458 |
| Brier score | 0.10207 |
| Weights | Spline GAM 0.333 · EBM 0.333 · CatBoost 0.333 |
| Output file | `submissions/submission_13_core3_honest.csv` |
| Team | Data Lab |

The earlier models give these scores:

| Model | Log Loss |
|---|---|
| Constant prevalence (0.1259) | 0.37844 |
| Logistic regression | 0.35261 |
| Best single model (CatBoost) | 0.35038 |
| **Core-3 Stack** | **0.34855** |

The optimizer gives an equal weight of 0.333 to each model. This result is important. No model
controls the blend. Thus the result does not depend on one model. This property makes the stack
more robust.

## Quick start

```bash
pixi install
pixi run lab       # open the notebook
pixi run report    # run the notebook from start to end (approximately 20 minutes)
```

The command `pixi run report` writes `submissions/submission_13_core3_honest.csv` again. A test
shows that the new file is the same as the file in this repository, byte for byte.

```
data/          the competition csv files (read only)
notebooks/     final_model_report.ipynb -- the full project
submissions/   submission_13_core3_honest.csv (final)
legacy/        the earlier notebooks and the superseded submissions
COMPETITION.md the competition brief, the rules and the data dictionary
```

---

## The steps in the notebook

### 1. Data audit and validation

The train set has 7,000 rows and 25 columns. The test set has 3,000 rows and 24 columns. The base
readmission rate is 12.59%.

A **variable audit table** shows all 24 features together. For each feature, the table gives
these items:

- the data type;
- the missing percentage in the train set and the test set;
- the number of unique values;
- a summary of the distribution;
- the relation to the target.

The table uses a correlation for a numeric feature. It uses the spread of the readmission rate
for a categorical feature.

These features have the strongest signals:

| Feature | Signal |
|---|---|
| `heart_failure` | the rate increases from 11.9% to **27.3%** (15.3 pp) |
| `chronic_kidney_disease` | the rate increases from 12.0% to **20.3%** (8.3 pp) |
| `age` | correlation **+0.180** |
| `prior_admissions_12m` | correlation **+0.146** |
| `comorbidity_count` | correlation **+0.117** |
| `rurality` | the rate increases from 10.9% to 15.6% (4.7 pp) |
| `socioeconomic_index` | correlation −0.002, thus there is no marginal signal |

A **validation schema** then examines each column. The schema has a biological range for each
column. It also has a rule for the permitted missing values. The schema gives a report with a
pass or a failure for each rule. This report is better than a manual examination of
`.describe()`. The report is the evidence of data integrity for the Trust Card.

The report shows no failures. Only the known missing values occur. But the missing values are not
random. The test set has more missing values than the train set in all six affected columns.

| Column | Missing in train | Missing in test |
|---|---|---|
| `creatinine_mg_dl` | 4.83% | **7.07%** |
| `heart_rate_bpm` | 4.81% | **6.87%** |
| `hemoglobin_g_dl` | 4.63% | **6.57%** |
| `sodium_mmol_l` | 4.44% | **6.40%** |
| `systolic_bp_mmhg` | 4.41% | **5.90%** |
| `followup_days` | 2.33% | **3.33%** |

A missing value also gives a signal about the outcome:

| Column | Rate when the value is missing | Rate when the value is present | Difference |
|---|---|---|---|
| `creatinine_mg_dl` | 13.9% | 12.5% | **+1.4 pp** |
| `heart_rate_bpm` | 13.6% | 12.5% | +1.1 pp |
| `systolic_bp_mmhg` | 13.3% | 12.6% | +0.7 pp |
| `sodium_mmol_l` | 10.9% | 12.7% | **−1.7 pp** |

The sodium result has a clinical explanation. A patient who does not get a full electrolyte panel
is usually a more healthy patient. The direction of the effect is not the same for all columns.
Thus the notebook keeps the missing values as features. It does not remove them with imputation
only.

### 2. Exploratory data analysis

The notebook shows each categorical feature and each binary feature as a bar chart. Each chart
also shows the readmission rate. The notebook shows each discrete count with its risk trend. It
shows each continuous feature as a density curve for train and test together. These curves make a
change in the distribution visible for each feature.

Age is the strongest single driver. It has the highest numeric correlation (+0.180). The EBM
shape function shows where the risk changes. The risk changes from low to high at an age of
approximately 58. The mix of the categories also changes between train and test. This occurs in
`region`, `hospital_type` and `care_pathway`. This change is the reason for the robustness tests.

### 3. Feature engineering: 25 columns to 38 columns

Each new feature comes from a result in the analysis above:

- Six `*_isna` flags and `n_missing`. A missing value gives a signal in two directions.
- `severe_cardiorenal` = heart failure AND chronic kidney disease. These are the two strongest
  signals (27.3% and 20.3%).
- `high_utilizer` (three or more prior admissions) and `risk_load`
  (`2 × prior_admissions + comorbidity_count`). The risk increases quickly with the number of
  prior admissions. Then it becomes stable.
- `age_x_comorbidity`. Age and the comorbidity count are the two strongest numeric features, and
  they interact.
- `anemia_flag` (haemoglobin less than 11.5) and `elevated_creatinine` (creatinine more than
  1.3). These are clinical limits at the positions where the density curves separate.

The notebook calculates the median of each continuous laboratory value on the train set only.
Then it puts this median in the empty positions. A median from the train set and the test set
together would put test information into the model. This is a leak. A leak increases the local
score and decreases the leaderboard score.

### 4. The model families

The notebook trains four different model families. Different families make different errors. Thus
a blend of these families is better than a blend of similar models.

| Family | Reason for this family | 5-fold OOF Log Loss |
|---|---|---|
| **CatBoost** (symmetric trees) | It is strong on tabular data. It uses a categorical column directly. It overfits 7,000 rows less than a leaf-wise model. | 0.35038 |
| **EBM** (glass-box GA²M) | It gives an exact shape function for each feature. It also learns two-way interactions. | 0.35034 |
| **Spline GAM** (cubic B-splines and L2 logistic regression) | It is parametric, smooth and fully readable. It is the logistic baseline without the linear limitation. | 0.35039 |
| **LightGBM** (leaf-wise GBDT) | It is the usual strong baseline. The notebook removes it later. | 0.35262 |
| Random Forest | The optimizer gives it a weight of 0.000. | 0.35289 |

All four families are better than the logistic baseline (0.35261). The four scores are within
0.002 of each other. The models have almost the same accuracy, but they do not agree about each
patient. This condition makes a blend useful.

### 5. Explainability

The notebook trains one EBM on all of the train data. This model shows the decision surface. It
is not an approximation after the calculation.

- The base intercept β₀ is **−2.1242** in logit space. This value is a base probability of
  10.68%. This probability applies before a feature makes a contribution.
- The notebook plots a shape function for each feature. It also prints the exact contribution
  table as odds ratios.
- `creatinine_mg_dl` is flat below 1.5. Then it increases steeply at **1.8 mg/dL to 2.0 mg/dL**.
  These values agree with acute kidney injury and severe renal disease. Thus the model learned
  correct clinical behaviour.
- `age` changes from a low risk to a high risk at **58**.
- `prior_admissions_12m` increases to three admissions. Then it becomes stable. The odds ratios
  are 0.91, then 1.03, then 1.13, then 1.29.

Two of the three models in the final stack are glass-box models. Thus you can read two thirds of
each prediction. This is the primary argument for the theme "Beyond the Black Box".

### 6. The final stack

**Seed bagging.** The data has only 7,000 rows. Thus one 5-fold split has a large variance. The
notebook trains each family with three different fold seeds (42, 7 and 123). Then it calculates
the average. This procedure improves each model:

| Model | One seed | Three seeds | Improvement |
|---|---|---|---|
| EBM | 0.35034 | **0.34928** | −0.00106 |
| CatBoost | 0.35038 | **0.34965** | −0.00073 |
| Spline GAM | 0.35039 | **0.34991** | −0.00048 |

**Honest scoring.** The blend weights are parameters. The optimizer calculates them from the
out-of-fold predictions. If you calculate the weights on all 7,000 rows and then measure the loss
on the same rows, the score is too good. The weights already saw these rows.

Thus the notebook uses nested cross-validation. It calculates the weights on four fifths of the
rows. It measures the loss on the other fifth. It does this five times. No row is in the
calculation and in the measurement at the same time. The weights for the test predictions use
100% of the rows. This is correct, because the notebook does not measure a score with them.

The measured difference on the four-model stack is 0.34913 with the first method and 0.34964 with
nested cross-validation.

**The removal of LightGBM.** The notebook removed one model at a time from the four-model stack:

| Configuration | Honest Log Loss | Difference from the four-model stack |
|---|---|---|
| All four models | 0.34912 | — |
| **Without LightGBM** | **0.34855** | **−0.00057 (better)** |
| Without CatBoost | 0.34913 | +0.00001 |
| Without EBM | 0.34933 | +0.00021 |
| Without Spline GAM | 0.34937 | +0.00025 |

LightGBM is the only model that makes the stack worse. It is the weakest single model. It also
gives no new information, because CatBoost is in the stack. Both models are GBDT models on the
same features. The nested-CV weight of LightGBM changes between 0.000 and 0.064 in the
different folds. Thus the optimizer cannot find a stable weight for it. The first scoring method does not
show this problem.

---

## Robustness and transportability

The notebook does an **adversarial validation** test. The test puts the label 0 on each train row
and the label 1 on each test row. Then a model tries to find the difference between the two sets.
The model uses the same features as the final model.

- The OOF ROC-AUC is **0.555**. This value is more than 0.50. Thus there is a real difference
  between the two sets, but the difference is small.
- These features show the largest difference: `age` (54.2), `length_of_stay_days` (35.8),
  `n_missing` (34.6), `care_pathway` (34.0), `socioeconomic_index` (27.8), `followup_days` (27.6)
  and `hospital_type` (26.4).
- The high position of `n_missing` is important. It shows that the different number of missing
  values in the test set is a real difference between the populations. It is not a sample error.
- A correction with importance weights (`w = p/(1−p)`) did not improve the honest score. The
  difference between the two sets is real, but it is too small for this correction.

This result is important for the Trust Card. The risk is measured, and a correction for the risk
was tested. The score can decrease on the private test set, because the two populations are not
the same.

## The treatments that were tested and rejected

The team made each treatment below and measured it against the honest score of 0.34855. No
treatment was better. This is the argument for the simple final model.

| Treatment | Result |
|---|---|
| **LightGBM in the stack** | Rejected. It is worse by 0.00057. |
| **K-Means clinical phenotypes** (cluster identifier, distance to the centroid, shock index) | Rejected. It made each model worse. The clusters come from columns that the models already use. Thus they add collinear noise. |
| **Tabular MLP** (two layers, 64 × 32, early stop) | Rejected. It was not sufficiently strong on 7,000 rows. |
| **Stacking in logit space** | Rejected. It was not better than the blend in probability space. |
| **LDA and complementary log-log models** | Rejected. They are weaker alone (0.35591 and 0.35408). They also add no new information. Core-3 with LDA gives 0.34879. Core-3 with cloglog gives 0.34870. Core-3 with both gives 0.34885. The seven-model stack gives 0.34912. All of these values are worse than 0.34855. |
| **Log transformation of the physiological values** (`log(creatinine)`, `log(LOS+1)`) | Rejected. It was not better than 0.34855. |
| **Weights for the difference between the populations** | Rejected. It did not improve the honest score. |
| **Random Forest** | Rejected. The optimizer gave it a weight of 0.000. |

## The reasons to trust this model

- Two of the three models are glass-box models. Only CatBoost is not readable, and its weight is
  one third.
- Each new feature comes from a result in the analysis. The limits are clinical limits. They are
  not arbitrary values.
- There is no leak. The notebook calculates each median on the train set only. It does not use
  `patient_id` as a feature. The prefix `TR` or `TE` in this column would show the set directly.
- The notebook controls the variance. It uses three fold seeds. It does not use one split.
- The score is honest. The notebook uses nested cross-validation for each reported number.
- The rejected treatments are in this document. Thus the simple final model is a measured
  decision.

## The limitations

- The ROC-AUC of 0.695 is a limit of this data. The model puts the patients in a useful risk
  sequence, but it is not exact. Use the model as an aid for triage. Do not use it as a decision
  rule.
- The test population has more missing values than the train population. The missing-value flags
  keep the decrease in performance small. But the score on the private test set can be worse than
  0.34855.
- The local score is not a leaderboard score. The value 0.34855 applies to the train
  distribution. The test distribution is a little different.
- `socioeconomic_index` shows almost no signal (correlation −0.002). This is a property of this
  synthetic dataset. Do not use it as a statement about the real world.
- The data is synthetic and educational. The model is not clinically validated. Do not use the
  model for a real patient.

## The next steps

- Do a subgroup error audit with confidence intervals for sex, rurality and age group.
- Make a calibration curve and a reliability diagram.
- Add a referral rule that uses the uncertainty. The rule sends the least reliable predictions to
  a person.

## The environment

The project uses pixi. The file `pixi.lock` contains the exact versions. The primary packages are
Python 3.14, pandas, scikit-learn, CatBoost, LightGBM and `interpret` for the EBM. The platform
is `osx-arm64`. To add a different platform, use `pixi workspace platform add`.
