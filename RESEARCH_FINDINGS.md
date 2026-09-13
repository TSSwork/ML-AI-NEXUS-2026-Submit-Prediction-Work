# Research Findings — 30-Day Readmission Prediction

A consolidated report of everything done across `scripts/*.ipynb` and `score_cal.ipynb`: data curation, EDA, modeling methodology, and results, in the order the work was actually carried out (per commit history: baseline v1.1 → submission 1 → estimated score calculator → logistic regression v2 → linear reg submission).

---

## 1. Data Integrity Checks (`scripts/check-data.ipynb`)

First pass before any modeling — pure data-quality and leakage audit.

- **Shapes:** train `(7000, 25)`, test `(3000, 24)`. Only column difference: `readmitted_30d` (target), present only in train.
- **Target distribution:** 6,119 negatives / 881 positives → base rate **p₀ = 0.125857** (12.6% readmission rate). This value matches `sample_submission.csv`'s constant exactly, confirming the sample submission is a naive constant-rate baseline.
- **Missing values** (train% / test%, all higher in test):

  | Column | Train missing | Test missing |
  |---|---|---|
  | creatinine_mg_dl | 4.83% | 7.07% |
  | heart_rate_bpm | 4.81% | 6.87% |
  | hemoglobin_g_dl | 4.63% | 6.57% |
  | sodium_mmol_l | 4.44% | 6.40% |
  | systolic_bp_mmhg | 4.41% | 5.90% |
  | followup_days | 2.33% | 3.33% |

  All other columns are complete. **Test missingness is consistently 1.5–2.2 points higher than train** for every affected column — an early, direct signal of train/test distribution shift that later work treats as a robustness risk.
- **Dtypes (train):** 9 float64, 9 int64, 7 object.
- **Leakage / ID checks:** `patient_id` is unique in both sets, zero ID overlap between train/test, and IDs carry a `TR…`/`TE…` prefix — flagged explicitly as **never usable as a feature** (prefix alone would leak train/test membership). Target confirmed absent from test.
- **Data dictionary trustworthy-AI notes** were pulled in programmatically at this stage (not just read manually) — `sex` (sensitive/subgroup audit), `rurality` & `socioeconomic_index` (context/subgroup audit), `hospital_type` & `region` (transportability), `care_pathway` (potentially unstable context variable) — these flags directly shaped which slices got audited in every later notebook.

**Outcome:** dataset is clean (no leakage, no duplicate IDs), but flagged for (a) missingness that's worse in test, and (b) several categorical variables explicitly at risk of population shift or unfair subgroup treatment.

---

## 2. Exploratory Data Analysis (`scripts/describe_data.ipynb`)

- **Target:** 12.6% positive class → substantial imbalance, reinforcing that log loss / calibration (not accuracy) is the right lens.
- **Numeric feature distributions** (age, socioeconomic_index, prior_admissions_12m, comorbidity_count, length_of_stay_days, medication_count, missed_appointments_12m, followup_days): compared train vs. test density plots side by side to check shape drift.
- **Categorical proportions, train vs. test** — cardinalities matched exactly (sex: 2, rurality: 3, hospital_type: 3, region: 4, discharge_disposition: 4, care_pathway: 4 — same categories in both, no new/unseen levels), but *mix* shifted between splits, most notably:
  - `region`: West 37% → 33%, North-East 19% → 22%
  - `hospital_type`: District 18% → 23%, Teaching 44% → 38.5%
  - `care_pathway`: P1 33% → 28.5%, P4 13.3% → 17.8%

  These specific shift numbers were reused verbatim in later notebooks to justify *why* `region`, `hospital_type`, and `care_pathway` needed dedicated robustness/sensitivity tests.
- **Subgroup readmission rates** computed for sex, rurality, hospital_type, region, and an engineered `age_group` (bins: `<30`, `30-50`, `50-70`, `70+`) — this is the same age binning reused in every subsequent notebook. The **age gradient was the strongest signal found in the whole EDA**: readmission rate climbs from ~3.8% (`<30`) to ~22.3% (`70+`).
- **Missingness-vs-outcome analysis:** for each of the 6 partially-missing columns, compared the readmission rate among rows *with* the value missing vs. present, to check whether missingness itself carries signal (informative missingness) rather than being purely random.

**Outcome:** established the exact feature engineering (age bins, missingness flags) and the exact shift magnitudes (region/hospital_type/care_pathway) that every downstream model and audit is built around.

---

## 3. Baseline 1 — Constant Prevalence Model (`scripts/basline_model_1.ipynb`, submitted via `baseline_model_1_submission.ipynb`)

### Method
No features at all — every prediction is the global training prevalence `p₀ = 0.125857`. This is deliberately the "floor" model: any real model must beat its per-slice numbers to justify its added complexity.

### Overall performance (on train, since predictions are constant)

| Metric | Value |
|---|---|
| Log Loss | **0.378435** |
| Brier score | 0.110017 |
| ROC-AUC | 0.500000 (expected — no discrimination possible) |

### Trustworthy-AI audit performed (Q1–Q7 structure, reused in every later notebook)

- **Q1 — Missingness strata:** performance invariant by construction (same prediction regardless of missingness), but *observed* rates differ slightly by stratum (1-missing: 13.1% vs 0-missing: 12.5% vs 2+-missing: 12.4%) → small calibration gaps within strata. A synthetic **+20% extra missingness stress test** was run to prove the harness works (confirmed invariant, as expected for a feature-less model).
- **Q2 — Region slices:** South (13.4% event rate, LogLoss 0.395) vs North-East (11.5%, LogLoss 0.357) — global constant produces ~±1pp calibration gaps purely from regional prevalence differences.
- **Q3 — Hospital type slices:** smaller gaps (±0.5pp) — Teaching (13.0%), District (12.4%), General (12.2%).
- **Q4 — care_pathway:** invariant by construction (no features used), but per-pathway rates already show the divergence that matters later: P4 = 15.7% vs P1/P2 ≈ 11.9–12.0%. Combined with the EDA's known test-set shift (P1 down, P4 up), this foreshadows a **calibration-mix problem in the private leaderboard**.
- **Q5 — Calibration:** perfect calibration-in-the-large on train (intercept = 0 by construction), but breaks down badly in slices — most severely the **70+ age group: predicted 12.6% vs observed 22.3% (calibration intercept 0.688 in logit space)**, and `<30`: predicted 12.6% vs observed 3.8% (intercept −1.296). A **covariate-reweighting exercise** (reweighting train rows by test-set category marginals, without using test labels) estimated the constant model's implied test-time rate shifts by +0.17–0.18pp for rurality/care_pathway.
- **Q6 — Subgroup reliability (sex, rurality, SES quintile):** sex gap ≈ 0 (12.8% Male vs 12.4% Female — no material fairness concern under a constant model); rurality showed a **real gap: Rural 15.6% vs Semi-urban 10.9%** (+3pp, CI [0.137,0.177] vs [0.096,0.124] — non-overlapping); SES quintiles showed a mild pattern (lowest and highest quintiles both ~13.2–13.4% vs middle quintiles ~12.0–12.2%). All reported with Wilson 95% confidence intervals to avoid over-interpreting small-sample subgroup gaps.
- **Q7 — Abstention/uncertainty:** entropy is identical for every patient under a constant model, so "referring the most uncertain 10%" is undefined — a random 10% was referred instead purely to validate the harness (LogLoss barely moved: 0.3796 vs 0.3784). Explicitly documented: **a constant model can never be overconfident (bounded failure mode = underfitting), but also cannot discriminate (AUC 0.5)** — its abstention mechanism only becomes meaningful once real per-patient probabilities exist.

### Submission
`submission_01_constant.csv` — all 3,000 test rows at 0.125857, with ID-order and probability-bounds assertions before saving.

---

## 4. Baseline 2 — Logistic Regression v1 (`scripts/baseline_linear_reg.ipynb`)

### Feature engineering / data curation
- Missingness indicator flags added for each of the 6 partially-missing columns (`<col>_flag`), plus an `n_missing` count column.
- `age_group` bins added (`<30`, `30-50`, `50-70`, `70+`), matching the EDA notebook exactly for numeric comparability.
- **Numeric features (14 + 6 flags):** age, socioeconomic_index, prior_admissions_12m, comorbidity_count, length_of_stay_days, medication_count, missed_appointments_12m, followup_days, hemoglobin_g_dl, creatinine_mg_dl, sodium_mmol_l, heart_rate_bpm, systolic_bp_mmhg, n_missing + 6 missing-flags.
- **Categorical features (6):** sex, rurality, hospital_type, region, discharge_disposition, care_pathway.
- `patient_id` explicitly dropped (leakage risk established in the data-integrity notebook).

### Preprocessing pipeline
- Numeric: `SimpleImputer(strategy='median')` → `StandardScaler`.
- Categorical: `SimpleImputer(strategy='constant', fill_value='missing')` → `OneHotEncoder(handle_unknown='ignore')`.
- Combined via `ColumnTransformer`, feeding a `LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)` — chosen deliberately as "the standard interpretable baseline" with default regularization to keep probabilities naturally calibrated.

### Validation
5-fold `StratifiedKFold` (shuffle=True, seed=42) generating **out-of-fold (OOF)** predictions — i.e., every train row scored only by a model that never saw it, avoiding optimistic in-sample bias.

### Overall OOF performance

| Metric | Value | vs. constant baseline |
|---|---|---|
| Log Loss | **0.352613** | −0.025822 (↓ 6.8%) |
| Brier score | 0.103179 | −0.006838 |
| ROC-AUC | **0.685290** | +0.185 (from 0.5) |

### Trustworthy-AI audit (same Q1–Q7 structure, now on a real model)

- **Q1 — Missingness strata:** LogLoss now varies meaningfully by stratum (1-missing: 0.370, 0-missing: 0.348, 2+-missing: 0.357) since the model actually uses the missingness flags.
- **Q4 — care_pathway sensitivity (new test):** trained the identical pipeline *with* vs. *without* `care_pathway` as a feature. LogLoss with: 0.352613, without: 0.353296 — a difference of only 0.0007, below the notebook's own 0.002 threshold → **conclusion: `care_pathway` adds negligible predictive value, and given it's flagged as an unstable/shifting variable, it's safe to drop for robustness.**
- **Q2/Q3 — Region & hospital type:** LogLoss and calibration gaps recomputed per slice; gaps stayed small (calib_gap within ±0.0007) — model calibration held up reasonably well across these slices, unlike the constant baseline's larger regional gaps.
- **Q5 — Calibration by slice:** age_group calibration much improved vs. the constant model — 70+ calib_gap essentially closed (−0.0008 vs the constant model's +0.097); remaining largest gap is 30-50 at +0.0069 (intercept +0.10).
- **Q6 — Subgroups:** sex, rurality, SES-quintile calibration gaps all near-zero (max ~0.004), a large improvement over the constant baseline's rurality gap of +0.030.
- **Q7 — Abstention:** now meaningful, since predicted probabilities vary per patient. Referring the top-10% highest-entropy (most uncertain) patients to a human reviewer **improved LogLoss on the retained 90% from 0.352613 → 0.324062 (−0.0286)** — interpreted as a genuine success signal (the model's uncertainty estimate is informative, not just noise).

### Submission
None yet at this stage — this notebook is exploratory/audit-only; the actual logistic-regression submission is produced in v2 below.

---

## 5. Logistic Regression v2 — Extended Robustness & Explainability (`scripts/baseline_linear_reg_v2.ipynb`)

Same pipeline, features, and OOF setup as v1 (confirmed identical OOF Log Loss 0.35261 / AUC 0.68529), extended with deeper statistical validation and interpretability work — this is the notebook that produced the final submitted model.

### New analysis added

1. **Subgroup "skill score"** (LogLoss saved vs. a per-slice constant floor, not just the global floor) — measures how much real value the model adds *within* each slice, not just overall:

   | Slice | Best skill (Δ LogLoss) | Worst skill |
   |---|---|---|
   | rurality | Rural: 0.0316 | Urban: 0.0208 |
   | region | Central: 0.0347 | West: 0.0172 |
   | hospital_type | District: 0.0346 | Teaching: 0.0213 |
   | age_group | 70+: 0.0280 | **<30: −0.0039 (model is *worse* than a local constant here)** |

   This is an important nuance missed by the overall metric: the model actively **underperforms a naive local baseline for the youngest age group** (small n=343, low base rate 3.8%), a candidate limitation to disclose in the Trust Card.

2. **Abstention control test** — entropy-based top-10% referral vs. a random 10% referral, to prove the improvement isn't a statistical artifact of just removing any 10% of cases:
   - All 100%: 0.35261
   - Entropy-based referral (10%): 0.32406 (**real improvement of 0.02855**)
   - Random referral (10%): 0.35579 (**slightly worse — confirms entropy-based selection is doing real work, not just variance reduction from a smaller sample**)
   - Clinical framing: the most uncertain 10% of patients contains **222 of 881 total readmission events (25.2%)** — i.e., referring 10% of the population for human review would surface a quarter of all true readmissions for extra scrutiny.

3. **`care_pathway` sensitivity, now with Repeated Stratified CV** (5-fold × 3 repeats = 15 folds, for a variance estimate rather than a single-split comparison):
   - With care_pathway: 0.35245 ± 0.00681
   - Without care_pathway: 0.35310 ± 0.00668
   - Mean difference: 0.00065 ± 0.00077 (difference is smaller than its own standard deviation → **not statistically distinguishable from zero**, reinforcing the v1 conclusion that this unstable feature is safe to drop).

4. **Missingness stress test under CV** (not just a single split): artificially masked an extra 20% of lab/vital values in the *validation* fold only, across all 5 folds:
   - Normal CV LogLoss: 0.35261
   - Stressed CV LogLoss: 0.35340
   - Degradation: **+0.00078** — small and consistent, suggesting the model (with missingness flags + median imputation) degrades gracefully rather than catastrophically under worse missingness, which is plausible given the observed test set actually does have higher missingness than train.

5. **Calibration curve (reliability diagram):** 10-bin uniform calibration curve plotted against the perfect-calibration diagonal (visual output, not reproduced here as text, but generated and reviewed).

6. **Explainability — logistic regression coefficients as odds ratios** (model refit on full training data for final interpretation):

   | Feature | Odds Ratio | Interpretation |
   |---|---|---|
   | age | 1.581 | Strongest single risk driver — older age raises odds substantially |
   | prior_admissions_12m | 1.391 | Second-strongest risk driver |
   | comorbidity_count | 1.241 | Additional comorbidities raise risk |
   | discharge_disposition_Home | 0.685 | Discharge home (vs. other dispositions) lowers risk |
   | sex_Female | 0.705 | (one-hot level; interpret jointly with sex_Male below) |
   | rurality_Semi-urban | 0.709 | Lower risk than reference level |
   | sex_Male | 0.724 | |
   | care_pathway_P1 | 0.751 | |
   | hospital_type_District | 0.756 | |
   | care_pathway_P2 | 0.762 | |

   Age, prior admissions, and comorbidity count stand out as the clinically sensible, dominant, *stable* predictors — reassuring from a trustworthiness standpoint since these are exactly the kind of established clinical risk factors a reviewer would expect, rather than the model leaning on unstable context variables like `care_pathway`.

### Final submission generation
- Final model refit on **full training data** (not just OOF folds).
- Predicted on test set, probabilities clipped to `[1e-6, 1-1e-6]` to avoid infinite log-loss penalties from a 0/1 edge case.
- Sanity checks before saving: row count match, ID order match, probability bounds check.
- **Mean predicted probability on test: 0.13747** vs. **train prevalence: 0.12586** — the model predicts a noticeably higher average risk on test than the observed train base rate, consistent with the EDA finding that test skews toward higher-risk categories (more Rural/District/P4 mix).
- Saved as `outputs/submission_02_logreg.csv`.

---

## 6. Pre-Submission Expected-Score Estimator (`score_cal.ipynb`)

A label-free sanity check built to estimate what a submission might score on the leaderboard **before spending one of the limited daily submissions**, since true test labels are never available locally.

### Method
For each test row, assigns a pseudo-truth rate equal to the **train event rate within its (age_group × rurality × care_pathway) slice**, smoothed toward the global rate with a Beta prior (`k=100`) so small/sparse slice combinations don't produce extreme pseudo-rates. Then computes the expected log loss of the submitted probabilities against these pseudo-rates. Explicitly documented as **an estimate, not the true leaderboard score**.

### Results

| Submission | Pred mean/min/max | Expected Log Loss (slice-rate estimate) |
|---|---|---|
| `submission_01_constant.csv` | 0.1259 / 0.1259 / 0.1259 | **0.381996** |
| `submission_02_logreg.csv` | 0.1375 / 0.0149 / 0.7924 | **0.399708** |

### ⚠️ Notable finding — a discrepancy worth flagging in the Trust Card

Under this slice-based pseudo-truth estimate, the **logistic regression submission scores *worse* (0.3997) than the constant baseline (0.3820)** — the opposite of what the rigorous 5-fold OOF cross-validation showed on train (LR: 0.3526 vs constant: 0.3784, LR clearly better). Plausible explanations to investigate/disclose:

- The estimator's pseudo-truth is itself derived only from a 3-way slice interaction on **train** rates — it does not reflect genuine test labels, so it may simply be a poor proxy, especially since the logistic regression's much wider probability range (0.015–0.792) will be penalized heavily by log loss if the crude slice-rate proxy disagrees with the model at the tails, even in cases where the model is actually right.
- Alternatively, this could be an early warning that the logistic regression is **overconfident in some slices relative to how the true (unknown) test population is distributed** — exactly the kind of miscalibration-under-shift risk the competition's private leaderboard is designed to expose.
- This tension should be explicitly discussed in the Model Trust Card's **Robustness** and **Calibration** sections rather than silently resolved — it's a genuine, documented piece of statistical evidence the judges will want to see addressed, not hidden.

---

## 7. Summary of Models & Performance

| Model | File(s) | Validation method | Log Loss | Brier | ROC-AUC | Status |
|---|---|---|---|---|---|---|
| **Constant baseline** (p₀ = 0.1259) | `basline_model_1.ipynb` → `submission_01_constant.csv` | N/A (no features) | 0.378435 | 0.110017 | 0.500000 | Submitted |
| **Logistic Regression v1** | `baseline_linear_reg.ipynb` | 5-fold StratifiedKFold OOF | 0.352613 | 0.103179 | 0.685290 | Exploratory only, not submitted |
| **Logistic Regression v2 (final)** | `baseline_linear_reg_v2.ipynb` → `submission_02_logreg.csv` | 5-fold OOF + 15-fold RepeatedStratifiedKFold sensitivity checks | 0.352613 (OOF); 0.35245 ± 0.00681 (repeated CV) | — | — | **Submitted (current best)** |

**Expected/estimated leaderboard log loss** (label-free slice-rate proxy, not a true score): constant = 0.382, logistic regression = 0.400 — flagged above as an open question rather than a settled result.

---

## 8. Consolidated Trustworthy-AI Findings (for the Model Trust Card)

- **Calibration:** Logistic regression is well-calibrated in aggregate and across most slices (gaps mostly < 1pp), a large improvement over the constant baseline's largest gap (70+ age group, +9.7pp). Weakest calibration remains in the 30-50 age band (+0.7pp) — still small in absolute terms.
- **Robustness / stability:** `care_pathway` — flagged by the data dictionary as unstable and confirmed to shift meaningfully between train/test (EDA) — was formally tested twice (single-split and 15-fold repeated CV) and shown to add negligible, statistically indistinguishable-from-zero predictive value (0.00065 ± 0.00077). Recommendation: consider dropping it, or at minimum treat it as a documented risk rather than a relied-upon predictor.
- **Missingness robustness:** an artificial +20% missingness stress test degraded CV log loss by only +0.00078 — graceful, not catastrophic, degradation. Relevant because test missingness is empirically higher than train for every affected column.
- **Subgroup reliability:** sex shows no material gap under either model. Rurality and the 70+ age group show the largest gaps under the constant model, both substantially closed by the logistic regression, with Wilson 95% CIs reported throughout rather than bare point estimates, and explicit language that differences are **associational, not causal**.
- **Uncertainty / human referral:** entropy-based abstention on the logistic regression is genuinely informative (real 10% referral beats random 10% referral: 0.324 vs 0.356 LogLoss on the retained set) and would surface 25.2% of true readmission events for extra human review if the top-10%-most-uncertain patients were referred.
- **Explainability:** age, prior admissions, and comorbidity count are the dominant, clinically sensible risk drivers (odds ratios 1.58, 1.39, 1.24) — the model is not leaning on the flagged unstable variable (`care_pathway`) for its main signal.
- **Known limitation:** the model underperforms a naive local baseline specifically in the `<30` age group (skill score −0.0039, small n=343) — a concrete failure mode to disclose.
- **Open risk to resolve before final submission:** the label-free expected-score estimator ranks the logistic regression *behind* the constant baseline, contradicting the OOF cross-validation. This conflict is unresolved in the current notebooks and should be investigated (e.g., by refining the slice-rate estimator, checking calibration specifically in the tails, or testing an isotonic/Platt calibration layer) before treating the logistic regression as a confirmed improvement over the constant floor.
