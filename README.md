# ML & AI NEXUS 2026 — Round 1: Kaggle Challenge

**Theme:** *Beyond the Black Box: Statistics for Trustworthy AI*

## Problem Statement

You are part of an **AI Validation and Statistical Assurance Team** supporting a health system. The task is to predict the **probability that a patient will experience an unplanned readmission within 30 days**, using a synthetic hospital dataset.

**Central question:** *Can you build an AI model we can trust — not merely one that tops the leaderboard?*

The objective is not only an accurate predictive model, but one that is well-calibrated, robust, interpretable, and reliable across different subgroups and data conditions. Final evaluation weighs robustness, subgroup performance, statistical validation, explainability, uncertainty, reproducibility, and limitations — not leaderboard score alone.

Kaggle competition: [kaggle.com/competitions/ml-nexus-2026/data](https://www.kaggle.com/competitions/ml-nexus-2026/data)

## Final Model & Reproduction

**Core-3 honest stack** — seed-bagged **Spline GAM + EBM + CatBoost**, blended with SLSQP convex
weights and scored by nested cross-validation.

| | |
|---|---|
| Honest (nested-CV) OOF LogLoss | **0.34855** |
| ROC-AUC | 0.69458 |
| Brier | 0.10207 |
| Weights | Spline 0.333 · EBM 0.333 · CatBoost 0.333 |
| Submission | `submissions/submission_13_core3_honest.csv` |

LightGBM is deliberately excluded — the leave-one-out ablation (`evidence/ablation_table.py`)
showed that dropping it *improves* the honest score, 0.34912 → 0.34855, because it is redundant
with CatBoost and was being given a small weight that fit fold-specific noise. Scoring is
nested (weights fit on 4/5 of the OOF rows, scored on the held-out 1/5), since fitting the blend
weights and reading the score off the same rows overstates it.

### Reproduce

```bash
pixi install
pixi run all          # data -> features -> base learners -> seed-bagging -> final submission
pixi run evidence     # the ablation, adversarial-validation and honest-stacking analyses
pixi run report       # execute notebooks/final_model_report.ipynb top to bottom
pixi run lab          # interactive JupyterLab
```

`pixi run all` rebuilds `pipeline/artifacts/` (gitignored, fully regenerable) and rewrites
`submissions/submission_13_core3_honest.csv`.

### Layout

```
data/          raw competition csvs (read-only)
notebooks/     final_model_report.ipynb -- EDA, feature engineering, models, explainability, final stack
pipeline/      01-07, the submission path; artifacts/ is regenerable and gitignored
evidence/      analyses the report cites but that are not on the submission path
submissions/   the final submission
legacy/        superseded work, kept as the evidence trail -- see legacy/README.md
```

---

## Dataset

Synthetic hospital readmission data. **7,000 training** rows and **3,000 test** rows. The test set is deliberately designed so that public-leaderboard optimization alone may not produce the winning solution.

| File | Description |
|---|---|
| `data/train.csv` | Training set (includes `readmitted_30d` target) |
| `data/test.csv` | Test set (no target) |
| `data/sample_submission.csv` | Sample submission in the correct format |
| `data/data_dictionary.csv` | Variable descriptions (below) |

### Required submission format

Exactly two columns, with `readmitted_30d` a probability strictly between 0 and 1:

```
patient_id,readmitted_30d
TE00001,0.187421
TE00002,0.731005
```

### Data Dictionary (`data/data_dictionary.csv`)

| Variable | Type | Description | Trustworthy AI Note |
|---|---|---|---|
| patient_id | Identifier | Unique synthetic patient identifier | No |
| age | Numeric | Age in years | No |
| sex | Categorical | Recorded sex category | Sensitive attribute for subgroup audit |
| rurality | Categorical | Urban / Semi-urban / Rural residence context | Context / subgroup audit |
| socioeconomic_index | Numeric | Synthetic standardized socioeconomic index; higher = greater advantage | Context / subgroup audit |
| hospital_type | Categorical | Teaching / General / District hospital | Potential transportability factor |
| region | Categorical | Synthetic broad region | Potential transportability factor |
| prior_admissions_12m | Integer | Admissions during prior 12 months | Predictor |
| comorbidity_count | Integer | Count of documented chronic comorbidities | Predictor |
| diabetes | Binary | 1 if documented | Predictor |
| hypertension | Binary | 1 if documented | Predictor |
| chronic_kidney_disease | Binary | 1 if documented | Predictor |
| heart_failure | Binary | 1 if documented | Predictor |
| length_of_stay_days | Numeric | Index-admission length of stay | Predictor |
| medication_count | Integer | Number of medications at discharge | Predictor |
| missed_appointments_12m | Integer | Missed appointments in prior 12 months | Predictor |
| followup_days | Numeric | Days until planned follow-up; may be missing | Predictor / access-related |
| hemoglobin_g_dl | Numeric | Hemoglobin; may be missing | Predictor |
| creatinine_mg_dl | Numeric | Creatinine; may be missing | Predictor |
| sodium_mmol_l | Numeric | Serum sodium; may be missing | Predictor |
| heart_rate_bpm | Numeric | Heart rate; may be missing | Predictor |
| systolic_bp_mmhg | Numeric | Systolic blood pressure; may be missing | Predictor |
| discharge_disposition | Categorical | Discharge destination | Predictor |
| care_pathway | Categorical | Synthetic pathway/workflow code | Potentially unstable context variable |
| readmitted_30d | Binary target | 1 = unplanned readmission within 30 days | Target; train only |

## Evaluation Metric

**Binary Log Loss** (lower is better):

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(p_i) + (1-y_i)\log(1-p_i)\Big]$$

Log loss is used (rather than a 0/1 accuracy metric) because the challenge rewards **well-calibrated probabilities**, not just correct classifications — confidently wrong predictions are penalized heavily.

- **Public leaderboard:** ~30% of the test set
- **Private leaderboard:** ~70% of the test set, intentionally containing cases that test transportability and robustness
- The **overall event winner is decided by the Final Trustworthiness Score**, not Kaggle rank alone

## Competition Timeline

| Milestone | Date & Time |
|---|---|
| Start | Sunday, 13 September 2026 — 8:00 AM |
| End | Monday, 14 September 2026 — 8:00 AM (24 hours from start) |
| Submission limit | Max 10 per team, per day cap while the window is open |
| Trust Card + notebook link deadline | 14 September 2026 (Wed), 8:00 AM |
| Presentation upload deadline | 16 September 2026 (Wed), 11:59 PM |

**Submission window split:**
- First 5 submissions: Sunday 8:00 AM → Sunday midnight
- Next 5 submissions: Sunday midnight → Monday 8:00 AM
- No more than 2 submissions recommended in the final hour

**Links:**
- Competition link: shared by email and via the WhatsApp group at 8:00 AM on Sunday, 13 September
- Trust Card + notebook submission: https://forms.gle/6rcYfFzsz5L94Pwn7
- Presentation submission: https://forms.gle/ZBa72adq7aHei7iu9
- Presentation/viva time slots: shared separately to each team after the presentation submission deadline

## Team & Account Rules

- One Kaggle account per team; no multi-account signups or submissions.
- Use your **TEAM NAME** as the Kaggle screen name / username so organizers can identify you on the leaderboard.
- After logging into Kaggle, use the given link to reach the competition page and start working.
- Team mergers are **not** allowed.
- No private sharing of the test set's response variable — disqualification.
- Private sharing of code or external datasets outside the registered team is strictly prohibited.
- Recommended team size: 2–4 participants.
- All participants must follow standard Kaggle guidelines and the competition instructions, including the announced submission deadline, format, and upload method.

### Integrity / Prohibited

- Leaderboard probing or hidden-label reconstruction
- Manual label reconstruction
- Account sharing / submission collusion between teams
- Using personally identifiable or real patient data
- Editing the test set to manufacture an advantage
- Any method that cannot be documented and reproduced

### Allowed (unless organizers announce otherwise)

- Open-source Python/R packages
- Pretrained generic software libraries
- Standard statistical / ML algorithms
- Public package documentation
- **No external datasets** — do not enrich rows using outside patient-level, hospital-level, regional, or demographic data

### Responsible Interpretation

This is a **synthetic, educational dataset**. Do not present the model as clinically validated or suitable for real deployment.

### Final Ranking Verification

Organizers reserve the right to verify code and evidence before confirming awards. A non-reproducible or rule-violating submission may be removed from final consideration.

## What Judges Expect the Solution to Address

- Predictive discrimination and probability quality
- Calibration
- Missing values and data quality
- Model stability / robustness
- Subgroup reliability
- Uncertainty and when the model should defer to a human
- Explainability
- Reproducibility
- Limitations and safe-use boundaries

### Statistical evidence expected

Whenever claiming one model is better than another, avoid relying on a single validation score. Useful evidence includes:

- Repeated cross-validation
- Bootstrap confidence intervals
- Paired bootstrap comparisons
- Calibration curves / intercept / slope
- Brier score
- Subgroup confidence intervals
- Sensitivity analyses

## Final Deliverables

1. **Final Kaggle prediction file**
2. **Reproducible notebook/script** that recreates the final predictions (organizers must be able to rerun it against the supplied data and reproduce the prediction file within reasonable numerical tolerance)
3. **Two-page Model Trust Card** (template below)
4. **Presentation** (10 min slides + 5 min viva)

## Scoring & Marks Breakdown

### Overall Evaluation Structure

| Assessment Component | Weight | Main Focus |
|---|---|---|
| Kaggle Leaderboard Challenge | 40% | Predictive performance |
| Innovative Pitch Presentation | 40% | Approach, analysis, innovation, communication |
| Viva Session | 20% | Understanding and ability to defend the work |

**Final Score (out of 100) = Kaggle Leaderboard Marks (40) + Presentation Marks (40) + Viva Marks (20)**, summed directly. A high Kaggle score alone does not determine final standing — strong performance across all three components is required.

### Kaggle Leaderboard Challenge (40 Marks)

- **25 marks** — Kaggle Leaderboard Score (by rank, see table below)
- **15 marks** — Submitted Trust Card

Ties on the leaderboard receive identical marks corresponding to the higher rank position. Teams that do not submit a valid leaderboard entry receive 0 marks for this component and are **not eligible for the presentation/viva stage**.

| Rank | Marks | Rank | Marks |
|---|---|---|---|
| 1 | 25.0 | 10 | 16.0 |
| 2 | 24.0 | 11 | 15.0 |
| 3 | 23.0 | 12 | 14.0 |
| 4 | 22.0 | 13 | 13.0 |
| 5 | 21.0 | 14 | 12.0 |
| 6 | 20.0 | 15 | 11.0 |
| 7 | 19.0 | 16 | 10.0 |
| 8 | 18.0 | 17 | 9.0 |
| 9 | 17.0 | 18 | 8.0 |

### Presentation (40 Marks)

Weighted toward this year's theme — trustworthy AI, causal inference, statistical reasoning, explainable AI — alongside sound analytical practice and clear communication.

| Criterion | Marks |
|---|---|
| Problem understanding & industry perspective | 4 |
| Dataset description & exploratory findings | 4 |
| Analytical approach: preprocessing, model architecture & validation method | 8 |
| Results & Kaggle performance, compared with a suitable baseline | 6 |
| Model trustworthiness, interpretability, fairness, uncertainty & limitations | 8 |
| Challenges faced, solutions used & suggestions for improvement | 4 |
| Practical value, innovation & conclusion | 4 |
| Clarity of communication & time management (10 min) | 2 |
| **Total** | **40** |

**Format:** 15 minutes per team total — 10 minutes slides, 5 minutes viva. Briefly cover: team/institute intro; problem & industry perspective; dataset description & EDA findings; analytical approach (preprocessing, model architecture, validation, model selection rationale); main results vs. a suitable baseline; problems encountered & solutions; practical value & conclusion. Keep it focused on **interpretation of results**, not methodology minutiae.

### Viva Session (20 Marks)

| Criterion | Marks |
|---|---|
| Depth of understanding of the team's own work and analysis | 6 |
| Ability to justify and defend methodology, model, and design choices | 6 |
| Quality of responses on trustworthiness, interpretability, and the Stat Day theme | 5 |
| Team coordination and clarity in answering under time pressure | 3 |
| **Total** | **20** |

**Eligibility:** Only teams that (a) complete the Kaggle Challenge and (b) upload their presentation on time are considered for viva.

## Model Trust Card — Template

A two-page document summarizing why the model should or should not be trusted, complementing the leaderboard score.

**Team**
- Team name
- Members
- Final Kaggle submission filename

1. **Model summary** — final model(s); key preprocessing; key hyperparameters; why this model was selected
2. **Validation design** — train/validation strategy and why it's appropriate; uncertainty/variability across splits/resamples
3. **Performance** — Log Loss, Brier score, ROC-AUC, sensitivity/specificity at a chosen threshold (not just the best single split)
4. **Calibration** — calibration plot/summary; calibration method if used; evidence before vs. after calibration
5. **Robustness** — what might change between development and deployment populations; sensitivity tests run; which variables/modeling choices appeared unstable
6. **Subgroup reliability** — at minimum: sex, rurality, age group, hospital type; report group sizes and uncertainty; do not treat differences as automatically discriminatory or causal
7. **Uncertainty / human referral** — how cautious predictions were identified; effect on performance if abstaining/referring the 10% most uncertain cases
8. **Explainability** — most influential predictors; one local explanation each for a high-risk and a low-risk prediction; distinguish association from causation
9. **Failure modes** — at least three concrete ways the model may fail
10. **Deployment recommendation** — one of: *Ready for limited prospective validation* / *Requires additional model development* / *Should not be deployed* — justified in ≤100 words
11. **Reproducibility** — software/package versions; random seed(s); approximate training time; AI-assistant use (if permitted)
12. **One-sentence conclusion** — "We trust this model only when…"

## Note on Model Choice: XGBoost/LightGBM/CatBoost Is NOT Required

**It is absolutely NOT a requirement to use XGBoost, LightGBM, or CatBoost.** The competition rules do not restrict the choice of algorithm. Given the theme — *"Beyond the Black Box: Statistics for Trustworthy AI"* — blindly reaching for a heavy gradient-boosted tree model can actually work against a team if they cannot explain or defend it.

### Why people default to GBDTs

- On raw tabular data with mixed numeric/categorical features, gradient boosted decision trees (GBDTs) handle missing values and non-linearities automatically with little feature engineering.
- They often post strong numbers on the public leaderboard.

**But in this competition:**
- The public leaderboard is only ~30% of the test set; the private 70% is explicitly designed to test **robustness and transportability** across hospitals/regions.
- The Kaggle score is only **25% of the total mark**. The other **75%** comes from the Trust Card, presentation, and viva.
- Heavily boosted trees often memorize spurious correlations (e.g. `care_pathway`, which the data dictionary flags as a *"potentially unstable context variable"*) and can fail when deployed on new hospital distributions.

### Strong alternatives that fit the theme

**A. Explainable Boosting Machines (EBMs, via `interpret`)**
- A modern Generalized Additive Model (GAM) from Microsoft Research (`pip install interpret`).
- Matches XGBoost/LightGBM-level accuracy on tabular data.
- **100% glass-box interpretable** — exact curves for how each variable (e.g. `creatinine`, `age`, `socioeconomic_index`) affects readmission risk, no SHAP approximation needed.

**B. Generalized Additive Models (GAMs / splines)**
- Fits smooth curves per feature instead of a straight line:
  $$\text{logit}(p) = f_1(\text{age}) + f_2(\text{creatinine}) + \dots$$
- Captures non-linear clinical thresholds (e.g. a sharp risk jump when `hemoglobin` drops below 10) without the erratic step-functions of tree models.

**C. Enhanced Logistic Regression (with feature engineering)**
- Plain logistic regression plus domain-informed features: interaction terms (`age * comorbidity_count`, `socioeconomic_index * rurality`), spline transforms or clinical binning for lab values (`creatinine`, `sodium`, `systolic_bp`).
- Naturally smooth, close-to-calibrated output probabilities.
- Generalizes better across hospitals since it avoids overfitting complex high-order interactions; fast, transparent, and easy to audit across subgroups (sex, rurality).

**D. Random Forests / Extra Trees**
- Bagged (not boosted) tree ensembles.
- Averaging trees gives smoother probability estimates than boosting, which aggressively targets hard cases and can yield overconfident probabilities — less prone to catastrophic log-loss penalties.

### Turning the model choice into marks

The rubric directly rewards this reasoning:
- **8 marks** — Analytical approach & validation method
- **8 marks** — Trustworthiness, interpretability, fairness, uncertainty & limitations
- **6 marks (viva)** — Ability to justify and defend methodology and design choices

A defensible pitch for a simpler/glass-box model (EBM or penalized logistic regression with splines) over XGBoost:

> "We evaluated black-box gradient boosted trees, but chose an Explainable Boosting Machine / regularized model because: (1) it provides exact transparency into risk drivers for clinical staff; (2) it avoids memorizing unstable hospital workflow artifacts (`care_pathway`); (3) it produces stable, well-calibrated probabilities across patient subgroups without risking extreme overconfidence on unseen test data."

Judges in a "Beyond the Black Box" competition should score that reasoning higher than a team that threw XGBoost at the data without understanding why. **Use whatever model you understand well and can thoroughly validate and explain.**

## Repository Contents

This repo tracks the team's working solution for the challenge.

```
csvs/       Provided competition data (train/test/sample submission/data dictionary)
scripts/    Exploratory and modeling notebooks (EDA, baselines, logistic regression)
outputs/    Generated submission files
score_cal.ipynb   Expected-score estimator for sanity-checking submissions pre-upload
```
