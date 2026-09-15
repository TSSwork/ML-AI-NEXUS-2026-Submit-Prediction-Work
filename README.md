# 30-Day Hospital Readmission — ML & AI NEXUS 2026

Predicting the probability of unplanned readmission within 30 days on a synthetic hospital
dataset, for a competition judged on **trustworthiness** — calibration, robustness, subgroup
fairness, explainability and reproducibility — not leaderboard rank alone.
The full competition brief, rules and data dictionary are in [COMPETITION.md](COMPETITION.md).

**Everything is in one notebook: [`notebooks/final_model_report.ipynb`](notebooks/final_model_report.ipynb).**
It runs top to bottom from `data/` and writes the final submission.

---

## Final model

**Core-3 honest stack** — seed-bagged **Spline GAM + EBM + CatBoost**, convex-blended with SLSQP
weights and scored by nested cross-validation.

| | |
|---|---|
| **Honest (nested-CV) OOF Log Loss** | **0.34855** |
| ROC-AUC | 0.69458 |
| Brier | 0.10207 |
| Weights | Spline GAM 0.333 · EBM 0.333 · CatBoost 0.333 |
| Output | `submissions/submission_13_core3_honest.csv` |

Against the baselines built along the way: **0.37844** (constant prevalence) → **0.35261**
(logistic regression) → **0.34855** (Core-3).

The optimizer settling on **equal thirds** is a result in itself: no family dominates, so the
blend is not one model with two decorations, and it is about as robust to any single model's
failure mode as a three-model blend can be.

## Quick start

```bash
pixi install
pixi run lab       # open the notebook interactively
pixi run report    # execute it end to end in place (~20 min)
```

`pixi run report` regenerates `submissions/submission_13_core3_honest.csv`. This has been
verified to reproduce the committed file **byte for byte**.

```
data/          raw competition csvs (read-only)
notebooks/     final_model_report.ipynb -- the entire project
submissions/   submission_13_core3_honest.csv (final)
legacy/        earlier baseline notebooks and superseded submissions
COMPETITION.md the competition brief, rules and data dictionary
```

---

## How the notebook is organised

### 1. Data audit and validation (cells 0–5)

Train `(7000, 25)`, test `(3000, 24)`, base readmission rate **12.59%**.

A **variable audit table** profiles all 24 features at once — dtype, train/test missingness,
cardinality, distribution summary, and each feature's readmission signal (correlation for
numerics, rate spread for categoricals). Strongest raw signals:

| Feature | Signal |
|---|---|
| `age` | corr **+0.180** |
| `prior_admissions_12m` | corr **+0.146** |
| `comorbidity_count` | corr **+0.117** |
| `chronic_kidney_disease` | rate spread 12.0% → **20.3%** (Δ 8.3pp) |
| `heart_failure` | rate spread → **27.3%** |
| `rurality` | rate spread 10.9% → 15.6% (Δ 4.7pp) |
| `socioeconomic_index` | corr −0.002 — essentially no marginal signal |

A **form-style validation schema** then checks every column against biological plausibility
ranges and permitted-missingness rules, producing a pass/fail report rather than an eyeballed
`.describe()`. This is the reproducibility and data-integrity evidence for the Trust Card.

**Missingness is not random, and it is worse in test** (train 2.3–4.8%, test 3.3–7.1% on the six
affected columns). Missingness itself carries signal:

| Column | Rate when missing | Rate when present | Δ |
|---|---|---|---|
| `creatinine_mg_dl` | 13.9% | 12.5% | **+1.4pp** |
| `heart_rate_bpm` | 13.6% | 12.5% | +1.1pp |
| `systolic_bp_mmhg` | 13.3% | 12.6% | +0.7pp |
| `sodium_mmol_l` | 10.9% | 12.7% | **−1.7pp** |

The sodium result has a clinical reading: patients who never get a full electrolyte panel tend to
be the straightforward, healthier admissions. Because missingness is informative in *both*
directions, it is encoded as explicit features rather than silently imputed away.

### 2. EDA (cells 6–12)

Categorical and binary features as bar charts with readmission rate overlaid; discrete counts
with their risk trend; continuous labs, vitals and demographics as train-vs-test KDE curves — so
distribution shift is visible per feature, not just asserted.

The age gradient is the dominant pattern: readmission climbs from ~3.8% (under 30) to ~22.3%
(70+). Category *mixes* also shift between train and test (`region`, `hospital_type`,
`care_pathway`), which is what motivates the robustness work later.

### 3. Feature engineering (cells 13–14) — 25 → 38 columns

Every added feature is traceable to something visible in the EDA:

- **Missingness flags** per affected column, plus `n_missing` — because missingness is
  informative (above).
- **`severe_cardiorenal`** — heart failure ∧ CKD, the two highest-risk comorbidities (27.3% and
  20.3%).
- **`high_utilizer`** (≥3 prior admissions) and **`risk_load`** (`2 × prior_admissions +
  comorbidity_count`) — from the monotonic, steep prior-admissions trend.
- **`age_x_comorbidity`** — the age/comorbidity interaction the KDE curves show.
- **`anemia_flag`** (Hb < 11.5) and **`elevated_creatinine`** (Cr > 1.3) — clinical thresholds
  placed where the KDE tails diverge.

Continuous labs are imputed with **train medians only**, so no test information leaks into
training.

### 4. Model families (cells 15–25)

Four deliberately different families, so a blend averages over genuinely different inductive
biases rather than re-runs of one:

| Family | Why it is here | 5-fold OOF Log Loss |
|---|---|---|
| **CatBoost** (symmetric trees) | Strong on tabular data, native categoricals, less prone to overfitting 7,000 rows than leaf-wise boosting | 0.35038 |
| **EBM** (glass-box GA²M) | Exact per-feature shape functions + learned two-way interactions | 0.35034 |
| **Spline GAM** (natural cubic B-splines → L2 logistic) | Parametric, smooth, fully inspectable; the logistic baseline without the linearity assumption | 0.35039 |
| **LightGBM** (leaf-wise GBDT) | The standard strong baseline — a candidate, ultimately dropped | 0.35262 |
| Random Forest | Tested, given weight 0.000 by the optimizer | 0.35289 |

All four beat the logistic baseline (0.35261), and they land within 0.002 of each other — close
in strength but disagreeing on individual patients, which is exactly the condition under which
blending pays.

### 5. Explainability (cells 19–23)

An EBM trained on the full dataset exposes the model's **actual decision surface**, not a
post-hoc approximation of it:

- **Base intercept β₀ = −2.1242** in logit space → a 10.68% base probability, before any feature
  contributes.
- **Shape function plots** per feature, with the exact piecewise contribution table printed as
  odds ratios.
- `creatinine_mg_dl` is flat and safe below 1.5 and then **spikes at 1.8–2.0 mg/dL** — which is
  where acute kidney injury and severe renal impairment actually sit clinically. The model
  learned real pathophysiology, not an artefact.
- `age` flips from protective to high-risk at **58**.
- `prior_admissions_12m` climbs linearly to 3 admissions, then flattens (OR 0.91 → 1.03 → 1.13 →
  1.29).

This is the core of the "Beyond the Black Box" argument: **two of the three models in the final
stack are glass-box**, so two thirds of the final prediction is directly inspectable.

### 6. The final stack (cells 26–29)

**Seed bagging.** On 7,000 rows a single 5-fold split carries real fold-to-fold variance, so each
family is trained across 3 `StratifiedKFold` seeds (42, 7, 123) and averaged. This is worth
0.0007–0.0014 on its own:

| Model | Single seed | 3-seed bagged |
|---|---|---|
| EBM | 0.35034 | **0.34928** |
| CatBoost | 0.35038 | **0.34965** |
| Spline GAM | 0.35039 | **0.34991** |
| LightGBM | 0.35262 | 0.35124 |

**Honest scoring.** Fitting the blend weights on all 7,000 OOF rows and then reporting the loss
on those same rows overstates the result. Every number above is *nested*: weights fit on 4/5 of
the OOF rows, scored on the held-out 1/5, rotated over 5 folds. Measured optimism on the 4-way
stack: same-data 0.34913 vs honest 0.34964 — small, but the same order of magnitude as the
differences being used to choose models, which is precisely why it had to be corrected first.

**Why LightGBM is dropped.** Leave-one-out over the seed-bagged models:

| Configuration | Honest Log Loss | vs. 4-way |
|---|---|---|
| All four | 0.34912 | — |
| **drop LightGBM** | **0.34855** | **−0.00057 (better)** |
| drop CatBoost | 0.34913 | +0.00001 |
| drop EBM | 0.34933 | +0.00021 |
| drop Spline GAM | 0.34937 | +0.00025 |

LightGBM is the only model whose removal *improves* the stack. It is the weakest solo model and
adds nothing once CatBoost is present — both are GBDTs over the same features. Its honest
per-fold weight swings between 0.000 and 0.064, i.e. the optimizer cannot find a stable role for
it. Same-data scoring hides this completely.

---

## Robustness and transportability

**Adversarial validation** — label train rows 0 and test rows 1, then try to tell them apart
using the same features the model uses:

- **OOF ROC-AUC 0.555** — above 0.50, so the shift is **real but mild**. The populations are only
  weakly separable jointly, even though individual columns shift visibly.
- Top separating features: `age` (54.2), `length_of_stay_days` (35.8), `n_missing` (34.6),
  `care_pathway` (34.0), `socioeconomic_index` (27.8), `followup_days` (27.6),
  `hospital_type` (26.4). `n_missing` appearing this high confirms the higher test missingness is
  a genuine distributional difference, not sampling noise.
- Converting this into importance-sampling weights (`w = p/(1−p)`, clipped and renormalized) and
  retraining **did not improve** the honest score. The shift is real but too mild for the
  correction to pay for the variance it adds.

**For the Trust Card:** this is a quantified transportability risk with a measured attempt to
correct it, not an unexamined assumption. The honest local score should be expected to degrade
somewhat on a private test set drawn from a mildly shifted population.

## Treatments tested and rejected

Each was implemented and measured against the 0.34855 honest baseline. None beat it — and that
is the argument for the final model's simplicity.

| Treatment | Outcome |
|---|---|
| **LightGBM in the stack** | Rejected — *worse* by 0.00057 honest. |
| **K-Means clinical phenotypes** (cluster id, distance-to-centroid, shock index) | Rejected — hurt every base model. Built from columns the models already see, so they add collinear noise. |
| **Tabular MLP** (2-layer 64×32, early stopping) | Never competitive on 7,000 rows. |
| **Logit-space stacking** | No gain over the probability-space convex blend. |
| **LDA / complementary log-log learners** | Rejected — weaker solo (0.35591 / 0.35408) and not complementary. Core-3 + LDA 0.34879, + cloglog 0.34870, + both 0.34885, 7-way 0.34912 — all worse than Core-3's 0.34855. |
| **Log-normal physiological transform** (`log(creatinine)`, `log(LOS+1)`) | Did not beat 0.34855. |
| **Transportability / shift reweighting** | Did not improve the honest score (above). |
| **Random Forest** | Given weight 0.000 by the blend optimizer. |

## Why this model is defensible, not just accurate

- **Two of three models are glass-box.** Only CatBoost is opaque, at one third of the weight.
- **Every engineered feature traces to a visible EDA finding**, with clinical thresholds rather
  than arbitrary cutoffs.
- **No leakage**: median imputation fitted on train only; `patient_id` (whose `TR`/`TE` prefix
  would leak split membership) is never a feature.
- **Variance is controlled explicitly** — seed bagging over 3 splits, not one lucky fold.
- **Scoring is honest by construction** — nested CV everywhere a number is reported.
- **The negative results are documented**, so the model's simplicity is a measured choice.

## Known limitations

- Honest OOF ≠ leaderboard. The test set is mildly shifted (AUC 0.555); expect some degradation.
- `socioeconomic_index` carries almost no marginal signal here (corr −0.002) — a synthetic-data
  property that should not be read as a real-world claim about deprivation and readmission.
- ROC-AUC ~0.695 is a genuine ceiling on this dataset; the model ranks risk usefully but is far
  from deterministic, and should be presented as a triage aid, not a decision rule.
- Synthetic, educational data. Not clinically validated and not suitable for deployment.

## Environment

Pixi, pinned via `pixi.lock` (Python 3.14, pandas, scikit-learn, CatBoost, LightGBM,
`interpret` for the EBM). Platform `osx-arm64`; add another with `pixi workspace platform add`.
