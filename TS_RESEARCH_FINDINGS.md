# STEPS.md — Full Audit Trail: 30-Day Readmission Prediction (ML & AI Nexus 2026)
Purpose: single source of truth for writing the 2-page Trust Card, the presentation, and viva defence.
Everything below happened in order. [FILL] = number that exists in your notebooks/leaderboard but must be copied in.

---

## 0. Competition context & constraints (frames every decision)
- Task: predict P(unplanned readmission within 30 days); metric = Binary Log Loss (lower better).
- Window: Sun 13 Sep 08:00 → Mon 14 Sep 08:00. Max 10 submissions (5 before midnight, 5 after). One Kaggle account per team, username = TEAM NAME.
- Marks: Kaggle 40 (25 leaderboard + 15 Trust Card), Presentation 40, Viva 20. Leaderboard rank 1 = 25 marks … rank 18 = 8. No valid entry = 0 and no viva.
- Explicit warning: test population may differ from train; leaderboard chasing is risky.
- Trust Card 12 sections: 1 Model summary, 2 Validation, 3 Performance, 4 Calibration, 5 Robustness, 6 Subgroups, 7 Uncertainty/human oversight, 8 Explainability, 9 Model comparison, 10 Failure modes, 11 Deployment recommendation, 12 Reproducibility.
- Data dictionary flags (used as design constraints, not afterthoughts): sex = sensitive audit; rurality, socioeconomic_index = context/subgroup audit; hospital_type, region = transportability; care_pathway = potentially unstable; 6 lab/vital/follow-up columns "may be missing"; target train-only.

---

## 1. Step 1 — Data loading & health check
What: shapes, columns, target, missingness, dtypes, leakage checks.
Results:
- train (7000, 25); test (3000, 24); sample_submission (3000, 2). Only extra train column = readmitted_30d.
- Target: 6119 zeros / 881 ones → prevalence 0.125857. sample_submission constant = 0.125857 → sample file IS the naive prevalence baseline.
- Missingness (train% / test%): creatinine 4.83/7.07, heart_rate 4.81/6.87, hemoglobin 4.63/6.57, sodium 4.44/6.40, systolic_bp 4.41/5.90, followup_days 2.33/3.33 → test missingness ≈1.4× train (first shift evidence).
- Dtypes: 9 float, 9 int, 7 object (incl. patient_id).
- Leakage checks: IDs unique in both; train∩test IDs = 0; target absent from test; prefixes TR/TE → patient_id must never be a feature (prefix alone would leak split membership).
Opinion recorded at the time: data is clean; the real challenge is calibration + shift + subgroup reliability, not raw prediction.

## 2. Step 2 — EDA (distributions, cardinality, subgroups, missingness-vs-target)
Numeric train-vs-test: largely aligned; medication_count flatter/heavier tails in test; followup_days peak shifted right (~11 vs ~10 days) → mild access-related drift.
Categorical train→test proportions (the shift story):
- sex: stable ~50/50. discharge_disposition: stable.
- rurality: Rural 17.5→22.5%, Urban 56→51%.
- hospital_type: District 18→23%, Teaching 44→38.5%.
- region: West 37→33%, North-East 19→22%.
- care_pathway: P1 33→28.5%, P4 13.3→17.8%.
→ Shift is concentrated exactly in the dictionary-flagged context/transportability/unstable variables.
Subgroup event rates (train): sex Male 12.79% (n3457) vs Female 12.39% (n3543) → NO sex signal; rurality Rural 15.61% > Urban 12.43% > Semi-urban 10.91%; hospital_type Teaching 13.0 / District 12.37 / General 12.20 (weak); region South 13.4 → NE 11.5 (weak); age_group <30 3.79% (n1477), 30-50 7.83% (n3150), 50-70 12.06% (n2030), 70+ 22.27% (n343) → age is the dominant marginal signal; 70+ and <30 are small groups (uncertainty!).
Missingness vs target (risk if missing vs present): creatinine +1.39pp, heart_rate +1.12pp, systolic +0.71pp, hemoglobin +0.40pp, followup +0.30pp, sodium −1.73pp → missingness is weakly but genuinely informative → justifies missing-indicator features.

## 3. Step 3 — Baseline 1: constant model + 7-question robustness harness
Why a constant baseline: (a) format/metric sanity, (b) leaderboard floor, (c) drift detector, (d) denominator for all later "skill" claims.
Harness built (reused for every later model): per-slice table (n, events, obs rate, Wilson 95% CI, LogLoss, Brier, calibration gap, calibration intercept); covariate-reweighting shift estimator (uses ONLY public test marginals, no labels); entropy abstention with random-referral fallback.
Results: LL 0.378435, Brier 0.110017, AUC 0.500.
- Missingness strata rates 0.1247 / 0.1306 / 0.1236 → constant invariant by construction.
- Slice calibration gaps: South +0.86pp, NE −1.12pp, P4 +3.14pp, Rural +3.02pp, 70+ +9.69pp, <30 −8.80pp → shows WHERE a global probability fails.
- Shift reweighting (Q5c): implied test prevalence 0.1275 (rurality), 0.1255 (hospital), 0.1277 (care_pathway) → pre-registered LB for constant = 0.3817–0.3820 if shift real, 0.3784 if not. THIS IS THE DRIFT TEST.
- Sex gaps ±0.2pp (fair); SES quintiles Q1 13.43 / Q5 13.21 vs Q2–Q4 ≈12.0–12.2 (weak, non-monotone).
- Abstention: entropy uniform → undefined for constant; random-10% demo only.
Submission #1 file: submission_01_constant.csv (guards: ID order, strict (0,1), row count). Pre-registered before upload.

## 4. Step 4 — Baseline 2: Logistic Regression + first critique cycle
Pipeline: median impute + StandardScaler (numerics), constant impute + one-hot (categoricals), 6 missing flags + n_missing, age_group bins [0,30,50,70,120]; sklearn Pipeline fit INSIDE each CV fold (no leakage).
First 5-fold OOF (C=1.0): LL 0.352613, Brier 0.103179, AUC 0.685290 (vs constant 0.378435).
7-question run: slice calibration gaps collapse to ≈0 (70+ gap +9.69pp → −0.08pp); care_pathway with/without 0.352613 vs 0.353296 (Δ 0.0007); abstention kept-90 0.324062 vs all 0.352613.
CRITIQUE RECEIVED (external review) and accepted — 3 corrections + overlooked items:
 (C1) Raw per-slice log loss is confounded by prevalence floors → use SKILL = slice-constant-floor LL − model LL.
 (C2) Abstention gain is partly mechanical (log loss maximal at p=0.5) → need random-referral control + event capture.
 (C3) care_pathway single-split Δ is noise → need repeated CV + drift two-condition rule.
 Overlooked: LB score of #1; uncertainty (repeated CV, bootstrap CIs, paired tests); Q1b masking stress on LR; pipeline-leakage confirmation; explainability (odds ratios, reliability diagram).
Fixes implemented (v2 LR notebook): quantile-bin reliability plot with n= annotations; honest ORs (OneHotEncoder drop='first', C=10 for OR reporting, units = per 1 SD for scaled numerics); <30 bootstrap CI; repeated 3×5 CV; masking stress (+0.00078 degradation); paired bootstrap LR-vs-constant (significant).
Key LR numbers after fixes: skill scores Rural +0.0316 / Semi-urban +0.0285 / Urban +0.0208; Central +0.0347 / West +0.0172; District +0.0346 / Teaching +0.0213; 70+ +0.0280 / 50-70 +0.0124 / 30-50 +0.0043 / <30 −0.0039 (CI crossed 0 → "no detectable benefit", not proven harm); abstention with control: entropy 0.32406 vs random 0.35579 (neutral) → real uncertainty signal; referred 10% contains 222/881 = 25.2% of all readmissions; repeated CV care_pathway Δ 0.00065 ± 0.00077.
MISTAKE CAUGHT HERE (ours, in code given to user): `from sklearn.metrics import calibration_curve` → wrong module; correct is `sklearn.calibration`. Also OR table without drop='first' produced sex_Female 0.705 AND sex_Male 0.724 (both <1 = collinearity signature), and age OR 1.58 is per-SD not per-year. Both fixed and documented.
Frozen spec decision: one-time C grid {0.1, 1, 10} on a single 5-fold → C=0.1 wins (0.35203 vs 0.35237 / 0.35242) → FROZEN, never re-tuned; mild selection optimism acknowledged as second-order. More regularization winning = signal mostly linear/monotone.

## 5. Step 5 — LightGBM challenger (baseline_lgbm.ipynb) + second critique cycle
Design principles (defensible line-by-line): no leakage (preprocessing fit inside folds; early stopping on an INNER 80/20 split so the scored fold is never touched); SAME engineered feature set as LR so paired tests compare models not features; every score carries variability; nothing uploaded before team meeting.
Feature engineering (deterministic, fit-free, identical function train/test): n_chronic (sum of 4 comorbidity flags), prior_x_comorb interaction, clinical threshold flags (creatinine>1.2, sodium<135, hemoglobin<12, sysbp≥140 or <100, HR>100), n_abnormal_labs, plus 6 `_wasna` missing indicators → 31 numeric + 6 categorical.
LR reference on same features (frozen C=0.1): OOF LL 0.35203, Brier 0.10304, AUC 0.68606.
LGBM params (conservative on purpose, 7k rows/881 events): lr 0.05, num_leaves 31, min_data_in_leaf 40, feature_fraction 0.85, bagging_fraction 0.85 freq 1, lambda_l2 1.0, seed 42.
Repeated 3×5 CV: repeats 0.35644 / 0.35489 / 0.35542 → 0.35559 ± 0.00065; pooled Brier 0.10359, AUC 0.68094; median best_iteration 41 (shallow booster → capacity was never the bottleneck).
Paired bootstrap: LGBM vs LR CI [−0.00167, +0.00443] → NOT significant (point estimate favours LR); LGBM vs constant CI [−0.03015, −0.01986] → significant; LGBM LL 95% CI [0.33877, 0.36827].
Calibration (nested, honest): uncalibrated 0.35348; Platt 0.35342 (gain +0.00007); isotonic 0.35590 (HURTS — data-hungry with 880 events); slope 1.081 / intercept 0.176 (slight UNDER-confidence, opposite of typical boosting failure); quantile-bin reliability tracks diagonal.
Subgroup skill (LGBM): positive everywhere except <30: −0.0090, CI [−0.0233, −0.0027] → CI EXCLUDES 0 → statistically significant no-benefit/harm → mandatory referral rule for <30.
Abstention: entropy kept-90 0.32273 vs random control 0.35347 ± 0.00267; referred set event rate 33.0%, captures 231/881 = 26.2% of all readmissions.
care_pathway ablation (identical folds): with 0.35559 ± 0.00065 vs without 0.35594 ± 0.00056; drop-cost CI [−0.00063, +0.00153] → within noise → KEEP with, ablation documented as §5 sensitivity.
Saved submission_03_lgbm.csv (Platt), mean test pred 0.13861 vs train prevalence 0.12586.
PRE-REGISTERED DECISION RULE: submit LGBM only if paired bootstrap shows significant gain over LR OR calibration closes the gap. Neither happened → **LR (C=0.1) is the FINAL model; LGBM is the documented, rejected challenger.**
SECOND CRITIQUE RECEIVED and accepted:
 (K1) 8b stress test was deployment-unfaithful: injected NaN without flipping `_wasna` (real deployment hits both channels) → +0.00001 was optimistic.
 (K2) Mean test pred 0.1386 vs 0.1259 (+1.3pp) unexplained vs reweighting estimate ~0.1277 → needs features-only decomposition (drift table + flag-neutralization counterfactual).
 (K3) Unfair comparison: LGBM pooled 3-repeat ensemble vs LR single OOF → LR must get same repeated protocol.
 (K4) <30 significant harm needs mitigation/referral rule on the FINAL model.
 (K5) A cell executed 3× (triple-printed subgroup table) → harmless (no accumulation) but re-run clean for the reproducibility deliverable.
v2 notebook built to fix F1–F5: corrected stress (flag flip, both models), diagnostic A (marginal drift table + flags-neutralized counterfactual, test features only — not leaderboard probing), LR repeated 3×5 + pooled-vs-pooled paired bootstrap, <30 CI + referral rule on final LR, LR care_pathway ablation, honest ORs, pre-registration table, and FINAL submission written ONLY in the last cell → submission_final_logreg.csv.
[FILL: v2 corrected-stress degradations for LR and LGBM; diagnostic-A neutralized mean; LR pooled repeated-CV mean±sd; LR calibration slope/intercept; honest OR top-10.]

## 6. Step 6 — Submissions & descriptions
- #1 constant (drift test; pre-reg 0.3817–0.3820 / 0.3784). [FILL actual LB]
- #2 LR C=0.1 = FINAL model (pre-reg ≈0.3555–0.3570). Description submitted with file. [FILL actual LB]
- #3 LGBM+Platt = challenger submitted for comparison with ≤500-char description stating it was rejected as primary by the pre-registered rule. [FILL actual LB]
- Planned hypothesis test when scores land: LB(#2) − LB(#1) ≈ 0.0264 (the CV skill gap). If it holds → shift and skill transferred exactly as modelled (Trust Card §2/§5 gold). If LR degrades more than constant → features (not base rate) are being damaged by shift.
- Trust Card scope decision: NOT only the final model — baselines AND the rejected challenger must appear (§9 comparison + justification of selection).

## 7. Environment / operational incidents (Presentation "challenges faced")
- Kaggle CPU/GPU quota exhausted mid-run → migrated to local Fedora 43, conda `base`, Python 3.13.5.
- ModuleNotFoundError lightgbm → conda-forge/pip install; Jupyter kernel mismatch fixed via `python -m ipykernel install --user --name base` + kernel switch.
- ImportError calibration_curve from sklearn.metrics → correct module is sklearn.calibration (our code bug); scikit-learn upgraded for py3.13.
- Versions (record for §12): numpy 2.1.3, pandas 2.2.3, scikit-learn 1.9.0, lightgbm 4.7.0; SEED 42 everywhere.
- AI coding assistants used throughout → MUST be disclosed in Trust Card §12.

---

## 8. MASTER NUMBERS TABLE (copy-paste source for Trust Card §3/§9)
| Model | CV LogLoss | Brier | AUC | Calibration | Status |
|---|---|---|---|---|---|
| Constant 0.125857 | 0.378435 | 0.110017 | 0.500 | perfect in-the-large by construction | baseline / drift probe |
| LR C=0.1 (FINAL) | 0.35203 (frozen grid); repeated [FILL] | 0.10304 | 0.68606 | native, slope≈1 [FILL exact] | submitted #2 |
| LGBM + Platt | 0.35559 ± 0.00065 (repeats); pooled+Platt 0.35342 | 0.10359 | 0.68094 | slope 1.081/int 0.176; Platt gain +0.00007; isotonic hurts | rejected challenger (#3) |
Paired bootstrap: LGBM−LR [−0.00167, +0.00443] NS; LR−constant significant; LGBM−constant significant.
Abstention: LR entropy kept-90 0.32406 vs random 0.35579; capture 25.2%. LGBM 0.32273 vs 0.35347±0.00267; capture 26.2%, referred event rate 33%.
Stress (missingness ×1.4): LR +0.00078 (original method); corrected flag-flip values [FILL v2]. LGBM original +0.00001 (unfaithful, superseded).
care_pathway: LR Δ 0.00065±0.00077; LGBM drop-cost CI [−0.00063,+0.00153] → keep, ablation documented.
Subgroup skill (LR): Rural +0.0316, Urban +0.0208, Semi-urb +0.0285; West +0.0172, Central +0.0347; District +0.0346, Teaching +0.0213; sex ≈+0.025 both; 70+ +0.0280, <30 −0.0039 (CI∋0). LGBM <30 −0.0090 CI[−0.0233,−0.0027] → referral rule.

## 9. MISTAK