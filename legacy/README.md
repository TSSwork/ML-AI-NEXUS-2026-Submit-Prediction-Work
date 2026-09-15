# Legacy — superseded work

Nothing here feeds the final submission. It is kept because the negative results are the
evidence for the choices the final model makes: the Core-3 stack excludes LightGBM, phenotype
clustering, an MLP and two shift corrections, and each of those exclusions was *measured*, not
assumed.

All scripts still run — they read `pipeline/artifacts/`, so run `pixi run bag` first to
regenerate the inputs. Any submission they write goes to `legacy/submissions/`.

## `experiments/` — research line that did not survive ablation

| Script | What it tried | Why it is not the final model |
|---|---|---|
| `19_stacking_super_learner_original.py` | First 4-way SLSQP stack (Spline, EBM, CatBoost, LightGBM) | Weights fit and scored on the same OOF rows — overstates the score. Replaced by nested-CV scoring in `evidence/honest_nested_stacking.py`. |
| `21_stat_models_lda_cloglog.py` | LDA and complementary log-log link as extra base learners | Weaker solo, no complementary signal in the blend. |
| `24_final_honest_bagged_super_learner.py` | Honest 4-way stack, seed-bagged | 0.34912 — beaten by dropping LightGBM (0.34855). |
| `26_logit_space_stacking.py` | Blending in logit space instead of probability space | No improvement over the probability-space convex blend. |
| `28_swap_lgbm_for_diverse_model.py`, `29_diverse_model_ablation.py` | Swap LightGBM for a more diverse learner (LDA / cloglog) | Swapping did not beat simply removing it. |
| `30_phenotype_core3.py` | K-Means clinical phenotypes + distance-to-centroid features | Hurt every base model — built from columns the models already see, so they add collinear noise. |
| `31_verify_notebook_core3_cells.py` | Standalone re-run of the notebook's Core-3 cells | Verification harness, kept for reference; `pipeline/07_final_core3_stack.py` is the maintained path (both produce the identical file). |
| `33_log_transform_core3.py` | `log(creatinine)`, `log(LOS+1)` alongside the raw columns | Did not beat the 0.34855 honest baseline. |
| `34_shift_reweighted_core3.py` | Importance-sampling weights from adversarial validation | The train/test shift is real but mild (OOF AUC 0.555); reweighting did not improve the honest score. |

## `notebooks/` — the early baseline line

| Notebook | What it was |
|---|---|
| `check-data.ipynb` | Data-integrity and leakage audit (shapes, missingness, ID checks). |
| `describe_data.ipynb` | EDA — distributions, train/test category mix shift, subgroup rates. |
| `basline_model_1.ipynb`, `baseline_model_1_submission.ipynb` | Constant-prevalence baseline, p₀ = 0.1259 → LogLoss 0.3784. |
| `baseline_linear_reg.ipynb`, `baseline_linear_reg_v2.ipynb` | Logistic regression, OOF LogLoss 0.3526 — the best model *before* the research line. |
| `baseline_lgbm.ipynb`, `baseline_lgbm_v2.ipynb` | First gradient-boosting attempts. |
| `score_cal.ipynb` | Label-free expected-score estimator (slice-rate pseudo-truth, Beta-smoothed). |
| `run_experiment.py` | Early single-file calibrated-LightGBM pipeline, superseded by `pipeline/`. |

The EDA and integrity findings from these notebooks were carried forward into
`notebooks/final_model_report.ipynb` and `RESEARCH_FINDINGS.md`.

## `arrays/`

OOF and test prediction arrays from the early logistic-regression / LightGBM baselines
(`p_oof_*.npy`, `p_test_*.npy`).

## `submissions/`

Every submission except the final one. Note `submission_12_final_3model_stack.csv` and
`submission_13_core3_honest.csv` are numerically identical — the same Core-3 stack reached
through two code paths (the pipeline scripts and the notebook cells); 13 is the canonical one.
