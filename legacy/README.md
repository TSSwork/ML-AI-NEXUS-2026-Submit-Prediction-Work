# Legacy — earlier work

Nothing here feeds the final submission. The project is
[`notebooks/final_model_report.ipynb`](../notebooks/final_model_report.ipynb); this folder is the
trail that led to it.

## `notebooks/` — the early baseline line

| Notebook | What it was |
|---|---|
| `check-data.ipynb` | Data-integrity and leakage audit — shapes, missingness, duplicate/ID checks. |
| `describe_data.ipynb` | First EDA — distributions, train/test category mix shift, subgroup rates. |
| `basline_model_1.ipynb`, `baseline_model_1_submission.ipynb` | Constant-prevalence baseline, p₀ = 0.1259 → Log Loss 0.37844. |
| `baseline_linear_reg.ipynb`, `baseline_linear_reg_v2.ipynb` | Logistic regression, OOF 0.35261 — the best model before the final notebook. |
| `baseline_lgbm.ipynb`, `baseline_lgbm_v2.ipynb` | First gradient-boosting attempts. |
| `score_cal.ipynb` | Label-free expected-score estimator (slice-rate pseudo-truth, Beta-smoothed). Its ranking conflicted with cross-validation; adversarial validation in the main notebook is the better answer to that question. |
| `run_experiment.py` | Early single-file calibrated-LightGBM pipeline. |

These notebooks expect to be run from this folder (`../../data/`, `../submissions/`). Their EDA
and integrity findings were carried forward into the main notebook.

## `arrays/`

OOF and test prediction arrays from the early logistic-regression and LightGBM baselines.

## `submissions/`

Every submission except the final one. `submission_12_final_3model_stack.csv` and the final
`submission_13_core3_honest.csv` are numerically identical — the same Core-3 stack reached
through two code paths.
