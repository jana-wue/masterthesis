# Definition of Done (Benchmark)

## A) Data and Scenario Integrity
- [x] All 24 scenario slots exist (3 datasets x 4 scenario families x 2 rates).
- [x] `MCAR_GLOBAL` excludes IDs/labels/non-sensible columns exactly as specified.
- [x] `MAR_TARGET` dependency columns follow spec per dataset.
- [x] `MNAR_TARGET` is applied on target column only.

## B) Run Completeness
- [x] Classical: all runs completed.
  - [x] Mean/Mode
  - [x] MICE
  - [x] MissForest
  - [x] DAE
- [x] LLM finetuned final scope completed (`10%`, `seed=42`, no `MCAR_GLOBAL`).
- [x] LLM prompt final scope completed (few-shot `k=2`, `10%`, `seed=42`).
- [x] No missing seed/method/model combinations relative to `scenario_manifest.csv`.

## C) Metric Completeness
- [x] Every run has RMSE and NRMSE.
- [x] NRMSE uses benchmark definition (std of evaluated ground truth cells).
- [x] LLM runs include fallback rate.

## D) Result Hygiene
- [x] No duplicate primary keys (`dataset_key, scenario_family, rate_pct, seed, method, model_key`).
- [x] No smoke/test rows in final benchmark tables.
- [x] Final tables are generated from clean benchmark result files only.

## E) Final Outputs for Thesis
- [x] Primary leaderboard: target-only comparison for all final-scope methods.
- [x] Mean ± std across seeds for each classical setting.
- [x] Separate section for `MCAR_GLOBAL` secondary analysis for classical methods only.
- [x] Explicit note of remaining limitations and compute constraints.
