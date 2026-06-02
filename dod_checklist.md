# Definition of Done (Benchmark)

## A) Data and Scenario Integrity
- [x] All 24 scenario slots exist (3 datasets x 4 scenario families x 2 rates).
- [x] `MCAR_GLOBAL` excludes IDs/labels/non-sensible columns exactly as specified.
- [x] `MAR_TARGET` dependency columns follow spec per dataset.
- [x] `MNAR_TARGET` is applied on target column only.

## B) Run Completeness
- [ ] Classical: all runs completed.
  - [x] Mean/Mode
  - [x] MICE
  - [x] MissForest
  - [ ] DAE
- [ ] LLM finetuned: all runs completed (Mistral + Llama + Qwen).
- [ ] No missing seed/method/model combinations relative to `scenario_manifest.csv`.

## C) Metric Completeness
- [ ] Every run has RMSE and NRMSE.
- [ ] NRMSE uses benchmark definition (std of evaluated ground truth cells).
- [ ] LLM runs include fallback rate.

## D) Result Hygiene
- [ ] No duplicate primary keys (`dataset_key, scenario_family, rate_pct, seed, method, model_key`).
- [ ] No smoke/test rows in final benchmark tables.
- [ ] Final tables are generated from clean benchmark result files only.

## E) Final Outputs for Thesis
- [ ] Primary leaderboard: target-only comparison for all methods.
- [ ] Mean ± std across seeds for each method/setting.
- [ ] Separate section for `MCAR_GLOBAL` secondary analysis (if included).
- [ ] Explicit note of remaining limitations and compute constraints.
