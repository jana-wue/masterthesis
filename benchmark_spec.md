# Benchmark Specification

**Version:** v1.1
**Frozen on:** 2026-05-20
**Owner:** Jana Wüsten

## 1) Goal
Fair and reproducible comparison of imputation methods across all datasets for a common target-variable evaluation setup.

## 2) In-Scope Methods
### Classical (5 seeds each)
- Mean/Mode
- Median/Mode
- MICE (single-imputation baseline)
- MICE posterior mean (m=5 draws, secondary variant)
- MissForest
- DAE

### LLM Finetuned (3 seeds per model)
- Mistral-7B
- Llama-3.1-8B
- Qwen-2.5-7B

### Out of Scope for this benchmark pass
- Prompt-based approaches (zero-shot/few-shot/paper-prompt variants)
  - Maybe later if enough time.

## 3) Datasets and Targets
- **Telco** (`telco`): target = `TotalCharges`
- **German Statlog** (`statlog`): target = `X2`
- **German Credit Card** (`creditcard`): target = `BILL_AMT1`

## 4) Scenario Families (two MCAR variants)
Per dataset, run all four families:
1. `MAR_TARGET`: target missingness depends on one observed feature
2. `MNAR_TARGET`: target missingness depends on target itself
3. `MCAR_TARGET`: random missingness only in target column
4. `MCAR_GLOBAL`: random missingness across all eligible columns

### MAR dependency columns
- Telco: `TotalCharges <- tenure`
- Statlog: `X2 <- X5`
- Credit: `BILL_AMT1 <- PAY_0`

## 5) Missingness Rates
- 10%
- 15%

## 6) Column Rules for `MCAR_GLOBAL`
Apply MCAR to all columns **except** IDs/names and non-sensible columns:
- Telco exclude: `customerID`, `Churn`
- Statlog exclude: `class`
- Credit exclude: `ID`, `default payment next month`

## 7) Seeds
Using multiple random seeds improves the reliability and robustness of the evaluation by reducing the influence of 
chance effects in missingness generation, data splitting and model training.
- Classical seeds: `[42, 202, 303, 404, 505]`
- LLM finetuned seeds: `[42, 202, 303]`

## 8) Models (LLM Finetuned)
`model_key` and default `model_name`:
- `mistral`: `mistralai/Mistral-7B-Instruct-v0.3`
- `llama`: `meta-llama/Llama-3.1-8B-Instruct`
- `qwen`: `Qwen/Qwen2.5-7B-Instruct`

(If a final model ID changes, update only `model_name`, keep `model_key` stable.)

## 9) Metrics
- RMSE
- NRMSE

NRMSE definition for this benchmark:
- `NRMSE = RMSE / std(ground_truth_on_evaluated_cells)`
- Add epsilon `1e-8` to denominator for numerical stability.

## 10) Evaluation Protocol
### Primary leaderboard
- Evaluate **target column only** for all methods (classical + LLM finetuned).
- Score only artificially masked target cells.
- Keep `mice` (single-imputation baseline) as the primary MICE entry.

### Secondary analysis (if time)
- `MCAR_GLOBAL` full-matrix analysis for classical methods only.
- This is supplementary and not mixed into the primary leaderboard.
- `mice_post_mean` is also treated as supplementary/secondary unless explicitly promoted.

## 11) Logging Schema (required fields)
Each run must log at least:
- `run_id`
- `run_timestamp_utc`
- `dataset_key`
- `target_column`
- `scenario_family`
- `missingness_type`
- `mcar_scope` (`target` or `global`)
- `rate_pct`
- `seed`
- `method_family` (`classical` or `llm_finetuned`)
- `method`
- `model_key` (LLM only)
- `model_name` (LLM only)
- `n_masked_target`
- `rmse`
- `nrmse`
- `fallback_rate` (LLM only)

## 12) Planned Run Volume (this spec)
- Scenario grid: `3 datasets x 4 scenario_families x 2 rates = 24 setting-slots`
- Classical: `24 x 6 methods x 5 seeds = 720 runs`
- LLM finetuned: `24 x 3 models x 3 seeds = 216 runs`
- **Total: 936 runs**

## 13) Reproducibility Rules
- Keep one frozen benchmark manifest (`scenario_manifest.csv`) as source of truth.
- No appending into old mixed result files for final thesis tables.
- Every final number must be traceable to seed + manifest row + output file.
