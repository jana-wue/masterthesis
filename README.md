# Are LLMSs Better at Missing Data Imputation? - Master Thesis Benchmark Repository

This repository contains the final benchmark workflow, result files and supporting scripts.

The repository has three benchmark parts:
- `classical`: local benchmark for classical imputation methods
- `llm_finetuned`: final thesis scope for finetuned LLM evaluation
- `llm_prompt`: final thesis scope for prompt-based few-shot baseline

The current repository state is aligned to the final thesis scope documented below. The central source of truth for the benchmark rows is:
- `scenario_manifest.csv`

## Final Scope

### Classical benchmark
- Datasets: `telco`, `statlog`, `creditcard`
- Scenario families: `MAR_TARGET`, `MNAR_TARGET`, `MCAR_TARGET`, `MCAR_GLOBAL`
- Rates: `10`, `15`
- Seeds: `42`, `202`, `303`, `404`, `505`
- Methods: `meanmode`, `medianmode`, `mice`, `missforest`, `dae`

### LLM finetuned benchmark
- Datasets: `telco`, `statlog`, `creditcard`
- Scenario families: `MAR_TARGET`, `MNAR_TARGET`, `MCAR_TARGET`
- Rate: `10`
- Seed: `42`
- Models: `mistral`, `llama`, `qwen`

### LLM prompt benchmark
- Datasets: `telco`, `statlog`, `creditcard`
- Scenario families: `MAR_TARGET`, `MNAR_TARGET`, `MCAR_TARGET`
- Rate: `10`
- Seed: `42`
- Baseline: few-shot `k=2`
- Models: `mistral`, `llama`, `qwen`

### Out of final scope
- `mice_post_mean`
- `MCAR_GLOBAL` for `llm_finetuned`
- prompt variants other than few-shot `k=2`
- additional LLM seeds or additional LLM rates

## Main Entry Points

Use these entry points for the final benchmark workflow:

- `main.py`
  - canonical entry point for the classical benchmark
- `src/scripts/run_classical_manifest.py`
  - same classical benchmark runner with full CLI options
- `src/scripts/build_llm_finetuning_final_results.py`
  - rebuilds the final finetuned LLM result table from detailed eval files and refreshes the final-scope finetuned manifest schema
- `src/scripts/build_llm_prompt_results_final.py`
  - cleans prompt result rows and refreshes the unified benchmark table
- `src/scripts/build_unified_benchmark_results.py`
  - rebuilds the unified final benchmark CSV from the canonical classical runner output as primary source, supplemented by legacy classical final exports if needed, plus the finalized LLM result files

Legacy scripts are intentionally disabled or redirected:
- `src/scripts/imputation_pipeline.py` is a legacy shim and must not be used for the final thesis benchmark
- `src/scripts/run_mice_post_mean.py` exits intentionally because `mice_post_mean` is out of scope

## Repository Structure

- `main.py`: canonical classical benchmark entry point
- `scenario_manifest.csv`: full final-scope manifest
- `scenario_manifest_local_classical.csv`: classical benchmark manifest
- `scenario_manifest_hpc_llm_finetuned.csv`: final-scope finetuned LLM manifest
- `scenario_manifest_llm_prompt_10pct_k2_final.csv`: final-scope prompt manifest
- `src/imputation/`: imputer implementations
- `src/evaluation/`: metrics and logging helpers
- `src/scripts/`: benchmark runners, result builders, plotting scripts
- `data/raw/`: raw datasets
- `data/processed/`: generated missingness scenarios
- `data/results/`: result tables and figure outputs
- `jobs/`: HPC job scripts for LLM runs

## Environment Setup

Run all commands from the project root.

Tested with Python `3.9+`.

### Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements
The Python dependencies are listed in:
- `requirements.txt`

## Running The Classical Benchmark

### Recommended command
```bash
.venv/bin/python main.py --resume --skip_missing_files
```

This runs the final-scope classical benchmark from `scenario_manifest_local_classical.csv`.

### Equivalent direct command
```bash
.venv/bin/python src/scripts/run_classical_manifest.py --resume --skip_missing_files
```

### Dry run
```bash
.venv/bin/python main.py --dry_run --limit 20
```

### Run only one dataset
```bash
.venv/bin/python main.py --dataset_keys telco --resume --skip_missing_files
```

### Run only selected methods
```bash
.venv/bin/python main.py --method_keys meanmode,mice,missforest --resume --skip_missing_files
```

### Run only selected scenario families
```bash
.venv/bin/python main.py --scenario_families MAR_TARGET,MCAR_TARGET --resume --skip_missing_files
```

### Run only selected rates and seeds
```bash
.venv/bin/python main.py --rates 10 --seeds 42,202 --resume --skip_missing_files
```

### Write into a custom output file
```bash
.venv/bin/python main.py --output_csv data/results/my_classical_runs.csv --resume --skip_missing_files
```

## Classical Runner Options

`main.py` forwards directly to `src/scripts/run_classical_manifest.py`, so both support the same options.

- `--manifest PATH`
  - default: `scenario_manifest_local_classical.csv`
  - classical manifest to execute
- `--output_csv PATH`
  - default: `data/results/benchmark_local_classical_runs.csv`
  - output CSV for run-level metrics
- `--dataset_keys CSV`
  - example: `telco,statlog,creditcard`
- `--method_keys CSV`
  - supported in final scope: `meanmode,medianmode,mice,missforest,dae`
- `--scenario_families CSV`
  - example: `MAR_TARGET,MNAR_TARGET,MCAR_TARGET,MCAR_GLOBAL`
- `--rates CSV`
  - example: `10,15`
- `--seeds CSV`
  - example: `42,202,303,404,505`
- `--limit N`
  - run only the first `N` filtered rows
- `--skip_missing_files`
  - log missing scenario files as `missing_file` instead of crashing
- `--resume`
  - skip rows already completed with `status=success` in the output CSV
- `--dry_run`
  - print the selected manifest rows and exit without executing
- `--fail_fast`
  - stop on the first error

## Result Files

Important result files:

- `data/results/benchmark_all_completed_results.csv`
  - final unified benchmark table
- `data/results/benchmark_local_classical_runs.csv`
  - classical run-level output from the manifest runner
- `data/results/benchmark_llm_prompt_runs_final.csv`
  - cleaned final prompt rows
- `data/results/llm_finetuning_results_final.csv`
  - final finetuned LLM summary table
- `data/results/imputation_results.csv`
  - RMSE summary table
- `data/results/imputation_results_nrmse.csv`
  - NRMSE summary table

## Final Output Files

These are the files that should be treated as the final thesis benchmark outputs:

- `README.md`
  - central project documentation and run instructions
- `scenario_manifest.csv`
  - final benchmark source of truth
- `scenario_manifest_local_classical.csv`
  - final classical benchmark manifest
- `scenario_manifest_hpc_llm_finetuned.csv`
  - final finetuned LLM manifest
- `scenario_manifest_llm_prompt_10pct_k2_final.csv`
  - final prompt benchmark manifest
- `data/results/benchmark_all_completed_results.csv`
  - final unified benchmark result table
- `data/results/benchmark_local_classical_runs.csv`
  - final classical run-level benchmark table
- `data/results/benchmark_llm_prompt_runs_final.csv`
  - final cleaned prompt benchmark table
- `data/results/llm_finetuning_results_final.csv`
  - final finetuned LLM summary table
- `data/results/imputation_results.csv`
  - final RMSE summary table
- `data/results/imputation_results_nrmse.csv`
  - final NRMSE summary table
- `data/results/figures/`
  - final exported thesis figures and figure tables

## Reproducibility Notes

- The final manifest and the unified benchmark CSV are currently aligned.
- `run_classical_manifest.py` covers the classical benchmark only, and `benchmark_local_classical_runs.csv` is the canonical classical source preferred by the unified builder.
- LLM prompt and finetuned results are not produced by `main.py`; they are finalized through their dedicated builder scripts on a HPC.
- `prepare_llm_data.py` is standalone and no longer depends on the full finetuning runner module for its JSONL schema.
- The repository contains legacy files and archived artifacts in `data/results/` and `logs/`, but the final benchmark workflow should use only the entry points listed in this README.
