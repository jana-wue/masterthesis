# Benchmark README (Local + HPC seperated)

Current workflow:
- **Classic methods locally**
- **LLM finetuning on HPC** 

## 1) Files
- Specs: `benchmark_spec.md`
- DoD checklist: `dod_checklist.md`
- Complete manifest: `scenario_manifest.csv`
- local (classic methods): `scenario_manifest_local_classical.csv`
-  HPC only (LLM-finetuned): `scenario_manifest_hpc_llm_finetuned.csv`
- local runner: `src/scripts/run_classical_manifest.py`

## 2) Prerequisites
- start from project root
- `.venv` is there

## 3) Fast start (local, classic methods)

### 3.1 Dry-Run (for testing only)
```bash
.venv/bin/python src/scripts/run_classical_manifest.py --dry_run --limit 20
```

### 3.2 Full run
```bash
.venv/bin/python src/scripts/run_classical_manifest.py --resume --skip_missing_files
```

Results are ih:
- `data/results/benchmark_local_classical_runs.csv`

## 4) Options

### `--resume`
Skip runs, which are already in output with `status=success`.

### `--skip_missing_files`
If i.e. 15% files are missing,the run is shown as `missing_file` instead of failing.

### `--limit N`
Limit on the first `N` lines of manifest (good for testing).

### Filter options
- `--dataset_keys telco,statlog,creditcard`
- `--method_keys meanmode,medianmode,mice,mice_post_mean,missforest,dae`
- `--scenario_families MAR_TARGET,MNAR_TARGET,MCAR_TARGET,MCAR_GLOBAL`
- `--rates 10,15`
- `--seeds 42,202,303,404,505`

Example:
```bash
.venv/bin/python src/scripts/run_classical_manifest.py \
  --dataset_keys telco \
  --method_keys mice,mice_post_mean,missforest \
  --rates 10 \
  --seeds 42,202 \
  --resume \
  --skip_missing_files
```

MICE variants:
- `mice`: single-imputation baseline (`sample_posterior=False`, `m=1`)
- `mice_post_mean`: posterior-sampling variant (`sample_posterior=True`, mean over `m=5` draws)

## 5) What is calculated per run?
- Primary calculation on  **target col**:
  - `RMSE`
  - `NRMSE`
- Result status per run:
  - `success`
  - `missing_file`
  - `error`

  
