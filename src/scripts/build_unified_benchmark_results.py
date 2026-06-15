from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_RESULTS


CLASSICAL_FILES = [
    DATA_RESULTS / "benchmark_local_classical_meanmode_final.csv",
    DATA_RESULTS / "benchmark_local_classical_medianmode_final.csv",
    DATA_RESULTS / "benchmark_local_classical_runs_mice_mean.csv",
    DATA_RESULTS / "benchmark_local_classical_missforest_final.csv",
    DATA_RESULTS / "benchmark_local_classical_dae_final.csv",
]
PROMPT_RUNS_FILE = DATA_RESULTS / "benchmark_llm_prompt_runs_final.csv"
PROMPT_RUNS_FALLBACK_FILE = DATA_RESULTS / "benchmark_llm_prompt_runs.csv"


DATASET_META = {
    "telco": {
        "dataset_name": "Telco",
        "target_column": "TotalCharges",
    },
    "statlog": {
        "dataset_name": "German Statlog",
        "target_column": "X2",
    },
    "credit": {
        "dataset_name": "German Credit Card",
        "target_column": "BILL_AMT1",
    },
}


MODEL_META = {
    "mistral": {
        "model_key": "mistral",
    },
    "llama31": {
        "model_key": "llama",
    },
    "qwen25": {
        "model_key": "qwen",
    },
}


SCENARIO_META = {
    "mar": {
        "scenario_family": "MAR_TARGET",
        "missingness_type": "MAR",
        "mcar_scope": "target",
    },
    "mnar": {
        "scenario_family": "MNAR_TARGET",
        "missingness_type": "MNAR",
        "mcar_scope": "target",
    },
    "mcar": {
        "scenario_family": "MCAR_TARGET",
        "missingness_type": "MCAR",
        "mcar_scope": "target",
    },
}


OUTPUT_COLUMNS = [
    "run_id",
    "run_timestamp_utc",
    "dataset_key",
    "dataset_name",
    "target_column",
    "scenario_family",
    "missingness_type",
    "mcar_scope",
    "rate_pct",
    "seed",
    "method_family",
    "method_key",
    "method_name",
    "model_key",
    "model_name",
    "status",
    "error_message",
    "full_path",
    "missing_path",
    "n_masked_target",
    "mean_rmse",
    "mean_nrmse",
    "runtime_seconds",
    "fallback_rate",
    "source_file",
    "source_type",
    "provenance_note",
]


def _load_classical_rows():
    frames = []

    for csv_path in CLASSICAL_FILES:
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["fallback_rate"] = pd.NA
        df["source_file"] = str(csv_path)
        df["source_type"] = "classical_run_csv"
        df["provenance_note"] = pd.NA
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["run_id"], keep="last").copy()

    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    return combined[OUTPUT_COLUMNS]


def _parse_llm_filename(csv_path):
    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[-2:] != ["finetuned", "eval"]:
        raise ValueError(f"Unexpected LLM finetuned eval filename: {csv_path.name}")

    dataset_token = parts[0]
    model_token = parts[1]
    scenario_token = parts[-3]
    return dataset_token, model_token, scenario_token


def _build_llm_row(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    if "run_timestamp_utc" not in df.columns:
        raise ValueError(f"Missing run_timestamp_utc in {csv_path}")

    latest_timestamp = df["run_timestamp_utc"].dropna().astype(str).max()
    latest = df[df["run_timestamp_utc"].astype(str) == latest_timestamp].copy()
    if latest.empty:
        return None

    dataset_token, model_token, scenario_token = _parse_llm_filename(csv_path)
    dataset_meta = DATASET_META[dataset_token]
    model_meta = MODEL_META[model_token]
    scenario_meta = SCENARIO_META[scenario_token]

    if "prediction" not in latest.columns or "ground_truth" not in latest.columns:
        raise ValueError(f"Missing prediction columns in {csv_path}")

    valid = latest.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid.empty:
        return None

    errors = valid["prediction"] - valid["ground_truth"]
    mean_rmse = float(math.sqrt(float((errors ** 2).mean())))
    denom = float(valid["ground_truth"].std(ddof=0)) + 1e-8
    mean_nrmse = float(mean_rmse / denom)

    fallback_rate = pd.NA
    if "source" in valid.columns:
        fallback_rate = float((valid["source"].astype(str) != "model_first_try").mean())

    model_name = pd.NA
    if "model_name" in valid.columns and valid["model_name"].notna().any():
        model_name = str(valid["model_name"].dropna().iloc[0])

    note = (
        "Derived from the latest run_timestamp group in a finetuned eval prediction CSV; "
        "mapped to completed 10% target-scenario LLM finetuning results only. "
        "Seed set to 42 based on the checked HPC job scripts using the default run_llm_finetune.py seed."
    )

    return {
        "run_id": f"inferred::{csv_path.stem}::{latest_timestamp}",
        "run_timestamp_utc": latest_timestamp,
        "dataset_key": dataset_token if dataset_token != "credit" else "creditcard",
        "dataset_name": dataset_meta["dataset_name"],
        "target_column": dataset_meta["target_column"],
        "scenario_family": scenario_meta["scenario_family"],
        "missingness_type": scenario_meta["missingness_type"],
        "mcar_scope": scenario_meta["mcar_scope"],
        "rate_pct": 10,
        "seed": 42,
        "method_family": "llm_finetuned",
        "method_key": "llm_finetuned",
        "method_name": "LLM Finetuned",
        "model_key": model_meta["model_key"],
        "model_name": model_name,
        "status": "success",
        "error_message": pd.NA,
        "full_path": pd.NA,
        "missing_path": pd.NA,
        "n_masked_target": int(len(valid)),
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse,
        "runtime_seconds": pd.NA,
        "fallback_rate": fallback_rate,
        "source_file": str(csv_path),
        "source_type": "llm_finetuned_eval_latest_timestamp",
        "provenance_note": note,
    }


def _load_llm_rows():
    rows = []

    for csv_path in sorted(DATA_RESULTS.glob("*_finetuned_eval.csv")):
        row = _build_llm_row(csv_path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    llm_df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in llm_df.columns:
            llm_df[col] = pd.NA

    return llm_df[OUTPUT_COLUMNS]


def _load_prompt_rows():
    prompt_path = PROMPT_RUNS_FILE if PROMPT_RUNS_FILE.exists() else PROMPT_RUNS_FALLBACK_FILE
    if not prompt_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_csv(prompt_path)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.drop_duplicates(subset=["run_id"], keep="last").copy()
    return df[OUTPUT_COLUMNS]


def main():
    output_path = DATA_RESULTS / "benchmark_all_completed_results.csv"

    classical = _load_classical_rows()
    llm = _load_llm_rows()
    prompt = _load_prompt_rows()

    combined = pd.concat([classical, llm, prompt], ignore_index=True, sort=False)
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[OUTPUT_COLUMNS].copy()
    combined = combined.sort_values(
        [
            "method_family",
            "dataset_key",
            "scenario_family",
            "rate_pct",
            "seed",
            "model_key",
            "run_timestamp_utc",
        ],
        na_position="last",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("=" * 80)
    print("UNIFIED BENCHMARK RESULTS BUILT")
    print("=" * 80)
    print(f"Classical rows: {len(classical)}")
    print(f"LLM rows: {len(llm)}")
    print(f"Prompt rows: {len(prompt)}")
    print(f"Total rows: {len(combined)}")
    print(f"Output CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
