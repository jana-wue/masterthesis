import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
RMSE_PATH = RESULTS_DIR / "imputation_results.csv"
NRMSE_PATH = RESULTS_DIR / "imputation_results_nrsme.csv"
FINAL_PATH = RESULTS_DIR / "llm_finetuning_results_final.csv"
MANIFEST_PATH = ROOT / "scenario_manifest_llm_finetuning_10pct_final.csv"


DATASET_META = {
    "credit": {
        "dataset": "German Credit Card",
        "target_column": "BILL_AMT1",
        "detail_target_slug": "billamt1",
    },
    "statlog": {
        "dataset": "German Statlog",
        "target_column": "X2",
        "detail_target_slug": "x2",
    },
    "telco": {
        "dataset": "Telco",
        "target_column": "TotalCharges",
        "detail_target_slug": "totalcharges",
    },
}


MODEL_META = {
    "llama31": {
        "model_key": "llama31",
        "model_label": "Llama-3.1-8B-Instruct",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral": {
        "model_key": "mistral",
        "model_label": "Mistral-7B-Instruct-v0.3",
        "hf_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "qwen25": {
        "model_key": "qwen25",
        "model_label": "Qwen2.5-7B-Instruct",
        "hf_model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
}


SCENARIO_META = {
    ("credit", "mar"): {
        "missingness_type": "MAR",
        "missing_rate_label": "billamt1 -> pay0, 10%",
    },
    ("credit", "mcar"): {
        "missingness_type": "MCAR",
        "missing_rate_label": "billamt1, 10%",
    },
    ("credit", "mnar"): {
        "missingness_type": "MNAR",
        "missing_rate_label": "billamt1, 10%",
    },
    ("statlog", "mar"): {
        "missingness_type": "MAR",
        "missing_rate_label": "X2, 10%",
    },
    ("statlog", "mcar"): {
        "missingness_type": "MCAR",
        "missing_rate_label": "X2, 10%",
    },
    ("statlog", "mnar"): {
        "missingness_type": "MNAR",
        "missing_rate_label": "X2, 10%",
    },
    ("telco", "mar"): {
        "missingness_type": "MAR",
        "missing_rate_label": "missing totalCharges -> tenure, 10%",
    },
    ("telco", "mcar"): {
        "missingness_type": "MCAR",
        "missing_rate_label": "missing TotalCharges, 10%",
    },
    ("telco", "mnar"): {
        "missingness_type": "MNAR",
        "missing_rate_label": "missing TotalCharges, 10%",
    },
}


def _parse_eval_filename(csv_path: Path) -> tuple[str, str, str]:
    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[-2:] != ["finetuned", "eval"]:
        raise ValueError(f"Unexpected LLM finetuned eval filename: {csv_path.name}")
    return parts[0], parts[1], parts[-3]


def _load_latest_valid_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Detailed eval CSV is empty: {csv_path}")
    if "run_timestamp_utc" not in df.columns:
        raise ValueError(f"Missing run_timestamp_utc in {csv_path}")

    latest_timestamp = df["run_timestamp_utc"].dropna().astype(str).max()
    latest = df[df["run_timestamp_utc"].astype(str) == latest_timestamp].copy()
    if latest.empty:
        raise ValueError(f"No latest timestamp group found in {csv_path}")

    valid = latest.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid.empty:
        raise ValueError(f"No valid prediction/ground_truth rows in {csv_path}")

    valid["prediction"] = pd.to_numeric(valid["prediction"], errors="coerce")
    valid["ground_truth"] = pd.to_numeric(valid["ground_truth"], errors="coerce")
    valid = valid.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid.empty:
        raise ValueError(f"No numeric prediction/ground_truth rows in {csv_path}")
    return valid


def _compute_metrics(valid_eval: pd.DataFrame) -> tuple[float, float]:
    errors = valid_eval["ground_truth"] - valid_eval["prediction"]
    mean_rmse = float(math.sqrt(float((errors ** 2).mean())))
    std_true = float(valid_eval["ground_truth"].std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true + 1e-8))
    return mean_rmse, mean_nrmse


def _build_final_row(csv_path: Path) -> dict:
    dataset_token, model_token, scenario_token = _parse_eval_filename(csv_path)
    dataset_meta = DATASET_META[dataset_token]
    model_meta = MODEL_META[model_token]
    scenario_meta = SCENARIO_META[(dataset_token, scenario_token)]
    valid_eval = _load_latest_valid_eval(csv_path)
    mean_rmse, mean_nrmse = _compute_metrics(valid_eval)

    return {
        "dataset": dataset_meta["dataset"],
        "missingness_type": scenario_meta["missingness_type"],
        "missing_rate_label": scenario_meta["missing_rate_label"],
        "target_column": dataset_meta["target_column"],
        "model_key": model_meta["model_key"],
        "model_label": model_meta["model_label"],
        "hf_model_name": model_meta["hf_model_name"],
        "rmse": mean_rmse,
        "nrmse": mean_nrmse,
        "detail_eval_file": csv_path.name,
    }


def build_final_rows() -> list[dict]:
    rows = []
    for csv_path in sorted(RESULTS_DIR.glob("*_finetuned_eval.csv")):
        rows.append(_build_final_row(csv_path))
    return rows


def write_final_results(final_rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "missingness_type",
        "missing_rate_label",
        "target_column",
        "model_key",
        "model_label",
        "hf_model_name",
        "rmse",
        "nrmse",
        "detail_eval_file",
    ]
    with FINAL_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)


def write_manifest(final_rows: list[dict]) -> None:
    fieldnames = [
        "run_group_id",
        "dataset",
        "target_column",
        "missingness_type",
        "model_key",
        "model_label",
        "hf_model_name",
        "job_script",
        "mech_env",
        "detail_eval_file",
    ]
    job_script_map = {
        ("Telco", "mistral"): "jobs/run_telco_mistral_safe.sbatch",
        ("Telco", "qwen25"): "jobs/run_telco_qwen25_safe.sbatch",
        ("Telco", "llama31"): "jobs/run_telco_llama31_safe.sbatch",
        ("German Statlog", "mistral"): "jobs/run_statlog_x2_safe.sbatch",
        ("German Statlog", "qwen25"): "jobs/run_statlog_qwen25_safe.sbatch",
        ("German Statlog", "llama31"): "jobs/run_statlog_llama31_safe.sbatch",
        ("German Credit Card", "mistral"): "jobs/run_credit_mistral_safe.sbatch",
        ("German Credit Card", "qwen25"): "jobs/run_credit_qwen25_safe.sbatch",
        ("German Credit Card", "llama31"): "jobs/run_credit_llama31_safe.sbatch",
    }
    with MANIFEST_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(final_rows, start=1):
            writer.writerow(
                {
                    "run_group_id": f"LLM10PCT_{idx:03d}",
                    "dataset": row["dataset"],
                    "target_column": row["target_column"],
                    "missingness_type": row["missingness_type"],
                    "model_key": row["model_key"],
                    "model_label": row["model_label"],
                    "hf_model_name": row["hf_model_name"],
                    "job_script": job_script_map[(row["dataset"], row["model_key"])],
                    "mech_env": row["missingness_type"],
                    "detail_eval_file": row["detail_eval_file"],
                }
            )


def _sync_global_rmse(final_rows: list[dict]) -> None:
    rmse_rows = []
    for row in final_rows:
        rmse_rows.append(
            {
                "dataset": row["dataset"],
                "missingness_type": row["missingness_type"],
                "missing_rate": row["missing_rate_label"],
                "imputation_method": f"LLM Finetuned ({row['model_label']})",
                "mean_rmse": row["rmse"],
            }
        )

    if RMSE_PATH.exists():
        existing = pd.read_csv(RMSE_PATH)
        keep = ~existing["imputation_method"].astype(str).str.startswith("LLM Finetuned (")
        combined = pd.concat([existing.loc[keep].copy(), pd.DataFrame(rmse_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(rmse_rows)

    required_columns = [
        "dataset",
        "missingness_type",
        "missing_rate",
        "imputation_method",
        "mean_rmse",
    ]
    for col in required_columns:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[required_columns]
    combined.to_csv(RMSE_PATH, index=False)


def _sync_global_nrmse(final_rows: list[dict]) -> None:
    nrmse_rows = []
    for row in final_rows:
        nrmse_rows.append(
            {
                "dataset": row["dataset"],
                "missingness_type": row["missingness_type"],
                "missing_rate": row["missing_rate_label"],
                "imputation_method": f"LLM Finetuned ({row['model_label']})",
                "mean_rmse": row["rmse"],
                "mean_nrmse": row["nrmse"],
            }
        )

    if NRMSE_PATH.exists():
        existing = pd.read_csv(NRMSE_PATH)
        keep = ~existing["imputation_method"].astype(str).str.startswith("LLM Finetuned (")
        combined = pd.concat([existing.loc[keep].copy(), pd.DataFrame(nrmse_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(nrmse_rows)

    required_columns = [
        "dataset",
        "missingness_type",
        "missing_rate",
        "imputation_method",
        "mean_rmse",
        "mean_nrmse",
    ]
    for col in required_columns:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[required_columns]
    combined.to_csv(NRMSE_PATH, index=False)


def main() -> None:
    final_rows = build_final_rows()
    write_final_results(final_rows)
    write_manifest(final_rows)
    _sync_global_rmse(final_rows)
    _sync_global_nrmse(final_rows)
    print(f"Wrote {len(final_rows)} final LLM rows to {FINAL_PATH}")
    print(f"Wrote {len(final_rows)} manifest rows to {MANIFEST_PATH}")
    print(f"Synchronized LLM rows in {RMSE_PATH}")
    print(f"Synchronized LLM rows in {NRMSE_PATH}")


if __name__ == "__main__":
    main()
