import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
LLM_FINETUNED_DIR = RESULTS_DIR / "new_without_domain_hints"
RMSE_PATH = RESULTS_DIR / "imputation_results.csv"
NRMSE_PATH = RESULTS_DIR / "imputation_results_nrmse.csv"
FINAL_PATH = RESULTS_DIR / "llm_finetuning_results_final.csv"
MANIFEST_PATH = ROOT / "scenario_manifest_hpc_llm_finetuned.csv"


DATASET_META = {
    "credit": {
        "dataset": "German Credit Card",
        "dataset_key": "creditcard",
        "target_column": "BILL_AMT1",
        "mar_dependency_column": "PAY_0",
        "mnar_driver_column": "BILL_AMT1",
    },
    "statlog": {
        "dataset": "German Statlog",
        "dataset_key": "statlog",
        "target_column": "X2",
        "mar_dependency_column": "X5",
        "mnar_driver_column": "X2",
    },
    "telco": {
        "dataset": "Telco",
        "dataset_key": "telco",
        "target_column": "TotalCharges",
        "mar_dependency_column": "tenure",
        "mnar_driver_column": "TotalCharges",
    },
}


MODEL_META = {
    "llama31": {
        "model_key": "llama",
        "model_label": "Llama-3.1-8B-Instruct",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral": {
        "model_key": "mistral",
        "model_label": "Mistral-7B-Instruct-v0.3",
        "hf_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "qwen25": {
        "model_key": "qwen",
        "model_label": "Qwen2.5-7B-Instruct",
        "hf_model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
}


SCENARIO_META = {
    ("credit", "mar"): {
        "missingness_type": "MAR",
        "scenario_family": "MAR_TARGET",
        "missing_rate_label": "billamt1 -> pay0, 10%",
    },
    ("credit", "mcar"): {
        "missingness_type": "MCAR",
        "scenario_family": "MCAR_TARGET",
        "missing_rate_label": "billamt1, 10%",
    },
    ("credit", "mnar"): {
        "missingness_type": "MNAR",
        "scenario_family": "MNAR_TARGET",
        "missing_rate_label": "billamt1, 10%",
    },
    ("statlog", "mar"): {
        "missingness_type": "MAR",
        "scenario_family": "MAR_TARGET",
        "missing_rate_label": "X2, 10%",
    },
    ("statlog", "mcar"): {
        "missingness_type": "MCAR",
        "scenario_family": "MCAR_TARGET",
        "missing_rate_label": "X2, 10%",
    },
    ("statlog", "mnar"): {
        "missingness_type": "MNAR",
        "scenario_family": "MNAR_TARGET",
        "missing_rate_label": "X2, 10%",
    },
    ("telco", "mar"): {
        "missingness_type": "MAR",
        "scenario_family": "MAR_TARGET",
        "missing_rate_label": "missing totalCharges -> tenure, 10%",
    },
    ("telco", "mcar"): {
        "missingness_type": "MCAR",
        "scenario_family": "MCAR_TARGET",
        "missing_rate_label": "missing TotalCharges, 10%",
    },
    ("telco", "mnar"): {
        "missingness_type": "MNAR",
        "scenario_family": "MNAR_TARGET",
        "missing_rate_label": "missing TotalCharges, 10%",
    },
}


def _parse_eval_filename(csv_path: Path) -> tuple[str, str, str]:
    """Parse evaluation filename."""
    stem = csv_path.stem
    if not stem.endswith("_finetuned_eval"):
        raise ValueError(f"Unexpected LLM finetuned eval filename: {csv_path.name}")

    if "_credit_billamt1_" in stem:
        dataset_token = "credit"
    elif "_statlog_x2_" in stem:
        dataset_token = "statlog"
    elif "_telco_totalcharges" in stem:
        dataset_token = "telco"
    else:
        raise ValueError(f"Could not infer dataset token from filename: {csv_path.name}")

    if "_qwen25_" in stem:
        model_token = "qwen25"
    elif "_llama31_" in stem:
        model_token = "llama31"
    elif "_mistral_" in stem:
        model_token = "mistral"
    else:
        raise ValueError(f"Could not infer model token from filename: {csv_path.name}")

    if "_mnar_lora_finetuned_eval" in stem:
        scenario_token = "mnar"
    elif "_mcar_lora_finetuned_eval" in stem:
        scenario_token = "mcar"
    elif "_mar_lora_finetuned_eval" in stem or "_totalcharges_lora_finetuned_eval" in stem:
        scenario_token = "mar"
    else:
        raise ValueError(f"Could not infer scenario token from filename: {csv_path.name}")

    return dataset_token, model_token, scenario_token


def _load_latest_valid_eval(csv_path: Path) -> tuple[str, pd.DataFrame]:
    """Load latest valid evaluation."""
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
    return latest_timestamp, valid


def _compute_metrics(valid_eval: pd.DataFrame) -> tuple[float, float]:
    """Compute metrics."""
    errors = valid_eval["ground_truth"] - valid_eval["prediction"]
    mean_rmse = float(math.sqrt(float((errors ** 2).mean())))
    std_true = float(valid_eval["ground_truth"].std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true + 1e-8))
    return mean_rmse, mean_nrmse


def _build_final_row(csv_path: Path) -> dict:
    """Build final row."""
    dataset_token, model_token, scenario_token = _parse_eval_filename(csv_path)
    dataset_meta = DATASET_META[dataset_token]
    model_meta = MODEL_META[model_token]
    scenario_meta = SCENARIO_META[(dataset_token, scenario_token)]
    run_timestamp_utc, valid_eval = _load_latest_valid_eval(csv_path)
    mean_rmse, mean_nrmse = _compute_metrics(valid_eval)

    return {
        "run_id": f"inferred::{csv_path.stem}::{run_timestamp_utc}",
        "run_timestamp_utc": run_timestamp_utc,
        "dataset_key": dataset_meta["dataset_key"],
        "dataset": dataset_meta["dataset"],
        "missingness_type": scenario_meta["missingness_type"],
        "scenario_family": scenario_meta["scenario_family"],
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
    """Build final rows."""
    rows = []
    finetuned_dir = LLM_FINETUNED_DIR if LLM_FINETUNED_DIR.exists() else RESULTS_DIR
    for csv_path in sorted(finetuned_dir.glob("*_finetuned_eval.csv")):
        rows.append(_build_final_row(csv_path))
    return rows


def write_final_results(final_rows: list[dict]) -> None:
    """Write final results."""
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
        writer.writerows({field: row.get(field) for field in fieldnames} for row in final_rows)


def write_manifest(final_rows: list[dict]) -> None:
    """Write manifest."""
    fieldnames = [
        "run_id",
        "dataset_key",
        "dataset_name",
        "target_column",
        "scenario_family",
        "missingness_type",
        "mcar_scope",
        "mar_dependency_column",
        "mnar_driver_column",
        "mcar_global_excluded_columns",
        "rate_pct",
        "seed",
        "method_family",
        "method_key",
        "method_name",
        "model_key",
        "model_name",
        "include_in_primary_leaderboard",
        "include_in_secondary_full_matrix",
    ]
    with MANIFEST_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            dataset_meta = next(
                meta for meta in DATASET_META.values() if meta["dataset_key"] == row["dataset_key"]
            )
            mar_dependency_column = (
                dataset_meta["mar_dependency_column"]
                if row["scenario_family"] == "MAR_TARGET"
                else None
            )
            mnar_driver_column = (
                dataset_meta["mnar_driver_column"]
                if row["scenario_family"] == "MNAR_TARGET"
                else None
            )
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "dataset_key": row["dataset_key"],
                    "dataset_name": row["dataset"],
                    "target_column": row["target_column"],
                    "scenario_family": row["scenario_family"],
                    "missingness_type": row["missingness_type"],
                    "mcar_scope": "target",
                    "mar_dependency_column": mar_dependency_column,
                    "mnar_driver_column": mnar_driver_column,
                    "mcar_global_excluded_columns": None,
                    "rate_pct": 10,
                    "seed": 42,
                    "method_family": "llm_finetuned",
                    "method_key": "llm_finetuned",
                    "method_name": "LLM Finetuned",
                    "model_key": row["model_key"],
                    "model_name": row["hf_model_name"],
                    "include_in_primary_leaderboard": True,
                    "include_in_secondary_full_matrix": False,
                }
            )


def _sync_global_rmse(final_rows: list[dict]) -> None:
    """Handle sync global RMSE."""
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
    """Handle sync global NRMSE."""
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
    """Run the script entry point."""
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
