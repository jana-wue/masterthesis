import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
RMSE_PATH = RESULTS_DIR / "imputation_results.csv"
NRMSE_PATH = RESULTS_DIR / "imputation_results_nrsme.csv"
FINAL_PATH = RESULTS_DIR / "llm_finetuning_results_final.csv"
MANIFEST_PATH = ROOT / "scenario_manifest_llm_finetuning_10pct_final.csv"


MODEL_INFO = {
    "Mistral-7B-Instruct-v0.3": {
        "model_key": "mistral",
        "hf_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "Qwen2.5-7B-Instruct": {
        "model_key": "qwen25",
        "hf_model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
    "Llama-3.1-8B-Instruct": {
        "model_key": "llama31",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


TARGET_INFO = {
    "Telco": {
        "target_column": "TotalCharges",
        "detail_target_slug": "totalcharges",
    },
    "German Statlog": {
        "target_column": "X2",
        "detail_target_slug": "x2",
    },
    "German Credit Card": {
        "target_column": "BILL_AMT1",
        "detail_target_slug": "billamt1",
    },
}


def parse_model_label(method_name: str) -> str:
    prefix = "LLM Finetuned ("
    if not method_name.startswith(prefix) or not method_name.endswith(")"):
        raise ValueError(f"Unexpected method label: {method_name}")
    return method_name[len(prefix) : -1]


def load_rmse_rows():
    rows = {}
    with RMSE_PATH.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5 or not row[3].startswith("LLM Finetuned"):
                continue
            dataset, missingness_type, missing_rate, method_name, rmse = row[:5]
            model_label = parse_model_label(method_name)
            rows[(dataset, missingness_type, missing_rate, model_label)] = {
                "dataset": dataset,
                "missingness_type": missingness_type,
                "missing_rate": missing_rate,
                "model_label": model_label,
                "rmse": rmse,
            }
    return rows


def merge_nrmse(rows):
    with NRMSE_PATH.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5 or not row[3].startswith("LLM Finetuned"):
                continue
            dataset, missingness_type, missing_rate, method_name, nrmse = row[:5]
            model_label = parse_model_label(method_name)
            key = (dataset, missingness_type, missing_rate, model_label)
            if key not in rows:
                continue
            rows[key]["nrmse"] = nrmse


def build_detail_filename(dataset: str, model_key: str, missingness_type: str) -> str:
    dataset_slug = {
        "Telco": "telco",
        "German Statlog": "statlog",
        "German Credit Card": "credit",
    }[dataset]
    target_slug = TARGET_INFO[dataset]["detail_target_slug"]
    mech_slug = missingness_type.lower()
    return f"{dataset_slug}_{model_key}_{target_slug}_{mech_slug}_finetuned_eval.csv"


def build_final_rows(rows):
    final_rows = []
    for key in sorted(rows):
        row = rows[key]
        model_meta = MODEL_INFO[row["model_label"]]
        target_meta = TARGET_INFO[row["dataset"]]
        final_rows.append(
            {
                "dataset": row["dataset"],
                "missingness_type": row["missingness_type"],
                "missing_rate_label": row["missing_rate"],
                "target_column": target_meta["target_column"],
                "model_key": model_meta["model_key"],
                "model_label": row["model_label"],
                "hf_model_name": model_meta["hf_model_name"],
                "rmse": row["rmse"],
                "nrmse": row.get("nrmse", ""),
                "detail_eval_file": build_detail_filename(
                    row["dataset"], model_meta["model_key"], row["missingness_type"]
                ),
            }
        )
    return final_rows


def write_final_results(final_rows):
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


def write_manifest(final_rows):
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


def main():
    rows = load_rmse_rows()
    merge_nrmse(rows)
    final_rows = build_final_rows(rows)
    write_final_results(final_rows)
    write_manifest(final_rows)
    print(f"Wrote {len(final_rows)} final LLM rows to {FINAL_PATH}")
    print(f"Wrote {len(final_rows)} manifest rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
