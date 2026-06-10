from __future__ import annotations

import argparse
import gc
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_RESULTS


DEFAULT_MANIFEST_PATH = Path("scenario_manifest_llm_finetuning_10pct_final.csv")
RUNS_SUMMARY_PATH = DATA_RESULTS / "benchmark_llm_prompt_runs.csv"


MODEL_META = {
    "mistral": {
        "benchmark_model_key": "mistral",
        "hf_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_label": "Mistral-7B-Instruct-v0.3",
    },
    "llama31": {
        "benchmark_model_key": "llama",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_label": "Llama-3.1-8B-Instruct",
    },
    "qwen25": {
        "benchmark_model_key": "qwen",
        "hf_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_label": "Qwen2.5-7B-Instruct",
    },
}


DATASET_META = {
    "German Credit Card": {
        "dataset_token": "credit",
        "benchmark_dataset_key": "creditcard",
        "target_column": "BILL_AMT1",
        "target_slug": "billamt1",
        "dataset_name": "German Credit Card",
        "raw_csv": DATA_RAW / "default_of_credit_card_clients.csv",
        "id_column": "ID",
        "missing_paths": {
            "MAR": Path("data/processed/MAR/german_credit_mar_billamt1_pay0_10pct.csv"),
            "MCAR": Path("data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv"),
            "MNAR": Path("data/processed/MNAR/german_credit_mnar_billamt1_10pct.csv"),
        },
        "missing_rate_labels": {
            "MAR": "billamt1 -> pay0, 10%",
            "MCAR": "billamt1, 10%",
            "MNAR": "billamt1, 10%",
        },
        "domain_hints": [
            "BILL_AMT1 is a credit-card bill amount and should stay on a plausible monetary scale.",
            "Use the related bill and payment history columns to keep the estimate consistent.",
        ],
    },
    "German Statlog": {
        "dataset_token": "statlog",
        "benchmark_dataset_key": "statlog",
        "target_column": "X2",
        "target_slug": "x2",
        "dataset_name": "German Statlog",
        "raw_csv": DATA_RAW / "german_statlog_numeric.csv",
        "id_column": None,
        "missing_paths": {
            "MAR": Path("data/processed/MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv"),
            "MCAR": Path("data/processed/MCAR/german_statlog_numeric_mcar_duration_10pct.csv"),
            "MNAR": Path("data/processed/MNAR/german_statlog_numeric_mnar_duration_10pct.csv"),
        },
        "missing_rate_labels": {
            "MAR": "X2, 10%",
            "MCAR": "X2, 10%",
            "MNAR": "X2, 10%",
        },
        "domain_hints": [
            "Keep the estimate numerically plausible relative to the observed target distribution.",
            "Use the other observed numeric features for consistency, but return only one number.",
        ],
    },
    "Telco": {
        "dataset_token": "telco",
        "benchmark_dataset_key": "telco",
        "target_column": "TotalCharges",
        "target_slug": "totalcharges",
        "dataset_name": "Telco",
        "raw_csv": DATA_RAW / "Telco-Customer-Churn_cleaned.csv",
        "id_column": "customerID",
        "missing_paths": {
            "MAR": Path("data/processed/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv"),
            "MCAR": Path("data/processed/MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv"),
            "MNAR": Path("data/processed/MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv"),
        },
        "missing_rate_labels": {
            "MAR": "missing totalCharges -> tenure, 10%",
            "MCAR": "missing TotalCharges, 10%",
            "MNAR": "missing TotalCharges, 10%",
        },
        "domain_hints": [
            "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
            "Respect plausible values from the observed target distribution.",
        ],
    },
}


SCENARIO_META = {
    "MAR": {
        "scenario_token": "mar",
        "scenario_family": "MAR_TARGET",
    },
    "MCAR": {
        "scenario_token": "mcar",
        "scenario_family": "MCAR_TARGET",
    },
    "MNAR": {
        "scenario_token": "mnar",
        "scenario_family": "MNAR_TARGET",
    },
}


SUMMARY_COLUMNS = [
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
    "few_shot_k",
]


def model_name_to_file_token(model_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def extract_first_number(text: str) -> str | None:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)
    return None


def apply_chat_template(messages: list[dict], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(message["content"] for message in messages)


def _load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def _append_rows(csv_path: Path, required_columns: list[str], rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    normalized_rows = [{col: row.get(col, pd.NA) for col in required_columns} for row in rows]
    df = pd.concat([df, pd.DataFrame(normalized_rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _prompt_method_meta(few_shot_k: int) -> tuple[str, str, str]:
    if few_shot_k == 0:
        return "llm_prompt_zero", "LLM Prompt Zero-shot", "zero"
    return (
        f"llm_prompt_fewshot_k{few_shot_k}",
        f"LLM Prompt Few-shot k={few_shot_k}",
        f"few{few_shot_k}",
    )


def _build_detail_eval_path(dataset_meta: dict, model_key: str, missingness_type: str, few_shot_k: int) -> Path:
    _, _, prompt_token = _prompt_method_meta(few_shot_k)
    scenario_token = SCENARIO_META[missingness_type]["scenario_token"]
    filename = (
        f"{dataset_meta['dataset_token']}_{model_key}_{dataset_meta['target_slug']}_"
        f"{scenario_token}_{prompt_token}_prompt_eval.csv"
    )
    return DATA_RESULTS / filename


def _fallback_numeric(row: pd.Series, target_column: str, target_median: float) -> tuple[float, str]:
    if target_column == "TotalCharges":
        tenure = row.get("tenure")
        monthly = row.get("MonthlyCharges")
        if pd.notna(tenure) and pd.notna(monthly):
            try:
                return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
            except Exception:
                pass
    return float(target_median), "fallback_median"


def _generate_raw_answer(messages: list[dict], tokenizer, model, max_new_tokens: int) -> str:
    prompt_text = apply_chat_template(messages, tokenizer)
    model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _predict_numeric(
    row: pd.Series,
    imputer: LLMImputer,
    tokenizer,
    model,
    few_shot_examples: list[pd.Series],
    target_column: str,
    target_median: float,
    max_new_tokens: int,
) -> tuple[float, str, str]:
    primary_messages = imputer.build_inference_messages(row, few_shot_examples=few_shot_examples)
    raw_primary = _generate_raw_answer(
        messages=primary_messages,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    pred_primary = extract_first_number(raw_primary)
    if pred_primary is not None:
        return float(pred_primary), raw_primary, "model_first_try"

    retry_messages = imputer.build_retry_messages(row)
    raw_retry = _generate_raw_answer(
        messages=retry_messages,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    pred_retry = extract_first_number(raw_retry)
    if pred_retry is not None:
        return float(pred_retry), raw_retry, "model_retry"

    fallback_value, fallback_source = _fallback_numeric(
        row=row,
        target_column=target_column,
        target_median=target_median,
    )
    combined_raw = f"first_try={raw_primary!r} | retry={raw_retry!r}"
    return fallback_value, combined_raw, fallback_source


def _compute_metrics(results_df: pd.DataFrame) -> tuple[float, float, float]:
    valid = results_df.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid.empty:
        raise ValueError("No valid numeric prediction/ground_truth rows produced.")

    valid["prediction"] = pd.to_numeric(valid["prediction"], errors="coerce")
    valid["ground_truth"] = pd.to_numeric(valid["ground_truth"], errors="coerce")
    valid = valid.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid.empty:
        raise ValueError("No numeric prediction/ground_truth rows produced.")

    errors = valid["ground_truth"] - valid["prediction"]
    mean_rmse = float(math.sqrt(float((errors ** 2).mean())))
    std_true = float(valid["ground_truth"].std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true + 1e-8))
    fallback_rate = float((valid["source"].astype(str) != "model_first_try").mean())
    return mean_rmse, mean_nrmse, fallback_rate


def _load_eval_frames(dataset_label: str, missingness_type: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset_meta = DATASET_META[dataset_label]
    missing_path = dataset_meta["missing_paths"][missingness_type]
    df_full = pd.read_csv(dataset_meta["raw_csv"])
    df_missing = pd.read_csv(missing_path)
    return df_full, df_missing, dataset_meta


def _build_imputer(dataset_meta: dict, df_missing: pd.DataFrame, model_name: str, few_shot_k: int) -> LLMImputer:
    target_column = dataset_meta["target_column"]
    feature_columns = [
        col for col in df_missing.columns if col not in {target_column, dataset_meta["id_column"]}
    ]
    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
            domain_hints=dataset_meta["domain_hints"],
            few_shot_k=few_shot_k,
        )
    )
    observed_rows = df_missing[df_missing[target_column].notna()].copy()
    imputer.fit_target_stats(observed_rows)
    return imputer


def _run_prompt_setting(
    dataset_label: str,
    missingness_type: str,
    model_key: str,
    hf_model_name: str,
    few_shot_k: int,
    n_examples: int | None,
    max_new_tokens: int,
    tokenizer,
    model,
) -> tuple[pd.DataFrame, dict]:
    df_full, df_missing, dataset_meta = _load_eval_frames(
        dataset_label=dataset_label,
        missingness_type=missingness_type,
    )
    target_column = dataset_meta["target_column"]
    id_column = dataset_meta["id_column"]
    missing_path = dataset_meta["missing_paths"][missingness_type]
    imputer = _build_imputer(
        dataset_meta=dataset_meta,
        df_missing=df_missing,
        model_name=hf_model_name,
        few_shot_k=few_shot_k,
    )

    missing_rows = df_missing[df_missing[target_column].isna()].copy()
    observed_rows = df_missing[df_missing[target_column].notna()].copy()
    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{target_column}'.")
    if few_shot_k > 0 and observed_rows.empty:
        raise ValueError("Few-shot requested but no observed reference rows are available.")
    if n_examples is not None:
        missing_rows = missing_rows.head(n_examples).copy()

    observed_target = pd.to_numeric(df_full[target_column], errors="coerce").dropna()
    if observed_target.empty:
        raise ValueError(f"No observed target values found in '{target_column}'.")
    target_median = float(observed_target.median())

    method_key, method_name_base, prompt_token = _prompt_method_meta(few_shot_k)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    results = []

    print("\n" + "=" * 80)
    print(
        f"PROMPT RUN | dataset={dataset_label} | missingness={missingness_type} | "
        f"model={model_key} | few_shot_k={few_shot_k} | rows={len(missing_rows)}"
    )
    print("=" * 80)

    for idx, row in missing_rows.iterrows():
        few_shot_examples = imputer.select_few_shot_examples(
            row=row,
            reference_df=observed_rows,
            exclude_indices={idx},
        )
        prediction, raw_output, source = _predict_numeric(
            row=row,
            imputer=imputer,
            tokenizer=tokenizer,
            model=model,
            few_shot_examples=few_shot_examples,
            target_column=target_column,
            target_median=target_median,
            max_new_tokens=max_new_tokens,
        )
        ground_truth = pd.to_numeric(df_full.loc[idx, target_column], errors="coerce")

        record = {
            "row_index": int(idx),
            "prediction": float(prediction),
            "ground_truth": ground_truth,
            "raw_output": raw_output,
            "source": source,
            "few_shot_k": few_shot_k,
            "n_fewshot_examples": len(few_shot_examples),
            "dataset": dataset_label,
            "dataset_key": dataset_meta["benchmark_dataset_key"],
            "target_column": target_column,
            "missingness_type": missingness_type,
            "missing_rate_label": dataset_meta["missing_rate_labels"][missingness_type],
            "model_key": model_key,
            "model_name": hf_model_name,
            "model_label": MODEL_META[model_key]["model_label"],
            "prompt_method_key": method_key,
            "prompt_method_name": method_name_base,
            "prompt_variant_token": prompt_token,
            "run_timestamp_utc": run_timestamp,
        }
        if id_column is not None and id_column in row.index:
            record[id_column] = row[id_column]
        results.append(record)

    results_df = pd.DataFrame(results)
    mean_rmse, mean_nrmse, fallback_rate = _compute_metrics(results_df)

    detail_eval_path = _build_detail_eval_path(
        dataset_meta=dataset_meta,
        model_key=model_key,
        missingness_type=missingness_type,
        few_shot_k=few_shot_k,
    )
    _append_rows(
        csv_path=detail_eval_path,
        required_columns=list(results_df.columns),
        rows=results_df.to_dict(orient="records"),
    )

    summary_row = {
        "run_id": (
            f"prompt::{dataset_meta['dataset_token']}::{model_key}::"
            f"{SCENARIO_META[missingness_type]['scenario_token']}::k{few_shot_k}::{run_timestamp}"
        ),
        "run_timestamp_utc": run_timestamp,
        "dataset_key": dataset_meta["benchmark_dataset_key"],
        "dataset_name": dataset_meta["dataset_name"],
        "target_column": target_column,
        "scenario_family": SCENARIO_META[missingness_type]["scenario_family"],
        "missingness_type": missingness_type,
        "mcar_scope": "target",
        "rate_pct": 10,
        "seed": 42,
        "method_family": "llm_prompt",
        "method_key": method_key,
        "method_name": method_name_base,
        "model_key": MODEL_META[model_key]["benchmark_model_key"],
        "model_name": hf_model_name,
        "status": "success",
        "error_message": pd.NA,
        "full_path": str(dataset_meta["raw_csv"]),
        "missing_path": str(missing_path),
        "n_masked_target": int(len(results_df)),
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse,
        "runtime_seconds": pd.NA,
        "fallback_rate": fallback_rate,
        "source_file": str(detail_eval_path),
        "source_type": "llm_prompt_eval_current_run",
        "provenance_note": (
            "Deterministic prompt baseline on the 10% target-only benchmark split."
            "Rows are evaluated on the masked target cells; seed is fixed to 42"
            "for alignment with the benchmark metadata, but generation uses do_sample=False."
        ),
        "few_shot_k": few_shot_k,
    }
    return results_df, summary_row


def _load_manifest_rows(
    manifest_path: Path,
    dataset_filters: set[str] | None,
    model_filters: set[str] | None,
    missingness_filters: set[str] | None,
) -> list[dict]:
    manifest = pd.read_csv(manifest_path)
    required_columns = {
        "dataset",
        "target_column",
        "missingness_type",
        "model_key",
        "hf_model_name",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing required columns: {missing_list}")

    work = manifest.copy()
    work = work.drop_duplicates(
        subset=["dataset", "target_column", "missingness_type", "model_key", "hf_model_name"]
    )

    if dataset_filters:
        work = work[work["dataset"].isin(dataset_filters)].copy()
    if model_filters:
        work = work[work["model_key"].isin(model_filters)].copy()
    if missingness_filters:
        work = work[work["missingness_type"].isin(missingness_filters)].copy()

    if work.empty:
        raise ValueError("No manifest rows left after filtering.")

    work = work.sort_values(by=["model_key", "dataset", "missingness_type"]).reset_index(drop=True)
    return work.to_dict(orient="records")


def _existing_success_keys(summary_path: Path) -> set[tuple[str, str, str, int]]:
    if not summary_path.exists():
        return set()

    df = pd.read_csv(summary_path)
    required = {"dataset_name", "model_name", "missingness_type", "few_shot_k", "status"}
    if not required.issubset(df.columns):
        return set()

    success = df[df["status"].astype(str) == "success"].copy()
    if success.empty:
        return set()

    success["few_shot_k"] = pd.to_numeric(success["few_shot_k"], errors="coerce")
    success = success.dropna(subset=["few_shot_k"])
    return {
        (
            str(row["dataset_name"]),
            str(row["model_name"]),
            str(row["missingness_type"]),
            int(row["few_shot_k"]),
        )
        for _, row in success.iterrows()
    }


def run_prompt_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    few_shot_settings: tuple[int, ...] = (0, 2),
    n_examples: int | None = None,
    max_new_tokens: int = 20,
    datasets: tuple[str, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
    missingness_types: tuple[str, ...] | None = None,
    max_runs: int | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    dataset_filters = set(datasets or [])
    model_filters = set(model_keys or [])
    missingness_filters = set(missingness_types or [])
    manifest_rows = _load_manifest_rows(
        manifest_path=manifest_path,
        dataset_filters=dataset_filters if dataset_filters else None,
        model_filters=model_filters if model_filters else None,
        missingness_filters=missingness_filters if missingness_filters else None,
    )

    success_keys = _existing_success_keys(RUNS_SUMMARY_PATH) if skip_existing else set()
    summary_rows = []
    current_model_name = None
    tokenizer = None
    model = None
    runs_started = 0

    try:
        for row in manifest_rows:
            dataset_label = str(row["dataset"])
            missingness_type = str(row["missingness_type"]).upper()
            model_key = str(row["model_key"])
            hf_model_name = str(row["hf_model_name"])

            if dataset_label not in DATASET_META:
                raise ValueError(f"Unsupported dataset in manifest: {dataset_label}")
            if model_key not in MODEL_META:
                raise ValueError(f"Unsupported model_key in manifest: {model_key}")
            if missingness_type not in SCENARIO_META:
                raise ValueError(f"Unsupported missingness_type in manifest: {missingness_type}")

            if current_model_name != hf_model_name:
                if model is not None:
                    del model
                    model = None
                if tokenizer is not None:
                    del tokenizer
                    tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("\n" + "=" * 80)
                print(f"LOADING MODEL | {hf_model_name}")
                print("=" * 80)
                tokenizer, model = _load_model(hf_model_name)
                current_model_name = hf_model_name

            for few_shot_k in few_shot_settings:
                success_key = (dataset_label, hf_model_name, missingness_type, int(few_shot_k))
                if skip_existing and success_key in success_keys:
                    print(
                        f"SKIP existing success | dataset={dataset_label} | "
                        f"missingness={missingness_type} | model={model_key} | few_shot_k={few_shot_k}"
                    )
                    continue

                if max_runs is not None and runs_started >= max_runs:
                    return pd.DataFrame(summary_rows)

                try:
                    _, summary_row = _run_prompt_setting(
                        dataset_label=dataset_label,
                        missingness_type=missingness_type,
                        model_key=model_key,
                        hf_model_name=hf_model_name,
                        few_shot_k=int(few_shot_k),
                        n_examples=n_examples,
                        max_new_tokens=max_new_tokens,
                        tokenizer=tokenizer,
                        model=model,
                    )
                except Exception as exc:
                    method_key, method_name_base, _ = _prompt_method_meta(int(few_shot_k))
                    dataset_meta = DATASET_META[dataset_label]
                    summary_row = {
                        "run_id": (
                            f"prompt::{dataset_meta['dataset_token']}::{model_key}::"
                            f"{SCENARIO_META[missingness_type]['scenario_token']}::k{few_shot_k}::"
                            f"{datetime.now(timezone.utc).isoformat()}"
                        ),
                        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset_key": dataset_meta["benchmark_dataset_key"],
                        "dataset_name": dataset_meta["dataset_name"],
                        "target_column": dataset_meta["target_column"],
                        "scenario_family": SCENARIO_META[missingness_type]["scenario_family"],
                        "missingness_type": missingness_type,
                        "mcar_scope": "target",
                        "rate_pct": 10,
                        "seed": 42,
                        "method_family": "llm_prompt",
                        "method_key": method_key,
                        "method_name": method_name_base,
                        "model_key": MODEL_META[model_key]["benchmark_model_key"],
                        "model_name": hf_model_name,
                        "status": "error",
                        "error_message": str(exc),
                        "full_path": str(dataset_meta["raw_csv"]),
                        "missing_path": str(dataset_meta["missing_paths"][missingness_type]),
                        "n_masked_target": pd.NA,
                        "mean_rmse": pd.NA,
                        "mean_nrmse": pd.NA,
                        "runtime_seconds": pd.NA,
                        "fallback_rate": pd.NA,
                        "source_file": str(
                            _build_detail_eval_path(
                                dataset_meta=dataset_meta,
                                model_key=model_key,
                                missingness_type=missingness_type,
                                few_shot_k=int(few_shot_k),
                            )
                        ),
                        "source_type": "llm_prompt_eval_current_run",
                        "provenance_note": "Prompt baseline run failed before metrics could be computed.",
                        "few_shot_k": int(few_shot_k),
                    }

                summary_rows.append(summary_row)
                runs_started += 1
    finally:
        if summary_rows:
            _append_rows(RUNS_SUMMARY_PATH, SUMMARY_COLUMNS, summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print("\n" + "=" * 80)
        print("PROMPT RUN SUMMARY")
        print("=" * 80)
        print(summary_df)
        print(f"\nSaved run summary to: {RUNS_SUMMARY_PATH}")
    return summary_df


def evaluate_telco_prompt_approach(few_shot_settings=(0, 2), n_examples=None) -> pd.DataFrame:
    return run_prompt_manifest(
        manifest_path=DEFAULT_MANIFEST_PATH,
        few_shot_settings=tuple(int(k) for k in few_shot_settings),
        n_examples=n_examples,
        datasets=("Telco",),
        missingness_types=("MAR",),
        model_keys=("mistral",),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot/few-shot LLM prompt baselines "
            "for the 10% target-only benchmark manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Scenario manifest to use. Defaults to the 10%% LLM benchmark manifest.",
    )
    parser.add_argument(
        "--few-shot-settings",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Few-shot k settings to evaluate, e.g. 0 2.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Optional cap on the number of masked rows evaluated per run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max tokens generated per prompt answer.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. Telco 'German Statlog'.",
    )
    parser.add_argument(
        "--model-keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional model-key filter, e.g. mistral qwen25 llama31.",
    )
    parser.add_argument(
        "--missingness-types",
        type=str,
        nargs="*",
        default=None,
        help="Optional missingness filter, e.g. MAR MCAR MNAR.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional hard cap on the number of prompt runs started.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip settings that already have a successful row in benchmark_llm_prompt_runs.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prompt_manifest(
        manifest_path=args.manifest,
        few_shot_settings=tuple(args.few_shot_settings),
        n_examples=args.n_examples,
        max_new_tokens=args.max_new_tokens,
        datasets=tuple(args.datasets) if args.datasets else None,
        model_keys=tuple(args.model_keys) if args.model_keys else None,
        missingness_types=tuple(m.upper() for m in args.missingness_types) if args.missingness_types else None,
        max_runs=args.max_runs,
        skip_existing=args.skip_existing,
    )
