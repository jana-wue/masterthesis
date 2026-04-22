from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.imputation.llm_paper_single import (
    build_paper_single_prompt,
    build_paper_single_retry_prompt,
    extract_numeric_value,
    summarize_target_distribution,
)
from src.paths import DATA_RAW, DATA_PROCESSED, DATA_RESULTS

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME_PROMPT = "Telco-Customer-Churn"
TARGET_COLUMN = "TotalCharges"
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def _load_telco_mar_data():
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )
    return df_full, df_missing


def model_name_to_file_token(model_name: str):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def _append_results_row(csv_path: Path, required_columns: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    row_normalized = {col: row.get(col, pd.NA) for col in required_columns}
    df = pd.concat([df, pd.DataFrame([row_normalized])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def append_to_global_results(
    dataset: str,
    missingness_type: str,
    missing_rate: str,
    imputation_method: str,
    mean_rmse: float,
    mean_nrmse: float,
) -> None:
    _append_results_row(
        csv_path=RESULTS_RMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
        },
    )

    _append_results_row(
        csv_path=RESULTS_NRMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
            "mean_nrmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
            "mean_nrmse": mean_nrmse,
        },
    )


def _append_prediction_rows(output_csv: Path, rows_df: pd.DataFrame) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        combined = pd.concat([existing_df, rows_df], ignore_index=True, sort=False)
    else:
        combined = rows_df.copy()

    combined.to_csv(output_csv, index=False)


def _load_or_get_model(model_name, tokenizer=None, model=None):
    owns_model = tokenizer is None or model is None
    if not owns_model:
        return tokenizer, model

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'transformers' package is required for model inference. "
            "Install project dependencies first (e.g. pip install -r requirements.txt)."
        ) from exc

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


def _apply_user_chat_template(prompt_text, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def _generate_answer(prompt_text, tokenizer, model, max_new_tokens):
    model_input = _apply_user_chat_template(prompt_text, tokenizer)
    model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = tokenizer(model_input, return_tensors="pt").to(model_device)
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


def fallback_totalcharges(row, totalcharges_median):
    tenure = row.get("tenure")
    monthly = row.get("MonthlyCharges")

    if pd.notna(tenure) and pd.notna(monthly):
        try:
            return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
        except Exception:
            pass

    return float(totalcharges_median), "fallback_median"


def _resolve_target_indices(df_missing, row_index: int | None) -> list[int]:
    missing_indices = df_missing.index[df_missing[TARGET_COLUMN].isna()].tolist()
    if not missing_indices:
        raise ValueError(f"No missing rows found for target column '{TARGET_COLUMN}'.")

    if row_index is None:
        return [int(idx) for idx in missing_indices]

    if row_index not in df_missing.index:
        raise ValueError(f"row_index={row_index} is not present in df_missing.")

    return [int(row_index)]


def impute_single_row_with_paper_prompt(row, dataset_name, target_column, feature_columns, target_stats: dict[str, float] | None,
    domain_hints: list[str] | None, tokenizer, model, max_new_tokens: int, fallback_median: float):
    primary_prompt = build_paper_single_prompt(
        dataset_name=dataset_name,
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
        target_stats=target_stats,
        domain_hints=domain_hints,
    )
    raw_primary = _generate_answer(
        prompt_text=primary_prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    parsed_primary = extract_numeric_value(raw_primary)
    if parsed_primary is not None:
        return float(parsed_primary), "paper_single_prompt", raw_primary

    retry_prompt = build_paper_single_retry_prompt(
        dataset_name=dataset_name,
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
        target_stats=target_stats,
        domain_hints=domain_hints,
    )
    raw_retry = _generate_answer(
        prompt_text=retry_prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    parsed_retry = extract_numeric_value(raw_retry)
    if parsed_retry is not None:
        return float(parsed_retry), "paper_single_prompt_retry", raw_retry

    fallback_value, source = fallback_totalcharges(
        row=row,
        totalcharges_median=fallback_median,
    )
    combined_raw = f"first_try={raw_primary!r} | retry={raw_retry!r}"
    return fallback_value, source, combined_raw


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-row imputation with a paper-style prompt. "
            "Outputs one numeric value only."
        )
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Row index to impute. If omitted, all rows with missing TotalCharges are imputed.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process only the first N selected rows (for test runs).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N rows to stderr (0 disables progress prints).",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Do not append to CSV result files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print row index, source, and raw model output to stderr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_full, df_missing = _load_telco_mar_data()
    target_indices = _resolve_target_indices(df_missing=df_missing, row_index=args.row_index)
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("--max-rows must be >= 1.")
        target_indices = target_indices[: args.max_rows]
    if not target_indices:
        raise ValueError("No target rows selected after applying filters.")

    feature_columns = [col for col in df_missing.columns if col not in [TARGET_COLUMN, "customerID"]]
    target_stats = summarize_target_distribution(df_full, TARGET_COLUMN)
    observed_target = pd.to_numeric(df_full[TARGET_COLUMN], errors="coerce").dropna()
    totalcharges_median = float(observed_target.median())

    tokenizer, model = _load_or_get_model(model_name=MODEL_NAME)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    result_rows = []
    total_rows = len(target_indices)
    start_time = time.perf_counter()
    for i, row_index in enumerate(target_indices, start=1):
        row = df_missing.loc[row_index]

        prediction, source, raw_output = impute_single_row_with_paper_prompt(
            row=row,
            dataset_name=DATASET_NAME_PROMPT,
            target_column=TARGET_COLUMN,
            feature_columns=feature_columns,
            target_stats=target_stats,
            domain_hints=[
                "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Respect plausible values from the observed target distribution.",
            ],
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            fallback_median=totalcharges_median,
        )

        ground_truth = pd.to_numeric(df_full.loc[row_index, TARGET_COLUMN], errors="coerce")
        ground_truth_value = float(ground_truth) if pd.notna(ground_truth) else pd.NA
        abs_error = (
            float(abs(prediction - ground_truth_value))
            if pd.notna(ground_truth_value)
            else pd.NA
        )
        sq_error = (
            float((prediction - ground_truth_value) ** 2)
            if pd.notna(ground_truth_value)
            else pd.NA
        )

        result_rows.append(
            {
                "row_index": int(row_index),
                "prediction": float(prediction),
                "ground_truth": ground_truth_value,
                "abs_error": abs_error,
                "sq_error": sq_error,
                "source": source,
                "raw_output": raw_output,
                "model_name": MODEL_NAME,
                "target_column": TARGET_COLUMN,
                "max_new_tokens": args.max_new_tokens,
                "run_timestamp_utc": run_timestamp,
            }
        )

        if args.progress_every > 0 and (
            i == 1 or i % args.progress_every == 0 or i == total_rows
        ):
            elapsed = time.perf_counter() - start_time
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = total_rows - i
            eta_seconds = (remaining / rate) if rate > 0 else float("inf")
            eta_text = f"{eta_seconds:.1f}s" if eta_seconds != float("inf") else "n/a"
            print(
                f"[progress] {i}/{total_rows} rows | elapsed={elapsed:.1f}s | eta={eta_text}",
                file=sys.stderr,
                flush=True,
            )

    if not args.no_save_results:
        model_token = model_name_to_file_token(MODEL_NAME)
        single_results_csv = DATA_RESULTS / f"telco_{model_token}_paper_single_results.csv"

        row_df = pd.DataFrame(result_rows)
        _append_prediction_rows(output_csv=single_results_csv, rows_df=row_df)

        valid_eval = row_df.dropna(subset=["ground_truth", "prediction"]).copy()
        if not valid_eval.empty:
            errors = valid_eval["ground_truth"] - valid_eval["prediction"]
            mean_rmse = float(((errors ** 2).mean()) ** 0.5)
            std_true_full = float(observed_target.std(ddof=0))
            mean_nrmse = float(mean_rmse / (std_true_full + 1e-8))
            model_short = MODEL_NAME.split("/")[-1]
            if len(target_indices) == 1:
                method_name = f"LLM Paper Single Prompt ({model_short})"
            else:
                method_name = f"LLM Paper Single Prompt All Missing ({model_short})"

            append_to_global_results(
                dataset="Telco",
                missingness_type="MAR",
                missing_rate="missing totalCharges -> tenure, 10%",
                imputation_method=method_name,
                mean_rmse=mean_rmse,
                mean_nrmse=mean_nrmse,
            )

    # Keep stdout as value-only output for easy piping/use in local workflows.
    for row in result_rows:
        print(row["prediction"])

    if args.debug:
        print(f"n_rows_imputed={len(result_rows)}", file=sys.stderr)
        print(f"model_name={MODEL_NAME}", file=sys.stderr)
        for row in result_rows:
            print(f"row_index={row['row_index']}", file=sys.stderr)
            print(f"source={row['source']}", file=sys.stderr)
            print(f"raw_output={row['raw_output']}", file=sys.stderr)
        if not args.no_save_results:
            model_token = model_name_to_file_token(MODEL_NAME)
            print(
                f"saved_single_results={DATA_RESULTS / f'telco_{model_token}_paper_single_results.csv'}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
