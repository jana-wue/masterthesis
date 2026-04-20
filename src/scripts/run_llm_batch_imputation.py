from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm_batch import (
    LLMBatchImputer,
    LLMBatchImputerConfig,
    model_name_to_file_token,
)
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_RESULTS

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT_STYLE_NAME = "BatchImputation"
DATASET_NAME = "Telco Customer Churn"
TARGET_COLUMN = "TotalCharges"
ID_COLUMN = "customerID"
DEFAULT_BATCH_SIZE = 20
DEFAULT_BATCH_COL_SIZE = 10
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def _append_results_row(csv_path, required_columns, row) -> None:
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


def append_to_global_results(dataset,missingness_type ,missing_rate ,imputation_method, mean_rmse,
    mean_nrmse):
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


def resolve_telco_batch_results_csv(model_name, prompt_style_name):
    """
    Resolve batch prediction CSV path for Telco runs.
    """
    if model_name is not None:
        model_token = model_name_to_file_token(model_name)
        csv_path = DATA_RESULTS / f"telco_{model_token}_{prompt_style_name}_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {csv_path}")
        return csv_path

    pattern = f"telco_*_{prompt_style_name}_results.csv"
    candidates = sorted(
        DATA_RESULTS.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No prediction files found for pattern '{pattern}' in: {DATA_RESULTS}")
    return candidates[0]


def append_global_results_from_telco_batch_predictions(results_csv,model_name, prompt_style_name, latest_per_setting_only,
    dataset,missingness_type, missing_rate):
    """
    Aggregate Telco batch prediction CSV into RMSE/NRMSE and append to global result files.
    """
    source_csv = results_csv or resolve_telco_batch_results_csv(
        model_name=model_name, prompt_style_name=prompt_style_name
    )
    df = pd.read_csv(source_csv)

    required_cols = {"prediction", "ground_truth"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns in {source_csv}: {missing_list}")

    work = df.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["ground_truth"] = pd.to_numeric(work["ground_truth"], errors="coerce")
    work = work.dropna(subset=["prediction", "ground_truth"]).copy()
    if work.empty:
        raise ValueError(
            f"No valid numeric prediction/ground_truth rows found in: {source_csv}"
        )

    if "model_name" not in work.columns:
        work["model_name"] = model_name if model_name is not None else MODEL_NAME
    if "prompt_style" not in work.columns:
        work["prompt_style"] = prompt_style_name
    if "batch_size" not in work.columns:
        work["batch_size"] = pd.NA

    work["batch_size"] = pd.to_numeric(work["batch_size"], errors="coerce")

    if latest_per_setting_only and "run_timestamp_utc" in work.columns:
        work["_run_ts"] = pd.to_datetime(work["run_timestamp_utc"], errors="coerce", utc=True)
        grouped_latest = []
        for _, group in work.groupby(
            ["model_name", "prompt_style", "batch_size"], dropna=False
        ):
            valid_ts = group["_run_ts"].dropna()
            if valid_ts.empty:
                grouped_latest.append(group)
                continue
            latest_ts = valid_ts.max()
            grouped_latest.append(group[group["_run_ts"] == latest_ts])
        work = pd.concat(grouped_latest, ignore_index=True)

    summary_rows = []
    for (model_name_value, prompt_style_value, batch_size_value), group in work.groupby(
        ["model_name", "prompt_style", "batch_size"], dropna=False
    ):
        errors = group["ground_truth"] - group["prediction"]
        mean_rmse = float(((errors ** 2).mean()) ** 0.5)
        std_true = float(group["ground_truth"].std(ddof=0))
        mean_nrmse = float(mean_rmse / (std_true + 1e-8))

        model_short = str(model_name_value).split("/")[-1]
        method_name = f"LLM Batch Prompt ({model_short}, {prompt_style_value})"
        if not pd.isna(batch_size_value):
            method_name = f"{method_name} [batch_size={int(batch_size_value)}]"

        append_to_global_results(
            dataset=dataset,
            missingness_type=missingness_type,
            missing_rate=missing_rate,
            imputation_method=method_name,
            mean_rmse=mean_rmse,
            mean_nrmse=mean_nrmse,
        )

        summary_rows.append(
            {
                "source_csv": str(source_csv),
                "model_name": model_name_value,
                "prompt_style": prompt_style_value,
                "batch_size": batch_size_value,
                "n_predictions": int(len(group)),
                "mean_rmse": mean_rmse,
                "mean_nrmse": mean_nrmse,
                "imputation_method": method_name,
                "appended_to_rmse_csv": str(RESULTS_RMSE_PATH),
                "appended_to_nrmse_csv": str(RESULTS_NRMSE_PATH),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("APPENDED GLOBAL RESULTS (BATCH)")
    print("=" * 80)
    print(summary_df)
    return summary_df


def run_telco_batch_llmsimputation(
    n_examples: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_col_size: int = DEFAULT_BATCH_COL_SIZE,
) -> None:
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )

    feature_columns = [c for c in df_missing.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
    observed_rows = df_missing[df_missing[TARGET_COLUMN].notna()].copy()
    missing_rows = df_missing[df_missing[TARGET_COLUMN].isna()].copy().head(n_examples)

    if observed_rows.empty:
        raise ValueError("No observed rows available for background knowledge.")
    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{TARGET_COLUMN}'.")

    imputer = LLMBatchImputer(
        LLMBatchImputerConfig(
            model_name=MODEL_NAME,
            dataset_name=DATASET_NAME,
            target_column=TARGET_COLUMN,
            prompt_style_name=PROMPT_STYLE_NAME,
            domain_hints=[
                "In subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Prefer values that are consistent with observed feature distributions and correlations.",
            ],
        )
    )

    totalcharges_median = float(
        pd.to_numeric(observed_rows[TARGET_COLUMN], errors="coerce").dropna().median()
    )
    background_knowledge = imputer.build_background_knowledge(
        observed_df=observed_rows,
        feature_columns=feature_columns,
    )

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("\n" + "=" * 80)
    print("BATCH PROMPT IMPUTATION")
    print("=" * 80)
    print(
        f"Rows to impute: {len(missing_rows)} | Batch rows: {batch_size} | Batch cols: {batch_col_size}"
    )

    parsed_prediction_by_index: dict[int, float] = {}
    raw_output_by_index: dict[int, str] = {}

    n_rows = len(missing_rows)
    n_feature_cols = len(feature_columns)
    matrix_base = missing_rows[feature_columns + [TARGET_COLUMN]].copy()

    iter_batch = 0
    for row_start in range(0, n_rows, batch_size):
        row_end = min(row_start + batch_size, n_rows)
        actual_start = row_start
        if (row_end - row_start) < batch_size and n_rows >= batch_size:
            actual_start = row_end - batch_size

        rows_needed = row_end - row_start
        eval_indices = missing_rows.iloc[row_start:row_end].index.astype(int).tolist()

        for col_start in range(0, n_feature_cols, batch_col_size):
            col_end = min(col_start + batch_col_size, n_feature_cols)
            feature_slice = feature_columns[col_start:col_end]

            block_df = matrix_base.iloc[actual_start:row_end][feature_slice + [TARGET_COLUMN]].copy()
            block_df.insert(0, "row_index", missing_rows.iloc[actual_start:row_end].index.astype(int))
            expected_columns = block_df.columns.tolist()

            messages = imputer.build_batch_messages(
                batch_df=block_df,
                feature_columns=feature_slice,
                background_knowledge=background_knowledge,
            )
            raw_output = imputer.generate_raw_answer(
                messages=messages,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max(256, batch_size * 20),
            )
            parsed_block = imputer.parse_imputed_matrix(
                response_text=raw_output,
                expected_columns=expected_columns,
                target_column=TARGET_COLUMN,
            )
            clean_block = parsed_block.iloc[-rows_needed:, :].copy()

            iter_batch += 1
            parsed_count = int(clean_block[TARGET_COLUMN].notna().sum()) if not clean_block.empty else 0
            print(
                f"\nBatch {iter_batch}: rows {row_start}..{row_end - 1}, feature cols {col_start}..{col_end - 1}"
            )
            print(f"Parsed predictions in eval slice: {parsed_count}/{rows_needed}")

            if clean_block.empty or clean_block.shape[0] != rows_needed:
                continue

            target_values = clean_block[TARGET_COLUMN].reset_index(drop=True)
            for pos, idx in enumerate(eval_indices):
                if pos >= len(target_values):
                    break
                parsed_pred = target_values.iloc[pos]
                if pd.notna(parsed_pred):
                    parsed_prediction_by_index[idx] = float(parsed_pred)
                    raw_output_by_index[idx] = raw_output

    all_results: list[dict] = []

    for idx, row in missing_rows.iterrows():
        parsed_pred = parsed_prediction_by_index.get(int(idx), pd.NA)
        source = "model_batch"

        if pd.isna(parsed_pred):
            parsed_pred, source = imputer.fallback_totalcharges(
                row=row,
                totalcharges_median=totalcharges_median,
            )

        ground_truth = df_full.loc[idx, TARGET_COLUMN] if idx in df_full.index else pd.NA

        all_results.append(
            {
                "row_index": int(idx),
                "prediction": float(parsed_pred),
                "ground_truth": ground_truth,
                "source": source,
                "batch_id": pd.NA,
                "raw_output": raw_output_by_index.get(int(idx), ""),
            }
        )

    results_df = pd.DataFrame(all_results)
    results_df["model_name"] = MODEL_NAME
    results_df["prompt_style"] = PROMPT_STYLE_NAME
    results_df["batch_size"] = batch_size
    results_df["batch_col_size"] = batch_col_size
    results_df["n_examples_requested"] = n_examples
    results_df["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results_df["leakage_guard_enabled"] = True
    results_df["reference_source"] = "df_missing_observed_rows_background_only"

    model_token = model_name_to_file_token(MODEL_NAME)
    output_csv = DATA_RESULTS / f"telco_{model_token}_{PROMPT_STYLE_NAME}_results.csv"
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df[["row_index", "prediction", "ground_truth", "source", "batch_id"]])
    print(f"\nSaved results to: {output_csv}")

    append_global_results_from_telco_batch_predictions(
        results_csv=output_csv,
        model_name=MODEL_NAME,
        prompt_style_name=PROMPT_STYLE_NAME,
        latest_per_setting_only=True,
        dataset="Telco",
        missingness_type="MAR",
        missing_rate="missing totalCharges -> tenure, 10%",
    )


if __name__ == "__main__":
    run_telco_batch_llmsimputation(n_examples=100, batch_size=20)
