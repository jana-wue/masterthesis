from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm_batch import (
    LLMBatchImputer,
    LLMBatchImputerConfig,
    model_name_to_file_token,
)
from src.paths import DATA_PROCESSED, DATA_RAW


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT_STYLE_NAME = "BatchImputation"
DATASET_NAME = "Telco Customer Churn"
TARGET_COLUMN = "TotalCharges"
ID_COLUMN = "customerID"
DEFAULT_BATCH_SIZE = 20


def run_telco_batch_llmsimputation(
    n_examples: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
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
    print(f"Rows to impute: {len(missing_rows)} | Batch size: {batch_size}")

    all_results: list[dict] = []

    for batch_id, start in enumerate(range(0, len(missing_rows), batch_size), start=1):
        batch_rows = missing_rows.iloc[start : start + batch_size].copy()
        batch_prompt_df = batch_rows[feature_columns + [TARGET_COLUMN]].copy()
        batch_prompt_df.insert(0, "row_index", batch_rows.index.astype(int))

        messages = imputer.build_batch_messages(
            batch_df=batch_prompt_df,
            feature_columns=feature_columns,
            background_knowledge=background_knowledge,
        )
        raw_output = imputer.generate_raw_answer(
            messages=messages,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max(256, batch_size * 20),
        )
        parsed = imputer.parse_batch_predictions(raw_output)
        parsed_map = dict(zip(parsed["row_index"], parsed[TARGET_COLUMN]))

        print(f"\nBatch {batch_id}: rows {start}..{start + len(batch_rows) - 1}")
        print(f"Parsed predictions: {len(parsed_map)}/{len(batch_rows)}")

        for idx, row in batch_rows.iterrows():
            parsed_pred = parsed_map.get(int(idx))
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
                    "batch_id": batch_id,
                    "raw_output": raw_output,
                }
            )

    results_df = pd.DataFrame(all_results)
    results_df["model_name"] = MODEL_NAME
    results_df["prompt_style"] = PROMPT_STYLE_NAME
    results_df["batch_size"] = batch_size
    results_df["n_examples_requested"] = n_examples
    results_df["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results_df["leakage_guard_enabled"] = True
    results_df["reference_source"] = "df_missing_observed_rows_background_only"

    model_token = model_name_to_file_token(MODEL_NAME)
    output_dir = DATA_PROCESSED / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"telco_{model_token}_{PROMPT_STYLE_NAME}_results.csv"
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df[["row_index", "prediction", "ground_truth", "source", "batch_id"]])
    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    run_telco_batch_llmsimputation(n_examples=100, batch_size=20)
