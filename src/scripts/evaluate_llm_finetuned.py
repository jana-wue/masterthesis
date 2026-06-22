# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_PROCESSED


ADAPTER_PATH = DATA_PROCESSED / "llm" / "mistral_telco_totalcharges_lora"
OUTPUT_CSV = DATA_PROCESSED / "llm" / "eval_mistral_telco_totalcharges_lora.csv"


def extract_first_number(text: str | None) -> float | None:
    """
    Returns float if a number can be extracted, else None.
    """
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None

    try:
        value = float(match.group(0))
        if math.isfinite(value):
            return value
    except ValueError:
        pass

    return None


def apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Apply the chat template to the message list."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_raw_answer(
    prompt_text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_new_tokens: int = 16,
) -> str:
    """Generate raw answer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
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
    raw_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw_generated_text


def fallback_totalcharges(row: pd.Series, totalcharges_median: float) -> tuple[float, str]:
    """
    Deterministic fallback so that we ALWAYS return a prediction.
    Priority:
    1) tenure * MonthlyCharges
    2) median TotalCharges
    """
    tenure = row.get("tenure")
    monthly = row.get("MonthlyCharges")

    if pd.notna(tenure) and pd.notna(monthly):
        try:
            return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
        except Exception:
            pass

    return float(totalcharges_median), "fallback_median"


def predict_numeric(
    row: pd.Series,
    imputer: LLMImputer,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    totalcharges_median: float,
) -> tuple[float, str, str]:
    """
    Multi-stage prediction:
    1) normal inference with chat template
    2) stricter retry prompt
    3) deterministic fallback
    """
    # First try
    messages = imputer.build_inference_messages(row)
    prompt_text = apply_chat_template(messages, tokenizer)
    raw_output_1 = generate_raw_answer(prompt_text, tokenizer, model, max_new_tokens=16)
    pred_1 = extract_first_number(raw_output_1)

    if pred_1 is not None:
        return pred_1, raw_output_1, "model_first_try"

    # Retry
    retry_messages = imputer.build_retry_messages(row)
    retry_prompt_text = apply_chat_template(retry_messages, tokenizer)
    raw_output_2 = generate_raw_answer(retry_prompt_text, tokenizer, model, max_new_tokens=16)
    pred_2 = extract_first_number(raw_output_2)

    if pred_2 is not None:
        return pred_2, raw_output_2, "model_retry"

    # Final fallback
    fallback_value, fallback_source = fallback_totalcharges(row, totalcharges_median)
    combined_raw = f"first_try={raw_output_1!r} | retry={raw_output_2!r}"
    return fallback_value, combined_raw, fallback_source


def main() -> None:
    """Run the script entry point."""
    target_column = "TotalCharges"

    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )

    observed_target = df_full[target_column].dropna()
    totalcharges_median = float(observed_target.median())

    feature_columns = [
        col for col in df_missing.columns if col not in [target_column, "customerID"]
    ]

    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=str(ADAPTER_PATH),
            target_column=target_column,
            feature_columns=feature_columns,
            domain_hints=[
                "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Respect plausible values from the observed target distribution.",
            ],
        )
    )
    imputer.fit_target_stats(df_missing)

    missing_rows = df_missing[df_missing[target_column].isna()].copy()

    print("=" * 80)
    print("LOADING FINETUNED MODEL")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("\n" + "=" * 80)
    print("RUNNING FINETUNED EVALUATION")
    print("=" * 80)

    results = []

    for idx, row in missing_rows.iterrows():
        prediction, raw_output, source = predict_numeric(
            row=row,
            imputer=imputer,
            tokenizer=tokenizer,
            model=model,
            totalcharges_median=totalcharges_median,
        )
        ground_truth = float(df_full.loc[idx, target_column])

        abs_error = abs(prediction - ground_truth)

        print(f"\nRow index    : {idx}")
        print(f"Raw output   : {raw_output}")
        print(f"Prediction   : {prediction}")
        print(f"Source       : {source}")
        print(f"Ground truth : {ground_truth}")
        print(f"Abs error    : {abs_error:.4f}")

        results.append(
            {
                "row_index": idx,
                "raw_output": raw_output,
                "prediction": prediction,
                "source": source,
                "ground_truth": ground_truth,
                "abs_error": abs_error,
            }
        )

    results_df = pd.DataFrame(results)
    rmse = float((((results_df["prediction"] - results_df["ground_truth"]) ** 2).mean()) ** 0.5)

    print("\n" + "=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)
    print(f"Number of predictions : {len(results_df)}")
    print(f"RMSE                  : {rmse:.6f}")
    print("\nSource counts:")
    print(results_df["source"].value_counts())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    summary_path = OUTPUT_CSV.with_suffix(".json")
    summary = {
        "adapter_path": str(ADAPTER_PATH),
        "n_predictions": int(len(results_df)),
        "rmse": float(rmse),
        "source_counts": results_df["source"].value_counts().to_dict(),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {OUTPUT_CSV}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
