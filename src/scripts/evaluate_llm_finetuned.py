# -*- coding: utf-8 -*-
from __future__ import annotations

import re

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_PROCESSED


ADAPTER_PATH = DATA_PROCESSED / "llm" / "mistral_telco_totalcharges_lora"


def extract_first_number(text: str) -> str:
    """
    Extract the first numeric value from text.
    Return NO_PREDICTION if no number can be found.
    """
    if text is None:
        return "NO_PREDICTION"

    text = text.strip()

    if not text:
        return "NO_PREDICTION"

    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)

    return "NO_PREDICTION"


def generate_answer(prompt: str, tokenizer, model) -> tuple[str, str]:
    """
    Returns:
        parsed_prediction, raw_generated_text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    raw_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed_prediction = extract_first_number(raw_generated_text)

    return parsed_prediction, raw_generated_text


def main() -> None:
    target_column = "TotalCharges"

    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )

    feature_columns = [
        col for col in df_missing.columns if col not in [target_column, "customerID"]
    ]

    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=str(ADAPTER_PATH),
            target_column=target_column,
            feature_columns=feature_columns,
        )
    )

    missing_rows = df_missing[df_missing[target_column].isna()].copy().head(10)

    print("=" * 80)
    print("LOADING FINETUNED MODEL")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\n" + "=" * 80)
    print("FINETUNED MODEL PREDICTIONS")
    print("=" * 80)

    results = []

    for idx, row in missing_rows.iterrows():
        prompt = imputer.build_prompt(row)
        prediction, raw_output = generate_answer(prompt, tokenizer, model)
        ground_truth = df_full.loc[idx, target_column]

        print(f"\nRow index    : {idx}")
        print(f"Raw output   : {raw_output}")
        print(f"Prediction   : {prediction}")
        print(f"Ground truth : {ground_truth}")

        results.append(
            {
                "row_index": idx,
                "raw_output": raw_output,
                "prediction": prediction,
                "ground_truth": ground_truth,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df)


if __name__ == "__main__":
    main()