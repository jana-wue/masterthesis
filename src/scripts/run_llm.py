from __future__ import annotations

import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_PROCESSED


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def extract_first_number(text: str) -> str:
    """
    Extract the first numeric value from model output.
    """
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)
    return text.strip()


def generate_answer(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded[len(prompt):].strip()
    answer = extract_first_number(answer)
    return answer


def run_telco_zero_shot_batch_preview(n_examples: int = 10) -> None:
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
            model_name=MODEL_NAME,
            target_column=target_column,
            feature_columns=feature_columns,
        )
    )

    missing_rows = df_missing[df_missing[target_column].isna()].copy()

    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{target_column}'.")

    missing_rows = missing_rows.head(n_examples)

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\n" + "=" * 80)
    print("ZERO-SHOT BATCH PREVIEW")
    print("=" * 80)

    results = []

    for idx, row in missing_rows.iterrows():
        prompt = imputer.build_prompt(row)
        prediction = generate_answer(prompt, tokenizer, model)
        ground_truth = df_full.loc[idx, target_column]

        print(f"\nRow index   : {idx}")
        print(f"Prediction  : {prediction}")
        print(f"Ground truth: {ground_truth}")

        results.append(
            {
                "row_index": idx,
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
    run_telco_zero_shot_batch_preview(n_examples=10)