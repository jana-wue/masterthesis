from __future__ import annotations

import json

import pandas as pd

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_PROCESSED


def save_jsonl(records: list[dict], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_telco_totalcharges_mar() -> None:
    target_column = "TotalCharges"

    input_path = DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    output_path = DATA_PROCESSED / "llm" / "telco_totalcharges_mar_train.jsonl"

    df = pd.read_csv(input_path)

    feature_columns = [
        col for col in df.columns if col not in [target_column, "customerID"]
    ]

    imputer = LLMImputer(
        LLMImputerConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            target_column=target_column,
            feature_columns=feature_columns,
        )
    )

    train_examples = imputer.build_training_dataset(df)

    save_jsonl(train_examples, output_path)

    print("=" * 80)
    print("LLM TRAINING DATA PREP FINISHED")
    print("=" * 80)
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Number of training examples: {len(train_examples)}")

    if train_examples:
        print("\nFirst example:")
        print(json.dumps(train_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare_telco_totalcharges_mar()
