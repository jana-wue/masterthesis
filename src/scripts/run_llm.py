from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_PROCESSED


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def model_name_to_file_token(model_name: str) -> str:
    """
    Convert model name into a filesystem-safe token.
    """
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def extract_first_number(text: str) -> str:
    """
    Extract the first numeric value from model output.
    """
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)
    return text.strip()


def apply_chat_template(messages, tokenizer) -> str:
    """
    Use tokenizer chat template when available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return "\n\n".join(message["content"] for message in messages)


def generate_answer(messages, tokenizer, model) -> tuple[str, str]:
    prompt_text = apply_chat_template(messages, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    raw_generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer = extract_first_number(raw_generated)
    return answer, raw_generated


def run_telco_zero_shot_batch_preview(n_examples: int = 10, few_shot_k: int = 2) -> None:
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
            domain_hints=[
                "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Respect plausible values from the observed target distribution.",
            ],
            few_shot_k=few_shot_k,
        )
    )
    imputer.fit_target_stats(df_full)

    missing_rows = df_missing[df_missing[target_column].isna()].copy()
    reference_df = df_full.copy()

    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{target_column}'.")

    missing_rows = missing_rows.head(n_examples)

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

    print("\n" + "=" * 80)
    print("ZERO-SHOT / FEW-SHOT BATCH PREVIEW")
    print("=" * 80)

    results = []

    for idx, row in missing_rows.iterrows():
        few_shot_examples = imputer.select_few_shot_examples(row, reference_df)
        messages = imputer.build_inference_messages(row, few_shot_examples=few_shot_examples)
        prediction, raw_output = generate_answer(messages, tokenizer, model)
        ground_truth = df_full.loc[idx, target_column]

        print(f"\nRow index    : {idx}")
        print(f"Raw output   : {raw_output}")
        print(f"Prediction   : {prediction}")
        print(f"Ground truth : {ground_truth}")

        results.append(
            {
                "row_index": idx,
                "prediction": prediction,
                "raw_output": raw_output,
                "ground_truth": ground_truth,
                "n_fewshot_examples": len(few_shot_examples),
            }
        )

    results_df = pd.DataFrame(results)
    results_df["model_name"] = MODEL_NAME
    results_df["few_shot_k"] = few_shot_k
    results_df["n_examples_requested"] = n_examples
    results_df["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    model_token = model_name_to_file_token(MODEL_NAME)
    output_dir = DATA_PROCESSED / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"telco_{model_token}_results.csv"

    file_exists = output_csv.exists()
    results_df.to_csv(
        output_csv,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df)
    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    run_telco_zero_shot_batch_preview(n_examples=10, few_shot_k=2)
