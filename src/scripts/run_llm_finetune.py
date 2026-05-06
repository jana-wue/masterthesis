# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_PROCESSED
from src.paths import DATA_RAW, DATA_RESULTS


DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_INPUT_JSONL = DATA_PROCESSED / "llm" / "telco_totalcharges_mar_train.jsonl"
DEFAULT_OUTPUT_DIR = DATA_PROCESSED / "llm" / "mistral_telco_totalcharges_lora"
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_example(example, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def format_prompt_only(example, tokenizer) -> str:
    messages = example.get("messages", [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected training example ending with an assistant message.")

    return tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_function(example, tokenizer, max_length: int = 512) -> dict:
    full_ids = tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    prompt_ids = tokenizer(
        example["prompt_text"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # Prompt boundary = longest common token prefix between prompt-only and full sequence
    prompt_len = 0
    for full_token, prompt_token in zip(full_ids, prompt_ids):
        if full_token != prompt_token:
            break
        prompt_len += 1

    # Keep the sequence tail so complteion tokens are preserved
    if len(full_ids) > max_length:
        drop = len(full_ids) - max_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    attention_mask = [1] * len(full_ids)
    labels = full_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires pad_token_id for fixed-length batches.")

    pad_len = max_length - len(full_ids)
    if pad_len > 0:
        full_ids = full_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_token_count": sum(1 for label in labels if label != -100),
    }


def model_name_to_file_token(model_name: str) -> str:
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


def extract_first_number(text: str):
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


def apply_chat_template(messages, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_raw_answer(prompt_text: str, tokenizer, model, max_new_tokens: int = 16) -> str:
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
    raw_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw_generated_text


def fallback_totalcharges(row: pd.Series, totalcharges_median: float) -> tuple[float, str]:
    tenure = row.get("tenure")
    monthly = row.get("MonthlyCharges")
    if pd.notna(tenure) and pd.notna(monthly):
        try:
            return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
        except Exception:
            pass
    return float(totalcharges_median), "fallback_median"


def predict_numeric(row, imputer, tokenizer, model, totalcharges_median: float, max_new_tokens: int):
    messages = imputer.build_inference_messages(row)
    prompt_text = apply_chat_template(messages, tokenizer)
    raw_output_1 = generate_raw_answer(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    pred_1 = extract_first_number(raw_output_1)
    if pred_1 is not None:
        return pred_1, raw_output_1, "model_first_try"

    retry_messages = imputer.build_retry_messages(row)
    retry_prompt_text = apply_chat_template(retry_messages, tokenizer)
    raw_output_2 = generate_raw_answer(
        prompt_text=retry_prompt_text,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    pred_2 = extract_first_number(raw_output_2)
    if pred_2 is not None:
        return pred_2, raw_output_2, "model_retry"

    fallback_value, fallback_source = fallback_totalcharges(row, totalcharges_median)
    combined_raw = f"first_try={raw_output_1!r} | retry={raw_output_2!r}"
    return fallback_value, combined_raw, fallback_source


def run_post_train_evaluation(args, adapter_path: Path) -> dict:
    df_full = pd.read_csv(args.eval_full_csv)
    df_missing = pd.read_csv(args.eval_missing_csv)
    target_column = args.eval_target_column

    if target_column not in df_full.columns or target_column not in df_missing.columns:
        raise ValueError(f"Target column '{target_column}' must exist in full and missing datasets.")

    observed_target = pd.to_numeric(df_full[target_column], errors="coerce").dropna()
    if observed_target.empty:
        raise ValueError(f"No observed target values found in '{target_column}' for evaluation.")
    target_median = float(observed_target.median())

    feature_columns = [
        col for col in df_missing.columns if col not in [target_column, args.eval_id_column]
    ]
    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=str(adapter_path),
            target_column=target_column,
            feature_columns=feature_columns,
            domain_hints=args.eval_domain_hints,
        )
    )
    imputer.fit_target_stats(df_missing)

    missing_rows = df_missing[df_missing[target_column].isna()].copy()
    if missing_rows.empty:
        raise ValueError(f"No missing rows found in evaluation file for '{target_column}'.")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), **model_kwargs)
    model.eval()

    results = []
    for idx, row in missing_rows.iterrows():
        prediction, raw_output, source = predict_numeric(
            row=row,
            imputer=imputer,
            tokenizer=tokenizer,
            model=model,
            totalcharges_median=target_median,
            max_new_tokens=args.eval_max_new_tokens,
        )
        ground_truth = pd.to_numeric(df_full.loc[idx, target_column], errors="coerce")
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
        results.append(
            {
                "row_index": int(idx),
                "prediction": float(prediction),
                "ground_truth": ground_truth_value,
                "abs_error": abs_error,
                "sq_error": sq_error,
                "source": source,
                "raw_output": raw_output,
                "model_name": args.model_name,
                "adapter_path": str(adapter_path),
            }
        )

    results_df = pd.DataFrame(results)
    valid_eval = results_df.dropna(subset=["prediction", "ground_truth"]).copy()
    if valid_eval.empty:
        raise ValueError("No valid numeric prediction/ground_truth rows after evaluation.")

    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_token = model_name_to_file_token(args.model_name)
    adapter_token = model_name_to_file_token(adapter_path.name)
    detailed_csv = DATA_RESULTS / f"telco_{model_token}_{adapter_token}_finetuned_eval.csv"

    results_df["run_timestamp_utc"] = run_timestamp
    _append_prediction_rows(detailed_csv, results_df)

    errors = valid_eval["ground_truth"] - valid_eval["prediction"]
    mean_rmse = float(((errors ** 2).mean()) ** 0.5)
    std_true_full = float(observed_target.std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true_full + 1e-8))

    model_short = args.model_name.split("/")[-1]
    method_name = (
        args.global_method_name
        if args.global_method_name
        else f"LLM Finetuned ({model_short})"
    )

    if not args.skip_append_global_results:
        append_to_global_results(
            dataset=args.global_dataset,
            missingness_type=args.global_missingness_type,
            missing_rate=args.global_missing_rate,
            imputation_method=method_name,
            mean_rmse=mean_rmse,
            mean_nrmse=mean_nrmse,
        )

    summary = {
        "run_timestamp_utc": run_timestamp,
        "adapter_path": str(adapter_path),
        "n_predictions": int(len(valid_eval)),
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse,
        "method_name": method_name,
        "detailed_csv": str(detailed_csv),
        "appended_to_rmse_csv": str(RESULTS_RMSE_PATH) if not args.skip_append_global_results else None,
        "appended_to_nrmse_csv": str(RESULTS_NRMSE_PATH) if not args.skip_append_global_results else None,
    }

    summary_path = adapter_path / "post_train_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved post-train detailed eval to: {detailed_csv}")
    print(f"Saved post-train eval summary to: {summary_path}")
    if not args.skip_append_global_results:
        print(f"Appended to global result files: {RESULTS_RMSE_PATH}, {RESULTS_NRMSE_PATH}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_append_global_results", action="store_true")
    parser.add_argument(
        "--eval_full_csv",
        type=Path,
        default=DATA_RAW / "Telco-Customer-Churn_cleaned.csv",
    )
    parser.add_argument(
        "--eval_missing_csv",
        type=Path,
        default=DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv",
    )
    parser.add_argument("--eval_target_column", type=str, default="TotalCharges")
    parser.add_argument("--eval_id_column", type=str, default="customerID")
    parser.add_argument("--eval_max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--eval_domain_hints",
        nargs="*",
        default=[
            "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
            "Respect plausible values from the observed target distribution.",
        ],
    )
    parser.add_argument("--global_dataset", type=str, default="Telco")
    parser.add_argument("--global_missingness_type", type=str, default="MAR")
    parser.add_argument(
        "--global_missing_rate",
        type=str,
        default="missing totalCharges -> tenure, 10%",
    )
    parser.add_argument("--global_method_name", type=str, default=None)
    return parser.parse_args()


def build_lora_config(args) -> LoraConfig:
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # Paper-aligned: LoRA across all linear layers.
        "target_modules": "all-linear",
    }
    if "use_rslora" in inspect.signature(LoraConfig.__init__).parameters:
        kwargs["use_rslora"] = True
    return LoraConfig(**kwargs)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)

    records = load_jsonl(input_path)
    print(f"Loaded examples total: {len(records)}")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    if args.train_size is not None and args.train_size > len(records):
        raise ValueError(
            f"Requested train_size={args.train_size}, but only {len(records)} examples exist."
        )

    if args.train_size is not None:
        records = records[: args.train_size]
    print(f"Using training examples: {len(records)}")

    save_jsonl(output_dir / "train_subset_used.jsonl", records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_records = [
        {
            "text": format_example(r, tokenizer),
            "prompt_text": format_prompt_only(r, tokenizer),
        }
        for r in records
    ]
    dataset = Dataset.from_list(formatted_records)

    print("\n" + "=" * 80)
    print("FIRST FORMATTED EXAMPLE")
    print("=" * 80)
    print(dataset[0]["text"])

    print("\n" + "=" * 80)
    print("TOKENIZING DATA")
    print("=" * 80)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_length),
        remove_columns=["text", "prompt_text"],
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: x["loss_token_count"] > 0
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["loss_token_count"])

    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    peft_config = build_lora_config(args)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 80)
    print("START TRAINING")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    config_path = output_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "input_jsonl": str(input_path),
                "train_size": args.train_size,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "seed": args.seed,
                "n_examples_used": len(records),
            },
            f,
            indent=2,
        )

    print(f"Saved adapter/tokenizer to: {output_dir}")
    print(f"Saved run config to: {config_path}")

    if not args.skip_eval:
        print("\n" + "=" * 80)
        print("RUNNING POST-TRAIN EVALUATION")
        print("=" * 80)
        run_post_train_evaluation(args=args, adapter_path=output_dir)


if __name__ == "__main__":
    main()
