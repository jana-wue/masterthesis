# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.paths import DATA_PROCESSED


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def format_example(example, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_function(example, tokenizer, max_length: int = 512) -> dict:
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def parse_train_size() -> int:
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    return 500


def main() -> None:
    train_size = parse_train_size()

    input_path = DATA_PROCESSED / "llm" / "telco_totalcharges_mar_train.jsonl"
    output_dir = DATA_PROCESSED / "llm" / f"mistral_telco_totalcharges_lora_{train_size}"

    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)

    records = load_jsonl(input_path)
    print(f"Loaded examples total: {len(records)}")

    records = records[:train_size]
    print(f"Using training examples: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_records = [{"text": format_example(r, tokenizer)} for r in records]
    dataset = Dataset.from_list(formatted_records)

    print("\n" + "=" * 80)
    print("FIRST FORMATTED EXAMPLE")
    print("=" * 80)
    print(dataset[0]["text"])

    print("\n" + "=" * 80)
    print("TOKENIZING DATA")
    print("=" * 80)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=512),
        remove_columns=["text"],
    )

    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
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

    print(f"Saved adapter/tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()