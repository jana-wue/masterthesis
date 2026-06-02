#!/bin/bash
#SBATCH --job-name=credit_qwen25
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd ~/work/masterthesis
export PYTHONPATH=~/work/masterthesis
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"
MECH="${MECH:?MECH not set}"

case "$MECH" in
  MAR)
    INCLUDE_GLOB='german_credit_mar_billamt1_pay0_10pct.csv'
    INPUT_JSONL='data/processed/llm/MAR/german_credit_mar_billamt1_pay0_10pct__target_BILL_AMT1_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_credit_billamt1_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_credit_mar_billamt1_pay0_10pct.csv'
    GLOBAL_RATE='billamt1 -> pay0, 10%'
    ;;
  MCAR)
    INCLUDE_GLOB='german_credit_mcar_billamt1_10pct.csv'
    INPUT_JSONL='data/processed/llm/MCAR/german_credit_mcar_billamt1_10pct__target_BILL_AMT1_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_credit_billamt1_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  MNAR)
    INCLUDE_GLOB='german_credit_mnar_billamt1_10pct.csv'
    INPUT_JSONL='data/processed/llm/MNAR/german_credit_mnar_billamt1_10pct__target_BILL_AMT1_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_credit_billamt1_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_credit_mnar_billamt1_10pct.csv'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  *) exit 1 ;;
esac

"$PYTHON_BIN" -c "import pandas,datasets,transformers,peft; print('Dependency check OK')"

"$PYTHON_BIN" src/scripts/prepare_llm_data.py \
  --mechanisms "$MECH" \
  --include_glob "$INCLUDE_GLOB" \
  --target_mode explicit \
  --target_columns BILL_AMT1

"$PYTHON_BIN" src/scripts/run_llm_finetune.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --input_jsonl "$INPUT_JSONL" \
  --output_dir "$OUT_DIR" \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --batch_size 1 \
  --grad_accum 1 \
  --max_length 512 \
  --lora_r 256 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --eval_full_csv data/raw/default_of_credit_card_clients.csv \
  --eval_missing_csv "$EVAL_MISSING" \
  --eval_target_column BILL_AMT1 \
  --eval_id_column ID \
  --eval_prompt_style current \
  --global_dataset 'German Credit Card' \
  --global_missingness_type "$MECH" \
  --global_missing_rate "$GLOBAL_RATE"
