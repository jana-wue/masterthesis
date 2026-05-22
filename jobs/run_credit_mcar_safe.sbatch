#!/bin/bash
#SBATCH --job-name=credit_mcar_llm
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/credit_mcar_%j.out
#SBATCH --error=logs/credit_mcar_%j.err

set -euo pipefail
cd ~/work/masterthesis
export PYTHONPATH=~/work/masterthesis
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"

"$PYTHON_BIN" -c "import pandas,datasets,transformers,peft; print('Dependency check OK')"

"$PYTHON_BIN" src/scripts/prepare_llm_data.py \
  --mechanisms MCAR \
  --include_glob 'german_credit_mcar_billamt1_10pct.csv' \
  --target_mode explicit \
  --target_columns BILL_AMT1

"$PYTHON_BIN" src/scripts/run_llm_finetune.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --input_jsonl data/processed/llm/MCAR/german_credit_mcar_billamt1_10pct__target_BILL_AMT1_train.jsonl \
  --output_dir data/processed/llm/mistral_credit_billamt1_mcar_lora \
  --num_train_epochs 3 --learning_rate 5e-5 --batch_size 1 --grad_accum 1 \
  --max_length 512 --lora_r 256 --lora_alpha 8 --lora_dropout 0.0 \
  --eval_full_csv data/raw/default_of_credit_card_clients.csv \
  --eval_missing_csv data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv \
  --eval_target_column BILL_AMT1 --eval_id_column ID \
  --global_dataset 'German Credit Card' \
  --global_missingness_type MCAR \
  --global_missing_rate 'billamt1, 10%'
