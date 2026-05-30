#!/bin/bash
#SBATCH --job-name=telco_qwen25
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
    INCLUDE_GLOB='telco_customer_churn_mar_totalcharges_tenure_10pct.csv'
    INPUT_JSONL='data/processed/llm/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct__target_TotalCharges_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_telco_totalcharges_mar_lora'
    EVAL_MISSING='data/processed/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv'
    GLOBAL_RATE='missing totalCharges -> tenure, 10%'
    ;;
  MCAR)
    INCLUDE_GLOB='telco_customer_churn_mcar_totalcharges_10pct.csv'
    INPUT_JSONL='data/processed/llm/MCAR/telco_customer_churn_mcar_totalcharges_10pct__target_TotalCharges_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_telco_totalcharges_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  MNAR)
    INCLUDE_GLOB='telco_customer_churn_mnar_totalcharges_10pct.csv'
    INPUT_JSONL='data/processed/llm/MNAR/telco_customer_churn_mnar_totalcharges_10pct__target_TotalCharges_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_telco_totalcharges_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  *) exit 1 ;;
esac

"$PYTHON_BIN" -c "import pandas,datasets,transformers,peft; print('Dependency check OK')"

"$PYTHON_BIN" src/scripts/prepare_llm_data.py \
  --mechanisms "$MECH" \
  --include_glob "$INCLUDE_GLOB" \
  --target_mode explicit \
  --target_columns TotalCharges

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
  --eval_full_csv data/raw/Telco-Customer-Churn_cleaned.csv \
  --eval_missing_csv "$EVAL_MISSING" \
  --eval_target_column TotalCharges \
  --eval_id_column customerID \
  --eval_prompt_style current \
  --global_dataset 'Telco' \
  --global_missingness_type "$MECH" \
  --global_missing_rate "$GLOBAL_RATE"
