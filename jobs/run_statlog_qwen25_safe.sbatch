#!/bin/bash
#SBATCH --job-name=statlog_qwen25
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
    INCLUDE_GLOB='german_statlog_numeric_mar_duration_creditamount_10pct.csv'
    INPUT_JSONL='data/processed/llm/MAR/german_statlog_numeric_mar_duration_creditamount_10pct__target_X2_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_statlog_x2_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv'
    ;;
  MCAR)
    INCLUDE_GLOB='german_statlog_numeric_mcar_duration_10pct.csv'
    INPUT_JSONL='data/processed/llm/MCAR/german_statlog_numeric_mcar_duration_10pct__target_X2_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_statlog_x2_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_statlog_numeric_mcar_duration_10pct.csv'
    ;;
  MNAR)
    INCLUDE_GLOB='german_statlog_numeric_mnar_duration_10pct.csv'
    INPUT_JSONL='data/processed/llm/MNAR/german_statlog_numeric_mnar_duration_10pct__target_X2_train.jsonl'
    OUT_DIR='data/processed/llm/qwen25_statlog_x2_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_statlog_numeric_mnar_duration_10pct.csv'
    ;;
  *) exit 1 ;;
esac

"$PYTHON_BIN" -c "import pandas,datasets,transformers,peft; print('Dependency check OK')"

"$PYTHON_BIN" src/scripts/prepare_llm_data.py \
  --mechanisms "$MECH" \
  --include_glob "$INCLUDE_GLOB" \
  --target_mode explicit \
  --target_columns X2

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
  --eval_full_csv data/raw/german_statlog_numeric.csv \
  --eval_missing_csv "$EVAL_MISSING" \
  --eval_target_column X2 \
  --eval_prompt_style current \
  --global_dataset 'German Statlog' \
  --global_missingness_type "$MECH" \
  --global_missing_rate 'X2, 10%'
