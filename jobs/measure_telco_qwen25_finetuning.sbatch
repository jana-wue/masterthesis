#!/bin/bash
# Measure the GPU energy demand of Telco / Qwen 2.5 LoRA fine-tuning.

# Example: sbatch --array=1-3 jobs/measure_telco_qwen25_finetuning.sbatch
#SBATCH --job-name=energy_telco_qwen25
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd ~/work/masterthesis
export PYTHONPATH=~/work/masterthesis
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"

# All repetitions use the benchmark seed.
SEED=42
RUN_LABEL="run_${SLURM_ARRAY_TASK_ID:-1}"
RESULT_DIR="data/results/energy/telco_qwen25_mcar_10pct/${RUN_LABEL}"
OUTPUT_DIR="data/processed/llm/energy_measurements/qwen25_telco_mcar_10pct_${RUN_LABEL}"
INPUT_JSONL="data/processed/llm/MCAR/telco_customer_churn_mcar_totalcharges_10pct__target_TotalCharges_train.jsonl"

mkdir -p "$RESULT_DIR"

"$PYTHON_BIN" src/scripts/prepare_llm_data.py \
  --mechanisms MCAR \
  --include_glob 'telco_customer_churn_mcar_totalcharges_10pct.csv' \
  --target_mode explicit \
  --target_columns TotalCharges

# Sample GPU power once per second
nvidia-smi \
  --query-gpu=timestamp,power.draw,utilization.gpu,memory.used \
  --format=csv,noheader,nounits \
  --loop=1 > "$RESULT_DIR/gpu_samples.csv" &
GPU_MONITOR_PID=$!

stop_gpu_monitor() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap stop_gpu_monitor EXIT

# /usr/bin/time captures elapsed time plus CPU and RAM statistics.
/usr/bin/time -v -o "$RESULT_DIR/time_verbose.txt" \
  "$PYTHON_BIN" src/scripts/run_llm_finetune.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --batch_size 1 \
    --grad_accum 1 \
    --max_length 512 \
    --lora_r 256 \
    --lora_alpha 8 \
    --lora_dropout 0.0 \
    --seed "$SEED" \
    --skip_eval \
    --skip_append_global_results

stop_gpu_monitor
trap - EXIT

"$PYTHON_BIN" src/scripts/summarize_gpu_energy.py \
  --samples_csv "$RESULT_DIR/gpu_samples.csv" \
  --method "Qwen 2.5 Fine-Tuning" \
  --run_label "$RUN_LABEL" \
  --output_csv "$RESULT_DIR/summary.csv"
