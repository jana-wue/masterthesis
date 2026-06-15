#!/bin/bash
set -euo pipefail

cd ~/work/masterthesis

export PYTHONPATH="${PYTHONPATH:-$HOME/work/masterthesis}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set before running this script}"

JOB_SCRIPT="jobs/run_llm_finetune_eval_only_safe.sbatch"

submit_triplet() {
  local dataset="$1"
  local model_key="$2"
  local base_name="$3"

  local jid_mar
  local jid_mcar
  local jid_mnar

  jid_mar=$(sbatch --job-name="${base_name}_eval_mar" --exclude=gpu004,gpu017 --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",DATASET="$dataset",MODEL_KEY="$model_key",MECH=MAR "$JOB_SCRIPT" | awk '{print $4}')
  jid_mcar=$(sbatch --job-name="${base_name}_eval_mcar" --exclude=gpu004,gpu017 --dependency=afterok:${jid_mar} --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",DATASET="$dataset",MODEL_KEY="$model_key",MECH=MCAR "$JOB_SCRIPT" | awk '{print $4}')
  jid_mnar=$(sbatch --job-name="${base_name}_eval_mnar" --exclude=gpu004,gpu017 --dependency=afterok:${jid_mcar} --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",DATASET="$dataset",MODEL_KEY="$model_key",MECH=MNAR "$JOB_SCRIPT" | awk '{print $4}')

  echo "${base_name}: MAR=${jid_mar} MCAR=${jid_mcar} MNAR=${jid_mnar}"
}

submit_triplet "Telco" "qwen25" "telco_qwen25"
submit_triplet "Telco" "llama31" "telco_llama31"
submit_triplet "Telco" "mistral" "telco_mistral"
submit_triplet "German Statlog" "mistral" "statlog_mistral"
submit_triplet "German Statlog" "qwen25" "statlog_qwen25"
submit_triplet "German Statlog" "llama31" "statlog_llama31"
submit_triplet "German Credit Card" "mistral" "credit_mistral"
submit_triplet "German Credit Card" "qwen25" "credit_qwen25"
submit_triplet "German Credit Card" "llama31" "credit_llama31"
