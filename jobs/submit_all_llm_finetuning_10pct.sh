#!/bin/bash
set -euo pipefail

cd ~/work/masterthesis

export PYTHONPATH="${PYTHONPATH:-$HOME/work/masterthesis}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set before running this script}"

submit_triplet() {
  local base_name="$1"
  local job_script="$2"

  local jid_mar
  local jid_mcar
  local jid_mnar

  jid_mar=$(sbatch --job-name="${base_name}_mar" --exclude=gpu004,gpu017 --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",MECH=MAR "$job_script" | awk '{print $4}')
  jid_mcar=$(sbatch --job-name="${base_name}_mcar" --exclude=gpu004,gpu017 --dependency=afterok:${jid_mar} --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",MECH=MCAR "$job_script" | awk '{print $4}')
  jid_mnar=$(sbatch --job-name="${base_name}_mnar" --exclude=gpu004,gpu017 --dependency=afterok:${jid_mcar} --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",MECH=MNAR "$job_script" | awk '{print $4}')

  echo "${base_name}: MAR=${jid_mar} MCAR=${jid_mcar} MNAR=${jid_mnar}"
}

submit_triplet "telco_qwen25" jobs/run_telco_qwen25_safe.sbatch
submit_triplet "telco_llama31" jobs/run_telco_llama31_safe.sbatch
submit_triplet "telco_mistral" jobs/run_telco_mistral_safe.sbatch
submit_triplet "statlog_mistral" jobs/run_statlog_x2_safe.sbatch
submit_triplet "statlog_qwen25" jobs/run_statlog_qwen25_safe.sbatch
submit_triplet "statlog_llama31" jobs/run_statlog_llama31_safe.sbatch
submit_triplet "credit_mistral" jobs/run_credit_mistral_safe.sbatch
submit_triplet "credit_qwen25" jobs/run_credit_qwen25_safe.sbatch
submit_triplet "credit_llama31" jobs/run_credit_llama31_safe.sbatch
