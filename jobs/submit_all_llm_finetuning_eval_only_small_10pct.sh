#!/bin/bash
set -euo pipefail

cd ~/work/masterthesis

export PYTHONPATH="${PYTHONPATH:-$HOME/work/masterthesis}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set before running this script}"

JOB_SCRIPT="jobs/run_llm_finetune_eval_only_safe.sbatch"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu004,gpu017}"

submit_eval_job() {
  local dataset="$1"
  local model_key="$2"
  local mech="$3"
  local base_name="$4"

  local jid
  jid=$(sbatch \
    --job-name="${base_name}_${mech,,}" \
    --exclude="$EXCLUDE_NODES" \
    --cpus-per-task=4 \
    --mem=64G \
    --time=03:00:00 \
    --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",DATASET="$dataset",MODEL_KEY="$model_key",MECH="$mech" \
    "$JOB_SCRIPT" | awk '{print $4}')

  echo "${dataset} | ${model_key} | ${mech}: ${jid}"
}

for mech in MAR MCAR MNAR; do
  submit_eval_job "Telco" "qwen25" "$mech" "fte_telco_qwen25"
  submit_eval_job "Telco" "llama31" "$mech" "fte_telco_llama31"
  submit_eval_job "Telco" "mistral" "$mech" "fte_telco_mistral"
  submit_eval_job "German Statlog" "mistral" "$mech" "fte_statlog_mistral"
  submit_eval_job "German Statlog" "qwen25" "$mech" "fte_statlog_qwen25"
  submit_eval_job "German Statlog" "llama31" "$mech" "fte_statlog_llama31"
  submit_eval_job "German Credit Card" "mistral" "$mech" "fte_credit_mistral"
  submit_eval_job "German Credit Card" "qwen25" "$mech" "fte_credit_qwen25"
  submit_eval_job "German Credit Card" "llama31" "$mech" "fte_credit_llama31"
done

echo
echo "Queued eval-only reruns with reduced resources."
