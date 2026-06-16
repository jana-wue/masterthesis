#!/bin/bash
set -euo pipefail

cd ~/work/masterthesis

export PYTHONPATH="${PYTHONPATH:-$HOME/work/masterthesis}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set before running this script}"

JOB_SCRIPT="jobs/run_llm_prompt_safe.sbatch"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu004,gpu017}"
FEW_SHOT_SETTINGS="${FEW_SHOT_SETTINGS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
MANIFEST_PATH="${MANIFEST_PATH:-scenario_manifest_llm_finetuning_10pct_final.csv}"
SKIP_EXISTING_FLAG="${SKIP_EXISTING_FLAG-}"
N_EXAMPLES="${N_EXAMPLES:-}"

submit_prompt_job() {
  local dataset_name="$1"
  local model_key="$2"
  local missingness_type="$3"

  local job_name="pr_${model_key}_$(echo "$dataset_name" | tr '[:upper:] ' '[:lower:]_' | tr -s '_')_${missingness_type,,}"
  local sbatch_args=(
    --job-name="$job_name"
    --exclude="$EXCLUDE_NODES"
    --cpus-per-task=4
    --mem=64G
    --time=04:00:00
    --export=ALL,HF_TOKEN="$HF_TOKEN",PYTHON_BIN="$PYTHON_BIN",DATASET_NAME="$dataset_name",MODEL_KEY="$model_key",MISSINGNESS_TYPE="$missingness_type",FEW_SHOT_SETTINGS="$FEW_SHOT_SETTINGS",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",MANIFEST_PATH="$MANIFEST_PATH",SKIP_EXISTING_FLAG="$SKIP_EXISTING_FLAG",N_EXAMPLES="$N_EXAMPLES"
  )

  local jid
  jid=$(sbatch "${sbatch_args[@]}" "$JOB_SCRIPT" | awk '{print $4}')
  echo "${dataset_name} | ${model_key} | ${missingness_type}: ${jid}"
}

for dataset_name in "German Statlog" "Telco" "German Credit Card"; do
  for model_key in llama31 mistral qwen25; do
    for missingness_type in MAR MCAR MNAR; do
      submit_prompt_job "$dataset_name" "$model_key" "$missingness_type"
    done
  done
done

echo
echo "Queued prompt reruns with FEW_SHOT_SETTINGS=${FEW_SHOT_SETTINGS}"
if [[ -n "$N_EXAMPLES" ]]; then
  echo "Row cap per run: ${N_EXAMPLES}"
else
  echo "Row cap per run: full masked split"
fi
