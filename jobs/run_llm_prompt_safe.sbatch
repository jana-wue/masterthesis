#!/bin/bash
#SBATCH --job-name=llm_prompt
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

DATASET_NAME="${DATASET_NAME:?DATASET_NAME not set}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY not set}"
MISSINGNESS_TYPE="${MISSINGNESS_TYPE:?MISSINGNESS_TYPE not set}"
FEW_SHOT_SETTINGS="${FEW_SHOT_SETTINGS:-0 2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
MANIFEST_PATH="${MANIFEST_PATH:-scenario_manifest_llm_finetuning_10pct_final.csv}"
SKIP_EXISTING_FLAG="${SKIP_EXISTING_FLAG:---skip-existing}"
N_EXAMPLES="${N_EXAMPLES:-}"

"$PYTHON_BIN" -c "import pandas, torch, transformers; print('Dependency check OK')"

CMD=(
  "$PYTHON_BIN"
  src/scripts/run_llm.py
  --manifest "$MANIFEST_PATH"
  --datasets "$DATASET_NAME"
  --model-keys "$MODEL_KEY"
  --missingness-types "$MISSINGNESS_TYPE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

for k in $FEW_SHOT_SETTINGS; do
  CMD+=(--few-shot-settings "$k")
done

if [[ -n "$N_EXAMPLES" ]]; then
  CMD+=(--n-examples "$N_EXAMPLES")
fi

if [[ -n "$SKIP_EXISTING_FLAG" ]]; then
  CMD+=("$SKIP_EXISTING_FLAG")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
