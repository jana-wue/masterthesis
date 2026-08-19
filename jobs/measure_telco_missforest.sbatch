#!/bin/bash
# Measure the CPU-side resource demand of the best classical Telco baseline.

# Example: sbatch --array=0-2 jobs/measure_telco_missforest.sbatch
#SBATCH --job-name=energy_telco_missforest
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd ~/work/masterthesis
export PYTHONPATH=~/work/masterthesis
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis/bin/python}"

SEEDS=(42 202 303 404 505)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
RUN_LABEL="seed_${SEED}"
RESULT_DIR="data/results/energy/telco_missforest_mcar_10pct/${RUN_LABEL}"
mkdir -p "$RESULT_DIR"

# The classical runner measures its own wall-clock runtime, /usr/bin/ time adds peak RAM and CPU statistics
/usr/bin/time -v -o "$RESULT_DIR/time_verbose.txt" \
  "$PYTHON_BIN" src/scripts/run_classical_manifest.py \
    --dataset_keys telco \
    --method_keys missforest \
    --scenario_families MCAR_TARGET \
    --rates 10 \
    --seeds "$SEED" \
    --output_csv "$RESULT_DIR/classical_result.csv" \
    --fail_fast
