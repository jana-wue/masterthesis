#!/bin/bash
#SBATCH --job-name=llm_ft_eval
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/work/masterthesis
export PYTHONPATH="${PYTHONPATH:-$HOME/work/masterthesis}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/masterthesis-llm/bin/python}"

DATASET="${DATASET:?DATASET not set}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY not set}"
MECH="${MECH:?MECH not set}"

case "$MODEL_KEY" in
  mistral)
    MODEL_NAME='mistralai/Mistral-7B-Instruct-v0.3'
    ;;
  qwen25)
    MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'
    ;;
  llama31)
    MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
    ;;
  *)
    echo "Unsupported MODEL_KEY=${MODEL_KEY}"
    exit 1
    ;;
esac

case "${DATASET}:${MODEL_KEY}:${MECH}" in
  Telco:mistral:MAR)
    if [[ -d 'data/processed/llm/mistral_telco_totalcharges_mar_lora' ]]; then
      ADAPTER_PATH='data/processed/llm/mistral_telco_totalcharges_mar_lora'
    else
      ADAPTER_PATH='data/processed/llm/mistral_telco_totalcharges_lora'
    fi
    EVAL_MISSING='data/processed/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing totalCharges -> tenure, 10%'
    ;;
  Telco:mistral:MCAR)
    ADAPTER_PATH='data/processed/llm/mistral_telco_totalcharges_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  Telco:mistral:MNAR)
    ADAPTER_PATH='data/processed/llm/mistral_telco_totalcharges_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  Telco:qwen25:MAR)
    ADAPTER_PATH='data/processed/llm/qwen25_telco_totalcharges_mar_lora'
    EVAL_MISSING='data/processed/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing totalCharges -> tenure, 10%'
    ;;
  Telco:qwen25:MCAR)
    ADAPTER_PATH='data/processed/llm/qwen25_telco_totalcharges_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  Telco:qwen25:MNAR)
    ADAPTER_PATH='data/processed/llm/qwen25_telco_totalcharges_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  Telco:llama31:MAR)
    ADAPTER_PATH='data/processed/llm/llama31_telco_totalcharges_mar_lora'
    EVAL_MISSING='data/processed/MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing totalCharges -> tenure, 10%'
    ;;
  Telco:llama31:MCAR)
    ADAPTER_PATH='data/processed/llm/llama31_telco_totalcharges_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  Telco:llama31:MNAR)
    ADAPTER_PATH='data/processed/llm/llama31_telco_totalcharges_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv'
    TARGET_COLUMN='TotalCharges'
    ID_COLUMN='customerID'
    EVAL_FULL='data/raw/Telco-Customer-Churn_cleaned.csv'
    GLOBAL_DATASET='Telco'
    GLOBAL_RATE='missing TotalCharges, 10%'
    ;;
  German\ Statlog:mistral:MAR)
    ADAPTER_PATH='data/processed/llm/mistral_statlog_x2_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:mistral:MCAR)
    ADAPTER_PATH='data/processed/llm/mistral_statlog_x2_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_statlog_numeric_mcar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:mistral:MNAR)
    ADAPTER_PATH='data/processed/llm/mistral_statlog_x2_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_statlog_numeric_mnar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:qwen25:MAR)
    ADAPTER_PATH='data/processed/llm/qwen25_statlog_x2_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:qwen25:MCAR)
    ADAPTER_PATH='data/processed/llm/qwen25_statlog_x2_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_statlog_numeric_mcar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:qwen25:MNAR)
    ADAPTER_PATH='data/processed/llm/qwen25_statlog_x2_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_statlog_numeric_mnar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:llama31:MAR)
    ADAPTER_PATH='data/processed/llm/llama31_statlog_x2_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:llama31:MCAR)
    ADAPTER_PATH='data/processed/llm/llama31_statlog_x2_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_statlog_numeric_mcar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Statlog:llama31:MNAR)
    ADAPTER_PATH='data/processed/llm/llama31_statlog_x2_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_statlog_numeric_mnar_duration_10pct.csv'
    TARGET_COLUMN='X2'
    ID_COLUMN=''
    EVAL_FULL='data/raw/german_statlog_numeric.csv'
    GLOBAL_DATASET='German Statlog'
    GLOBAL_RATE='X2, 10%'
    ;;
  German\ Credit\ Card:mistral:MAR)
    ADAPTER_PATH='data/processed/llm/mistral_credit_billamt1_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_credit_mar_billamt1_pay0_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1 -> pay0, 10%'
    ;;
  German\ Credit\ Card:mistral:MCAR)
    ADAPTER_PATH='data/processed/llm/mistral_credit_billamt1_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  German\ Credit\ Card:mistral:MNAR)
    ADAPTER_PATH='data/processed/llm/mistral_credit_billamt1_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_credit_mnar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  German\ Credit\ Card:qwen25:MAR)
    ADAPTER_PATH='data/processed/llm/qwen25_credit_billamt1_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_credit_mar_billamt1_pay0_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1 -> pay0, 10%'
    ;;
  German\ Credit\ Card:qwen25:MCAR)
    ADAPTER_PATH='data/processed/llm/qwen25_credit_billamt1_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  German\ Credit\ Card:qwen25:MNAR)
    ADAPTER_PATH='data/processed/llm/qwen25_credit_billamt1_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_credit_mnar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  German\ Credit\ Card:llama31:MAR)
    ADAPTER_PATH='data/processed/llm/llama31_credit_billamt1_mar_lora'
    EVAL_MISSING='data/processed/MAR/german_credit_mar_billamt1_pay0_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1 -> pay0, 10%'
    ;;
  German\ Credit\ Card:llama31:MCAR)
    ADAPTER_PATH='data/processed/llm/llama31_credit_billamt1_mcar_lora'
    EVAL_MISSING='data/processed/MCAR/german_credit_mcar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  German\ Credit\ Card:llama31:MNAR)
    ADAPTER_PATH='data/processed/llm/llama31_credit_billamt1_mnar_lora'
    EVAL_MISSING='data/processed/MNAR/german_credit_mnar_billamt1_10pct.csv'
    TARGET_COLUMN='BILL_AMT1'
    ID_COLUMN='ID'
    EVAL_FULL='data/raw/default_of_credit_card_clients.csv'
    GLOBAL_DATASET='German Credit Card'
    GLOBAL_RATE='billamt1, 10%'
    ;;
  *)
    echo "Unsupported combination: DATASET=${DATASET} MODEL_KEY=${MODEL_KEY} MECH=${MECH}"
    exit 1
    ;;
esac

"$PYTHON_BIN" -c "import pandas,transformers,peft; print('Dependency check OK')"

CMD=(
  "$PYTHON_BIN"
  src/scripts/run_llm_finetune.py
  --eval_only
  --model_name "$MODEL_NAME"
  --adapter_path "$ADAPTER_PATH"
  --eval_full_csv "$EVAL_FULL"
  --eval_missing_csv "$EVAL_MISSING"
  --eval_target_column "$TARGET_COLUMN"
  --eval_prompt_style current
  --global_dataset "$GLOBAL_DATASET"
  --global_missingness_type "$MECH"
  --global_missing_rate "$GLOBAL_RATE"
)

if [[ -n "$ID_COLUMN" ]]; then
  CMD+=(--eval_id_column "$ID_COLUMN")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
