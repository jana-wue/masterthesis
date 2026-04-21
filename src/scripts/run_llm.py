from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import nrmse, rmse
from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_RAW, DATA_PROCESSED, DATA_RESULTS


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def model_name_to_file_token(model_name):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def extract_first_number(text):
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)
    return text.strip()


def apply_chat_template(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(message["content"] for message in messages)


def generate_answer(messages, tokenizer, model):
    prompt_text = apply_chat_template(messages, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    raw_generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer = extract_first_number(raw_generated)
    return answer, raw_generated


def _load_telco_mar_data():
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )
    return df_full, df_missing


def _build_telco_imputer(df_missing, few_shot_k):
    target_column = "TotalCharges"
    feature_columns = [col for col in df_missing.columns if col not in [target_column, "customerID"]]

    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=MODEL_NAME,
            target_column=target_column,
            feature_columns=feature_columns,
            domain_hints=[
                "For subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Respect plausible values from the observed target distribution.",
            ],
            few_shot_k=few_shot_k,
        )
    )
    observed_rows = df_missing[df_missing[target_column].notna()].copy()
    imputer.fit_target_stats(observed_rows)
    return imputer


def _append_results_row(csv_path, required_columns, row) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    row_normalized = {col: row.get(col, pd.NA) for col in required_columns}
    df = pd.concat([df, pd.DataFrame([row_normalized])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def append_to_global_results(dataset,missingness_type, missing_rate, imputation_method,
    mean_rmse, mean_nrmse):
    _append_results_row(
        csv_path=RESULTS_RMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
        },
    )

    _append_results_row(
        csv_path=RESULTS_NRMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
            "mean_nrmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
            "mean_nrmse": mean_nrmse,
        },
    )


def resolve_telco_results_csv(model_name):
    """
    Resolve prediction CSV path for Telco runs.
    If `model_name` is provided, use the tokenized filename for that model.
    """
    if model_name is not None:
        model_token = model_name_to_file_token(model_name)
        csv_path = DATA_RESULTS / f"telco_{model_token}_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {csv_path}")
        return csv_path

    candidates = sorted(
        DATA_RESULTS.glob("telco_*_results.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No prediction files found in: {DATA_RESULTS}")
    return candidates[0]


def append_global_results_from_telco_predictions(results_csv, model_name, latest_per_setting_only, dataset,
    missingness_type, missing_rate,) -> pd.DataFrame:
    """
    Aggregate Telco prediction CSV into RMSE/NRMSE and append to global result files.
    """
    source_csv = results_csv or resolve_telco_results_csv(model_name=model_name)
    df = pd.read_csv(source_csv)

    required_cols = {"prediction", "ground_truth"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns in {source_csv}: {missing_list}")

    work = df.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["ground_truth"] = pd.to_numeric(work["ground_truth"], errors="coerce")
    work = work.dropna(subset=["prediction", "ground_truth"]).copy()
    if work.empty:
        raise ValueError(
            f"No valid numeric prediction/ground_truth rows found in: {source_csv}"
        )

    if "model_name" not in work.columns:
        work["model_name"] = model_name if model_name is not None else MODEL_NAME
    if "few_shot_k" not in work.columns:
        work["few_shot_k"] = pd.NA

    work["few_shot_k"] = pd.to_numeric(work["few_shot_k"], errors="coerce")

    if latest_per_setting_only and "run_timestamp_utc" in work.columns:
        work["_run_ts"] = pd.to_datetime(work["run_timestamp_utc"], errors="coerce", utc=True)
        grouped_latest = []
        for _, group in work.groupby(["model_name", "few_shot_k"], dropna=False):
            valid_ts = group["_run_ts"].dropna()
            if valid_ts.empty:
                grouped_latest.append(group)
                continue
            latest_ts = valid_ts.max()
            grouped_latest.append(group[group["_run_ts"] == latest_ts])
        work = pd.concat(grouped_latest, ignore_index=True)

    summary_rows = []
    for (model_name_value, few_shot_k_value), group in work.groupby(
        ["model_name", "few_shot_k"], dropna=False
    ):
        errors = group["ground_truth"] - group["prediction"]
        mean_rmse = float(((errors ** 2).mean()) ** 0.5)
        std_true = float(group["ground_truth"].std(ddof=0))
        mean_nrmse = float(mean_rmse / (std_true + 1e-8))

        model_short = str(model_name_value).split("/")[-1]
        if pd.isna(few_shot_k_value):
            method_name = f"LLM Prompt ({model_short})"
            few_shot_k_clean = pd.NA
        else:
            few_shot_k_int = int(few_shot_k_value)
            few_shot_k_clean = few_shot_k_int
            if few_shot_k_int == 0:
                method_name = f"LLM Prompt Zero-shot ({model_short})"
            else:
                method_name = f"LLM Prompt Few-shot k={few_shot_k_int} ({model_short})"

        append_to_global_results(
            dataset=dataset,
            missingness_type=missingness_type,
            missing_rate=missing_rate,
            imputation_method=method_name,
            mean_rmse=mean_rmse,
            mean_nrmse=mean_nrmse,
        )

        summary_rows.append(
            {
                "source_csv": str(source_csv),
                "model_name": model_name_value,
                "few_shot_k": few_shot_k_clean,
                "n_predictions": int(len(group)),
                "mean_rmse": mean_rmse,
                "mean_nrmse": mean_nrmse,
                "imputation_method": method_name,
                "appended_to_rmse_csv": str(RESULTS_RMSE_PATH),
                "appended_to_nrmse_csv": str(RESULTS_NRMSE_PATH),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("APPENDED GLOBAL RESULTS")
    print("=" * 80)
    print(summary_df)
    return summary_df


def run_telco_zero_shot_batch_preview(n_examples: int | None = 10, few_shot_k: int = 2, tokenizer=None,
    model=None) -> pd.DataFrame:
    target_column = "TotalCharges"
    df_full, df_missing = _load_telco_mar_data()
    imputer = _build_telco_imputer(df_missing=df_missing, few_shot_k=few_shot_k)

    missing_rows = df_missing[df_missing[target_column].isna()].copy()
    observed_rows = df_missing[df_missing[target_column].notna()].copy()
    reference_df = observed_rows

    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{target_column}'.")
    if few_shot_k > 0 and reference_df.empty:
        raise ValueError("Few-shot requested but no observed reference rows are available.")

    if n_examples is not None:
        missing_rows = missing_rows.head(n_examples)

    owns_model = tokenizer is None or model is None

    if owns_model:
        print("=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

    print("\n" + "=" * 80)
    print("ZERO-SHOT / FEW-SHOT BATCH PREVIEW")
    print("=" * 80)

    results = []

    for idx, row in missing_rows.iterrows():
        few_shot_examples = imputer.select_few_shot_examples(
            row,
            reference_df,
            exclude_indices={idx},
        )
        messages = imputer.build_inference_messages(row, few_shot_examples=few_shot_examples)
        prediction_raw, raw_output = generate_answer(messages, tokenizer, model)
        prediction = pd.to_numeric(prediction_raw, errors="coerce")
        ground_truth = pd.to_numeric(df_full.loc[idx, target_column], errors="coerce") if idx in df_full.index else pd.NA

        print(f"\nRow index    : {idx}")
        print(f"Raw output   : {raw_output}")
        print(f"Prediction   : {prediction}")
        print(f"Ground truth : {ground_truth}")

        results.append(
            {
                "row_index": int(idx),
                "prediction": prediction,
                "raw_output": raw_output,
                "ground_truth": ground_truth,
                "n_fewshot_examples": len(few_shot_examples),
            }
        )

    results_df = pd.DataFrame(results)
    results_df["model_name"] = MODEL_NAME
    results_df["few_shot_k"] = few_shot_k
    results_df["n_examples_requested"] = n_examples if n_examples is not None else len(missing_rows)
    results_df["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results_df["leakage_guard_enabled"] = True
    results_df["reference_source"] = "df_missing_observed_rows"

    model_token = model_name_to_file_token(MODEL_NAME)
    output_csv = DATA_RESULTS / f"telco_{model_token}_results.csv"

    file_exists = output_csv.exists()
    results_to_write = results_df
    if file_exists:
        existing_columns = pd.read_csv(output_csv, nrows=0).columns.tolist()
        if existing_columns:
            for col in existing_columns:
                if col not in results_to_write.columns:
                    results_to_write[col] = pd.NA
            results_to_write = results_to_write[existing_columns]

    results_to_write.to_csv(
        output_csv,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df)
    print(f"\nSaved predictions to: {output_csv}")
    return results_df


def evaluate_telco_prompt_approach(few_shot_settings = (0, 2),n_examples = None,) -> pd.DataFrame:
    target_column = "TotalCharges"
    df_full, _ = _load_telco_mar_data()

    print("=" * 80)
    print("LOADING MODEL (ONCE FOR ALL SETTINGS)")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    summary_rows = []

    for few_shot_k in few_shot_settings:
        print("\n" + "=" * 80)
        print(f"EVALUATING PROMPT APPROACH | few_shot_k={few_shot_k}")
        print("=" * 80)

        predictions_df = run_telco_zero_shot_batch_preview(
            n_examples=n_examples,
            few_shot_k=few_shot_k,
            tokenizer=tokenizer,
            model=model,
        )

        eval_indices = predictions_df["row_index"].astype(int).tolist()
        eval_true = df_full.drop(columns=["customerID"]).copy()
        eval_missing = eval_true.copy()
        eval_imputed = eval_true.copy()

        eval_missing.loc[eval_indices, target_column] = pd.NA
        eval_imputed.loc[eval_indices, target_column] = pd.to_numeric(
            predictions_df["prediction"].values,
            errors="coerce",
        )

        num_cols = eval_true.select_dtypes(include=["number"]).columns
        eval_true_num = eval_true[num_cols]
        eval_missing_num = eval_missing[num_cols]
        eval_imputed_num = eval_imputed[num_cols]

        rmse_values = rmse(eval_true_num, eval_missing_num, eval_imputed_num)
        nrmse_values = nrmse(eval_true_num, eval_missing_num, eval_imputed_num, norm="std")
        mean_rmse = float(rmse_values.mean(skipna=True))
        mean_nrmse = float(nrmse_values.mean(skipna=True))

        model_short = MODEL_NAME.split("/")[-1]
        if few_shot_k == 0:
            method_name = f"LLM Prompt Zero-shot ({model_short})"
        else:
            method_name = f"LLM Prompt Few-shot k={few_shot_k} ({model_short})"

        append_to_global_results(
            dataset="Telco",
            missingness_type="MAR",
            missing_rate="missing totalCharges -> tenure, 10%",
            imputation_method=method_name,
            mean_rmse=mean_rmse,
            mean_nrmse=mean_nrmse,
        )

        summary_rows.append(
            {
                "few_shot_k": few_shot_k,
                "imputation_method": method_name,
                "mean_rmse": mean_rmse,
                "mean_nrmse": mean_nrmse,
                "n_predictions": int(len(predictions_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    evaluate_telco_prompt_approach(few_shot_settings=(0, 2), n_examples=None)
