from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm_paper import (
    build_paper_prompt,
    clean_and_parse_llm_data,
    normalize_imputed_matrix,
)
from src.paths import DATA_RAW, DATA_PROCESSED, DATA_RESULTS


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME_PROMPT = "Telco-Customer-Churn"
TARGET_COLUMN = "TotalCharges"
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def model_name_to_file_token(model_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def _append_results_row(csv_path: Path, required_columns: list[str], row: dict) -> None:
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


def append_to_global_results(dataset, missingness_type, missing_rate, imputation_method,
    mean_rmse, mean_nrmse) -> None:
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


def _load_telco_mar_data():
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )
    return df_full, df_missing


def _build_prompt_matrix(df_missing, n_rows):
    work = df_missing.drop(columns=["customerID"], errors="ignore").copy()
    if n_rows is not None:
        work = work.head(n_rows).copy()

    original_indices = work.index.tolist()
    prompt_matrix = work.reset_index(drop=True)

    missing_positions = [
        pos for pos, is_missing in enumerate(prompt_matrix[TARGET_COLUMN].isna()) if is_missing
    ]
    if not missing_positions:
        raise ValueError(
            "Selected prompt matrix has no missing TotalCharges rows. "
            "Increase n_rows or set n_rows=None."
        )

    return prompt_matrix, original_indices, missing_positions


def apply_user_chat_template(prompt_text, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def generate_matrix_answer(prompt_text, tokenizer,model, max_new_tokens):
    model_input = apply_user_chat_template(prompt_text, tokenizer)
    inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def fallback_totalcharges(row, totalcharges_median):
    tenure = row.get("tenure")
    monthly = row.get("MonthlyCharges")

    if pd.notna(tenure) and pd.notna(monthly):
        try:
            return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
        except Exception:
            pass

    return float(totalcharges_median), "fallback_median"


def _append_prediction_rows(output_csv, rows_df):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        existing_columns = pd.read_csv(output_csv, nrows=0).columns.tolist()
    else:
        existing_columns = []

    to_write = rows_df.copy()
    if existing_columns:
        for col in existing_columns:
            if col not in to_write.columns:
                to_write[col] = pd.NA
        to_write = to_write[existing_columns]

    to_write.to_csv(
        output_csv,
        mode="a" if output_csv.exists() else "w",
        header=not output_csv.exists(),
        index=False,
    )


def run_telco_paper_prompt(n_rows: int | None = 40, max_new_tokens: int = 4096, tokenizer=None, model=None):
    df_full, df_missing = _load_telco_mar_data()

    prompt_matrix, original_indices, missing_positions = _build_prompt_matrix(
        df_missing=df_missing,
        n_rows=n_rows,
    )

    prompt_text = build_paper_prompt(
        dataset_name=DATASET_NAME_PROMPT,
        missing_data=prompt_matrix,
    )

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
    print("RUNNING PAPER PROMPT (NO BATCH LOOP)")
    print("=" * 80)
    print(f"Prompt rows: {len(prompt_matrix)}")
    print(f"Missing rows to evaluate: {len(missing_positions)}")

    raw_output = generate_matrix_answer(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    parse_error = None
    try:
        parsed_df = clean_and_parse_llm_data(
            response_text=raw_output,
            expected_shape=prompt_matrix.shape,
        )
        imputed_matrix = normalize_imputed_matrix(
            df_imputed=parsed_df,
            expected_columns=list(prompt_matrix.columns),
            expected_rows=len(prompt_matrix),
        )
    except Exception as exc:
        parse_error = str(exc)
        print(f"WARNING: parsing failed, using fallback-only for missing rows. ({parse_error})")
        imputed_matrix = prompt_matrix.copy()

    observed_target = pd.to_numeric(df_full[TARGET_COLUMN], errors="coerce").dropna()
    totalcharges_median = float(observed_target.median())

    results = []
    for pos in missing_positions:
        original_idx = original_indices[pos]
        row_for_fallback = prompt_matrix.iloc[pos]

        pred_value = pd.to_numeric(imputed_matrix.loc[pos, TARGET_COLUMN], errors="coerce")
        source = "paper_prompt"

        if pd.isna(pred_value):
            pred_value, source = fallback_totalcharges(
                row=row_for_fallback,
                totalcharges_median=totalcharges_median,
            )

        ground_truth = pd.to_numeric(df_full.loc[original_idx, TARGET_COLUMN], errors="coerce")

        results.append(
            {
                "row_index": int(original_idx),
                "prompt_row_position": int(pos),
                "prediction": float(pred_value),
                "ground_truth": float(ground_truth) if pd.notna(ground_truth) else pd.NA,
                "source": source,
                "parse_error": parse_error,
            }
        )

    results_df = pd.DataFrame(results)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_token = model_name_to_file_token(MODEL_NAME)
    output_csv = DATA_RESULTS / f"telco_{model_token}_paper_results.csv"
    raw_output_txt = DATA_RESULTS / f"telco_{model_token}_paper_raw_output.txt"

    results_df["model_name"] = MODEL_NAME
    results_df["prompt_rows"] = len(prompt_matrix)
    results_df["n_rows_requested"] = n_rows if n_rows is not None else len(prompt_matrix)
    results_df["max_new_tokens"] = max_new_tokens
    results_df["run_timestamp_utc"] = run_timestamp
    results_df["raw_output_preview"] = raw_output[:600]

    raw_output_txt.parent.mkdir(parents=True, exist_ok=True)
    raw_output_txt.write_text(raw_output, encoding="utf-8")
    _append_prediction_rows(output_csv=output_csv, rows_df=results_df)

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df.head())
    print(f"\nSaved predictions to: {output_csv}")
    print(f"Saved full model output to: {raw_output_txt}")

    return results_df


def evaluate_telco_paper_prompt(
    n_rows: int | None = 40,
    max_new_tokens: int = 4096,
) -> pd.DataFrame:
    results_df = run_telco_paper_prompt(
        n_rows=n_rows,
        max_new_tokens=max_new_tokens,
    )

    work = results_df.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["ground_truth"] = pd.to_numeric(work["ground_truth"], errors="coerce")
    work = work.dropna(subset=["prediction", "ground_truth"]).copy()

    if work.empty:
        raise ValueError("No valid numeric prediction/ground_truth rows to evaluate.")

    errors = work["ground_truth"] - work["prediction"]
    mean_rmse = float(((errors ** 2).mean()) ** 0.5)
    std_true = float(work["ground_truth"].std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true + 1e-8))

    model_short = MODEL_NAME.split("/")[-1]
    if n_rows is None:
        method_name = f"LLM Paper Prompt ({model_short})"
    else:
        method_name = f"LLM Paper Prompt rows={n_rows} ({model_short})"

    append_to_global_results(
        dataset="Telco",
        missingness_type="MAR",
        missing_rate="missing totalCharges -> tenure, 10%",
        imputation_method=method_name,
        mean_rmse=mean_rmse,
        mean_nrmse=mean_nrmse,
    )

    summary_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "imputation_method": method_name,
                "n_predictions": int(len(work)),
                "mean_rmse": mean_rmse,
                "mean_nrmse": mean_nrmse,
            }
        ]
    )

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(summary_df)
    print(f"\nAppended to: {RESULTS_RMSE_PATH}")
    print(f"Appended to: {RESULTS_NRMSE_PATH}")

    return summary_df


if __name__ == "__main__":
    evaluate_telco_paper_prompt(n_rows=40, max_new_tokens=4096)
