from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from src.evaluation.metrics import pop_eval_context_by_means


# Global result tables
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")

# Backward-compatible alias
RESULTS_PATH = RESULTS_NRMSE_PATH

# One result file per (dataset, missingness, method, model) combination
SETTING_RESULTS_DIR = Path("data/results/setting_runs")


def _to_file_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token.lower() or "unknown"


def _append_results_row(csv_path, required_columns, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[required_columns]
    row_normalized = {col: row.get(col, pd.NA) for col in required_columns}
    df = pd.concat([df, pd.DataFrame([row_normalized])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _append_results_rows(csv_path, rows_df):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    drop_cols = [col for col in ["mean_rmse", "mean_nrmse", "rmse"] if col in rows_df.columns]
    if drop_cols:
        rows_df = rows_df.drop(columns=drop_cols)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        drop_cols = [col for col in ["mean_rmse", "mean_nrmse", "rmse"] if col in existing_df.columns]
        if drop_cols:
            existing_df = existing_df.drop(columns=drop_cols)
        for col in existing_df.columns:
            if col not in rows_df.columns:
                rows_df[col] = pd.NA
        for col in rows_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = pd.NA
        rows_df = rows_df[existing_df.columns]
        combined = pd.concat([existing_df, rows_df], ignore_index=True, sort=False)
    else:
        combined = rows_df.copy()

    combined.to_csv(csv_path, index=False)


def _resolve_setting_results_path(dataset, missingness_type, imputation_method, model_name=None, results_path=None):
    if results_path is not None:
        return Path(results_path)

    dataset_token = _to_file_token(dataset)
    missingness_token = _to_file_token(missingness_type)
    method_token = _to_file_token(imputation_method)

    filename_parts = [dataset_token, missingness_token, method_token]
    if model_name is not None and str(model_name).strip() != "":
        filename_parts.append(_to_file_token(model_name))

    filename = "__".join(filename_parts) + ".csv"
    return SETTING_RESULTS_DIR / filename


def _build_detail_rows_from_context(ctx, run_timestamp_utc, dataset, missingness_type, missing_rate, imputation_method, model_name):
    df_true = ctx["df_true"]
    df_missing = ctx["df_missing"]
    df_imputed = ctx["df_imputed"]

    numeric_cols = df_true.select_dtypes(include=["number"]).columns
    rows = []

    for col in numeric_cols:
        mask = df_missing[col].isna() & df_true[col].notna()
        if mask.sum() == 0:
            continue

        idxs = df_true.index[mask].tolist()
        true_vals = pd.to_numeric(df_true.loc[mask, col], errors="coerce")
        pred_vals = pd.to_numeric(df_imputed.loc[mask, col], errors="coerce")

        for row_index, ground_truth, prediction in zip(idxs, true_vals, pred_vals):
            if pd.isna(ground_truth) or pd.isna(prediction):
                abs_error = pd.NA
                sq_error = pd.NA
            else:
                abs_error = float(abs(float(prediction) - float(ground_truth)))
                sq_error = float((float(prediction) - float(ground_truth)) ** 2)

            rows.append(
                {
                    "run_timestamp_utc": run_timestamp_utc,
                    "dataset": dataset,
                    "missingness_type": missingness_type,
                    "missing_rate": missing_rate,
                    "imputation_method": imputation_method,
                    "model_name": model_name if model_name is not None and str(model_name).strip() != "" else pd.NA,
                    "row_index": row_index,
                    "column_name": col,
                    "prediction": float(prediction) if pd.notna(prediction) else pd.NA,
                    "ground_truth": float(ground_truth) if pd.notna(ground_truth) else pd.NA,
                    "abs_error": abs_error,
                    "sq_error": sq_error,
                    "source": "tabular_imputer",
                }
            )

    return pd.DataFrame(rows)


def append_setting_result(dataset, missingness_type, missing_rate, imputation_method, model_name=None, results_path=None,
    extra=None, detail_rows=None):
    """
    Append detailed prediction rows to a setting-specific file.
    """
    output_path = _resolve_setting_results_path(
        dataset=dataset,
        missingness_type=missingness_type,
        imputation_method=imputation_method,
        model_name=model_name,
        results_path=results_path,
    )

    if detail_rows is None or detail_rows.empty:
        raise ValueError(
            "append_setting_result requires non-empty detail_rows. "
            "Summary fallback is disabled."
        )

    rows_df = detail_rows.copy()
    if extra:
        for key, value in extra.items():
            rows_df[key] = value
    _append_results_rows(output_path, rows_df)
    return output_path


def append_global_results(dataset, missingness_type, missing_rate, imputation_method, mean_rmse, mean_nrmse, model_name=None,
    results_path=None, extra=None, write_setting_file=True, detail_rows=None):
    """
    Append row to global result files
    """
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
            "mean_nrmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_nrmse": mean_nrmse,
        },
    )

    if not write_setting_file:
        return None

    if detail_rows is None or detail_rows.empty:
        raise ValueError(
            "append_global_results requires non-empty detail_rows when write_setting_file=True. "
            "Summary fallback is disabled."
        )

    return append_setting_result(
        dataset=dataset,
        missingness_type=missingness_type,
        missing_rate=missing_rate,
        imputation_method=imputation_method,
        model_name=model_name,
        results_path=results_path,
        extra=extra,
        detail_rows=detail_rows,
    )


def append_method_result(dataset, missingness_type, missing_rate, imputation_method, mean_rmse, mean_nrmse,
    method_results_path=None, extra=None):
    """
    Alias for older callers
    """
    raise RuntimeError(
        "append_method_result no longer supports summary-only rows. "
        "Use append_setting_result with explicit detail_rows."
    )


def log_result(dataset, missingness_type, missing_rate, imputation_method, mean_rmse, mean_nrmse, model_name=None,
    method_results_path=None, extra=None):
    """
    Entrypoint used by current classical run scripts.
    """
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    ctx = pop_eval_context_by_means(float(mean_rmse), float(mean_nrmse))
    if ctx is None:
        raise RuntimeError(
            "No evaluation context found for this log_result call. "
            "Call rmse(...) and nrmse(...) directly before log_result(...)."
        )

    detail_rows = _build_detail_rows_from_context(
        ctx=ctx,
        run_timestamp_utc=run_timestamp_utc,
        dataset=dataset,
        missingness_type=missingness_type,
        missing_rate=missing_rate,
        imputation_method=imputation_method,
        model_name=model_name,
    )

    if detail_rows.empty:
        raise RuntimeError(
            "log_result produced no detail rows. "
            "Ensure there are numeric missing values to evaluate."
        )

    output_path = append_global_results(
        dataset=dataset,
        missingness_type=missingness_type,
        missing_rate=missing_rate,
        imputation_method=imputation_method,
        mean_rmse=mean_rmse,
        mean_nrmse=mean_nrmse,
        model_name=model_name,
        results_path=method_results_path,
        extra=extra,
        write_setting_file=True,
        detail_rows=detail_rows,
    )

    print(
        f"Result saved for {dataset} - {missingness_type} - {imputation_method} | "
        f"global: {RESULTS_RMSE_PATH}, {RESULTS_NRMSE_PATH} | setting file: {output_path}"
    )
