from typing import Literal, TypedDict
import numpy as np
import pandas as pd


class EvalContext(TypedDict):
    """Store the last evaluated dataframes and aggregate metrics."""

    key: tuple[int, int, int]
    df_true: pd.DataFrame
    df_missing: pd.DataFrame
    df_imputed: pd.DataFrame
    mean_rmse: float | None
    mean_nrmse: float | None
    consumed: bool


_EVAL_CONTEXTS: list[EvalContext] = []


def _context_key(
    df_true: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame,
) -> tuple[int, int, int]:
    """Handle context key."""
    return (id(df_true), id(df_missing), id(df_imputed))


def _upsert_context(
    df_true: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame,
    mean_rmse: float | None = None,
    mean_nrmse: float | None = None,
) -> None:
    """Upsert context."""
    key = _context_key(df_true, df_missing, df_imputed)

    for ctx in reversed(_EVAL_CONTEXTS):
        if ctx["key"] == key and not ctx.get("consumed", False):
            if mean_rmse is not None:
                ctx["mean_rmse"] = float(mean_rmse)
            if mean_nrmse is not None:
                ctx["mean_nrmse"] = float(mean_nrmse)
            return

    _EVAL_CONTEXTS.append(
        {
            "key": key,
            "df_true": df_true,
            "df_missing": df_missing,
            "df_imputed": df_imputed,
            "mean_rmse": float(mean_rmse) if mean_rmse is not None else None,
            "mean_nrmse": float(mean_nrmse) if mean_nrmse is not None else None,
            "consumed": False,
        }
    )


def pop_eval_context_by_means(
    mean_rmse: float,
    mean_nrmse: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> EvalContext | None:
    """Pop evaluation context by means."""
    for ctx in reversed(_EVAL_CONTEXTS):
        if ctx.get("consumed", False):
            continue

        ctx_rmse = ctx.get("mean_rmse")
        ctx_nrmse = ctx.get("mean_nrmse")
        if ctx_rmse is None or ctx_nrmse is None:
            continue

        if np.isclose(ctx_rmse, mean_rmse, rtol=rtol, atol=atol) and np.isclose(
            ctx_nrmse, mean_nrmse, rtol=rtol, atol=atol
        ):
            ctx["consumed"] = True
            return ctx

    return None


def rmse(
    df_true: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame,
    numeric_only: bool = True,
) -> pd.Series:
    """Handle RMSE."""
    if numeric_only:
        cols = df_true.select_dtypes(include=["number"]).columns
    else:
        cols = df_true.columns

    rmses = {}

    for col in cols:
        mask = df_missing[col].isna() & df_true[col].notna()
        if mask.sum() == 0:
            rmses[col] = np.nan
            continue

        true_vals = df_true.loc[mask, col]
        imputed_vals = df_imputed.loc[mask, col]
        mse = ((true_vals - imputed_vals) ** 2).mean()
        rmses[col] = float(np.sqrt(mse))

    result = pd.Series(rmses)
    _upsert_context(
        df_true=df_true,
        df_missing=df_missing,
        df_imputed=df_imputed,
        mean_rmse=float(result.mean(skipna=True)),
    )
    return result


def nrmse(
    df_true: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame,
    norm: Literal["std", "range"] = "std",
) -> pd.Series:
    """Handle NRMSE."""
    nrmse_values = {}

    for col in df_true.columns:
        mask = df_missing[col].isna()
        if mask.sum() == 0:
            nrmse_values[col] = np.nan
            continue

        true_vals = df_true.loc[mask, col]
        imputed_vals = df_imputed.loc[mask, col]
        col_rmse = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))

        if norm == "std":
            denom = np.std(true_vals)
        elif norm == "range":
            denom = np.max(true_vals) - np.min(true_vals)
        else:
            raise ValueError("norm must be 'std' or 'range'")

        nrmse_values[col] = float(col_rmse / (denom + 1e-8))

    result = pd.Series(nrmse_values)
    _upsert_context(
        df_true=df_true,
        df_missing=df_missing,
        df_imputed=df_imputed,
        mean_nrmse=float(result.mean(skipna=True)),
    )
    return result
