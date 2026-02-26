import pandas as pd
import numpy as np

import numpy as np
import pandas as pd


def rmse(df_true: pd.DataFrame,
         df_missing: pd.DataFrame,
         df_imputed: pd.DataFrame,
         numeric_only: bool = True) -> pd.Series:
    """
    RMSE per col for missing values.
    """

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
        rmses[col] = np.sqrt(mse)

    return pd.Series(rmses)


def nrmse(df_true, df_missing, df_imputed, norm="std"):
    """
    Compute NRMSE per column, only on originally missing cells.

    norm:
        "std"   -> divide by std of true values (recommended)
        "range" -> divide by (max - min)
    """

    nrmse_values = {}

    for col in df_true.columns:
        mask = df_missing[col].isna()

        if mask.sum() == 0:
            nrmse_values[col] = np.nan
            continue

        true_vals = df_true.loc[mask, col]
        imputed_vals = df_imputed.loc[mask, col]

        rmse = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))

        if norm == "std":
            denom = np.std(true_vals)
        elif norm == "range":
            denom = np.max(true_vals) - np.min(true_vals)
        else:
            raise ValueError("norm must be 'std' or 'range'")

        nrmse_values[col] = rmse / (denom + 1e-8)

    return pd.Series(nrmse_values)
