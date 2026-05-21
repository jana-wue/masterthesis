from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.remove_data import mar, mcar, mcar_single_feature, mnar
from src.paths import DATA_PROCESSED, DATA_RAW


def _pct_token(rate: float) -> str:
    return f"{int(round(rate * 100))}pct"


def _save(df, relative_output_path):
    output_path = DATA_PROCESSED / relative_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[saved] {output_path}")


def mcar_global_excluding(df, missing_frac, exclude_cols, random_state):
    """
    MCAR over all eligible columns except excluded ones (e.g. IDs or labels).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    excluded = set(exclude_cols)
    eligible_cols = [col for col in out.columns if col not in excluded]
    if not eligible_cols:
        return out

    for col in eligible_cols:
        out[col] = out[col].astype("object")

    mask = rng.random((len(out), len(eligible_cols))) < float(missing_frac)
    for j, col in enumerate(eligible_cols):
        out.loc[mask[:, j], col] = np.nan

    return out


def run_generate_missingness(rates, random_state):
    """
    Generate missingness files for all datasets.

    Creates for each rate:
    - MAR_TARGET
    - MNAR_TARGET
    - MCAR_TARGET
    - MCAR_GLOBAL (with excluded columns)

    Also writes legacy 10%-only files used by older experiments.
    """
    df_credit = pd.read_csv(DATA_RAW / "default_of_credit_card_clients.csv")
    df_telco = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_statlog = pd.read_csv(DATA_RAW / "german_statlog_numeric.csv")

    for rate in rates:
        token = _pct_token(rate)

        # Telco
        _save(
            mar(
                df_telco,
                feature_dep="tenure",
                missing_feature="TotalCharges",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MAR/telco_customer_churn_mar_totalcharges_tenure_{token}.csv",
        )
        _save(
            mnar(
                df_telco,
                feature="TotalCharges",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MNAR/telco_customer_churn_mnar_totalcharges_{token}.csv",
        )
        _save(
            mcar_single_feature(
                df_telco,
                missing_frac=rate,
                feature="TotalCharges",
                random_state=random_state,
            ),
            f"MCAR/telco_customer_churn_mcar_totalcharges_{token}.csv",
        )
        _save(
            mcar_global_excluding(
                df_telco,
                missing_frac=rate,
                exclude_cols=["customerID", "Churn"],
                random_state=random_state,
            ),
            f"MCAR/telco_customer_churn_mcar_{token}.csv",
        )

        # German Statlog
        _save(
            mar(
                df_statlog,
                feature_dep="X5",
                missing_feature="X2",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MAR/german_statlog_numeric_mar_duration_creditamount_{token}.csv",
        )
        _save(
            mnar(
                df_statlog,
                feature="X2",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MNAR/german_statlog_numeric_mnar_duration_{token}.csv",
        )
        _save(
            mcar_single_feature(
                df_statlog,
                missing_frac=rate,
                feature="X2",
                random_state=random_state,
            ),
            f"MCAR/german_statlog_numeric_mcar_duration_{token}.csv",
        )
        _save(
            mcar_global_excluding(
                df_statlog,
                missing_frac=rate,
                exclude_cols=["class"],
                random_state=random_state,
            ),
            f"MCAR/german_statlog_numeric_mcar_{token}.csv",
        )

        # German Credit Card
        _save(
            mar(
                df_credit,
                feature_dep="PAY_0",
                missing_feature="BILL_AMT1",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MAR/german_credit_mar_billamt1_pay0_{token}.csv",
        )
        _save(
            mnar(
                df_credit,
                feature="BILL_AMT1",
                missing_frac=rate,
                random_state=random_state,
            ),
            f"MNAR/german_credit_mnar_billamt1_{token}.csv",
        )
        _save(
            mcar_single_feature(
                df_credit,
                missing_frac=rate,
                feature="BILL_AMT1",
                random_state=random_state,
            ),
            f"MCAR/german_credit_mcar_billamt1_{token}.csv",
        )
        _save(
            mcar_global_excluding(
                df_credit,
                missing_frac=rate,
                exclude_cols=["ID", "default payment next month"],
                random_state=random_state,
            ),
            f"MCAR/german_credit_mcar_{token}.csv",
        )


if __name__ == "__main__":
    run_generate_missingness()
