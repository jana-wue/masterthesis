import pandas as pd
from src.paths import *

def xls_to_csv(path_to_xls, path_to_save):
    xls_file = path_to_xls

    df = pd.read_excel(xls_file, sheet_name=0)

    csv_file = path_to_save
    df.to_csv(csv_file, index=False)

    print(f"CSV wurde erstellt: {csv_file}")

def data_numeric_to_csv(data_file, filename_save):
    df = pd.read_csv(
    DATA_RAW / data_file,
    sep=r"\s+",
    header=None
    )

    print(df.shape)
    print(df.head())

    df.columns = [f"X{i}" for i in range(1, 26)]
    df.columns.values[-1] = "class"

    df.to_csv(DATA_PROCESSED / filename_save, index=False)


import pandas as pd
import numpy as np

def make_numeric_columns_numeric(df: pd.DataFrame, exclude: list[str] | None = None,
                                 threshold: float = 0.95) -> pd.DataFrame:
    """
    Make cols numeric, if they are numeric, but were read as 'object'.
    (i.e. Telco TotalCharges)

    exclude: cols that shouldn't be converted
    threshold: min part of numeric values
    """
    df = df.copy()
    exclude = set(exclude or [])

    # only look at object cols
    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]

    for col in object_cols:
        cleaned = df[col].astype(str).str.strip()
        cleaned = cleaned.replace("", np.nan)

        # try to convert into numvers
        as_num = pd.to_numeric(cleaned, errors="coerce")

        convertible_ratio = as_num.notna().mean()

        if convertible_ratio >= threshold:
            df[col] = as_num
    return df

