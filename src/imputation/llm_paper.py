from __future__ import annotations

import re
from io import StringIO

import pandas as pd


def clean_and_parse_llm_data(response_text: str, expected_shape: tuple[int, int]) -> pd.DataFrame:
    """
    Parse LLM output into a DataFrame.
    """
    match = re.search(r"```(?:csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    content = match.group(1).strip() if match else response_text.strip()

    for separator in [",", r"\s+"]:
        try:
            df_imputed = pd.read_csv(StringIO(content), sep=separator, engine="python")
            return df_imputed
        except Exception:
            continue

    raise ValueError(f"Could not parse imputed matrix. Expected shape: {expected_shape}")


def build_paper_prompt(dataset_name: str, missing_data: pd.DataFrame) -> str:
    """
    Prompt copied 1:1 from ArthurMangussi/LLMsImputation (adjust_prompt).
    """
    headers_str = ", ".join(missing_data.columns)
    string_missing = missing_data.to_string()
    prompt = f"""
    You are an expert data analyst. I am providing a subset of the {dataset_name} Dataset.
    Task: Use your knowledge of this specific dataset's statistical properties (feature ranges, class distributions, and correlations) to perform data imputation.
    Constraint: DO NOT execute Python code. DO NOT provide any conversational text. Do NOT return any NaN or ? value.

    The matrix below contains missing values. Impute them to be as consistent as possible with the original dataset.
    Matrix:
    {string_missing}

    Output Format:
    Return the complete imputed matrix inside a single Markdown code block. 
    Use CSV format (comma-separated values) with the original headers.
    Expected Columns ({missing_data.shape[1]}): 
    [{headers_str}]

    Strict Rules:
    1. Start directly with the code block: ```csv
    2. End exactly with: ```
    3. Ensure the exact same number of rows as the input.
    4. No explanations, no introductory text, no "Here is the matrix".
    5. Use commas as delimiters. Every row MUST have exactly {missing_data.shape[1] - 1} commas.
    """
    return prompt


def normalize_imputed_matrix(df_imputed, expected_columns, expected_rows) -> pd.DataFrame:
    """
    Align parsed matrix to the expected Telco shape/columns.
    """
    work = df_imputed.copy()

    if work.empty:
        raise ValueError("Parsed matrix is empty.")

    # Common LLM behavior: extra index-like column.
    if len(work.columns) == len(expected_columns) + 1:
        first_col = str(work.columns[0]).strip().lower()
        if first_col.startswith("unnamed") or work.columns[0] not in expected_columns:
            work = work.iloc[:, 1:]

    if set(expected_columns).issubset(set(work.columns)):
        work = work.loc[:, expected_columns]
    elif len(work.columns) == len(expected_columns):
        work.columns = expected_columns
    else:
        raise ValueError(
            "Could not align output columns. "
            f"Expected {len(expected_columns)} cols, got {len(work.columns)}."
        )

    if len(work) < expected_rows:
        raise ValueError(
            f"LLM returned fewer rows than expected ({len(work)} < {expected_rows})."
        )

    if len(work) > expected_rows:
        # Same idea as the paper code path that trims to needed rows.
        work = work.tail(expected_rows)

    return work.reset_index(drop=True)
