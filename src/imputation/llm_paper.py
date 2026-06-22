from __future__ import annotations

import re
from io import StringIO

import pandas as pd


def _extract_code_block_content(response_text: str) -> str:
    """Extract code block content."""
    blocks = re.findall(r"```(?:csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    if not blocks:
        return response_text.strip()
    # Prefer the longest block if multiple code blocks are returned.
    return max(blocks, key=len).strip()


def _score_candidate(df: pd.DataFrame, expected_shape: tuple[int, int]) -> int:
    """Handle score candidate."""
    expected_rows, expected_cols = expected_shape
    rows, cols = df.shape
    one_col_penalty = 1000 if expected_cols > 1 and cols == 1 else 0
    return one_col_penalty + abs(cols - expected_cols) * 100 + abs(rows - expected_rows)


def _parse_markdown_table(content: str) -> pd.DataFrame | None:
    """Parse markdown table."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    if not all("|" in line for line in lines[:2]):
        return None

    cleaned = []
    for line in lines:
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        cleaned.append(parts)

    if len(cleaned) < 2:
        return None

    header = cleaned[0]
    data_rows = cleaned[1:]
    if not header or not data_rows:
        return None

    try:
        return pd.DataFrame(data_rows, columns=header)
    except Exception:
        return None


def clean_and_parse_llm_data(
    response_text: str,
    expected_shape: tuple[int, int],
) -> pd.DataFrame:
    """
    Parse LLM output into a DataFrame.
    """
    content = _extract_code_block_content(response_text)
    if not content:
        raise ValueError(f"Empty model output. Expected shape: {expected_shape}")

    candidates: list[tuple[int, pd.DataFrame]] = []
    separators = [",", ";", "\t", "|", r"\s+"]
    header_modes = ["infer", None]

    for sep in separators:
        for header in header_modes:
            try:
                kwargs = {"sep": sep, "engine": "python"}
                if header is None:
                    kwargs["header"] = None
                df = pd.read_csv(StringIO(content), **kwargs)
            except Exception:
                continue

            if df.empty:
                continue
            candidates.append((_score_candidate(df, expected_shape), df))

    markdown_df = _parse_markdown_table(content)
    if markdown_df is not None and not markdown_df.empty:
        candidates.append((_score_candidate(markdown_df, expected_shape), markdown_df))

    if not candidates:
        raise ValueError(f"Could not parse imputed matrix. Expected shape: {expected_shape}")

    candidates.sort(key=lambda x: x[0])
    best_df = candidates[0][1]
    if expected_shape[1] > 1 and best_df.shape[1] == 1:
        raise ValueError(
            "Parser produced a single-column table. Model output was not valid tabular CSV."
        )

    return best_df


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
    Return the complete imputed matrix as plain CSV text (no markdown code block).
    Use comma-separated values with the original headers.
    Expected Columns ({missing_data.shape[1]}):
    [{headers_str}]

    Strict Rules:
    1. First line must be exactly the CSV header with these columns.
    2. Return exactly {missing_data.shape[0]} data rows after the header.
    3. Every data row must have exactly {missing_data.shape[1] - 1} commas.
    4. No explanations, no introductory text, no markdown.
    5. Do not add an index column.
    """
    return prompt


def build_paper_retry_prompt(dataset_name: str, missing_data: pd.DataFrame) -> str:
    """
    Stricter retry prompt used only when first parsing fails.
    """
    base = build_paper_prompt(dataset_name=dataset_name, missing_data=missing_data)
    return (
        f"{base}\n"
        "\n"
        "CRITICAL OUTPUT VALIDATION:\n"
        "- Return plain CSV only (no markdown code block).\n"
        f"- The CSV must contain exactly {missing_data.shape[1]} columns.\n"
        f"- The CSV must contain exactly {missing_data.shape[0]} data rows.\n"
        "- Do not return markdown tables using pipes.\n"
        "- Do not return explanations.\n"
    )


def normalize_imputed_matrix(
    df_imputed: pd.DataFrame,
    expected_columns: Sequence[str],
    expected_rows: int,
) -> pd.DataFrame:
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
