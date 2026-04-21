from __future__ import annotations

import math
import re

import pandas as pd


def format_prompt_value(value):
    if pd.isna(value):
        return "MISSING"

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def summarize_target_distribution(df, target_column):
    observed = pd.to_numeric(df[target_column], errors="coerce").dropna()
    if observed.empty:
        return {}

    return {
        "min": float(observed.min()),
        "q1": float(observed.quantile(0.25)),
        "median": float(observed.median()),
        "q3": float(observed.quantile(0.75)),
        "max": float(observed.max()),
        "mean": float(observed.mean()),
    }


def row_to_feature_text(row, target_column, feature_columns=None):
    if feature_columns is None:
        columns = [col for col in row.index if col != target_column]
    else:
        columns = [col for col in feature_columns if col != target_column]

    return "\n".join(f"{col}: {format_prompt_value(row[col])}" for col in columns)


def _format_target_stats(target_stats: dict[str, float] | None):
    if not target_stats:
        return "Target distribution summary: unavailable."

    ordered_keys = ["min", "q1", "median", "q3", "max", "mean"]
    lines = ["Target distribution summary (observed rows):"]
    for key in ordered_keys:
        if key in target_stats:
            lines.append(f"- {key}: {format_prompt_value(target_stats[key])}")
    return "\n".join(lines)


def build_paper_single_prompt(dataset_name, row, target_column, feature_columns, target_stats: dict[str, float] | None = None,
    domain_hints: list[str] | None = None):
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    stats_text = _format_target_stats(target_stats)

    lines = [
        f"You are an expert data analyst. I am providing one record from the {dataset_name} Dataset.",
        (
            "Task: Use your knowledge of this specific dataset's statistical properties "
            "(feature ranges, class distributions, and correlations) to perform data imputation."
        ),
        (
            "Constraint: DO NOT execute Python code. DO NOT provide any conversational text. "
            "Do NOT return any NaN or ? value."
        ),
        "",
        "The record below contains one missing target value. Impute it to be as consistent as possible with the original dataset.",
        f"Target column: {target_column}",
        stats_text,
        "",
        "Observed feature values:",
        feature_text,
    ]

    if domain_hints:
        lines.extend(["", "Domain hints:"])
        for hint in domain_hints:
            lines.append(f"- {hint}")

    lines.extend(
        [
            "",
            "Output Format:",
            f"Return exactly one numeric value for {target_column} as plain text.",
            "",
            "Strict Rules:",
            "1. Output must contain exactly one number.",
            "2. No explanations, no introductory text, no markdown.",
            "3. No units and no thousands separators.",
            "4. Do not echo the input features.",
            "Imputed value:",
        ]
    )
    return "\n".join(lines)


def build_paper_single_retry_prompt(dataset_name, row, target_column, feature_columns=None, target_stats: dict[str, float] | None = None,
    domain_hints: list[str] | None = None):
    base_prompt = build_paper_single_prompt(
        dataset_name=dataset_name,
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
        target_stats=target_stats,
        domain_hints=domain_hints,
    )

    return (
        f"{base_prompt}\n"
        "\n"
        "CRITICAL OUTPUT VALIDATION:\n"
        "- Return one plain number only.\n"
        "- Do not output a sentence.\n"
        "- Do not output markdown.\n"
        "- Do not output multiple candidates.\n"
    )


def _extract_code_block_content(response_text):
    blocks = re.findall(r"```(?:text|csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    if not blocks:
        return response_text.strip()
    return max(blocks, key=len).strip()


def _normalize_number_token(token):
    token = token.strip()

    # Handle "1,234.56" vs "1.234,56" and plain "123,45".
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")

    return token


def extract_numeric_value(response_text) -> float | None:
    if response_text is None:
        return None

    content = _extract_code_block_content(response_text)
    if not content:
        return None

    # Fast path when the whole output is already a number.
    for candidate in [content.strip(), content.strip().splitlines()[0].strip()]:
        if not candidate:
            continue
        normalized = _normalize_number_token(candidate)
        try:
            value = float(normalized)
            if math.isfinite(value):
                return value
        except ValueError:
            continue

    match = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", content)
    if not match:
        return None

    token = _normalize_number_token(match.group(0))
    try:
        value = float(token)
    except ValueError:
        return None

    return value if math.isfinite(value) else None
