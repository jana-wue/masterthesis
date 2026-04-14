from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


def format_value(value) -> str:
    """
    Format dataframe cell value into string representation.
    """
    if pd.isna(value):
        return "MISSING"

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def row_to_feature_text(row: pd.Series, target_column: str, feature_columns = None,) -> str:
    """
    Convert one dataframe row into 'feature: value' lines,
    excluding target column.
    """
    if feature_columns is None:
        columns = [col for col in row.index if col != target_column]
    else:
        columns = [col for col in feature_columns if col != target_column]

    lines = []
    for col in columns:
        lines.append(f"{col}: {format_value(row[col])}")

    return "\n".join(lines)


def summarize_target_distribution( df: pd.DataFrame, target_column: str,) -> dict[str, float]:
    """
    Build robust summary for the target values.
    """
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


def _format_stats_lines(target_stats: Optional[dict[str, float]]) -> str:
    if not target_stats:
        return "Target distribution summary: not there."

    ordered_keys = ["min", "q1", "median", "q3", "max", "mean"]
    lines = ["Target distribution summary (observed rows):"]
    for key in ordered_keys:
        if key in target_stats:
            lines.append(f"- {key}: {format_value(target_stats[key])}")
    return "\n".join(lines)


def _build_system_message() -> str:
    return (
        "You are an expert in missing-value imputation for tabular data. "
        "Your task is to estimate one numeric target value from observed feature values. "
        "Use statistical consistency and feature relationships. "
        "Think silently and return only the final numeric answer."
    )


def _build_user_message(row: pd.Series, target_column: str, feature_columns = None, target_stats = None,
    domain_hints = None) -> str:
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    lines = [
        "Impute the missing target value for this record.",
        "",
        f"Target column: {target_column}",
        _format_stats_lines(target_stats),
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
            "Here are the output constraints:",
            "- Return exactly one number.",
            "- No words, no units, no commas, no explanation, no markdown.",
            "Imputed value:",
        ]
    )

    return "\n".join(lines)


def build_imputation_prompt(row: pd.Series, target_column: str, feature_columns = None, target_stats = None,
    domain_hints = None) -> str:
    """
    Build plain-text prompt for one row.
    """
    return (
        f"{_build_system_message()}\n\n"
        f"{_build_user_message(row, target_column, feature_columns, target_stats, domain_hints)}"
    )


def build_training_example(row: pd.Series, target_column: str, feature_columns = None, target_stats = None,
    domain_hints = None) -> dict:
    """
    Build one training example in a chat-style format.
    """
    if pd.isna(row[target_column]):
        raise ValueError(
            f"Target column '{target_column}' is missing in this row. "
            "Training examples require observed target values."
        )

    user_message = _build_user_message(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
        target_stats=target_stats,
        domain_hints=domain_hints
    )

    assistant_message = format_value(row[target_column])

    return {
        "messages": [
            {"role": "system", "content": _build_system_message()},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def _numeric_similarity(query_value: float, candidate_value: float, scale: float) -> float:
    return 1.0 / (1.0 + abs(query_value - candidate_value) / scale)


def _build_numeric_scales(reference_df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}

    for col in columns:
        series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        if series.empty:
            continue

        iqr = float(series.quantile(0.75) - series.quantile(0.25))
        std = float(series.std()) if len(series) > 1 else 0.0
        median_abs = abs(float(series.median()))
        scale = iqr if iqr > 0 else std
        if scale <= 0:
            scale = median_abs if median_abs > 0 else 1.0

        scales[col] = scale

    return scales


def select_similar_examples(query_row: pd.Series,reference_df: pd.DataFrame,target_column: str, feature_columns=None,
    n_examples: int = 3, exclude_indices = None) -> list[pd.Series]:
    """
    Select similar observed rows for few-shot.
    """
    if n_examples <= 0:
        return []

    observed_df = reference_df[reference_df[target_column].notna()].copy()
    if observed_df.empty:
        return []

    if feature_columns is None:
        columns = [col for col in query_row.index if col != target_column and col in observed_df.columns]
    else:
        columns = [col for col in feature_columns if col != target_column and col in observed_df.columns]

    if not columns:
        return []

    numeric_columns = [col for col in columns if is_numeric_dtype(observed_df[col])]
    scales = _build_numeric_scales(observed_df, numeric_columns)

    excluded = set(exclude_indices or [])

    scored_rows: list[tuple[float, pd.Series]] = []
    for idx, candidate in observed_df.iterrows():
        if idx in excluded:
            continue

        score = 0.0
        used = 0

        for col in columns:
            query_val = query_row.get(col)
            candidate_val = candidate.get(col)

            if pd.isna(query_val) or pd.isna(candidate_val):
                continue

            if col in numeric_columns:
                scale = scales.get(col, 1.0)
                score += _numeric_similarity(float(query_val), float(candidate_val), scale)
            else:
                score += 1.0 if str(query_val) == str(candidate_val) else 0.0

            used += 1

        if used > 0:
            scored_rows.append((score / used, candidate.copy()))

    if not scored_rows:
        return []

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    return [candidate for _, candidate in scored_rows[:n_examples]]


def build_inference_messages(row: pd.Series, target_column: str, feature_columns = None, target_stats = None,
    domain_hints = None, few_shot_examples = None) -> list[dict]:
    """
    Build inference messages in chat-style format.
    Optionally with few-shot.
    """
    messages = [{"role": "system", "content": _build_system_message()}]

    if few_shot_examples:
        for example_row in few_shot_examples:
            if pd.isna(example_row[target_column]):
                continue
            demo_user = _build_user_message(
                row=example_row,
                target_column=target_column,
                feature_columns=feature_columns,
                target_stats=target_stats,
                domain_hints=domain_hints,
            )
            demo_assistant = format_value(example_row[target_column])
            messages.append({"role": "user", "content": demo_user})
            messages.append({"role": "assistant", "content": demo_assistant})

    query_user = _build_user_message(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
        target_stats=target_stats,
        domain_hints=domain_hints,
    )
    messages.append({"role": "user", "content": query_user})
    return messages


def build_retry_messages(row: pd.Series, target_column: str, feature_columns = None, target_stats = None,
    domain_hints = None) -> list[dict]:
    """
    Stricter retry prompt if the first generation is empty or non-numeric.
    """
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    stats_lines = _format_stats_lines(target_stats)

    hints_lines = []
    if domain_hints:
        hints_lines = ["Domain hints:"] + [f"- {hint}" for hint in domain_hints]

    system_message = (
        "You perform numeric imputation for tabular data. "
        "Return exactly one numeric value only."
    )

    user_lines = [
        f"Target column: {target_column}",
        stats_lines,
        "Observed feature values:",
        feature_text,
    ]
    if hints_lines:
        user_lines.extend([""] + hints_lines)

    user_lines.extend(
        [
            "",
            "Strict output:",
            "One number only. No text.",
            "Answer:",
        ]
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


@dataclass
class LLMImputerConfig:
    model_name: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    target_stats: Optional[dict[str, float]] = None
    domain_hints: Optional[list[str]] = None
    few_shot_k: int = 0


class LLMImputer:
    """
    Prompt and training-example builder for LLM-based tabular imputation.
    """

    def __init__(self, config: LLMImputerConfig):
        self.config = config

    def fit_target_stats(self, df: pd.DataFrame) -> None:
        self.config.target_stats = summarize_target_distribution(
            df=df,
            target_column=self.config.target_column,
        )

    def build_prompt(self, row: pd.Series) -> str:
        return build_imputation_prompt(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
            target_stats=self.config.target_stats,
            domain_hints=self.config.domain_hints,
        )

    def build_training_example(self, row: pd.Series) -> dict:
        return build_training_example(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
            target_stats=self.config.target_stats,
            domain_hints=self.config.domain_hints,
        )

    def build_training_dataset(self, df: pd.DataFrame) -> list[dict]:
        """
        Build training examples from rows where the target is there.
        """
        df_obs = df[df[self.config.target_column].notna()]
        return [self.build_training_example(row) for _, row in df_obs.iterrows()]

    def select_few_shot_examples(self, row: pd.Series,reference_df: pd.DataFrame, n_examples=None,
        exclude_indices = None) -> list[pd.Series]:
        k = self.config.few_shot_k if n_examples is None else n_examples
        return select_similar_examples(
            query_row=row,
            reference_df=reference_df,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
            n_examples=k,
            exclude_indices=exclude_indices,
        )

    def build_inference_messages(self, row: pd.Series, few_shot_examples = None) -> list[dict]:
        return build_inference_messages(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
            target_stats=self.config.target_stats,
            domain_hints=self.config.domain_hints,
            few_shot_examples=few_shot_examples,
        )

    def build_retry_messages(self, row: pd.Series) -> list[dict]:
        return build_retry_messages(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
            target_stats=self.config.target_stats,
            domain_hints=self.config.domain_hints,
        )
