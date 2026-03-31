from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


def format_value(value) -> str:
    """
    Format a dataframe cell value into a string representation
    """
    if pd.isna(value):
        return "MISSING"

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def row_to_feature_text(
    row: pd.Series,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
) -> str:
    """
    Convert one dataframe row into 'feature: value' lines,
    excluding the target column.
    """
    if feature_columns is None:
        columns = [col for col in row.index if col != target_column]
    else:
        columns = [col for col in feature_columns if col != target_column]

    lines = []
    for col in columns:
        lines.append(f"{col}: {format_value(row[col])}")

    return "\n".join(lines)


def build_imputation_prompt(
    row: pd.Series,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
) -> str:
    """
    Build a deterministic imputation prompt for one row.
    """
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    prompt = (
        "You are performing a missing value imputation task for tabular data.\n"
        f"Predict the missing numeric value of the target column '{target_column}' "
        "using the observed values of this record.\n"
        "Return only one numeric value.\n"
        "Do not return words, explanations, units, or extra text.\n\n"
        "Observed values:\n"
        f"{feature_text}\n\n"
        f"Target column: {target_column}\n"
        "Numeric imputed value:"
    )
    return prompt


def build_training_example(
    row: pd.Series,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
) -> dict:
    """
    Build one training example in a chat-style format.
    """
    if pd.isna(row[target_column]):
        raise ValueError(
            f"Target column '{target_column}' is missing in this row. "
            "Training examples require observed target values."
        )

    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    target_value = format_value(row[target_column])

    system_message = (
        "You are performing a missing value imputation task for tabular data. "
        "Given observed feature values, predict the missing target value. "
        "Return only the imputed value and no explanation."
    )

    user_message = (
        f"Observed values:\n{feature_text}\n\n"
        f"Target column: {target_column}\n"
        "Imputed value:"
    )

    assistant_message = target_value

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }

def build_inference_messages(
    row: pd.Series,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
) -> list[dict]:
    """
    Build inference messages in the same chat-style structure
    that is used for training examples.
    """
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    system_message = (
        "You are performing a missing value imputation task for tabular data. "
        "Given observed feature values, predict the missing target value. "
        "Return exactly one numeric value and no explanation."
    )

    user_message = (
        f"Observed values:\n{feature_text}\n\n"
        f"Target column: {target_column}\n"
        "Imputed value:"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_retry_messages(
    row: pd.Series,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
) -> list[dict]:
    """
    Stricter retry prompt if the first generation is empty 
    """
    feature_text = row_to_feature_text(
        row=row,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    system_message = (
        "You are performing numeric imputation for tabular data. "
        "You must answer with exactly one numeric value only. "
        "No words. No sentence. No explanation. No unit."
    )

    user_message = (
        f"Observed values:\n{feature_text}\n\n"
        f"Target column: {target_column}\n"
        "Answer with one number only:"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


@dataclass
class LLMImputerConfig:
    model_name: str
    target_column: str
    feature_columns: Optional[list[str]] = None


class LLMImputer:
    """
    Minimal scaffold for LLM-based imputation.

    This class currently focuses on prompt/example creation.
    Model loading and fine-tuning are added in the next step.
    """

    def __init__(self, config: LLMImputerConfig):
        self.config = config

    def build_prompt(self, row: pd.Series) -> str:
        return build_imputation_prompt(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
        )

    def build_training_example(self, row: pd.Series) -> dict:
        return build_training_example(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
        )

    def build_training_dataset(self, df: pd.DataFrame) -> list[dict]:
        """
        Build training examples only from rows where the target is observed.
        """
        df_obs = df[df[self.config.target_column].notna()].copy()
        return [self.build_training_example(row) for _, row in df_obs.iterrows()]

    def build_inference_messages(self, row: pd.Series) -> list[dict]:
        return build_inference_messages(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
        )

    def build_retry_messages(self, row: pd.Series) -> list[dict]:
        return build_retry_messages(
            row=row,
            target_column=self.config.target_column,
            feature_columns=self.config.feature_columns,
        )