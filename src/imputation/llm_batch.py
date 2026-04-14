from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import re
from typing import Optional

import pandas as pd
import torch


def model_name_to_file_token(model_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


@dataclass
class LLMBatchImputerConfig:
    model_name: str
    dataset_name: str
    target_column: str
    prompt_style_name: str = "BatchImputation"
    domain_hints: Optional[list[str]] = None


class LLMBatchImputer:
    """
    Batch-oriented prompt builder and parser for matrix-style LLM imputation.
    """

    def __init__(self, config: LLMBatchImputerConfig):
        self.config = config

    @staticmethod
    def apply_chat_template(messages, tokenizer) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(message["content"] for message in messages)

    @staticmethod
    def format_cell(value) -> str:
        if pd.isna(value):
            return "MISSING"
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def extract_code_block(text: str) -> str:
        match = re.search(r"```(?:csv)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def parse_batch_predictions(self, response_text: str) -> pd.DataFrame:
        """
        Parse model output to a dataframe with columns ['row_index', target_column].
        """
        content = self.extract_code_block(response_text)
        target_column = self.config.target_column

        for sep in [",", r"\s+"]:
            try:
                parsed = pd.read_csv(StringIO(content), sep=sep, engine="python")
                if parsed.empty:
                    continue

                if "row_index" not in parsed.columns and len(parsed.columns) >= 1:
                    parsed = parsed.rename(columns={parsed.columns[0]: "row_index"})

                if target_column not in parsed.columns and len(parsed.columns) >= 2:
                    parsed = parsed.rename(columns={parsed.columns[1]: target_column})

                if "row_index" not in parsed.columns or target_column not in parsed.columns:
                    continue

                parsed = parsed[["row_index", target_column]].copy()
                parsed["row_index"] = pd.to_numeric(parsed["row_index"], errors="coerce")
                parsed[target_column] = pd.to_numeric(parsed[target_column], errors="coerce")
                parsed = parsed.dropna(subset=["row_index"]).copy()
                parsed["row_index"] = parsed["row_index"].astype(int)
                parsed = parsed.drop_duplicates(subset=["row_index"], keep="last")
                return parsed
            except Exception:
                continue

        return pd.DataFrame(columns=["row_index", target_column])

    def build_background_knowledge(
        self,
        observed_df: pd.DataFrame,
        feature_columns: list[str],
        max_numeric_features: int = 8,
        max_categorical_features: int = 5,
    ) -> str:
        target_column = self.config.target_column

        lines = ["Dataset-specific background knowledge:"]
        target_obs = pd.to_numeric(observed_df[target_column], errors="coerce").dropna()
        if not target_obs.empty:
            lines.extend(
                [
                    f"- {target_column} observed min: {target_obs.min():.4f}",
                    f"- {target_column} observed median: {target_obs.median():.4f}",
                    f"- {target_column} observed max: {target_obs.max():.4f}",
                ]
            )

        numeric_features = [
            col for col in feature_columns if pd.api.types.is_numeric_dtype(observed_df[col])
        ]
        categorical_features = [col for col in feature_columns if col not in numeric_features]

        ranked_numeric = numeric_features
        if numeric_features and target_column in observed_df.columns:
            corr_df = observed_df[numeric_features + [target_column]].apply(
                pd.to_numeric,
                errors="coerce",
            )
            corr_series = corr_df.corr(numeric_only=True)[target_column].drop(
                labels=[target_column],
                errors="ignore",
            )
            corr_series = corr_series.abs().sort_values(ascending=False)
            ranked_numeric = corr_series.index.tolist()

        lines.append("- Numeric feature ranges (min/median/max):")
        for col in ranked_numeric[:max_numeric_features]:
            series = pd.to_numeric(observed_df[col], errors="coerce").dropna()
            if series.empty:
                continue
            lines.append(
                f"  - {col}: {series.min():.4f} / {series.median():.4f} / {series.max():.4f}"
            )

        lines.append("- Categorical feature priors (top values):")
        for col in categorical_features[:max_categorical_features]:
            top_vals = observed_df[col].dropna().astype(str).value_counts().head(3)
            if top_vals.empty:
                continue
            values_text = ", ".join([f"{name} ({count})" for name, count in top_vals.items()])
            lines.append(f"  - {col}: {values_text}")

        if self.config.domain_hints:
            lines.append("- Domain hints:")
            for hint in self.config.domain_hints:
                lines.append(f"  - {hint}")

        return "\n".join(lines)

    def build_batch_prompt(
        self,
        batch_df: pd.DataFrame,
        feature_columns: list[str],
        background_knowledge: str,
    ) -> str:
        target_column = self.config.target_column

        prompt_df = batch_df[["row_index"] + feature_columns + [target_column]].copy()
        prompt_df = prompt_df.apply(lambda col: col.map(self.format_cell))
        matrix_csv = prompt_df.to_csv(index=False)

        return (
            "System Role:\n"
            f"You are an expert data analyst. I am providing a subset of the {self.config.dataset_name} dataset.\n\n"
            "Task:\n"
            f"Use your knowledge of this dataset's statistical properties (feature ranges, distributions, and correlations) "
            f"to impute missing values in the target column '{target_column}'.\n\n"
            f"{background_knowledge}\n\n"
            "Constraint:\n"
            "- DO NOT execute Python code.\n"
            "- DO NOT provide conversational text.\n"
            "- DO NOT return any NaN, null, or '?'.\n"
            "- DO NOT change row_index values.\n"
            "- Return one imputed value per row_index.\n\n"
            "Input matrix (CSV):\n"
            f"{matrix_csv}\n"
            "Output format:\n"
            "Return only one markdown code block with CSV and exactly these columns:\n"
            f"row_index,{target_column}\n"
            "Example format:\n"
            "```csv\n"
            f"row_index,{target_column}\n"
            "10,123.45\n"
            "11,234.56\n"
            "```\n"
        )

    def build_batch_messages(
        self,
        batch_df: pd.DataFrame,
        feature_columns: list[str],
        background_knowledge: str,
    ) -> list[dict]:
        prompt = self.build_batch_prompt(
            batch_df=batch_df,
            feature_columns=feature_columns,
            background_knowledge=background_knowledge,
        )
        return [
            {
                "role": "system",
                "content": "You perform strict tabular imputation and must follow output format exactly.",
            },
            {"role": "user", "content": prompt},
        ]

    def generate_raw_answer(
        self,
        messages,
        tokenizer,
        model,
        max_new_tokens: int = 512,
    ) -> str:
        prompt_text = self.apply_chat_template(messages, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def fallback_totalcharges(row: pd.Series, totalcharges_median: float) -> tuple[float, str]:
        tenure = row.get("tenure")
        monthly = row.get("MonthlyCharges")

        if pd.notna(tenure) and pd.notna(monthly):
            try:
                return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
            except Exception:
                pass

        return float(totalcharges_median), "fallback_median"
