from __future__ import annotations

import ast
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

    @staticmethod
    def _safe_eval_numeric_expression(expr: str) -> float:
        """
        Evaluate simple arithmetic expression safely.
        """
        allowed_bin_ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
        }
        allowed_unary_ops = {
            ast.UAdd: lambda a: a,
            ast.USub: lambda a: -a,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Num):
                return float(node.n)
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
                return allowed_unary_ops[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin_ops:
                return allowed_bin_ops[type(node.op)](_eval(node.left), _eval(node.right))
            raise ValueError("Unsupported expression")

        parsed = ast.parse(expr, mode="eval")
        return float(_eval(parsed))

    @classmethod
    def _coerce_numeric_or_expression(cls, value):
        if pd.isna(value):
            return pd.NA

        direct = pd.to_numeric(value, errors="coerce")
        if pd.notna(direct):
            return float(direct)

        text = str(value).strip()
        if not text:
            return pd.NA

        # Handle patterns like "60 * 74.85 + 60 * 10 = 4788"
        if "=" in text:
            rhs = text.split("=")[-1].strip()
            rhs_num = pd.to_numeric(rhs, errors="coerce")
            if pd.notna(rhs_num):
                return float(rhs_num)
            text = text.split("=")[0].strip()

        # Keep only characters needed for simple arithmetic expressions.
        sanitized = re.sub(r"[^0-9eE\+\-\*\/\.\(\)\s]", "", text)
        if not sanitized.strip():
            return pd.NA

        try:
            return cls._safe_eval_numeric_expression(sanitized)
        except Exception:
            # Last resort: first numeric token
            token = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if token:
                return float(token.group(0))
            return pd.NA

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
                parsed[target_column] = parsed[target_column].apply(self._coerce_numeric_or_expression)
                parsed = parsed.dropna(subset=["row_index"]).copy()
                parsed["row_index"] = parsed["row_index"].astype(int)
                parsed = parsed.drop_duplicates(subset=["row_index"], keep="last")
                return parsed
            except Exception:
                continue

        return pd.DataFrame(columns=["row_index", target_column])

    def parse_imputed_matrix(
        self,
        response_text: str,
        expected_columns: list[str],
        target_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Parse an imputed matrix block and align it to expected columns.
        """
        content = self.extract_code_block(response_text)

        for sep in [",", r"\s+"]:
            try:
                parsed = pd.read_csv(StringIO(content), sep=sep, engine="python")
                if parsed.empty:
                    continue

                parsed = parsed.dropna(axis=1, how="all")
                parsed = parsed.loc[
                    :,
                    [col for col in parsed.columns if not str(col).startswith("Unnamed")],
                ]

                if parsed.shape[1] == len(expected_columns) + 1:
                    parsed = parsed.iloc[:, 1:]

                if parsed.shape[1] > len(expected_columns):
                    if all(col in parsed.columns for col in expected_columns):
                        parsed = parsed[expected_columns]
                    else:
                        parsed = parsed.iloc[:, : len(expected_columns)]

                if parsed.shape[1] != len(expected_columns):
                    continue

                parsed = parsed.copy()
                parsed.columns = expected_columns

                if target_column is not None and target_column in parsed.columns:
                    parsed[target_column] = parsed[target_column].apply(
                        self._coerce_numeric_or_expression
                    )

                return parsed.reset_index(drop=True)
            except Exception:
                continue

        return pd.DataFrame(columns=expected_columns)

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
        # Keep arguments for API compatibility, but mirror paper prompt structure.
        _ = feature_columns, background_knowledge

        prompt_df = batch_df.copy().apply(lambda col: col.map(self.format_cell))
        headers_str = ", ".join(prompt_df.columns)
        string_missing = prompt_df.to_string()

        return f"""
You are an expert data analyst. I am providing a subset of the {self.config.dataset_name} Dataset.
Task: Use your knowledge of this specific dataset's statistical properties (feature ranges, class distributions, and correlations) to perform data imputation.
Constraint: DO NOT execute Python code. DO NOT provide any conversational text. Do NOT return any NaN or ? value.

The matrix below contains missing values. Impute them to be as consistent as possible with the original dataset.
Matrix:
{string_missing}

Output Format:
Return the complete imputed matrix inside a single Markdown code block.
Use CSV format (comma-separated values) with the original headers.
Expected Columns ({prompt_df.shape[1]}):
[{headers_str}]

Strict Rules:
1. Start directly with the code block: ```csv
2. End exactly with: ```
3. Ensure the exact same number of rows as the input.
4. No explanations, no introductory text, no "Here is the matrix".
5. Use commas as delimiters. Every row MUST have exactly {prompt_df.shape[1] - 1} commas.
""".strip()

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
        return [{"role": "user", "content": prompt}]

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
