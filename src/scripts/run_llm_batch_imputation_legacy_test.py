from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.paths import DATA_PROCESSED, DATA_RAW, DATA_RESULTS


DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT_STYLE_NAME = "BatchImputation"
DATASET_NAME = "Telco Customer Churn"
TARGET_COLUMN = "TotalCharges"
ID_COLUMN = "customerID"
DEFAULT_BATCH_SIZE = 40
DEFAULT_BATCH_COL_SIZE = 10


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
    Legacy batch-oriented prompt builder and parser for matrix-style LLM imputation.
    """

    def __init__(self, config: LLMBatchImputerConfig):
        self.config = config

    @staticmethod
    def apply_chat_template(messages: list[dict], tokenizer) -> str:
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

        if "=" in text:
            rhs = text.split("=")[-1].strip()
            rhs_num = pd.to_numeric(rhs, errors="coerce")
            if pd.notna(rhs_num):
                return float(rhs_num)
            text = text.split("=")[0].strip()

        sanitized = re.sub(r"[^0-9eE\+\-\*\/\.\(\)\s]", "", text)
        if not sanitized.strip():
            return pd.NA

        try:
            return cls._safe_eval_numeric_expression(sanitized)
        except Exception:
            token = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if token:
                return float(token.group(0))
            return pd.NA

    def parse_imputed_matrix(
        self,
        response_text: str,
        expected_columns: list[str],
        target_column: str | None = None,
    ) -> pd.DataFrame:
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
        messages: list[dict],
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


def _resolve_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {', '.join(mapping)}")
    return mapping[dtype_name]


def run_telco_batch_llm_legacy_test(
    model_name: str,
    n_examples: int | None,
    batch_size: int,
    batch_col_size: int,
    max_new_tokens: int | None,
    dtype_name: str,
    device_map: str,
    save_raw_output: bool,
    output_csv: str | None,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_col_size < 1:
        raise ValueError("batch_col_size must be >= 1")
    if n_examples is not None and n_examples < 1:
        raise ValueError("n_examples must be >= 1 or omitted for full set")

    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )

    feature_columns = [c for c in df_missing.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
    observed_rows = df_missing[df_missing[TARGET_COLUMN].notna()].copy()
    missing_rows = df_missing[df_missing[TARGET_COLUMN].isna()].copy()

    if n_examples is not None:
        missing_rows = missing_rows.head(n_examples)

    if observed_rows.empty:
        raise ValueError("No observed rows available for background knowledge.")
    if missing_rows.empty:
        raise ValueError(f"No missing rows found for target column '{TARGET_COLUMN}'.")

    imputer = LLMBatchImputer(
        LLMBatchImputerConfig(
            model_name=model_name,
            dataset_name=DATASET_NAME,
            target_column=TARGET_COLUMN,
            prompt_style_name=PROMPT_STYLE_NAME,
            domain_hints=[
                "In subscription billing data, TotalCharges is often close to tenure * MonthlyCharges.",
                "Prefer values that are consistent with observed feature distributions and correlations.",
            ],
        )
    )

    totalcharges_median = float(
        pd.to_numeric(observed_rows[TARGET_COLUMN], errors="coerce").dropna().median()
    )
    background_knowledge = imputer.build_background_knowledge(
        observed_df=observed_rows,
        feature_columns=feature_columns,
    )

    dtype = _resolve_dtype(dtype_name)

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"dtype: {dtype_name} | device_map: {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    print("\n" + "=" * 80)
    print("LEGACY BATCH PROMPT IMPUTATION TEST")
    print("=" * 80)
    print(
        f"Rows to impute: {len(missing_rows)} | Batch rows: {batch_size} | Batch cols: {batch_col_size}"
    )

    parsed_prediction_by_index: dict[int, float] = {}
    raw_output_by_index: dict[int, str] = {}

    n_rows = len(missing_rows)
    n_feature_cols = len(feature_columns)
    matrix_base = missing_rows[feature_columns + [TARGET_COLUMN]].copy()

    iter_batch = 0
    for row_start in range(0, n_rows, batch_size):
        row_end = min(row_start + batch_size, n_rows)
        actual_start = row_start
        if (row_end - row_start) < batch_size and n_rows >= batch_size:
            actual_start = row_end - batch_size

        rows_needed = row_end - row_start
        eval_indices = missing_rows.iloc[row_start:row_end].index.astype(int).tolist()

        for col_start in range(0, n_feature_cols, batch_col_size):
            col_end = min(col_start + batch_col_size, n_feature_cols)
            feature_slice = feature_columns[col_start:col_end]

            block_df = matrix_base.iloc[actual_start:row_end][feature_slice + [TARGET_COLUMN]].copy()
            block_df.insert(0, "row_index", missing_rows.iloc[actual_start:row_end].index.astype(int))
            expected_columns = block_df.columns.tolist()

            messages = imputer.build_batch_messages(
                batch_df=block_df,
                feature_columns=feature_slice,
                background_knowledge=background_knowledge,
            )
            dynamic_tokens = max_new_tokens if max_new_tokens is not None else max(256, batch_size * 20)
            raw_output = imputer.generate_raw_answer(
                messages=messages,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=dynamic_tokens,
            )
            parsed_block = imputer.parse_imputed_matrix(
                response_text=raw_output,
                expected_columns=expected_columns,
                target_column=TARGET_COLUMN,
            )
            clean_block = parsed_block.iloc[-rows_needed:, :].copy()

            iter_batch += 1
            parsed_count = int(clean_block[TARGET_COLUMN].notna().sum()) if not clean_block.empty else 0
            print(
                f"Batch {iter_batch}: rows {row_start}..{row_end - 1}, feature cols {col_start}..{col_end - 1} "
                f"| parsed {parsed_count}/{rows_needed}"
            )

            if clean_block.empty or clean_block.shape[0] != rows_needed:
                continue

            target_values = clean_block[TARGET_COLUMN].reset_index(drop=True)
            for pos, idx in enumerate(eval_indices):
                if pos >= len(target_values):
                    break
                parsed_pred = target_values.iloc[pos]
                if pd.notna(parsed_pred):
                    parsed_prediction_by_index[idx] = float(parsed_pred)
                    if save_raw_output:
                        raw_output_by_index[idx] = raw_output

    all_results: list[dict] = []

    for idx, row in missing_rows.iterrows():
        parsed_pred = parsed_prediction_by_index.get(int(idx), pd.NA)
        source = "model_batch"

        if pd.isna(parsed_pred):
            parsed_pred, source = imputer.fallback_totalcharges(
                row=row,
                totalcharges_median=totalcharges_median,
            )

        ground_truth = df_full.loc[idx, TARGET_COLUMN] if idx in df_full.index else pd.NA

        record = {
            "row_index": int(idx),
            "prediction": float(parsed_pred),
            "ground_truth": ground_truth,
            "source": source,
            "batch_id": pd.NA,
        }
        if save_raw_output:
            record["raw_output"] = raw_output_by_index.get(int(idx), "")

        all_results.append(record)

    results_df = pd.DataFrame(all_results)
    results_df["model_name"] = model_name
    results_df["prompt_style"] = PROMPT_STYLE_NAME
    results_df["batch_size"] = batch_size
    results_df["batch_col_size"] = batch_col_size
    results_df["n_examples_requested"] = n_examples if n_examples is not None else len(missing_rows)
    results_df["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results_df["leakage_guard_enabled"] = True
    results_df["reference_source"] = "df_missing_observed_rows_background_only"

    model_token = model_name_to_file_token(model_name)
    default_output = DATA_RESULTS / f"telco_{model_token}_{PROMPT_STYLE_NAME}_legacy_test_results.csv"
    out_path = Path(output_csv) if output_csv else default_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    valid_eval = results_df.copy()
    valid_eval["prediction"] = pd.to_numeric(valid_eval["prediction"], errors="coerce")
    valid_eval["ground_truth"] = pd.to_numeric(valid_eval["ground_truth"], errors="coerce")
    valid_eval = valid_eval.dropna(subset=["prediction", "ground_truth"])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Saved: {out_path}")
    print(f"Rows imputed: {len(results_df)}")
    print(f"Model predictions: {(results_df['source'] == 'model_batch').sum()}")
    print(f"Fallback predictions: {(results_df['source'] != 'model_batch').sum()}")

    if not valid_eval.empty:
        errors = valid_eval["ground_truth"] - valid_eval["prediction"]
        mean_rmse = float(((errors ** 2).mean()) ** 0.5)
        std_true = float(valid_eval["ground_truth"].std(ddof=0))
        mean_nrmse = float(mean_rmse / (std_true + 1e-8))
        print(f"mean_rmse: {mean_rmse:.6f}")
        print(f"mean_nrmse: {mean_nrmse:.6f}")
    else:
        print("No valid numeric rows for RMSE/NRMSE summary.")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy batch-imputation test (old BatchImputation logic) on Telco MAR data."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of missing rows to process. Omit for all missing rows.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-col-size", type=int, default=DEFAULT_BATCH_COL_SIZE)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for generation length per batch call.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--save-raw-output",
        action="store_true",
        help="Store the raw LLM output for each imputed row in the result CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_telco_batch_llm_legacy_test(
        model_name=args.model_name,
        n_examples=args.n_examples,
        batch_size=args.batch_size,
        batch_col_size=args.batch_col_size,
        max_new_tokens=args.max_new_tokens,
        dtype_name=args.dtype,
        device_map=args.device_map,
        save_raw_output=args.save_raw_output,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
