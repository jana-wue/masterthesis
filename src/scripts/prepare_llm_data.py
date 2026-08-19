from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.data.helper_dataprocessing import make_numeric_columns_numeric
from src.imputation.llm import LLMImputer, LLMImputerConfig
from src.paths import DATA_PROCESSED


DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MECHANISMS = ("MAR", "MCAR", "MNAR")


class ChatMessage(TypedDict):
    """Represent one chat-formatted training message."""

    role: str
    content: str


class TrainingRecord(TypedDict):
    """Represent one JSONL finetuning training example."""

    messages: list[ChatMessage]


class PrepareManifestRow(TypedDict):
    """Represent one preparation-manifest entry."""

    input_csv: str
    mechanism: str
    target_column: str | None
    status: str
    reason: str | None
    output_jsonl: str | None
    n_rows: int
    n_examples: int
    n_missing_target: int
    n_observed_target: int


def save_jsonl(records: list[TrainingRecord], output_path: Path) -> None:
    """Save JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_file_token(value: object) -> str:
    """Convert a value into a file-safe token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token or "unknown"


def _is_id_like_name(column_name: str) -> bool:
    """Handle is id like name."""
    name = column_name.strip().lower()
    return (
        name == "id"
        or name.endswith("id")
        or name.startswith("id_")
        or "_id" in name
    )


def detect_id_columns(df: pd.DataFrame, explicit_id_columns: list[str]) -> set[str]:
    """Detect id columns."""
    explicit = {col for col in explicit_id_columns if col in df.columns}
    inferred = {col for col in df.columns if _is_id_like_name(col)}
    return explicit | inferred


def resolve_target_columns(
    df: pd.DataFrame,
    id_columns: set[str],
    target_mode: str,
    explicit_targets: list[str] | None,
) -> list[str]:
    """Resolve target columns."""
    missing_counts = df.isna().sum()
    numeric_df = make_numeric_columns_numeric(df, exclude=list(id_columns))
    numeric_columns = set(numeric_df.select_dtypes(include=["number"]).columns)

    eligible = [
        col
        for col in df.columns
        if col in numeric_columns and col not in id_columns and missing_counts.get(col, 0) > 0
    ]

    if target_mode == "explicit":
        if not explicit_targets:
            raise ValueError("target_mode='explicit' requires --target_columns.")
        explicit_valid = [
            col
            for col in explicit_targets
            if col in df.columns and col in numeric_columns and col not in id_columns
        ]
        return explicit_valid

    if not eligible:
        return []

    if target_mode == "all_missing_numeric":
        return sorted(eligible)

    # primary_missing: take numeric target with most missing cells
    primary = max(eligible, key=lambda col: int(missing_counts[col]))
    return [primary]


def build_output_path(
    output_root: Path,
    mechanism: str,
    input_path: Path,
    target_column: str,
) -> Path:
    """Build output path."""
    return (
        output_root
        / mechanism
        / f"{input_path.stem}__target_{to_file_token(target_column)}_train.jsonl"
    )


def build_training_jsonl_for_target(
    df: pd.DataFrame,
    target_column: str,
    id_columns: set[str],
    model_name: str,
) -> list[TrainingRecord]:
    """Build training JSONL for target."""
    feature_columns = [col for col in df.columns if col not in {target_column, *id_columns}]

    imputer = LLMImputer(
        LLMImputerConfig(
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
        )
    )
    imputer.fit_target_stats(df)
    return imputer.build_training_dataset(df)


def process_input_file(
    input_path: Path,
    output_root: Path,
    model_name: str,
    target_mode: str,
    explicit_targets: list[str] | None,
    explicit_id_columns: list[str],
    dry_run: bool,
    print_first_example: bool,
) -> list[PrepareManifestRow]:
    """Process input file."""
    df_raw = pd.read_csv(input_path)
    mechanism = input_path.parent.name
    id_columns = detect_id_columns(df_raw, explicit_id_columns)
    df = make_numeric_columns_numeric(df_raw, exclude=list(id_columns))

    target_columns = resolve_target_columns(
        df=df,
        id_columns=id_columns,
        target_mode=target_mode,
        explicit_targets=explicit_targets,
    )

    rows: list[PrepareManifestRow] = []
    if not target_columns:
        rows.append(
            {
                "input_csv": str(input_path),
                "mechanism": mechanism,
                "target_column": None,
                "status": "skipped",
                "reason": "no_numeric_missing_target_found",
                "output_jsonl": None,
                "n_rows": int(len(df)),
                "n_examples": 0,
            }
        )
        return rows

    for target_column in target_columns:
        training_examples = build_training_jsonl_for_target(
            df=df,
            target_column=target_column,
            id_columns=id_columns,
            model_name=model_name,
        )
        output_path = build_output_path(
            output_root=output_root,
            mechanism=mechanism,
            input_path=input_path,
            target_column=target_column,
        )

        if not dry_run:
            save_jsonl(training_examples, output_path)

        row = {
            "input_csv": str(input_path),
            "mechanism": mechanism,
            "target_column": target_column,
            "status": "prepared",
            "reason": None,
            "output_jsonl": str(output_path),
            "n_rows": int(len(df)),
            "n_missing_target": int(df[target_column].isna().sum()),
            "n_observed_target": int(df[target_column].notna().sum()),
            "n_examples": int(len(training_examples)),
        }
        rows.append(row)

        print("=" * 80)
        print("LLM TRAINING DATA PREP")
        print("=" * 80)
        print(f"Input file   : {input_path}")
        print(f"Mechanism    : {mechanism}")
        print(f"Target column: {target_column}")
        print(f"Examples     : {len(training_examples)}")
        if dry_run:
            print("Dry-run      : output NOT written")
        else:
            print(f"Output file  : {output_path}")

        if print_first_example and training_examples:
            print("\nFirst example:")
            print(json.dumps(training_examples[0], indent=2, ensure_ascii=False))

    return rows


def discover_input_files(
    input_root: Path,
    mechanisms: list[str],
    include_glob: str,
) -> list[Path]:
    """Discover input files."""
    paths: list[Path] = []
    for mechanism in mechanisms:
        mechanism_dir = input_root / mechanism
        if not mechanism_dir.exists():
            continue
        paths.extend(sorted(mechanism_dir.glob(include_glob)))
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LLM finetuning JSONL data from processed missingness CSV files. "
            "Works across datasets and missingness mechanisms."
        )
    )
    parser.add_argument("--input_root", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--output_root", type=Path, default=DATA_PROCESSED / "llm")
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(DEFAULT_MECHANISMS),
        help="Subdirectories under input_root, e.g. MAR MCAR MNAR",
    )
    parser.add_argument("--include_glob", type=str, default="*.csv")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["primary_missing", "all_missing_numeric", "explicit"],
        default="primary_missing",
        help=(
            "primary_missing: one numeric target per file (most missing cells). "
            "all_missing_numeric: one JSONL per numeric missing column. "
            "explicit: only columns from --target_columns."
        ),
    )
    parser.add_argument(
        "--target_columns",
        nargs="*",
        default=None,
        help="Used when target_mode=explicit. Columns not present are skipped.",
    )
    parser.add_argument(
        "--id_columns",
        nargs="*",
        default=["customerID", "ID"],
        help="Columns to always exclude from features/targets.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_first_example", action="store_true")
    return parser.parse_args()


def prepare_telco_totalcharges_mar() -> None:
    # Backward-compatible helper for the original one-off use case.
    """Prepare Telco totalcharges MAR."""
    input_path = DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    process_input_file(
        input_path=input_path,
        output_root=DATA_PROCESSED / "llm",
        model_name=DEFAULT_MODEL_NAME,
        target_mode="explicit",
        explicit_targets=["TotalCharges"],
        explicit_id_columns=["customerID", "ID"],
        dry_run=False,
        print_first_example=True,
    )


def main() -> None:
    """Run the script entry point."""
    args = parse_args()
    input_files = discover_input_files(
        input_root=args.input_root,
        mechanisms=args.mechanisms,
        include_glob=args.include_glob,
    )

    if not input_files:
        raise FileNotFoundError(
            f"No input CSV files found in {args.input_root} for mechanisms={args.mechanisms} "
            f"and include_glob='{args.include_glob}'."
        )

    all_rows: list[dict] = []
    for input_path in input_files:
        all_rows.extend(
            process_input_file(
                input_path=input_path,
                output_root=args.output_root,
                model_name=args.model_name,
                target_mode=args.target_mode,
                explicit_targets=args.target_columns,
                explicit_id_columns=args.id_columns,
                dry_run=args.dry_run,
                print_first_example=args.print_first_example,
            )
        )

    manifest_df = pd.DataFrame(all_rows)
    summary = manifest_df["status"].value_counts(dropna=False).to_dict()

    print("\n" + "=" * 80)
    print("PREP SUMMARY")
    print("=" * 80)
    print(f"Input files processed: {len(input_files)}")
    print(f"Rows in manifest     : {len(manifest_df)}")
    print(f"Status counts        : {summary}")

    if not args.dry_run:
        manifest_path = args.output_root / "prepare_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Manifest written     : {manifest_path}")

if __name__ == "__main__":
    main()
