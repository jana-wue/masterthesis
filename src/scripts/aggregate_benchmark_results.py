from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_RESULTS


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Create simple benchmark aggregation (mean/std) from run-level results."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=DATA_RESULTS / "benchmark_local_classical_runs.csv",
        help="Run-level input CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DATA_RESULTS / "benchmark_local_classical_aggregated.csv",
        help="Aggregated output CSV.",
    )
    return parser


def main() -> None:
    """Run the script entry point."""
    args = _build_parser().parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "status" not in df.columns:
        raise ValueError("Input CSV is missing required column 'status'.")

    success = df[df["status"] == "success"].copy()
    if success.empty:
        raise ValueError("No rows with status='success' found.")

    id_columns = [
        "dataset_key",
        "dataset_name",
        "target_column",
        "scenario_family",
        "missingness_type",
        "mcar_scope",
        "rate_pct",
        "method_family",
        "method_key",
        "method_name",
        "model_key",
        "model_name",
    ]

    group_cols = [col for col in id_columns if col in success.columns]
    if not group_cols:
        raise ValueError("No grouping columns found in input CSV.")

    agg = (
        success.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            n_masked_target_mean=("n_masked_target", "mean"),
            n_masked_target_std=("n_masked_target", "std"),
            mean_rmse_mean=("mean_rmse", "mean"),
            mean_rmse_std=("mean_rmse", "std"),
            mean_nrmse_mean=("mean_nrmse", "mean"),
            mean_nrmse_std=("mean_nrmse", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
        )
        .reset_index()
    )

    std_cols = [c for c in agg.columns if c.endswith("_std")]
    for col in std_cols:
        agg[col] = agg[col].fillna(0.0)

    agg = agg.sort_values(group_cols).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output_csv, index=False)

    print("=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Input rows: {len(df)}")
    print(f"Success rows used: {len(success)}")
    print(f"Groups written: {len(agg)}")
    print(f"Output CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
