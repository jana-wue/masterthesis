import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_classical_manifest import main as run_classical_manifest_main


AVAILABLE_METHODS = ["meanmode", "medianmode", "mice", "missforest", "dae"]
AVAILABLE_DATASETS = ["telco", "statlog", "creditcard"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the final-scope classical benchmark via the manifest runner.",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        help="Method key(s): meanmode, medianmode, mice, missforest, dae or all.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Dataset key(s): telco, statlog, creditcard or all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available methods and datasets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print selection without executing experiments.",
    )
    return parser


def _resolve_selection(
    values: list[str],
    available_values: list[str],
    label: str,
) -> list[str]:
    """Resolve selection."""
    normalized = [value.strip().lower() for value in values]
    if "all" in normalized:
        return available_values

    unknown = sorted(set(normalized) - set(available_values))
    if unknown:
        available = ", ".join(available_values)
        raise ValueError(f"Unknown {label}(s): {', '.join(unknown)}. Available: {available}, all")

    # keep user order, remove duplicates
    ordered: list[str] = []
    seen = set()
    for value in normalized:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def main() -> None:
    """Run the script entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    available_methods = AVAILABLE_METHODS
    available_datasets = AVAILABLE_DATASETS

    if args.list:
        print("Methods:", ", ".join(available_methods))
        print("Datasets:", ", ".join(available_datasets))
        return

    method_keys = _resolve_selection(args.method, available_methods, "method key")
    dataset_keys = _resolve_selection(args.dataset, available_datasets, "dataset key")

    print("Selected methods:", ", ".join(method_keys))
    print("Selected datasets:", ", ".join(dataset_keys))

    if args.dry_run:
        forwarded_args = ["--dry_run"]
    else:
        forwarded_args = []

    if method_keys != available_methods:
        forwarded_args.extend(["--method_keys", ",".join(method_keys)])
    if dataset_keys != available_datasets:
        forwarded_args.extend(["--dataset_keys", ",".join(dataset_keys)])

    sys.argv = [sys.argv[0]] + forwarded_args
    run_classical_manifest_main()


if __name__ == "__main__":
    main()
