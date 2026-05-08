import argparse

from src.scripts.imputation_pipeline import (
    get_available_dataset_keys,
    get_available_method_keys,
    run_all_imputation_methods,
    run_imputation_method,
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run deduplicated imputation experiments by method and dataset filter.",
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


def _resolve_selection(values, available_values, label):
    normalized = [value.strip().lower() for value in values]
    if "all" in normalized:
        return available_values

    unknown = sorted(set(normalized) - set(available_values))
    if unknown:
        available = ", ".join(available_values)
        raise ValueError(f"Unknown {label}(s): {', '.join(unknown)}. Available: {available}, all")

    # keep user order, remove duplicates
    ordered = []
    seen = set()
    for value in normalized:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def main():
    parser = _build_parser()
    args = parser.parse_args()

    available_methods = get_available_method_keys()
    available_datasets = get_available_dataset_keys()

    if args.list:
        print("Methods:", ", ".join(available_methods))
        print("Datasets:", ", ".join(available_datasets))
        return

    method_keys = _resolve_selection(args.method, available_methods, "method key")
    dataset_keys = _resolve_selection(args.dataset, available_datasets, "dataset key")

    print("Selected methods:", ", ".join(method_keys))
    print("Selected datasets:", ", ".join(dataset_keys))

    if args.dry_run:
        print("Dry run only. No experiments executed.")
        return

    if len(method_keys) == 1:
        run_imputation_method(method_keys[0], dataset_keys=dataset_keys)
        return

    run_all_imputation_methods(dataset_keys=dataset_keys, method_keys=method_keys)


if __name__ == "__main__":
    main()
