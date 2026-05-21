from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import nrmse, rmse
from src.imputation.machine_learning import MICEImputer, MissForestImputer
from src.imputation.statistical import MeanModeImputer, MedianModeImputer
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_RESULTS


def _build_dae_imputer(seed: int):
    from src.imputation.deep_learning import DenoisingAutoencoder

    return DenoisingAutoencoder(
        hidden_dims=(128, 64),
        epochs=200,
        batch_size=256,
        lr=1e-3,
        corruption_rate=0.2,
        dropout=0.0,
        verbose=False,
        seed=seed,
    )


DATASET_CONFIG = {
    "telco": {
        "dataset_name": "Telco",
        "full_path": DATA_RAW / "Telco-Customer-Churn_cleaned.csv",
        "target_column": "TotalCharges",
        "drop_columns": ["customerID", "Churn"],
        "numeric_only_features": True,
        "path_templates": {
            "MAR_TARGET": "MAR/telco_customer_churn_mar_totalcharges_tenure_{rate}pct.csv",
            "MNAR_TARGET": "MNAR/telco_customer_churn_mnar_totalcharges_{rate}pct.csv",
            "MCAR_TARGET": "MCAR/telco_customer_churn_mcar_totalcharges_{rate}pct.csv",
            "MCAR_GLOBAL": "MCAR/telco_customer_churn_mcar_{rate}pct.csv",
        },
    },
    "statlog": {
        "dataset_name": "German Statlog",
        "full_path": DATA_RAW / "german_statlog_numeric.csv",
        "target_column": "X2",
        "drop_columns": ["class"],
        "numeric_only_features": False,
        "path_templates": {
            "MAR_TARGET": "MAR/german_statlog_numeric_mar_duration_creditamount_{rate}pct.csv",
            "MNAR_TARGET": "MNAR/german_statlog_numeric_mnar_duration_{rate}pct.csv",
            "MCAR_TARGET": "MCAR/german_statlog_numeric_mcar_duration_{rate}pct.csv",
            "MCAR_GLOBAL": "MCAR/german_statlog_numeric_mcar_{rate}pct.csv",
        },
    },
    "creditcard": {
        "dataset_name": "German Credit Card",
        "full_path": DATA_RAW / "default_of_credit_card_clients.csv",
        "target_column": "BILL_AMT1",
        "drop_columns": ["ID", "default payment next month"],
        "numeric_only_features": False,
        "path_templates": {
            "MAR_TARGET": "MAR/german_credit_mar_billamt1_pay0_{rate}pct.csv",
            "MNAR_TARGET": "MNAR/german_credit_mnar_billamt1_{rate}pct.csv",
            "MCAR_TARGET": "MCAR/german_credit_mcar_billamt1_{rate}pct.csv",
            "MCAR_GLOBAL": "MCAR/german_credit_mcar_{rate}pct.csv",
        },
    },
}


METHODS = {
    "meanmode": "Mean/Mode",
    "medianmode": "Median/Mode",
    "mice": "MICE",
    "mice_post_mean": "MICE posterior mean (m=5)",
    "missforest": "MissForest",
    "dae": "DAE",
}


def _build_imputer(method_key, seed):
    if method_key == "meanmode":
        return MeanModeImputer()
    if method_key == "medianmode":
        return MedianModeImputer()
    if method_key == "mice":
        return MICEImputer(
            random_state=seed,
            sample_posterior=False,
            n_imputations=1,
        )
    if method_key == "mice_post_mean":
        return MICEImputer(
            random_state=seed,
            sample_posterior=True,
            n_imputations=5,
        )
    if method_key == "missforest":
        return MissForestImputer(random_state=seed)
    if method_key == "dae":
        return _build_dae_imputer(seed=seed)
    raise ValueError(f"Unknown classical method_key: {method_key}")


def _resolve_missing_path(dataset_key: str, scenario_family: str, rate_pct: int) -> Path:
    if dataset_key not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset_key '{dataset_key}'.")
    cfg = DATASET_CONFIG[dataset_key]
    template = cfg["path_templates"].get(scenario_family)
    if template is None:
        raise ValueError(
            f"Unknown scenario_family '{scenario_family}' for dataset_key '{dataset_key}'."
        )
    return DATA_PROCESSED / template.format(rate=int(rate_pct))


def _prepare_feature_frames(dataset_key, missing_path):
    cfg = DATASET_CONFIG[dataset_key]

    df_full = pd.read_csv(cfg["full_path"])
    df_missing = pd.read_csv(missing_path)

    drop_cols = list(cfg.get("drop_columns", []))
    if drop_cols:
        df_full = df_full.drop(columns=drop_cols, errors="ignore")
        df_missing = df_missing.drop(columns=drop_cols, errors="ignore")

    if cfg.get("numeric_only_features", False):
        numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        df_full = df_full[numeric_cols].copy()
        df_missing = df_missing[numeric_cols].copy()

    if list(df_full.columns) != list(df_missing.columns):
        raise ValueError(
            f"Column mismatch after preprocessing for dataset '{dataset_key}'. "
            f"full_cols={len(df_full.columns)}, missing_cols={len(df_missing.columns)}"
        )

    return df_full, df_missing


def _evaluate_target_only(df_full,df_missing, df_imputed, target_column):
    if target_column not in df_full.columns:
        raise ValueError(f"Target column '{target_column}' not found in full data.")
    if target_column not in df_missing.columns:
        raise ValueError(f"Target column '{target_column}' not found in missing data.")
    if target_column not in df_imputed.columns:
        raise ValueError(f"Target column '{target_column}' not found in imputed data.")

    df_true_target = pd.DataFrame(
        {target_column: pd.to_numeric(df_full[target_column], errors="coerce")}
    )
    df_missing_target = pd.DataFrame(
        {target_column: pd.to_numeric(df_missing[target_column], errors="coerce")}
    )
    df_imputed_target = pd.DataFrame(
        {target_column: pd.to_numeric(df_imputed[target_column], errors="coerce")}
    )

    mask = df_missing_target[target_column].isna() & df_true_target[target_column].notna()
    n_masked_target = int(mask.sum())
    if n_masked_target == 0:
        raise ValueError("No evaluable masked target cells found.")

    rmse_values = rmse(
        df_true=df_true_target,
        df_missing=df_missing_target,
        df_imputed=df_imputed_target,
        numeric_only=False,
    )
    nrmse_values = nrmse(
        df_true=df_true_target,
        df_missing=df_missing_target,
        df_imputed=df_imputed_target,
        norm="std",
    )

    mean_rmse = float(rmse_values.mean(skipna=True))
    mean_nrmse = float(nrmse_values.mean(skipna=True))
    return mean_rmse, mean_nrmse, n_masked_target


def _parse_csv_list(values: str | None) -> set[str] | None:
    if values is None:
        return None
    items = [part.strip() for part in values.split(",") if part.strip()]
    if not items:
        return None
    return set(items)


def _append_results(output_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = pd.NA
        new_df = new_df[existing_df.columns]
        combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    combined.to_csv(output_csv, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run classical imputation benchmark rows from scenario manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scenario_manifest_local_classical.csv"),
        help="Manifest CSV with classical rows.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DATA_RESULTS / "benchmark_local_classical_runs.csv",
        help="Output CSV for run-level metrics.",
    )
    parser.add_argument(
        "--dataset_keys",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. telco,statlog",
    )
    parser.add_argument(
        "--method_keys",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. meanmode,mice,mice_post_mean",
    )
    parser.add_argument(
        "--scenario_families",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. MAR_TARGET,MCAR_GLOBAL",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. 10,15",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated filter, e.g. 42,202",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of manifest rows to execute.",
    )
    parser.add_argument(
        "--skip_missing_files",
        action="store_true",
        help="Skip rows whose scenario CSV does not exist (status=missing_file).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already completed with status=success in output_csv.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show filtered rows and exit without execution.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on first error.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest = pd.read_csv(args.manifest)
    if "method_family" not in manifest.columns:
        raise ValueError("Manifest is missing required column 'method_family'.")

    work = manifest[manifest["method_family"] == "classical"].copy()
    if work.empty:
        raise ValueError("No classical rows found in manifest.")

    dataset_filter = _parse_csv_list(args.dataset_keys)
    method_filter = _parse_csv_list(args.method_keys)
    scenario_filter = _parse_csv_list(args.scenario_families)
    rate_filter = _parse_csv_list(args.rates)
    seed_filter = _parse_csv_list(args.seeds)

    if dataset_filter is not None:
        work = work[work["dataset_key"].astype(str).isin(dataset_filter)]
    if method_filter is not None:
        work = work[work["method_key"].astype(str).isin(method_filter)]
    if scenario_filter is not None:
        work = work[work["scenario_family"].astype(str).isin(scenario_filter)]
    if rate_filter is not None:
        work = work[work["rate_pct"].astype(str).isin(rate_filter)]
    if seed_filter is not None:
        work = work[work["seed"].astype(str).isin(seed_filter)]

    if work.empty:
        print("No rows left after filtering.")
        return

    done_run_ids: set[str] = set()
    if args.resume and args.output_csv.exists():
        existing = pd.read_csv(args.output_csv)
        if {"run_id", "status"}.issubset(existing.columns):
            done_run_ids = set(
                existing.loc[existing["status"] == "success", "run_id"].astype(str).tolist()
            )
            work = work[~work["run_id"].astype(str).isin(done_run_ids)]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        work = work.head(args.limit)

    print("=" * 80)
    print("CLASSICAL MANIFEST RUNNER")
    print("=" * 80)
    print(f"Manifest rows selected: {len(work)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Resume mode: {args.resume}")
    if args.resume:
        print(f"Previously completed run_ids found: {len(done_run_ids)}")

    if args.dry_run:
        cols = ["run_id", "dataset_key", "scenario_family", "rate_pct", "seed", "method_key"]
        print("\nDry-run preview:")
        print(work[cols].head(30).to_string(index=False))
        return

    result_rows: list[dict] = []
    total = len(work)
    started = time.perf_counter()

    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        run_id = str(row["run_id"])
        dataset_key = str(row["dataset_key"])
        scenario_family = str(row["scenario_family"])
        method_key = str(row["method_key"])
        rate_pct = int(row["rate_pct"])
        seed = int(row["seed"])
        target_column = str(row["target_column"])

        run_started = time.perf_counter()
        missing_path = _resolve_missing_path(
            dataset_key=dataset_key,
            scenario_family=scenario_family,
            rate_pct=rate_pct,
        )

        print(
            f"[{idx}/{total}] run_id={run_id} "
            f"{dataset_key} {scenario_family} {rate_pct}% seed={seed} method={method_key}"
        )

        if not missing_path.exists():
            message = f"Missing scenario file: {missing_path}"
            if args.skip_missing_files:
                result_rows.append(
                    {
                        "run_id": run_id,
                        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset_key": dataset_key,
                        "dataset_name": row.get("dataset_name", pd.NA),
                        "target_column": target_column,
                        "scenario_family": scenario_family,
                        "missingness_type": row.get("missingness_type", pd.NA),
                        "mcar_scope": row.get("mcar_scope", pd.NA),
                        "rate_pct": rate_pct,
                        "seed": seed,
                        "method_family": "classical",
                        "method_key": method_key,
                        "method_name": row.get("method_name", METHODS.get(method_key, method_key)),
                        "model_key": pd.NA,
                        "model_name": pd.NA,
                        "status": "missing_file",
                        "error_message": message,
                        "full_path": str(DATASET_CONFIG[dataset_key]["full_path"]),
                        "missing_path": str(missing_path),
                        "n_masked_target": pd.NA,
                        "mean_rmse": pd.NA,
                        "mean_nrmse": pd.NA,
                        "runtime_seconds": float(time.perf_counter() - run_started),
                    }
                )
                continue
            raise FileNotFoundError(message)

        try:
            df_full, df_missing = _prepare_feature_frames(
                dataset_key=dataset_key,
                missing_path=missing_path,
            )
            imputer = _build_imputer(method_key=method_key, seed=seed)
            df_imputed = imputer.fit_transform(df_missing.copy())

            mean_rmse, mean_nrmse, n_masked_target = _evaluate_target_only(
                df_full=df_full,
                df_missing=df_missing,
                df_imputed=df_imputed,
                target_column=target_column,
            )

            result_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_key": dataset_key,
                    "dataset_name": row.get("dataset_name", DATASET_CONFIG[dataset_key]["dataset_name"]),
                    "target_column": target_column,
                    "scenario_family": scenario_family,
                    "missingness_type": row.get("missingness_type", pd.NA),
                    "mcar_scope": row.get("mcar_scope", pd.NA),
                    "rate_pct": rate_pct,
                    "seed": seed,
                    "method_family": "classical",
                    "method_key": method_key,
                    "method_name": row.get("method_name", METHODS.get(method_key, method_key)),
                    "model_key": pd.NA,
                    "model_name": pd.NA,
                    "status": "success",
                    "error_message": pd.NA,
                    "full_path": str(DATASET_CONFIG[dataset_key]["full_path"]),
                    "missing_path": str(missing_path),
                    "n_masked_target": int(n_masked_target),
                    "mean_rmse": float(mean_rmse),
                    "mean_nrmse": float(mean_nrmse),
                    "runtime_seconds": float(time.perf_counter() - run_started),
                }
            )
        except Exception as exc:
            result_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_key": dataset_key,
                    "dataset_name": row.get("dataset_name", pd.NA),
                    "target_column": target_column,
                    "scenario_family": scenario_family,
                    "missingness_type": row.get("missingness_type", pd.NA),
                    "mcar_scope": row.get("mcar_scope", pd.NA),
                    "rate_pct": rate_pct,
                    "seed": seed,
                    "method_family": "classical",
                    "method_key": method_key,
                    "method_name": row.get("method_name", METHODS.get(method_key, method_key)),
                    "model_key": pd.NA,
                    "model_name": pd.NA,
                    "status": "error",
                    "error_message": str(exc),
                    "full_path": str(DATASET_CONFIG.get(dataset_key, {}).get("full_path", "")),
                    "missing_path": str(missing_path),
                    "n_masked_target": pd.NA,
                    "mean_rmse": pd.NA,
                    "mean_nrmse": pd.NA,
                    "runtime_seconds": float(time.perf_counter() - run_started),
                }
            )
            if args.fail_fast:
                _append_results(args.output_csv, result_rows)
                raise

    _append_results(args.output_csv, result_rows)

    elapsed = time.perf_counter() - started
    summary = pd.DataFrame(result_rows)["status"].value_counts(dropna=False).to_dict()
    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Rows processed now: {len(result_rows)}")
    print(f"Status counts: {summary}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Written to: {args.output_csv}")


if __name__ == "__main__":
    main()
