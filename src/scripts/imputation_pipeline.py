import numpy as np
import pandas as pd

from src.paths import DATA_RAW, DATA_PROCESSED
from src.evaluation.metrics import rmse, nrmse
from src.evaluation.results_logger import log_result
from src.imputation.machine_learning import MICEImputer, MissForestImputer
from src.imputation.statistical import MeanModeImputer, MedianModeImputer


def _build_dae_imputer():
    from src.imputation.deep_learning import DenoisingAutoencoder

    return DenoisingAutoencoder(
        hidden_dims=(128, 64),
        epochs=200,
        batch_size=256,
        lr=1e-3,
        corruption_rate=0.2,
        dropout=0.0,
        verbose=True,
    )


DATASET_CONFIGS = [
    {
        "dataset_key": "telco",
        "dataset": "Telco",
        "full_path": DATA_RAW / "Telco-Customer-Churn_cleaned.csv",
        "drop_columns": ["customerID"],
        "numeric_only": True,
        "nrmse_norm": "std",
        "scenarios": [
            {
                "missingness_type": "MAR",
                "missing_rate": "missing totalCharges -> tenure, 10%",
                "missing_path": DATA_PROCESSED / "MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "tenure, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/telco_customer_churn_mnar_tenure_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "10%",
                "missing_path": DATA_PROCESSED / "MCAR/telco_customer_churn_mcar_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "totalcharges, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "totalcharges, 10%",
                "missing_path": DATA_PROCESSED / "MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv",
            },
        ],
    },
    {
        "dataset_key": "statlog",
        "dataset": "German Statlog",
        "full_path": DATA_RAW / "german_statlog_numeric.csv",
        "drop_columns": [],
        "numeric_only": False,
        "nrmse_norm": "std",
        "scenarios": [
            {
                "missingness_type": "MAR",
                "missing_rate": "missing duration -> credit amount, 10%",
                "missing_path": DATA_PROCESSED / "MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "credit amount, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/german_statlog_numeric_mnar_creditamount_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "10%",
                "missing_path": DATA_PROCESSED / "MCAR/german_statlog_numeric_mcar_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "duration, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/german_statlog_numeric_mnar_duration_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "duration, 10%",
                "missing_path": DATA_PROCESSED / "MCAR/german_statlog_numeric_mcar_duration_10pct.csv",
            },
        ],
    },
    {
        "dataset_key": "creditcard",
        "dataset": "German Credit Card",
        "full_path": DATA_RAW / "default_of_credit_card_clients.csv",
        "drop_columns": [],
        "numeric_only": False,
        "nrmse_norm": "std",
        "scenarios": [
            {
                "missingness_type": "MAR",
                "missing_rate": "missing pay_0 -> limitbal, 10%",
                "missing_path": DATA_PROCESSED / "MAR/german_credit_mar_pay0_limitbal_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "Amount Credit, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/german_credit_mnar_amountcredit_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "10%",
                "missing_path": DATA_PROCESSED / "MCAR/german_credit_mcar_10pct.csv",
            },
            {
                "missingness_type": "MAR",
                "missing_rate": "missing billamt1 -> pay0, 10%",
                "missing_path": DATA_PROCESSED / "MAR/german_credit_mar_billamt1_pay0_10pct.csv",
            },
            {
                "missingness_type": "MNAR",
                "missing_rate": "billamt1, 10%",
                "missing_path": DATA_PROCESSED / "MNAR/german_credit_mnar_billamt1_10pct.csv",
            },
            {
                "missingness_type": "MCAR",
                "missing_rate": "billamt1, 10%",
                "missing_path": DATA_PROCESSED / "MCAR/german_credit_mcar_billamt1_10pct.csv",
            },
        ],
    },
]


IMPUTATION_METHODS = {
    "meanmode": {"runner_name": "Mean/Mode", "factory": MeanModeImputer},
    "medianmode": {"runner_name": "Median/Mode", "factory": MedianModeImputer},
    "mice": {
        "runner_name": "MICE",
        "factory": lambda: MICEImputer(sample_posterior=False, n_imputations=1),
    },
    "mice_post_mean": {
        "runner_name": "MICE posterior mean (m=5)",
        "factory": lambda: MICEImputer(sample_posterior=True, n_imputations=5),
    },
    "missforest": {"runner_name": "MissForest", "factory": MissForestImputer},
    "dae": {"runner_name": "DAE", "factory": _build_dae_imputer},
}


def get_available_method_keys():
    return list(IMPUTATION_METHODS.keys())


def get_available_dataset_keys(dataset_configs=None):
    configs = dataset_configs if dataset_configs is not None else DATASET_CONFIGS
    return [cfg["dataset_key"] for cfg in configs]


def get_dataset_configs(dataset_keys=None, dataset_configs=None):
    configs = dataset_configs if dataset_configs is not None else DATASET_CONFIGS
    if dataset_keys is None:
        return configs

    wanted_keys = [str(key).strip().lower() for key in dataset_keys]
    config_by_key = {cfg["dataset_key"].lower(): cfg for cfg in configs}
    unknown = sorted(set(wanted_keys) - set(config_by_key.keys()))
    if unknown:
        available = ", ".join(sorted(config_by_key.keys()))
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}. Available: {available}")

    return [config_by_key[key] for key in wanted_keys]


def _prepare_dataset_context(dataset):
    df_full = pd.read_csv(dataset["full_path"])
    if dataset.get("drop_columns"):
        df_full = df_full.drop(columns=dataset["drop_columns"], errors="ignore")

    context = dict(dataset)
    if dataset.get("numeric_only", False):
        numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        context["numeric_cols"] = numeric_cols
        context["df_full_eval"] = df_full[numeric_cols].copy()
    else:
        context["numeric_cols"] = None
        context["df_full_eval"] = df_full.copy()

    return context


def _prepare_missing_for_dataset(dataset, scenario):
    df_missing = pd.read_csv(scenario["missing_path"])
    if dataset.get("drop_columns"):
        df_missing = df_missing.drop(columns=dataset["drop_columns"], errors="ignore")

    if dataset.get("numeric_only", False):
        return df_missing[dataset["numeric_cols"]].copy()

    return df_missing


def run_imputation_experiment(imputer, dataset, scenario):
    df_full_eval = dataset["df_full_eval"]
    df_missing_eval = _prepare_missing_for_dataset(dataset, scenario)

    df_imputed = imputer.fit_transform(df_missing_eval)

    rmse_values = rmse(df_full_eval, df_missing_eval, df_imputed)
    nrmse_values = nrmse(
        df_full_eval,
        df_missing_eval,
        df_imputed,
        norm=dataset.get("nrmse_norm", "std"),
    )

    mean_rmse = float(rmse_values.mean(skipna=True))
    mean_nrmse = float(nrmse_values.mean(skipna=True))

    print(
        f"{dataset['dataset']} | {scenario['missingness_type']} | {scenario['missing_rate']} | "
        f"RMSE={mean_rmse:.6f} | NRMSE={mean_nrmse:.6f}"
    )

    log_result(
        dataset=dataset["dataset"],
        missingness_type=scenario["missingness_type"],
        missing_rate=scenario["missing_rate"],
        imputation_method=imputer.name,
        mean_rmse=mean_rmse,
        mean_nrmse=mean_nrmse,
    )

    return {
        "dataset": dataset["dataset"],
        "missingness_type": scenario["missingness_type"],
        "missing_rate": scenario["missing_rate"],
        "imputation_method": imputer.name,
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse,
    }


def run_imputation_method(method_key, dataset_configs=None, dataset_keys=None):
    if method_key not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown method_key: {method_key}")

    method = IMPUTATION_METHODS[method_key]
    configs = get_dataset_configs(dataset_keys=dataset_keys, dataset_configs=dataset_configs)
    results = []

    print(f"\\n=== Running {method['runner_name']} across all datasets/scenarios ===")
    for dataset_cfg in configs:
        dataset_ctx = _prepare_dataset_context(dataset_cfg)
        for scenario in dataset_ctx["scenarios"]:
            imputer = method["factory"]()
            result = run_imputation_experiment(imputer, dataset_ctx, scenario)
            results.append(result)

    return results


def run_all_imputation_methods(dataset_configs=None, dataset_keys=None, method_keys=None):
    selected_method_keys = method_keys if method_keys is not None else get_available_method_keys()
    unknown_method_keys = [key for key in selected_method_keys if key not in IMPUTATION_METHODS]
    if unknown_method_keys:
        available = ", ".join(sorted(IMPUTATION_METHODS.keys()))
        raise ValueError(
            f"Unknown method key(s): {', '.join(unknown_method_keys)}. Available: {available}"
        )

    all_results = {}
    for method_key in selected_method_keys:
        all_results[method_key] = run_imputation_method(
            method_key,
            dataset_configs=dataset_configs,
            dataset_keys=dataset_keys,
        )
    return all_results
