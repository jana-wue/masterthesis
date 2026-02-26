import pandas as pd
from pathlib import Path


RESULTS_PATH = Path("data/results/imputation_results_nrsme.csv")


def log_result(dataset,
               missingness_type,
               missing_rate,
               imputation_method,
               mean_rmse,
               mean_nrmse):
    """
    Append new experiment result to CSV.

    Args:
        dataset: i.e. "Telco"
        missingness_type: "MCAR", "MAR", "MNAR"
        missing_rate: i.e. 0.1
        imputation_method: i.e. "Mean/Mode"
        mean_rmse: avg. RMSE for numeric cols
    """

    if not RESULTS_PATH.exists():
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        df_init = pd.DataFrame(columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
            "mean_nrmse"
        ])

        df_init.to_csv(RESULTS_PATH, index=False)

    df_results = pd.read_csv(RESULTS_PATH)

    new_row = {
        "dataset": dataset,
        "missingness_type": missingness_type,
        "missing_rate": missing_rate,
        "imputation_method": imputation_method,
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse

    }

    df_results = pd.concat(
        [df_results, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df_results.to_csv(RESULTS_PATH, index=False)

    print(f"Result saved for {dataset} - {missingness_type} - {imputation_method}")
