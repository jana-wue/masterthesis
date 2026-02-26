from src.imputation.machine_learning import MissForestImputer
from src.evaluation.metrics import rmse, nrmse
from src.data.helper_dataprocessing import *
from src.evaluation.results_logger import log_result


def run_missforest():
    # Telco
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing_MAR = pd.read_csv(DATA_PROCESSED / "MAR/telco_customer_churn_mar_totalcharges_tenure_10pct.csv")
    df_missing_MNAR = pd.read_csv(DATA_PROCESSED / "MNAR/telco_customer_churn_mnar_tenure_10pct.csv")
    df_missing_MCAR = pd.read_csv(DATA_PROCESSED / "MCAR/telco_customer_churn_mcar_10pct.csv")
    df_missing_MNAR_tc = pd.read_csv(DATA_PROCESSED / "MNAR/telco_customer_churn_mnar_totalcharges_10pct.csv")
    df_missing_MCAR_tc = pd.read_csv(DATA_PROCESSED / "MCAR/telco_customer_churn_mcar_totalcharges_10pct.csv")

    # drop ID
    df_full = df_full.drop(columns=["customerID"])
    df_missing_MAR = df_missing_MAR.drop(columns=["customerID"])
    df_missing_MNAR = df_missing_MNAR.drop(columns=["customerID"])
    df_missing_MCAR = df_missing_MCAR.drop(columns=["customerID"])
    df_missing_MNAR_tc = df_missing_MNAR_tc.drop(columns=["customerID"])
    df_missing_MCAR_tc = df_missing_MCAR_tc.drop(columns=["customerID"])

    num_cols = df_full.select_dtypes(include=[np.number]).columns

    df_full_num = df_full[num_cols]
    df_missing_MAR_num = df_missing_MAR[num_cols]
    df_missing_MNAR_num = df_missing_MNAR[num_cols]
    df_missing_MCAR_num = df_missing_MCAR[num_cols]
    df_missing_MNAR_tc_num = df_missing_MNAR_tc[num_cols]
    df_missing_MCAR_tc_num = df_missing_MCAR_tc[num_cols]

    imputer = MissForestImputer()

    df_imputed_MAR = imputer.fit_transform(df_missing_MAR_num)
    df_imputed_MNAR = imputer.fit_transform(df_missing_MNAR_num)
    df_imputed_MCAR = imputer.fit_transform(df_missing_MCAR_num)
    df_imputed_MNAR_tc = imputer.fit_transform(df_missing_MNAR_tc_num)
    df_imputed_MCAR_tc = imputer.fit_transform(df_missing_MCAR_tc_num)

    rmse_values_MAR = rmse(df_full_num, df_missing_MAR_num, df_imputed_MAR)
    rmse_values_MNAR = rmse(df_full_num, df_missing_MNAR_num, df_imputed_MNAR)
    rmse_values_MCAR = rmse(df_full_num, df_missing_MCAR_num, df_imputed_MCAR)
    rmse_values_MNAR_tc = rmse(df_full_num, df_missing_MNAR_tc_num, df_imputed_MNAR_tc)
    rmse_values_MCAR_tc = rmse(df_full_num, df_missing_MCAR_tc_num, df_imputed_MCAR_tc)

    nrmse_values_MAR = nrmse(df_full_num, df_missing_MAR_num, df_imputed_MAR, norm="std")
    nrmse_values_MNAR = nrmse(df_full_num, df_missing_MNAR_num, df_imputed_MNAR, norm="std")
    nrmse_values_MCAR = nrmse(df_full_num, df_missing_MCAR_num, df_imputed_MCAR, norm="std")
    nrmse_values_MNAR_tc = nrmse(df_full_num, df_missing_MNAR_tc_num, df_imputed_MNAR_tc, norm="std")
    nrmse_values_MCAR_tc = nrmse(df_full_num, df_missing_MCAR_tc_num, df_imputed_MCAR_tc, norm="std")

    print("MAR:", rmse_values_MAR.sort_values())
    mean_rmse_MAR = rmse_values_MAR.mean(skipna=True)
    mean_nrmse_MAR = nrmse_values_MAR.mean(skipna=True)
    print("Mean RMSE MAR:", mean_rmse_MAR)

    log_result(
        dataset="Telco",
        missingness_type="MAR",
        missing_rate="missing totalCharges -> tenure, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MAR,
        mean_nrmse=mean_nrmse_MAR
    )

    print("MNAR:", rmse_values_MNAR.sort_values())
    mean_rmse_MNAR = rmse_values_MNAR.mean(skipna=True)
    mean_nrmse_MNAR = nrmse_values_MNAR.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR)

    log_result(
        dataset="Telco",
        missingness_type="MNAR",
        missing_rate="tenure, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR,
        mean_nrmse=mean_nrmse_MNAR
    )

    print("MCAR:", rmse_values_MCAR.sort_values())
    mean_rmse_MCAR = rmse_values_MCAR.mean(skipna=True)
    mean_nrmse_MCAR = nrmse_values_MCAR.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR)

    log_result(
        dataset="Telco",
        missingness_type="MCAR",
        missing_rate="10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR,
        mean_nrmse=mean_nrmse_MCAR
    )

    print("MNAR:", rmse_values_MNAR_tc.sort_values())
    mean_rmse_MNAR_tc = rmse_values_MNAR_tc.mean(skipna=True)
    mean_nrmse_MNAR_tc = nrmse_values_MNAR_tc.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR_tc)

    log_result(
        dataset="Telco",
        missingness_type="MNAR",
        missing_rate="totalcharges, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR_tc,
        mean_nrmse=mean_nrmse_MNAR_tc
    )

    print("MCAR:", rmse_values_MCAR_tc.sort_values())
    mean_rmse_MCAR_tc = rmse_values_MCAR_tc.mean(skipna=True)
    mean_nrmse_MCAR_tc = nrmse_values_MCAR_tc.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR_tc)

    log_result(
        dataset="Telco",
        missingness_type="MCAR",
        missing_rate="totalcharges, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR_tc,
        mean_nrmse=mean_nrmse_MCAR_tc
    )

    # Statlog
    df_full = pd.read_csv(DATA_RAW / "german_statlog_numeric.csv")
    df_missing_MAR = pd.read_csv(DATA_PROCESSED / "MAR/german_statlog_numeric_mar_duration_creditamount_10pct.csv")
    df_missing_MNAR = pd.read_csv(DATA_PROCESSED / "MNAR/german_statlog_numeric_mnar_creditamount_10pct.csv")
    df_missing_MCAR = pd.read_csv(DATA_PROCESSED / "MCAR/german_statlog_numeric_mcar_10pct.csv")
    df_missing_MNAR_d = pd.read_csv(DATA_PROCESSED / "MNAR/german_statlog_numeric_mnar_duration_10pct.csv")
    df_missing_MCAR_d = pd.read_csv(DATA_PROCESSED / "MCAR/german_statlog_numeric_mcar_duration_10pct.csv")

    imputer = MissForestImputer()

    df_imputed_MAR = imputer.fit_transform(df_missing_MAR)
    df_imputed_MNAR = imputer.fit_transform(df_missing_MNAR)
    df_imputed_MCAR = imputer.fit_transform(df_missing_MCAR)
    df_imputed_MNAR_d = imputer.fit_transform(df_missing_MNAR_d)
    df_imputed_MCAR_d = imputer.fit_transform(df_missing_MCAR_d)

    rmse_values_MAR = rmse(df_full, df_missing_MAR, df_imputed_MAR)
    rmse_values_MNAR = rmse(df_full, df_missing_MNAR, df_imputed_MNAR)
    rmse_values_MCAR = rmse(df_full, df_missing_MCAR, df_imputed_MCAR)
    rmse_values_MNAR_d = rmse(df_full, df_missing_MNAR_d, df_imputed_MNAR_d)
    rmse_values_MCAR_d = rmse(df_full, df_missing_MCAR_d, df_imputed_MCAR_d)

    nrmse_values_MAR = nrmse(df_full, df_missing_MAR, df_imputed_MAR)
    nrmse_values_MNAR = nrmse(df_full, df_missing_MNAR, df_imputed_MNAR)
    nrmse_values_MCAR = nrmse(df_full, df_missing_MCAR, df_imputed_MCAR)
    nrmse_values_MNAR_d = nrmse(df_full, df_missing_MNAR_d, df_imputed_MNAR_d)
    nrmse_values_MCAR_d = nrmse(df_full, df_missing_MCAR_d, df_imputed_MCAR_d)

    print("MAR:", rmse_values_MAR.sort_values())
    mean_rmse_MAR = rmse_values_MAR.mean(skipna=True)
    mean_nrmse_MAR = nrmse_values_MAR.mean(skipna=True)
    print("Mean RMSE MAR:", mean_rmse_MAR)

    log_result(
        dataset="German Statlog",
        missingness_type="MAR",
        missing_rate="missing duration -> credit amount, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MAR,
        mean_nrmse=mean_nrmse_MAR
    )

    print("MNAR:", rmse_values_MNAR.sort_values())
    mean_rmse_MNAR = rmse_values_MNAR.mean(skipna=True)
    mean_nrmse_MNAR = nrmse_values_MNAR.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR)

    log_result(
        dataset="German Statlog",
        missingness_type="MNAR",
        missing_rate="credit amount, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR,
        mean_nrmse=mean_nrmse_MNAR
    )

    print("MCAR:", rmse_values_MCAR.sort_values())
    mean_rmse_MCAR = rmse_values_MCAR.mean(skipna=True)
    mean_nrmse_MCAR = nrmse_values_MCAR.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR)

    log_result(
        dataset="German Statlog",
        missingness_type="MCAR",
        missing_rate="10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR,
        mean_nrmse=mean_nrmse_MCAR
    )

    print("MNAR:", rmse_values_MNAR_d.sort_values())
    mean_rmse_MNAR_d = rmse_values_MNAR_d.mean(skipna=True)
    mean_nrmse_MNAR_d = nrmse_values_MNAR_d.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR_d)

    log_result(
        dataset="German Statlog",
        missingness_type="MNAR",
        missing_rate="duration, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR_d,
        mean_nrmse=mean_nrmse_MNAR_d
    )

    print("MCAR:", rmse_values_MCAR_d.sort_values())
    mean_rmse_MCAR_d = rmse_values_MCAR_d.mean(skipna=True)
    mean_nrmse_MCAR_d = nrmse_values_MCAR_d.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR_d)

    log_result(
        dataset="German Statlog",
        missingness_type="MCAR",
        missing_rate="duration, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR_d,
        mean_nrmse=mean_nrmse_MCAR_d
    )
    # Credit Card
    df_full = pd.read_csv(DATA_RAW / "default_of_credit_card_clients.csv")
    df_missing_MAR = pd.read_csv(DATA_PROCESSED / "MAR/german_credit_mar_pay0_limitbal_10pct.csv")
    df_missing_MNAR = pd.read_csv(DATA_PROCESSED / "MNAR/german_credit_mnar_amountcredit_10pct.csv")
    df_missing_MCAR = pd.read_csv(DATA_PROCESSED / "MCAR/german_credit_mcar_10pct.csv")
    df_missing_MAR_b = pd.read_csv(DATA_PROCESSED / "MAR/german_credit_mar_billamt1_pay0_10pct.csv")
    df_missing_MNAR_b = pd.read_csv(DATA_PROCESSED / "MNAR/german_credit_mnar_billamt1_10pct.csv")
    df_missing_MCAR_b = pd.read_csv(DATA_PROCESSED / "MCAR/german_credit_mcar_billamt1_10pct.csv")

    imputer = MissForestImputer()

    df_imputed_MAR = imputer.fit_transform(df_missing_MAR)
    df_imputed_MNAR = imputer.fit_transform(df_missing_MNAR)
    df_imputed_MCAR = imputer.fit_transform(df_missing_MCAR)
    df_imputed_MAR_b = imputer.fit_transform(df_missing_MAR_b)
    df_imputed_MNAR_b = imputer.fit_transform(df_missing_MNAR_b)
    df_imputed_MCAR_b = imputer.fit_transform(df_missing_MCAR_b)

    rmse_values_MAR = rmse(df_full, df_missing_MAR, df_imputed_MAR)
    rmse_values_MNAR = rmse(df_full, df_missing_MNAR, df_imputed_MNAR)
    rmse_values_MCAR = rmse(df_full, df_missing_MCAR, df_imputed_MCAR)
    rmse_values_MAR_b = rmse(df_full, df_missing_MAR_b, df_imputed_MAR_b)
    rmse_values_MNAR_b = rmse(df_full, df_missing_MNAR_b, df_imputed_MNAR_b)
    rmse_values_MCAR_b = rmse(df_full, df_missing_MCAR_b, df_imputed_MCAR_b)

    nrmse_values_MAR = nrmse(df_full, df_missing_MAR, df_imputed_MAR)
    nrmse_values_MNAR = nrmse(df_full, df_missing_MNAR, df_imputed_MNAR)
    nrmse_values_MCAR = nrmse(df_full, df_missing_MCAR, df_imputed_MCAR)
    nrmse_values_MAR_b = nrmse(df_full, df_missing_MAR_b, df_imputed_MAR_b)
    nrmse_values_MNAR_b = nrmse(df_full, df_missing_MNAR_b, df_imputed_MNAR_b)
    nrmse_values_MCAR_b = nrmse(df_full, df_missing_MCAR_b, df_imputed_MCAR_b)

    print("MAR:", rmse_values_MAR.sort_values())
    mean_rmse_MAR = rmse_values_MAR.mean(skipna=True)
    mean_nrmse_MAR = nrmse_values_MAR.mean(skipna=True)
    print("Mean RMSE MAR:", mean_rmse_MAR)

    log_result(
        dataset="German Credit Card",
        missingness_type="MAR",
        missing_rate="missing pay_0 -> limitbal, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MAR,
        mean_nrmse=mean_nrmse_MAR
    )

    print("MNAR:", rmse_values_MNAR.sort_values())
    mean_rmse_MNAR = rmse_values_MNAR.mean(skipna=True)
    mean_nrmse_MNAR = nrmse_values_MNAR.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR)

    log_result(
        dataset="German Credit Card",
        missingness_type="MNAR",
        missing_rate="Amount Credit, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR,
        mean_nrmse=mean_nrmse_MNAR
    )

    print("MCAR:", rmse_values_MCAR.sort_values())
    mean_rmse_MCAR = rmse_values_MCAR.mean(skipna=True)
    mean_nrmse_MCAR = nrmse_values_MCAR.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR)

    log_result(
        dataset="German Credit Card",
        missingness_type="MCAR",
        missing_rate="10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR,
        mean_nrmse=mean_nrmse_MCAR
    )

    print("MAR:", rmse_values_MAR_b.sort_values())
    mean_rmse_MAR_b = rmse_values_MAR_b.mean(skipna=True)
    mean_nrmse_MAR_b = nrmse_values_MAR_b.mean(skipna=True)
    print("Mean RMSE MAR:", mean_rmse_MAR_b)

    log_result(
        dataset="German Credit Card",
        missingness_type="MAR",
        missing_rate="missing billamt1 -> pay0, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MAR_b,
        mean_nrmse=mean_nrmse_MAR_b
    )

    print("MNAR:", rmse_values_MNAR_b.sort_values())
    mean_rmse_MNAR_b = rmse_values_MNAR_b.mean(skipna=True)
    mean_nrmse_MNAR_b = nrmse_values_MNAR_b.mean(skipna=True)
    print("Mean RMSE MNAR:", mean_rmse_MNAR_b)

    log_result(
        dataset="German Credit Card",
        missingness_type="MNAR",
        missing_rate="billamt1, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MNAR_b,
        mean_nrmse=mean_nrmse_MNAR_b
    )

    print("MCAR:", rmse_values_MCAR_b.sort_values())
    mean_rmse_MCAR_b = rmse_values_MCAR_b.mean(skipna=True)
    mean_nrmse_MCAR_b = nrmse_values_MCAR_b.mean(skipna=True)
    print("Mean RMSE MCAR:", mean_rmse_MCAR_b)

    log_result(
        dataset="German Credit Card",
        missingness_type="MCAR",
        missing_rate="billamt1, 10%",
        imputation_method="MissForest",
        mean_rmse=mean_rmse_MCAR_b,
        mean_nrmse=mean_nrmse_MCAR_b
    )




