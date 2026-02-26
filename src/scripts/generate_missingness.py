from src.data.remove_data import *
from src.data.helper_dataprocessing import *


def run_generate_missingness():
    # MCAR generell 10%
    df_credit_card = pd.read_csv(DATA_RAW / "default_of_credit_card_clients.csv")
    df_credit_card_mcar = mcar(df_credit_card, 0.1)
    df_credit_card_mcar.to_csv("german_credit_mcar_10pct.csv", index=False)

    df_telco = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_telco_mcar = mcar(df_telco, 0.1)
    df_telco_mcar.to_csv("telco_customer_churn_mcar_10pct.csv", index=False)

    df_statlog = pd.read_csv(DATA_RAW / "german_statlog_numeric.csv")
    df_statlog_mcar = mcar(df_statlog, 0.1)
    df_statlog_mcar.to_csv("german_statlog_numeric_mcar_10pct.csv", index=False)

    # MCAR Features 10%
    df_credit_card_mcar = mcar_single_feature(df_credit_card, 0.1, "BILL_AMT1")
    df_credit_card_mcar.to_csv("german_credit_mcar_billamt1_10pct.csv", index=False)

    df_telco_mcar = mcar_single_feature(df_telco, 0.1, "TotalCharges")
    df_telco_mcar.to_csv("telco_customer_churn_mcar_totalcharges_10pct.csv", index=False)

    df_statlog_mcar = mcar_single_feature(df_statlog, 0.1, "X2")
    df_statlog_mcar.to_csv("german_statlog_numeric_mcar_duration_10pct.csv", index=False)

    # MNAR
    # Statlog: Credit Amount
    df_statlog_mnar = mnar(df_statlog, feature="X5", missing_frac=0.1, random_state=42)
    df_statlog_mnar.to_csv("german_statlog_numeric_mnar_creditamount_10pct.csv", index=False)
    df_statlog_mnar = mnar(df_statlog, feature="X2", missing_frac=0.1, random_state=42)
    df_statlog_mnar.to_csv("german_statlog_numeric_mnar_duration_10pct.csv", index=False)

    # Telco: Tenure
    df_telco_mnar = mnar(df_telco, feature="tenure", missing_frac=0.1, random_state=42)
    df_telco_mnar.to_csv("telco_customer_churn_mnar_tenure_10pct.csv", index=False)
    df_telco_mnar = mnar(df_telco, feature="TotalCharges", missing_frac=0.1, random_state=42)
    df_telco_mnar.to_csv("telco_customer_churn_mnar_totalcharges_10pct.csv", index=False)

    # Credit Card: Amount of Credit
    df_credit_card_mnar = mnar(df_credit_card, feature="LIMIT_BAL", missing_frac=0.1, random_state=42)
    df_credit_card_mnar.to_csv("german_credit_mnar_amountcredit_10pct.csv", index=False)
    df_credit_card_mnar = mnar(df_credit_card, feature="BILL_AMT1", missing_frac=0.1, random_state=42)
    df_credit_card_mnar.to_csv("german_credit_mnar_billamt1_10pct.csv", index=False)

    # MAR
    # Statlog: Missing Duration -> Credit Amount
    df_statlog_mar = mar(df_statlog, feature_dep="X5", missing_feature="X2", missing_frac=0.1, random_state=42)
    df_statlog_mar.to_csv("german_statlog_numeric_mar_duration_creditamount_10pct.csv", index=False)

    # Telco: Missing Total Charges -> Tenure
    df_telco_mar = mar(df_telco, feature_dep="tenure", missing_feature="TotalCharges", missing_frac=0.1, random_state=42)
    df_telco_mar.to_csv("telco_customer_churn_mar_totalcharges_tenure_10pct.csv", index=False)

    # Credit Card: Missing PAY_0 -> Limit
    df_credit_card_mar = mar(df_credit_card, feature_dep="LIMIT_BAL", missing_feature="PAY_0", missing_frac=0.1, random_state=42)
    df_credit_card_mar.to_csv("german_credit_mar_pay0_limitbal_10pct.csv", index=False)
    df_credit_card_mar = mar(df_credit_card, feature_dep="PAY_0", missing_feature="BILL_AMT1", missing_frac=0.1,
                             random_state=42)
    df_credit_card_mar.to_csv("german_credit_mar_billamt1_pay0_10pct.csv", index=False)

    # TODO: Missingness mit derselben Zielvariable (TotalCharges, Duration, BillAmount) erklären