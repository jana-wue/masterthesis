from src.data.helper_dataprocessing import *

def clean_telco(df, feature, missing_frac):
    # Clean Telco: Make all what can be numeric numeric
    df_telco = pd.read_csv(DATA_PROCESSED / "MNAR/telco_customer_churn_mnar_tenure_p50.csv")
    print(df_telco.dtypes)

    df_telco = make_numeric_columns_numeric(df_telco, exclude=["customerID"], threshold=0.60
    )

    output_path = Path("data/processed/MNAR/telco_customer_churn_mnar_tenure_p50_cleaned.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_telco.to_csv(output_path, index=False)

    print("Fertig gespeichert.")
    print(df_telco.dtypes)