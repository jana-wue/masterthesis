from src.scripts.imputation_pipeline import run_imputation_method


def run_meanmode():
    return run_imputation_method("meanmode")


if __name__ == "__main__":
    run_meanmode()
