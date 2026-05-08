from src.scripts.imputation_pipeline import run_imputation_method


def run_medianmode():
    return run_imputation_method("medianmode")


if __name__ == "__main__":
    run_medianmode()
