from src.scripts.imputation_pipeline import run_imputation_method


def run_dae():
    return run_imputation_method("dae")


if __name__ == "__main__":
    run_dae()
