from src.scripts.imputation_pipeline import run_imputation_method


def run_mice():
    return run_imputation_method("mice")


if __name__ == "__main__":
    run_mice()
