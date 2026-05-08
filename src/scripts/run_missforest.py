from src.scripts.imputation_pipeline import run_imputation_method


def run_missforest():
    return run_imputation_method("missforest")


if __name__ == "__main__":
    run_missforest()
