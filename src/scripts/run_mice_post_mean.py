from src.scripts.imputation_pipeline import run_imputation_method


def run_mice_post_mean():
    return run_imputation_method("mice_post_mean")


if __name__ == "__main__":
    run_mice_post_mean()
