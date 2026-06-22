import numpy as np
import pandas as pd


def mcar(df: pd.DataFrame, missing_frac: float, random_state: int = 42) -> pd.DataFrame:
    """MCAR: Random Missing Values over all features"""
    np.random.seed(random_state)
    df_mcar = df.copy()
    mask = np.random.rand(*df.shape) < missing_frac
    df_mcar[mask] = np.nan
    return df_mcar


def mcar_single_feature(
    df: pd.DataFrame,
    missing_frac: float,
    feature: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate single-feature MCAR missingness."""
    np.random.seed(random_state)
    df_mcar = df.copy()

    n = len(df_mcar)
    n_missing = int(missing_frac * n)

    valid_idx = df_mcar[df_mcar[feature].notna()].index
    missing_indices = np.random.choice(valid_idx, size=n_missing, replace=False)

    df_mcar.loc[missing_indices, feature] = np.nan

    return df_mcar


def mar(
    df: pd.DataFrame,
    feature_dep: str,
    missing_feature: str,
    missing_frac: float,
    random_state: int,
) -> pd.DataFrame:
    """
    MAR: Missingness in `missing_feature` depends on `feature_dep` (observed).
    """

    if feature_dep == missing_feature:
        raise ValueError("feature_dep and missing_feature must be different (MAR!)")

    rng = np.random.default_rng(random_state)
    df_m = df.copy()

    # only candidates where we can set missing
    candidates = df_m.index[df_m[missing_feature].notna() & df_m[feature_dep].notna()]
    if len(candidates) == 0:
        return df_m

    n_missing = int(round(missing_frac * len(df_m)))
    n_missing = min(n_missing, len(candidates))
    if n_missing == 0:
        return df_m

    dep = df_m.loc[candidates, feature_dep].astype(float)

    # robust scaling via ranks
    ranks = dep.rank(method="average")
    weights = ranks / ranks.sum()

    chosen = rng.choice(candidates.to_numpy(), size=n_missing, replace=False, p=weights.to_numpy())
    df_m.loc[chosen, missing_feature] = np.nan

    return df_m


def mnar(
    df: pd.DataFrame,
    feature: str,
    missing_frac: float,
    random_state: int,
) -> pd.DataFrame:
    """
    MNAR: Missingness in `feature` depends on the value of `feature` itself.
    """

    rng = np.random.default_rng(random_state)
    df_m = df.copy()

    # Only consider rows where feature is not already missing
    candidates = df_m.index[df_m[feature].notna()]
    if len(candidates) == 0:
        return df_m

    n_missing = int(round(missing_frac * len(df_m)))
    n_missing = min(n_missing, len(candidates))

    if n_missing == 0:
        return df_m

    values = df_m.loc[candidates, feature].astype(float)

    # Robust scaling via ranks
    ranks = values.rank(method="average")
    weights = ranks / ranks.sum()

    chosen = rng.choice(
        candidates.to_numpy(),
        size=n_missing,
        replace=False,
        p=weights.to_numpy()
    )

    df_m.loc[chosen, feature] = np.nan

    return df_m
