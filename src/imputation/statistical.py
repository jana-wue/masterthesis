from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .base import BaseImputer
import pandas as pd
import numpy as np

class MeanModeImputer(BaseImputer):
    """
    Simple Baseline:
    - numeric cols: mean
    - categorial cols: mode
    """

    def __init__(self, fallback_cat: str = "missing") -> None:
        """Initialize the object state."""
        super().__init__("Mean/Mode")
        self.fallback_cat = fallback_cat
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.means_ = {}
        self.modes_ = {}

    def fit(self, X: pd.DataFrame) -> MeanModeImputer:
        """
        Saves value per col with which missing is going to be replaced.
        """
        X = X.copy()

        # sort cols into categoric and numeric
        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # mean for numeric
        for col in self.numeric_cols_:
            self.means_[col] = X[col].mean()

        # mode for categoric
        for col in self.categorical_cols_:
            non_missing = X[col].dropna()
            if non_missing.empty:
                # if complete col is missing
                self.modes_[col] = self.fallback_cat
            else:
                self.modes_[col] = non_missing.mode().iloc[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        replace missing values with means/modes from fit().
        """
        if self.numeric_cols_ is None:
            raise RuntimeError("First fit(), then transform().")

        X_out = X.copy()

        # numeric -> mean
        for col in self.numeric_cols_:
            fill_val = self.means_.get(col, np.nan)
            if pd.isna(fill_val):
                # mean nicht berechenbar (z.B. komplett missing) -> nichts tun
                continue
            X_out[col] = X_out[col].fillna(fill_val)

        # categoric -> mode
        for col in self.categorical_cols_:
            fill_val = self.modes_.get(col, self.fallback_cat)
            X_out[col] = X_out[col].fillna(fill_val)

        return X_out


class MedianModeImputer(BaseImputer):
    """
    Simple Baseline II:
    - numeric cols: median
    - categorical cols: mode
    """

    def __init__(self, fallback_cat: str = "missing") -> None:
        """Initialize the object state."""
        super().__init__("Median/Mode")
        self.fallback_cat = fallback_cat
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.medians_ = {}
        self.modes_ = {}

    def fit(self, X: pd.DataFrame) -> MedianModeImputer:
        """
        Saves value per col with which missing is going to be replaced.
        """
        X = X.copy()

        # sort cols into categoric and numeric
        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # median for numeric
        for col in self.numeric_cols_:
            self.medians_[col] = X[col].median()

        # mode for categorical
        for col in self.categorical_cols_:
            non_missing = X[col].dropna()
            if non_missing.empty:
                # if complete col is missing
                self.modes_[col] = self.fallback_cat
            else:
                self.modes_[col] = non_missing.mode().iloc[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing values with medians/modes from fit().
        """
        if self.numeric_cols_ is None:
            raise RuntimeError("First fit(), then transform().")

        X_out = X.copy()

        # numeric -> median
        for col in self.numeric_cols_:
            fill_val = self.medians_.get(col, np.nan)
            if pd.isna(fill_val):
                # median nicht berechenbar (z.B. komplett missing) -> nichts tun
                continue
            X_out[col] = X_out[col].fillna(fill_val)

        # categorical -> mode
        for col in self.categorical_cols_:
            fill_val = self.modes_.get(col, self.fallback_cat)
            X_out[col] = X_out[col].fillna(fill_val)

        return X_out
