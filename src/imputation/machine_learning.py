import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from src.imputation.base import BaseImputer
from tqdm import tqdm

class MICEImputer(BaseImputer):
    """
    MICE using sklearn IterativeImputer.
    Categorical features are ordinal encoded first using OrdinalEncoder.
    Numeric features are imputed via regression.
    """

    def __init__(self, random_state: int = 42):
        super().__init__("MICE")
        self.random_state = random_state
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.encoder_ = None
        self.imputer_ = None

    def fit(self, X):
        X = X.copy()

        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # encode categorical
        if self.categorical_cols_:
            self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)
            X[self.categorical_cols_] = self.encoder_.fit_transform(
                X[self.categorical_cols_]
            )

        self.imputer_ = IterativeImputer(
            random_state=self.random_state,
            max_iter=10
        )

        self.imputer_.fit(X)
        return self

    def transform(self, X):
        if self.imputer_ is None:
            raise RuntimeError("Call fit() before transform().")

        X_out = X.copy()

        # encode categorical
        if self.categorical_cols_:
            X_out[self.categorical_cols_] = self.encoder_.transform(
                X_out[self.categorical_cols_]
            )

        X_imputed = self.imputer_.transform(X_out)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MissForestImputer(BaseImputer):
    """
    Random-Forest-based iterative imputation (MissForest-style).

    1) Make initial fill so there are no NaNs (mean for numeric, mode for categorical)
    2) Encode categorical columns using OrdinalEncoder
    3) Repeat for several iterations:
        - For column that originally had missing values:
            - train RF on rows where that column is NOT missing
            - predict missing rows for that column
            - write predictions back
    4) Decode categorical columns
    """

    def __init__(
            self,
            n_estimators: int = 200,
            max_iter: int = 10,
            random_state: int = 42,
            n_jobs: int = -1,
            min_samples_leaf: int = 1,
            max_features: str | float | int = "sqrt",
    ):
        super().__init__("MissForest")
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.numeric_cols_: list[str] | None = None
        self.categorical_cols_: list[str] | None = None
        self.cols_with_missing_: list[str] | None = None
        self.encoder_: OrdinalEncoder | None = None

    def fit(self, X):
        X = X.copy()

        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        self.cols_with_missing_ = [c for c in X.columns if X[c].isna().any()]

        if self.categorical_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

            self.encoder_.fit(X[self.categorical_cols_])

        return self

    def transform(self, X):
        if self.numeric_cols_ is None or self.categorical_cols_ is None or self.cols_with_missing_ is None:
            raise RuntimeError("Call fit() before transform().")

        X_work = X.copy()
        original_missing_mask = {c: X_work[c].isna().to_numpy() for c in self.cols_with_missing_}

        # initial fill
        for col in self.numeric_cols_:
            if X_work[col].isna().any():
                m = X_work[col].mean()
                if pd.isna(m):
                    m = 0.0
                X_work[col] = X_work[col].fillna(m)

        for col in self.categorical_cols_:
            if X_work[col].isna().any():
                non_missing = X_work[col].dropna()
                fill_val = non_missing.mode().iloc[0] if not non_missing.empty else "missing"
                X_work[col] = X_work[col].fillna(fill_val)

        # encode categoricals to numeric
        if self.categorical_cols_:
            assert self.encoder_ is not None
            X_work[self.categorical_cols_] = self.encoder_.transform(X_work[self.categorical_cols_])

        X_mat = X_work.astype(float)

        # iterative imputation
        for iteration in tqdm(range(self.max_iter), desc="MissForest Iterations"):
            # loop each column that had missing in the ORIGINAL data
            for target_col in tqdm(self.cols_with_missing_,
                                   desc=f"Iteration {iteration + 1}",
                                   leave=False):
                miss_mask = original_missing_mask[target_col]
                if miss_mask.sum() == 0:
                    continue

                # build train/predict splits
                y_all = X_mat[target_col].to_numpy()
                X_all = X_mat.drop(columns=[target_col]).to_numpy(copy=True)

                train_idx = ~miss_mask
                pred_idx = miss_mask

                if train_idx.sum() == 0:
                    continue

                X_train = X_all[train_idx]
                y_train = y_all[train_idx]
                X_pred = X_all[pred_idx]

                # choose model type
                is_categorical_target = target_col in self.categorical_cols_

                if is_categorical_target:
                    y_train_int = np.rint(y_train).astype(int)

                    clf = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                    )
                    clf.fit(X_train, y_train_int)
                    y_pred = clf.predict(X_pred).astype(float)  # keep float for X_mat
                else:
                    reg = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                    )
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_pred)

                # write predictions back into the matrix
                X_mat.loc[pred_idx, target_col] = y_pred

        X_imputed = pd.DataFrame(X_mat, columns=X_work.columns, index=X_work.index)

        # decode categoricals
        if self.categorical_cols_:
            # round
            X_imputed[self.categorical_cols_] = np.rint(X_imputed[self.categorical_cols_]).astype(int)

            decoded = self.encoder_.inverse_transform(X_imputed[self.categorical_cols_].to_numpy())
            X_imputed[self.categorical_cols_] = decoded

        for col in self.numeric_cols_:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors="coerce")

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)
