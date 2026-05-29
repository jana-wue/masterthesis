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

    def __init__(self, random_state = 42, max_iter = 10, n_imputations = 1, sample_posterior = False):
        if sample_posterior and n_imputations > 1:
            super().__init__(f"MICE posterior mean (m={n_imputations})")
        else:
            super().__init__("MICE")
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_imputations = n_imputations
        self.sample_posterior = sample_posterior
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
            max_iter=self.max_iter,
            sample_posterior=self.sample_posterior,
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

        if self.sample_posterior and self.n_imputations > 1:
            draws = []
            for i in range(self.n_imputations):
                imputer_i = IterativeImputer(
                    random_state=self.random_state + i,
                    max_iter=self.max_iter,
                    sample_posterior=True,
                )
                Xi = imputer_i.fit_transform(X_out)
                draws.append(Xi)

            X_imputed = np.mean(np.stack(draws, axis=0), axis=0)
        else:
            X_imputed = self.imputer_.transform(X_out)

        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        # convert encoded categoricals back to labels
        if self.categorical_cols_:
            encoded = X_imputed[self.categorical_cols_].to_numpy(dtype=float, copy=True)
            for j, categories in enumerate(self.encoder_.categories_):
                encoded[:, j] = np.clip(np.rint(encoded[:, j]), 0, len(categories) - 1)
            X_imputed[self.categorical_cols_] = self.encoder_.inverse_transform(encoded)

        for col in self.numeric_cols_:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors="coerce")

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MissForestImputer(BaseImputer):
    """
    Random-Forest-based iterative imputation in the style of missForest:
    - initial mean/mode
    - per-variable RF imputation
    - stop when iteration differences increase
    """

    def __init__(
            self,
            n_estimators = 200,
            max_iter = 10,
            random_state = 42,
            n_jobs = -1,
            min_samples_leaf = 1,
            max_features = "sqrt",
            decreasing = False,
            verbose = False,
            fallback_cat = "missing",
    ):
        super().__init__("MissForest")
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.decreasing = decreasing
        self.verbose = verbose
        self.fallback_cat = fallback_cat

        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.cols_with_missing_ = None
        self.cols_with_missing_sorted_ = None
        self.encoder_: OrdinalEncoder = None
        self.initial_fill_values_ = None
        self.convergence_history_ = []
        self.n_iter_ = 0

    def fit(self, X):
        X = X.copy()

        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        self.cols_with_missing_ = [c for c in X.columns if X[c].isna().any()]
        missing_fraction = X[self.cols_with_missing_].isna().mean()
        self.cols_with_missing_sorted_ = (
            missing_fraction.sort_values(ascending=not self.decreasing).index.tolist()
        )

        self.initial_fill_values_ = {}
        for col in self.numeric_cols_:
            mean_val = X[col].mean()
            self.initial_fill_values_[col] = float(mean_val) if not pd.isna(mean_val) else 0.0
        for col in self.categorical_cols_:
            non_missing = X[col].dropna()
            self.initial_fill_values_[col] = (
                non_missing.mode().iloc[0] if not non_missing.empty else self.fallback_cat
            )

        if self.categorical_cols_:
            # Fit on initially filled categories
            X_cat = X[self.categorical_cols_].copy()
            for col in self.categorical_cols_:
                X_cat[col] = X_cat[col].fillna(self.initial_fill_values_[col])
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoder_.fit(X_cat)

        return self

    def _initial_imputation(self, X):
        X_work = X.copy()
        assert self.initial_fill_values_ is not None

        for col, fill_val in self.initial_fill_values_.items():
            if col in X_work.columns and X_work[col].isna().any():
                X_work[col] = X_work[col].fillna(fill_val)

        if self.categorical_cols_:
            assert self.encoder_ is not None
            X_cat = X_work[self.categorical_cols_].copy()
            for col in self.categorical_cols_:
                X_cat[col] = X_cat[col].fillna(self.initial_fill_values_[col])
            X_work[self.categorical_cols_] = self.encoder_.transform(X_cat)

        return X_work.astype(float)

    def _compute_numeric_delta(self, previous, current, original_missing_mask):
        numeric_cols = [
            c for c in self.numeric_cols_ if c in original_missing_mask and original_missing_mask[c].any()
        ]
        if not numeric_cols:
            return float("nan")

        numerator = 0.0
        denominator = 0.0
        for col in numeric_cols:
            mask = original_missing_mask[col]
            old_vals = previous.loc[mask, col].to_numpy(dtype=float)
            new_vals = current.loc[mask, col].to_numpy(dtype=float)
            numerator += float(np.sum((new_vals - old_vals) ** 2))
            denominator += float(np.sum(new_vals ** 2))

        if denominator <= 0.0:
            denominator = 1e-12
        return numerator / denominator

    def _compute_categorical_delta(self, previous, current, original_missing_mask):
        categorical_cols = [
            c for c in self.categorical_cols_ if c in original_missing_mask and original_missing_mask[c].any()
        ]
        if not categorical_cols:
            return float("nan")

        mismatches = 0
        total = 0
        for col in categorical_cols:
            mask = original_missing_mask[col]
            old_vals = np.rint(previous.loc[mask, col].to_numpy(dtype=float)).astype(int)
            new_vals = np.rint(current.loc[mask, col].to_numpy(dtype=float)).astype(int)
            mismatches += int(np.sum(old_vals != new_vals))
            total += int(mask.sum())

        if total == 0:
            return float("nan")
        return mismatches / total

    def transform(self, X):
        if (
            self.numeric_cols_ is None
            or self.categorical_cols_ is None
            or self.cols_with_missing_ is None
            or self.cols_with_missing_sorted_ is None
            or self.initial_fill_values_ is None
        ):
            raise RuntimeError("Call fit() before transform().")

        X_work = X.copy()
        original_missing_mask = {
            col: X_work[col].isna().to_numpy() for col in self.cols_with_missing_ if col in X_work.columns
        }

        if not self.cols_with_missing_:
            return X_work

        X_mat = self._initial_imputation(X_work)
        previous_numeric_delta = float("inf")
        previous_categorical_delta = float("inf")
        self.convergence_history_ = []
        self.n_iter_ = 0

        for iteration in tqdm(
            range(self.max_iter),
            desc="MissForest Iterations",
            disable=not self.verbose,
        ):
            X_old = X_mat.copy()

            for target_col in tqdm(
                self.cols_with_missing_sorted_,
                desc=f"Iteration {iteration + 1}",
                leave=False,
                disable=not self.verbose,
            ):
                miss_mask = original_missing_mask[target_col]
                if miss_mask.sum() == 0:
                    continue

                train_idx = ~miss_mask
                pred_idx = miss_mask
                if train_idx.sum() == 0:
                    continue

                y_all = X_mat[target_col].to_numpy()
                X_all = X_mat.drop(columns=[target_col]).to_numpy(copy=True)
                X_train = X_all[train_idx]
                y_train = y_all[train_idx]
                X_pred = X_all[pred_idx]

                is_categorical_target = target_col in self.categorical_cols_
                if is_categorical_target:
                    y_train_int = np.rint(y_train).astype(int)
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                    )
                    model.fit(X_train, y_train_int)
                    y_pred = model.predict(X_pred).astype(float)
                else:
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_pred)

                X_mat.loc[pred_idx, target_col] = y_pred

            numeric_delta = self._compute_numeric_delta(X_old, X_mat, original_missing_mask)
            categorical_delta = self._compute_categorical_delta(X_old, X_mat, original_missing_mask)
            has_numeric = not np.isnan(numeric_delta)
            has_categorical = not np.isnan(categorical_delta)

            stop_numeric = has_numeric and numeric_delta > previous_numeric_delta
            stop_categorical = has_categorical and categorical_delta > previous_categorical_delta
            if has_numeric and has_categorical:
                should_stop = stop_numeric and stop_categorical
            elif has_numeric:
                should_stop = stop_numeric
            elif has_categorical:
                should_stop = stop_categorical
            else:
                should_stop = False

            self.convergence_history_.append(
                {
                    "iteration": int(iteration + 1),
                    "numeric_delta": float(numeric_delta) if has_numeric else float("nan"),
                    "categorical_delta": float(categorical_delta) if has_categorical else float("nan"),
                    "stopped": bool(should_stop),
                }
            )

            if should_stop:
                # return previous imputation
                X_mat = X_old
                self.n_iter_ = int(iteration)
                break

            if has_numeric:
                previous_numeric_delta = numeric_delta
            if has_categorical:
                previous_categorical_delta = categorical_delta

            self.n_iter_ = int(iteration + 1)

        X_imputed = pd.DataFrame(X_mat, columns=X_work.columns, index=X_work.index)

        if self.categorical_cols_:
            assert self.encoder_ is not None
            encoded = X_imputed[self.categorical_cols_].to_numpy(dtype=float, copy=True)
            for j, categories in enumerate(self.encoder_.categories_):
                encoded[:, j] = np.clip(np.rint(encoded[:, j]), 0, len(categories) - 1)
            X_imputed[self.categorical_cols_] = self.encoder_.inverse_transform(encoded)

        for col in self.numeric_cols_:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors="coerce")

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)
