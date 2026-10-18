"""Fill in missing values (NaN)."""
import numpy as np
from ..base import BaseEstimator


class SimpleImputer(BaseEstimator):
    """Replace NaN with a statistic learned from each column.

    strategy: "mean", "median", "most_frequent" or "constant"
    fill_value: used for "constant", and for columns that are entirely NaN
    """

    _estimator_type = "transformer"

    def __init__(self, strategy="mean", fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        if self.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(
                "strategy must be 'mean', 'median', 'most_frequent' or 'constant'")
        X = self._check_X(X, allow_nan=True)

        statistics = []
        for j in range(X.shape[1]):
            column = X[:, j]
            column = column[~np.isnan(column)]  # keep only the real values

            if self.strategy == "constant" or column.size == 0:
                statistics.append(self.fill_value)
            elif self.strategy == "mean":
                statistics.append(column.mean())
            elif self.strategy == "median":
                statistics.append(np.median(column))
            else:  # most_frequent
                values, counts = np.unique(column, return_counts=True)
                statistics.append(values[np.argmax(counts)])

        self.statistics_ = np.array(statistics, dtype=float)
        return self

    def transform(self, X):
        self._check_is_fitted("statistics_")
        X = self._check_X(X, allow_nan=True).copy()  # copy: don't touch caller's data
        if X.shape[1] != len(self.statistics_):
            raise ValueError("X has a different number of features than during fit")
        rows, cols = np.where(np.isnan(X))       # positions of every NaN
        X[rows, cols] = self.statistics_[cols]   # fill each with its column's value
        return X
