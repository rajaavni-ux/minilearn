"""Feature scaling."""
import numpy as np
from ..base import BaseEstimator


class StandardScaler(BaseEstimator):
    """Rescale each feature to mean 0 and standard deviation 1.

    z = (x - mean) / std
    """

    _estimator_type = "transformer"

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0   # a constant column would divide by zero
        self.scale_ = std
        return self

    def transform(self, X):
        self._check_is_fitted("mean_")
        X = self._check_X(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        """Undo the scaling (handy for turning predictions back into real units)."""
        self._check_is_fitted("mean_")
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """Squeeze each feature into a range, by default 0 to 1.

    x_scaled = (x - min) / (max - min) * (high - low) + low
    """

    _estimator_type = "transformer"

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        low, high = self.feature_range
        if low >= high:
            raise ValueError("feature_range must be (low, high) with low < high")
        X = self._check_X(X)
        self.data_min_ = X.min(axis=0)
        data_range = X.max(axis=0) - self.data_min_
        data_range[data_range == 0] = 1.0   # constant column: avoid dividing by zero
        self.data_range_ = data_range
        return self

    def transform(self, X):
        self._check_is_fitted("data_min_")
        X = self._check_X(X)
        low, high = self.feature_range
        return (X - self.data_min_) / self.data_range_ * (high - low) + low

    def inverse_transform(self, X):
        self._check_is_fitted("data_min_")
        low, high = self.feature_range
        X = np.asarray(X, dtype=float)
        return (X - low) / (high - low) * self.data_range_ + self.data_min_
