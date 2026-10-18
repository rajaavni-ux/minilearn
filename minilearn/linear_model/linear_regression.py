"""Linear regression solved directly (no training loop needed)."""
import numpy as np
from ..base import BaseEstimator


class LinearRegression(BaseEstimator):
    """Fit y = X @ coef_ + intercept_ by least squares.

    np.linalg.lstsq solves the same problem as the "normal equation"
    (X^T X) w = X^T y, but is more stable when features are correlated.
    """

    _estimator_type = "regressor"

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        y = y.astype(float)

        if self.fit_intercept:
            ones = np.ones((len(X), 1))
            X = np.hstack([ones, X])        # first column of 1s learns the intercept

        weights, *_ = np.linalg.lstsq(X, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict(self, X):
        self._check_is_fitted("coef_")
        X = self._check_X(X)
        return X @ self.coef_ + self.intercept_
