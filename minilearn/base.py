"""Shared foundation for every model and transformer in MiniLearn."""
import inspect
import numpy as np

from .metrics import accuracy_score, r2_score


class BaseEstimator:
    """Parent class. Gives every estimator the same small set of tools.

    Rules for subclasses:
      * __init__ only stores its parameters, each under the same name
        (self.alpha = alpha). That lets get_params() and clone() work.
      * fit() learns from data, stores what it learned in attributes that
        end with an underscore (coef_, mean_ ...), and returns self.
      * Models define predict(); transformers define transform().
    """

    _estimator_type = None  # "classifier", "regressor" or "transformer"

    # ----- parameters -----
    def get_params(self):
        """Return the constructor parameters as a dictionary."""
        names = [
            name for name, p in inspect.signature(self.__init__).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        return {name: getattr(self, name) for name in names}

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    # ----- to be overridden -----
    def fit(self, X, y=None):
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict()")

    def transform(self, X):
        raise NotImplementedError("Subclasses must implement transform()")

    # ----- shared helpers -----
    def fit_transform(self, X, y=None):
        """fit() then transform() in one call (used by transformers)."""
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        """Accuracy for classifiers, R^2 for regressors. Higher is better."""
        predictions = self.predict(X)
        if self._estimator_type == "regressor":
            return r2_score(y, predictions)
        return accuracy_score(y, predictions)

    def _check_X(self, X, allow_nan=False):
        """Convert input to a 2D float array of shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if not allow_nan and np.isnan(X).any():
            raise ValueError(
                "X contains missing values (NaN). Use SimpleImputer first.")
        return X

    def _check_X_y(self, X, y):
        """Validate X and y together and make sure their lengths match."""
        X = self._check_X(X)
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        return X, y

    def _check_is_fitted(self, attribute):
        if not hasattr(self, attribute):
            raise RuntimeError(
                f"This {type(self).__name__} is not fitted yet. Call fit() first.")


def clone(estimator):
    """Make a fresh, unfitted copy with the same parameters.

    Cross-validation needs this so each fold starts from scratch.
    """
    if hasattr(estimator, "_clone"):  # e.g. Pipeline clones its own steps
        return estimator._clone()
    return type(estimator)(**estimator.get_params())
