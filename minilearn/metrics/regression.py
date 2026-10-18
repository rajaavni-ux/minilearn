"""Scoring functions for regression."""
import numpy as np


def _check_pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return y_true, y_pred


def mean_squared_error(y_true, y_pred):
    """Average of the squared differences. Lower is better."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Square root of the MSE, so the error is in the same units as y."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """1.0 is a perfect fit; 0.0 is no better than always predicting the mean."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - y_true.mean()) ** 2)
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    return float(1 - ss_residual / ss_total)


# Short aliases
mse = mean_squared_error
rmse = root_mean_squared_error
