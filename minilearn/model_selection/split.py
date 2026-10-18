"""Splitting data for honest evaluation."""
import numpy as np


def train_test_split(X, y=None, test_size=0.25, shuffle=True, random_state=None):
    """Split data into a training part and a testing part.

    test_size: a fraction (0.25 = 25%) or an exact number of test rows.
    random_state: any integer makes the shuffle repeatable.
    Returns X_train, X_test (and y_train, y_test if y is given).
    """
    X = np.asarray(X)
    n_samples = len(X)

    if isinstance(test_size, float):
        n_test = int(round(n_samples * test_size))
    else:
        n_test = int(test_size)
    if not 0 < n_test < n_samples:
        raise ValueError("test_size leaves no data for training or for testing")

    indices = np.arange(n_samples)
    if shuffle:
        np.random.default_rng(random_state).shuffle(indices)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    if y is None:
        return X[train_idx], X[test_idx]

    y = np.asarray(y)
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
