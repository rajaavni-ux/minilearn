"""K-Nearest Neighbors: predict from the most similar training examples."""
import numpy as np
from ..base import BaseEstimator


def _pairwise_distances(A, B):
    """Euclidean distance between every row of A and every row of B.

    Uses |a - b|^2 = |a|^2 + |b|^2 - 2 a.b so no big 3D array is needed.
    """
    squared = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(squared, 0))   # maximum() guards tiny negative rounding


class _BaseKNN(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        if not 1 <= self.n_neighbors <= len(X):
            raise ValueError("n_neighbors must be between 1 and the number of samples")
        self.X_train_ = X          # "training" is just remembering the data
        self.y_train_ = y
        return self

    def _neighbor_labels(self, X):
        """Labels of the k closest training rows for each query row."""
        self._check_is_fitted("X_train_")
        X = self._check_X(X)
        distances = _pairwise_distances(X, self.X_train_)
        nearest = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        return self.y_train_[nearest]          # shape (n_queries, k)


class KNeighborsClassifier(_BaseKNN):
    """Majority vote among the k nearest neighbors. Scale your features first."""

    _estimator_type = "classifier"

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X):
        predictions = []
        for labels in self._neighbor_labels(X):
            values, counts = np.unique(labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])   # ties go to the smaller label
        return np.array(predictions)


class KNeighborsRegressor(_BaseKNN):
    """Average of the k nearest neighbors' target values."""

    _estimator_type = "regressor"

    def fit(self, X, y):
        super().fit(X, y)
        self.y_train_ = self.y_train_.astype(float)
        return self

    def predict(self, X):
        return self._neighbor_labels(X).mean(axis=1)
