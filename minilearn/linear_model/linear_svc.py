"""Linear support vector classifier (maximum-margin) with gradient descent."""
import numpy as np
from ..base import BaseEstimator


class LinearSVC(BaseEstimator):
    """Find the straight boundary with the widest safety margin.

    Minimises:  0.5 * ||w||^2  +  C * average(hinge loss)
    where hinge loss = max(0, 1 - y * (w.x + b)) and y is -1 or +1.
    Only points inside the margin (or misclassified) push the boundary.

    C: higher = fit the training points harder, lower = wider margin
    Works best on scaled features.
    """

    _estimator_type = "classifier"

    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _fit_binary(self, X, y_pm):
        """Train one boundary. y_pm contains only -1 and +1."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.n_iterations):
            margins = y_pm * (X @ w + b)
            violators = margins < 1               # inside the margin or wrong side
            grad_w = w - self.C * (X[violators].T @ y_pm[violators]) / n_samples
            grad_b = -self.C * y_pm[violators].sum() / n_samples
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        return w, b

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two different classes")

        targets = self.classes_[1:] if len(self.classes_) == 2 else self.classes_

        coefs, intercepts = [], []
        for label in targets:
            y_pm = np.where(y == label, 1.0, -1.0)
            w, b = self._fit_binary(X, y_pm)
            coefs.append(w)
            intercepts.append(b)

        self.coef_ = np.array(coefs)
        self.intercept_ = np.array(intercepts)
        return self

    def decision_function(self, X):
        """Signed distance-like score. Positive means 'this class'."""
        self._check_is_fitted("coef_")
        X = self._check_X(X)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return self.classes_[(scores[:, 0] > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]
