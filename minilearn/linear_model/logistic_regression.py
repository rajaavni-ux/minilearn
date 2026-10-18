"""Logistic regression trained with gradient descent."""
import numpy as np
from ..base import BaseEstimator


def _sigmoid(z):
    """Squash any number into the range (0, 1). Clipping avoids overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(BaseEstimator):
    """Classifier that learns the probability of each class.

    Two classes: one model is trained.
    More classes: one-vs-rest, i.e. one model per class ("this class or not").

    learning_rate: size of each gradient descent step
    n_iterations:  number of steps
    l2:            strength of weight penalty (0 = none); fights overfitting
    Works best on scaled features.
    """

    _estimator_type = "classifier"

    def __init__(self, learning_rate=0.1, n_iterations=1000, l2=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2

    def _fit_binary(self, X, y01):
        """Train one yes/no model. y01 contains only 0s and 1s."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.n_iterations):
            p = _sigmoid(X @ w + b)               # 1. predict probabilities
            error = p - y01                       # 2. how wrong were we?
            grad_w = X.T @ error / n_samples + self.l2 * w   # 3. gradients
            grad_b = error.mean()
            w -= self.learning_rate * grad_w      # 4. step downhill
            b -= self.learning_rate * grad_b
        return w, b

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two different classes")

        # With two classes we only need a model for the second class.
        targets = self.classes_[1:] if len(self.classes_) == 2 else self.classes_

        coefs, intercepts = [], []
        for label in targets:
            w, b = self._fit_binary(X, (y == label).astype(float))
            coefs.append(w)
            intercepts.append(b)

        self.coef_ = np.array(coefs)              # shape (n_models, n_features)
        self.intercept_ = np.array(intercepts)
        return self

    def predict_proba(self, X):
        """Probability of each class, one column per class (rows sum to 1)."""
        self._check_is_fitted("coef_")
        X = self._check_X(X)
        p = _sigmoid(X @ self.coef_.T + self.intercept_)
        if len(self.classes_) == 2:
            return np.column_stack([1 - p[:, 0], p[:, 0]])
        return p / p.sum(axis=1, keepdims=True)   # make the per-class scores add to 1

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
