import numpy as np
from minilearn.base import BaseEstimator

class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Enforce the API contract
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape
        
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent Loop
        for _ in range(self.epochs):
            # 1. Calculate predictions
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 2. Calculate the gradients (derivatives)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 3. Update the weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X, _ = self._validate_input(X)
        return np.dot(X, self.weights) + self.bias