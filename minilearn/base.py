from abc import ABC, abstractmethod
import numpy as np

class BaseEstimator(ABC):
    def _validate_input(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if y is not None:
            y = np.asarray(y)
        return X, y

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass