"""Random forest: many different trees voting together."""
import numpy as np
from ..base import BaseEstimator
from ..tree import DecisionTreeClassifier


class RandomForestClassifier(BaseEstimator):
    """Train many trees, each on a different random view of the data, then vote.

    Two sources of randomness make the trees different from each other:
      1. Bootstrap sampling: each tree trains on rows drawn with replacement.
      2. Feature sampling:   each split only considers a random subset of features.
    Averaging many different trees cancels out their individual mistakes.
    """

    _estimator_type = "classifier"

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self.classes_, y_index = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            sample = rng.integers(0, n_samples, size=n_samples)   # bootstrap rows
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[sample], y_index[sample])   # trees see class indices 0..k-1
            self.estimators_.append(tree)

        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0)
        return self

    def predict_proba(self, X):
        """Share of trees voting for each class (one column per class)."""
        self._check_is_fitted("estimators_")
        X = self._check_X(X)
        votes = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        for tree in self.estimators_:
            votes[rows, tree.predict(X)] += 1      # each tree casts one vote per row
        return votes / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
