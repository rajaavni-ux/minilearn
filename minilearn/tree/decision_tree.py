"""Decision tree classifier using Gini impurity."""
import numpy as np
from ..base import BaseEstimator


class _Node:
    """One box in the tree: either a question (feature <= threshold) or an answer."""

    def __init__(self, prediction, feature=None, threshold=None, left=None, right=None):
        self.prediction = prediction   # majority class index at this node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None


def _gini(counts):
    """Impurity of a group: 0 = all one class, higher = more mixed."""
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return 1.0 - np.sum(proportions ** 2)


class DecisionTreeClassifier(BaseEstimator):
    """Learn a set of yes/no questions that split the data into pure groups.

    max_depth:         limit on tree depth (None = grow until pure)
    min_samples_split: don't split a group smaller than this
    min_samples_leaf:  every group after a split must have at least this many rows
    max_features:      features tried per split: None (all), int, "sqrt" or "log2"
                       (random forests use this to make trees different)
    random_state:      seed, only matters when max_features is not None
    """

    _estimator_type = "classifier"

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    # ----- training -----
    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        # Work with class indices 0..k-1 internally, map back to labels in predict().
        self.classes_, y_index = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_)

        self.tree_ = self._build(X, y_index, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total   # make them add up to 1
        return self

    def _build(self, X, y, depth):
        """Grow the tree recursively."""
        n_samples = len(y)
        counts = np.bincount(y, minlength=len(self.classes_))
        node = _Node(prediction=int(np.argmax(counts)))

        # Stop if the group is pure, too small, or the tree is deep enough.
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if len(np.unique(y)) == 1 or n_samples < self.min_samples_split or too_deep:
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None:                      # no useful split exists
            return node

        left_mask = X[:, feature] <= threshold
        self.feature_importances_[feature] += gain * n_samples

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _n_candidate_features(self):
        if self.max_features is None:
            return self.n_features_
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        return max(1, min(int(self.max_features), self.n_features_))

    def _best_split(self, X, y):
        """Try every sensible threshold and return the one that reduces Gini most."""
        n_samples = len(y)
        n_classes = len(self.classes_)
        parent_counts = np.bincount(y, minlength=n_classes)
        parent_gini = _gini(parent_counts)
        y_onehot = np.eye(n_classes)[y]          # row per sample, 1 in its class column

        k = self._n_candidate_features()
        features = self._rng.choice(self.n_features_, size=k, replace=False)

        best_gain, best_feature, best_threshold = 1e-12, None, None
        for feature in features:
            # Sort the rows by this feature; then a split is just "first i rows vs rest".
            order = np.argsort(X[:, feature], kind="stable")
            sorted_x = X[order, feature]

            # left_counts[i] = class counts among the first i+1 rows (running total)
            left_counts = np.cumsum(y_onehot[order], axis=0)[:-1]
            right_counts = parent_counts - left_counts
            n_left = np.arange(1, n_samples)
            n_right = n_samples - n_left

            gini_left = 1 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
            gini_right = 1 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
            weighted = (n_left * gini_left + n_right * gini_right) / n_samples
            gains = parent_gini - weighted

            # A split is only valid between two different values, and
            # must leave enough rows on both sides.
            valid = sorted_x[1:] != sorted_x[:-1]
            valid &= (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not valid.any():
                continue

            gains = np.where(valid, gains, -np.inf)
            i = int(np.argmax(gains))
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_feature = int(feature)
                best_threshold = (sorted_x[i] + sorted_x[i + 1]) / 2   # midpoint

        return best_feature, best_threshold, best_gain

    # ----- prediction -----
    def _predict_one(self, row):
        node = self.tree_
        while not node.is_leaf:                  # answer each question until a leaf
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        self._check_is_fitted("tree_")
        X = self._check_X(X)
        indices = np.array([self._predict_one(row) for row in X])
        return self.classes_[indices]
