"""k-Fold cross-validation splitter."""
import numpy as np


class KFold:
    """Cut the data into k folds; each fold takes one turn as the test set."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """Yield (train_indices, test_indices), once per fold."""
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot be larger than the number of samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(indices)

        # Spread the leftover rows over the first folds so sizes differ by at most 1.
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        start = 0
        for size in fold_sizes:
            test_idx = indices[start:start + size]
            train_idx = np.concatenate([indices[:start], indices[start + size:]])
            yield train_idx, test_idx
            start += size
