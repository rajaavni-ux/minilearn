"""Cross-validated scoring."""
import numpy as np
from ..base import clone
from .kfold import KFold


def cross_val_score(estimator, X, y, cv=5, scoring=None, random_state=None):
    """Score a model on k different train/test splits.

    cv: number of folds, or a KFold object.
    scoring: optional function(y_true, y_pred). Default is estimator.score
             (accuracy for classifiers, R^2 for regressors).
    Returns an array with one score per fold.
    """
    X, y = np.asarray(X), np.asarray(y)
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, test_idx in cv.split(X):
        model = clone(estimator)               # fresh copy every fold
        model.fit(X[train_idx], y[train_idx])
        if scoring is None:
            scores.append(model.score(X[test_idx], y[test_idx]))
        else:
            scores.append(scoring(y[test_idx], model.predict(X[test_idx])))
    return np.array(scores)
