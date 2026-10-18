"""Scoring functions for classification."""
import numpy as np


def _check_pair(y_true, y_pred):
    """Turn both inputs into 1D arrays and make sure they line up."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Fraction of predictions that match the true labels."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    """Of everything predicted as positive, how much really was positive?

    precision = true positives / (true positives + false positives)

    average="binary": score only the class given by pos_label.
    average="macro":  score every class separately, then average them.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)

    def precision_for(label):
        predicted_positive = (y_pred == label)
        n_predicted = predicted_positive.sum()
        if n_predicted == 0:
            return 0.0  # nothing was predicted as this class
        true_positive = np.sum(predicted_positive & (y_true == label))
        return true_positive / n_predicted

    if average == "binary":
        return float(precision_for(pos_label))
    if average == "macro":
        labels = np.unique(np.concatenate([y_true, y_pred]))
        return float(np.mean([precision_for(label) for label in labels]))
    raise ValueError("average must be 'binary' or 'macro'")
