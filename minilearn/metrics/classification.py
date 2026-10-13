import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the ratio of correctly predicted labels to total labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
        
    return float(np.mean(y_true == y_pred))

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates True Positives / (True Positives + False Positives) for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    if predicted_positives == 0:
        return 0.0
        
    return float(true_positives / predicted_positives)
