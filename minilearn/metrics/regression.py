import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the average squared difference between true and predicted values."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")
        
    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the square root of MSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))