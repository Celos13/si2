import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth target values.
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        The RMSE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Different shapes: {y_true.shape} vs {y_pred.shape}")

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
