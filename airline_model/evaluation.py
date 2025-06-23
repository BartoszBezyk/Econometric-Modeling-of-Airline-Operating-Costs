"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    return mean_squared_error(y_true, y_pred, squared=False)


def mape(y_true, y_pred) -> float:
    """Return mean absolute percentage error."""
    diff = np.abs((y_true - y_pred) / y_true)
    return float(diff.mean() * 100)
