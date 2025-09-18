"""Model evaluation metrics used throughout the project."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_parameters: int) -> float:
    """Return the adjusted coefficient of determination."""

    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return 1 - (1 - r2) * (n - 1) / (n - n_parameters - 1)


def aic(y_true: np.ndarray, y_pred: np.ndarray, n_parameters: int) -> float:
    """Compute Akaike's Information Criterion for a Gaussian linear model."""

    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    return n * np.log(rss / n) + 2 * n_parameters


def bic(y_true: np.ndarray, y_pred: np.ndarray, n_parameters: int) -> float:
    """Compute the Bayesian Information Criterion for a Gaussian linear model."""

    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    return n * np.log(rss / n) + n_parameters * np.log(n)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error between observed and predicted values."""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error between observed and predicted values."""

    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error expressed in percent."""

    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2) for completeness."""

    return float(r2_score(y_true, y_pred))
