"""Regression modelling utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_linear_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[LinearRegression, np.ndarray]:
    """Fit an ordinary least squares model using scikit-learn."""

    model = LinearRegression()
    model.fit(X, y)
    fitted_values = model.predict(X)
    # Return both estimator and in-sample fits for diagnostics.
    return model, fitted_values


def predict(model: LinearRegression, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained linear regression model."""

    return model.predict(X)
