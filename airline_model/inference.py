"""Model inference utilities."""

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_model(model_path: str):
    """Load a trained model from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for prediction."""
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def predict(model, data: Iterable[dict]) -> pd.Series:
    """Predict total costs from an iterable of row dictionaries."""
    df = pd.DataFrame(data)
    X = prepare_features(df)
    preds = model.predict(X)
    return pd.Series(preds ** 2, name="TotalCost")  # inverse sqrt transform
