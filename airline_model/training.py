"""Model training utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "PanelData.csv"

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .preprocessing import build_preprocessor, load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

@dataclass
class TrainConfig:
    data_path: Path = DEFAULT_DATA_PATH
    model_path: str = "model.joblib"
    test_size: float = 0.2
    random_state: int = 42

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns and target variable."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["TotalCost"].pow(0.5)  # sqrt transform as used in original analysis
    return X, y

def train_model(config: TrainConfig) -> Pipeline:
    """Train a regression model and persist it to disk."""
    df = load_data(config.data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    return model
