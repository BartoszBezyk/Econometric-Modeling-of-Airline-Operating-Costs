"""Data loading and preprocessing utilities."""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["FuelPrice", "LoadFactor", "Output", "Year"]
CATEGORICAL_FEATURES = ["AirlineID"]

def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV file into a pandas DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(path)

def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric data and encodes categoricals."""
    numeric_pipeline = StandardScaler()
    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
