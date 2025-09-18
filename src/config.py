"""Configuration values for the econometric modeling project."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "PanelData.csv"

TARGET_COLUMN = "TotalCost"
CANDIDATE_FEATURES = ["FuelPrice", "LoadFactor", "Output", "AirlineID", "Year"]
DEFAULT_RANDOM_STATE = 155
TRAIN_SIZE = 0.8
