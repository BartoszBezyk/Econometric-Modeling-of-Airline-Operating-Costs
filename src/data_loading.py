"""Data access utilities for the airline operating cost dataset."""

from __future__ import annotations

import pandas as pd

from .config import DATA_PATH, TARGET_COLUMN, CANDIDATE_FEATURES

# Map raw column codes to descriptive labels for readability downstream.
_COLUMN_RENAME_MAP = {
    "I": "AirlineID",
    "T": "Year",
    "C": "TotalCost",
    "Q": "Output",
    "PF": "FuelPrice",
    "LF": "LoadFactor",
}


def load_panel_data(path: str | None = None) -> pd.DataFrame:
    """Return the econometric panel dataset with consistent column names.

    Parameters
    ----------
    path : str or Path-like, optional
        Location of the `PanelData.csv` file. When `None` the value from
        :mod:src.config is used.

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame containing the target (`TotalCost`) and all candidate
        explanatory variables. The raw column symbols from the source file are
        mapped to descriptive names via `_COLUMN_RENAME_MAP`.

    Notes
    -----
    The dataset spans six airlines observed annually across fifteen years. It
    captures production volume (`Output`), fuel price, fleet utilisation
    (`LoadFactor`) and identifiers that enable panel-style modelling.
    """

    csv_path = DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_COLUMN_RENAME_MAP)
    ordered_columns = [TARGET_COLUMN, *CANDIDATE_FEATURES]
    # Ensure the frame only exposes the columns expected by the modelling code.
    return df[ordered_columns]
