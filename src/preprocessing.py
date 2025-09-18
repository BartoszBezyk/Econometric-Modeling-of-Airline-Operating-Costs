"""Data preparation helpers for modelling airline operating costs."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN, CANDIDATE_FEATURES, DEFAULT_RANDOM_STATE, TRAIN_SIZE


def stratified_train_test_split(
    df: pd.DataFrame,
    group_column: str = "AirlineID",
    train_size: float = TRAIN_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the panel data into stratified training and test sets.

    The split mirrors the original notebook: each airline contributes the same
    proportion of observations to the training set, preserving the
    cross-sectional balance of the panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Complete dataset containing feature columns and the target variable.
    group_column : str, default `"AirlineID"`
        Column used to stratify the split so every airline remains represented
        in both subsets.
    train_size : float, default `config.TRAIN_SIZE`
        Proportion of observations per group allocated to the training set.
    random_state : int, default `config.DEFAULT_RANDOM_STATE`
        Seed for the sampling procedure to maintain reproducibility.

    Returns
    -------
    (train_df, test_df) : tuple of pandas.DataFrame
        Stratified training and test subsets sorted by their original row order.
    """

    train_segments = []
    test_segments = []

    # Sample within each airline so the holdout remains balanced.
    for _, segment in df.groupby(group_column):
        n_train = int(np.floor(len(segment) * train_size))
        train_sample = segment.sample(n=n_train, random_state=random_state, replace=False)
        test_sample = segment.drop(train_sample.index)
        train_segments.append(train_sample)
        test_segments.append(test_sample)

    train_df = pd.concat(train_segments).sort_index()
    test_df = pd.concat(test_segments).sort_index()
    return train_df, test_df


def split_features_target(
    df: pd.DataFrame,
    features: Iterable[str] = CANDIDATE_FEATURES,
    target: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix `X` and target vector `y` from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing all required columns.
    features : iterable of str, optional
        Feature names to include in the design matrix. Defaults to the full
        candidate list from :mod:src.config.
    target : str, default `"TotalCost"`
        Dependent variable to be predicted.

    Returns
    -------
    X, y : tuple
        `X` is a DataFrame of predictors, `y` is a Series representing the
        target variable.
    """

    feature_list = list(features)
    # Maintain ordering so coefficients align with the requested feature set.
    return df[feature_list], df[target]
