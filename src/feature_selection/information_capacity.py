"""Information Capacity Index feature selection."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import TARGET_COLUMN


def _single_information_capacity(
    variable: str,
    target_correlations: pd.Series,
    absolute_feature_correlations: pd.DataFrame,
) -> float:
    r"""Return the information capacity of a single variable.

    The statistic follows Nowosad (1973). For a given explanatory variable `j`
    in combination `k` the information capacity is

    .. math:: h_{kj} = \frac{r_j^2}{1 + \sum_{\ell=1}^{m_k} |r_{j\ell}|},

    where `r_j` is the correlation between the dependent variable and the
    `j`\ th predictor, and `r_{j\ell}` are absolute correlations between
    predictor `j` and the remaining predictors. Higher values denote stronger
    individual contribution given multicollinearity among predictors.
    """

    r_j = target_correlations[variable]
    rj_squared = float(r_j) ** 2
    sum_rlj = float(absolute_feature_correlations.loc[variable].sum())
    # Higher denominators down-weight variables entangled with others.
    return rj_squared / (1.0 + sum_rlj)


def information_capacity_scores(
    df: pd.DataFrame,
    features: Iterable[str],
    target: str = TARGET_COLUMN,
) -> pd.Series:
    """Compute information capacity scores for individual variables."""

    columns = [target, *features]
    corr = df[columns].corr()
    target_correlations = corr.loc[target, features]
    absolute_feature_correlations = corr.loc[features, features].abs()

    scores = {
        feature: _single_information_capacity(feature, target_correlations, absolute_feature_correlations)
        for feature in features
    }
    return pd.Series(scores).sort_values(ascending=False)


def top_information_capacity_combinations(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = TARGET_COLUMN,
    top_k: int = 3,
) -> List[Tuple[Tuple[str, ...], float]]:
    r"""Return the top `k` feature combinations ranked by information capacity.

    The total capacity of a combination `k` is defined as

    .. math:: H_k = \sum_{j=1}^{m_k} h_{kj},

    i.e. the sum of the individual capacities for features included in the
    subset. The function enumerates all non-empty subsets, evaluates `H_k` and
    returns the `top_k` with the highest scores.
    """

    columns = [target, *features]
    corr = df[columns].corr()
    target_correlations = corr.loc[target, features]
    absolute_feature_correlations = corr.loc[features, features].abs()

    combination_scores = []

    # Iterate through every subset size and accumulate the combined capacity.
    for r in range(1, len(features) + 1):
        for combo in itertools.combinations(features, r):
            h_values = [
                _single_information_capacity(feature, target_correlations, absolute_feature_correlations)
                for feature in combo
            ]
            combination_scores.append((combo, float(np.sum(h_values))))

    combination_scores.sort(key=lambda item: item[1], reverse=True)
    return combination_scores[:top_k]
