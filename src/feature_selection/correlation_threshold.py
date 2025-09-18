"""Correlation-based variable screening methods."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..config import TARGET_COLUMN


def critical_r_value(sample_size: int, alpha: float = 0.05) -> float:
    r"""Return the critical correlation magnitude `r*` for a two-sided t-test.

    The statistic follows

    .. math:: r^* = \sqrt{ \frac{t_{\alpha/2, n-2}^2}{t_{\alpha/2, n-2}^2 + n - 2} },

    where :math:	_{\alpha/2, n-2} is the critical value of the Student's
    :math:	 distribution with `n-2` degrees of freedom. Correlations whose
    absolute value exceed `r*` are deemed statistically significant at level
    `alpha`.
    """

    t_critical = stats.t.ppf(1 - alpha / 2, df=sample_size - 2)
    return float(np.sqrt(t_critical**2 / (t_critical**2 + sample_size - 2)))


def select_via_critical_r(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = TARGET_COLUMN,
    alpha: float = 0.05,
) -> List[str]:
    """Select variables exceeding a critical correlation magnitude `r*`.

    The procedure mirrors the first correlation-screening variant in the
    notebook:

    1. Retain predictors with `|corr(target, feature)| > r*`.
    2. Choose the feature with the largest absolute correlation to the target.
    3. Remove any remaining feature whose correlation with the chosen one also
       exceeds `r*` (helps to limit multicollinearity).

    Returns the final feature list in selection order.
    """

    correlations = df[[target, *features]].corr()
    target_corr = correlations.loc[target, features]
    r_star = critical_r_value(sample_size=len(df), alpha=alpha)

    eligible = target_corr[np.abs(target_corr) > r_star]
    if eligible.empty:
        return []

    anchor = eligible.abs().idxmax()
    remaining = [
        feature
        for feature in eligible.index
        if np.abs(correlations.loc[anchor, feature]) <= r_star or feature == anchor
    ]

    ordered = [anchor] + [feat for feat in remaining if feat != anchor]
    return ordered


def select_via_custom_threshold(
    df: pd.DataFrame,
    features: Sequence[str],
    r_star: float,
    target: str = TARGET_COLUMN,
) -> List[str]:
    """Repeat the correlation-screening algorithm with a user-defined `r*`.

    The original notebook set `r*` to the smallest absolute entry of the
    target-correlation vector (:math:R_0). This helper makes the threshold
    explicit so that alternative heuristics can be explored.
    """

    correlations = df[[target, *features]].corr()
    target_corr = correlations.loc[target, features]

    eligible = target_corr[np.abs(target_corr) > r_star]
    if eligible.empty:
        return []

    anchor = eligible.abs().idxmax()
    remaining = [
        feature
        for feature in eligible.index
        if np.abs(correlations.loc[anchor, feature]) <= r_star or feature == anchor
    ]

    ordered = [anchor] + [feat for feat in remaining if feat != anchor]
    return ordered
