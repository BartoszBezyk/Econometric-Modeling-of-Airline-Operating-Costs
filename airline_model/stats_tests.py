"""Statistical tests used for residual analysis."""

from typing import Tuple
import numpy as np
from scipy import stats
import statsmodels.api as sm


def shapiro_wilk(residuals: np.ndarray) -> Tuple[float, float]:
    """Shapiro-Wilk test for normality."""
    stat, p = stats.shapiro(residuals)
    return float(stat), float(p)


def runs_test(residuals: np.ndarray) -> Tuple[float, float]:
    """Two-sided runs test for randomness."""
    signs = residuals > 0
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1 = np.sum(signs)
    n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    expected = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - expected) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for autocorrelation."""
    return float(sm.stats.durbin_watson(residuals))


def spearman_correlation(fitted: np.ndarray, residuals: np.ndarray) -> Tuple[float, float]:
    """Spearman rank correlation between fitted values and residuals."""
    rho, p = stats.spearmanr(fitted, residuals)
    return float(rho), float(p)


def skewness(residuals: np.ndarray) -> float:
    """Skewness of residuals."""
    return float(stats.skew(residuals))
