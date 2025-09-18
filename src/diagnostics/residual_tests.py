"""Residual diagnostic tests for regression models."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.stattools import durbin_watson


def run_residual_diagnostics(residuals: np.ndarray, fitted: np.ndarray) -> Dict[str, dict]:
    """Execute a suite of classical residual diagnostics."""

    residuals = np.asarray(residuals)
    fitted = np.asarray(fitted)

    diagnostics = {}

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    diagnostics["shapiro_wilk"] = {
        "statistic": float(shapiro_stat),
        "p_value": float(shapiro_p),
    }

    # Encode residual signs for the Wald-Wolfowitz runs test of randomness.
    signs = np.where(residuals >= 0, 1, -1)
    runs_stat, runs_p = runstest_1samp(signs, cutoff=0)
    diagnostics["runs_test"] = {
        "statistic": float(runs_stat),
        "p_value": float(runs_p),
    }

    dw_stat = durbin_watson(residuals)
    diagnostics["durbin_watson"] = {
        "statistic": float(dw_stat),
        "p_value": None,
    }

    spearman_stat, spearman_p = stats.spearmanr(fitted, residuals)
    diagnostics["spearman_rho"] = {
        "statistic": float(spearman_stat),
        "p_value": float(spearman_p),
    }

    diagnostics["skewness"] = {
        "statistic": float(stats.skew(residuals, bias=False)),
        "p_value": None,
    }

    return diagnostics
