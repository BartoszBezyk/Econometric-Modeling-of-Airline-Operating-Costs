"""Project pipeline orchestrating data preparation, modelling and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CANDIDATE_FEATURES, TARGET_COLUMN
from .data_loading import load_panel_data
from .diagnostics.residual_tests import run_residual_diagnostics
from .evaluation.metrics import aic, adjusted_r2, bic, mae, mape, r2, rmse
from .feature_selection.correlation_threshold import (
    select_via_critical_r,
    select_via_custom_threshold,
)
from .feature_selection.information_capacity import (
    top_information_capacity_combinations,
)
from .modeling.regression import fit_linear_regression, predict
from .preprocessing import split_features_target, stratified_train_test_split

Transform = Callable[[np.ndarray], np.ndarray]

# Support directory for saving artefacts such as prediction tables.
ARTIFACTS_DIR = Path("artifacts")


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _identity(values: np.ndarray) -> np.ndarray:
    return _as_array(values)


def _log_transform(values: np.ndarray) -> np.ndarray:
    return np.log(_as_array(values))


def _exp_inverse(values: np.ndarray) -> np.ndarray:
    return np.exp(_as_array(values))


def _sqrt_transform(values: np.ndarray) -> np.ndarray:
    return np.sqrt(_as_array(values))


def _square_inverse(values: np.ndarray) -> np.ndarray:
    arr = _as_array(values)
    return arr**2


@dataclass(frozen=True)
class ModelSpec:
    """Container describing a candidate regression specification."""

    name: str
    features: Sequence[str]
    target_transform: Transform = _identity
    inverse_transform: Transform = _identity
    notes: str | None = None


@dataclass
class ModelResult:
    """Holds fitted model artefacts and evaluation outputs."""

    spec: ModelSpec
    model: object
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    diagnostics: Dict[str, Dict[str, float]]
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def build_candidate_model_specs(train_df: pd.DataFrame) -> List[ModelSpec]:
    """Construct model specifications based on information capacity and correlation heuristics."""

    specs: List[ModelSpec] = []

    ica_candidates = top_information_capacity_combinations(
        train_df,
        features=CANDIDATE_FEATURES,
        target=TARGET_COLUMN,
        top_k=3,
    )
    for idx, (combo, _) in enumerate(ica_candidates, start=1):
        specs.append(
            ModelSpec(name=f"ICA_{idx}_raw", features=list(combo))
        )

    if ica_candidates:
        best_features = list(ica_candidates[0][0])
        specs.extend(
            [
                ModelSpec(
                    name="ICA_best_log",
                    features=best_features,
                    target_transform=_log_transform,
                    inverse_transform=_exp_inverse,
                    notes="Log-transform of the dependent variable",
                ),
                ModelSpec(
                    name="ICA_best_sqrt",
                    features=best_features,
                    target_transform=_sqrt_transform,
                    inverse_transform=_square_inverse,
                    notes="Square-root transform of the dependent variable",
                ),
            ]
        )

    corr_features = select_via_critical_r(train_df, CANDIDATE_FEATURES, TARGET_COLUMN)
    if corr_features:
        specs.append(ModelSpec(name="Corr_rstar_raw", features=corr_features))

    correlations = train_df[[TARGET_COLUMN, *CANDIDATE_FEATURES]].corr()
    r0 = correlations.loc[TARGET_COLUMN, CANDIDATE_FEATURES]
    r_star_alt = float(np.abs(r0).min())
    corr_alt = select_via_custom_threshold(
        train_df,
        CANDIDATE_FEATURES,
        r_star=r_star_alt,
        target=TARGET_COLUMN,
    )
    if corr_alt:
        specs.append(
            ModelSpec(
                name="Corr_custom_raw",
                features=corr_alt,
                notes=f"Custom threshold r*={r_star_alt:.3f}",
            )
        )

    return specs


def evaluate_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: ModelSpec,
) -> ModelResult:
    """Fit the specified model and return diagnostics and performance metrics."""

    X_train, y_train_series = split_features_target(train_df, spec.features, TARGET_COLUMN)
    X_test, y_test_series = split_features_target(test_df, spec.features, TARGET_COLUMN)

    y_train_original = y_train_series.to_numpy()
    y_test_original = y_test_series.to_numpy()

    # Apply target transformation when the specification requests it.
    y_train_transformed = spec.target_transform(y_train_original)
    model, fitted_values = fit_linear_regression(X_train, y_train_transformed)
    residuals = y_train_transformed - fitted_values

    n_features = X_train.shape[1]
    train_predictions_original = spec.inverse_transform(fitted_values)
    train_metrics = {
        "adj_r2": adjusted_r2(y_train_transformed, fitted_values, n_features),
        "r2_original": r2(y_train_original, train_predictions_original),
        "aic": aic(y_train_transformed, fitted_values, n_features + 1),
        "bic": bic(y_train_transformed, fitted_values, n_features + 1),
        "rmse_transformed": rmse(y_train_transformed, fitted_values),
        "rmse_original": rmse(y_train_original, train_predictions_original),
        "mae_original": mae(y_train_original, train_predictions_original),
        "mape_original": mape(y_train_original, train_predictions_original),
    }

    predictions_transformed = predict(model, X_test)
    predictions_original = spec.inverse_transform(predictions_transformed)

    test_metrics = {
        "r2": r2(y_test_original, predictions_original),
        "rmse": rmse(y_test_original, predictions_original),
        "mae": mae(y_test_original, predictions_original),
        "mape": mape(y_test_original, predictions_original),
    }

    diagnostics = run_residual_diagnostics(residuals, fitted_values)

    train_predictions = pd.DataFrame(
        {
            "y_true": y_train_original,
            "y_pred": train_predictions_original,
            "residual": y_train_original - train_predictions_original,
        },
        index=X_train.index,
    )

    test_predictions = pd.DataFrame(
        {
            "y_true": y_test_original,
            "y_pred": predictions_original,
            "residual": y_test_original - predictions_original,
        },
        index=X_test.index,
    )

    return ModelResult(
        spec=spec,
        model=model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        diagnostics=diagnostics,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
    )


def summarise_results(results: Sequence[ModelResult]) -> pd.DataFrame:
    """Create a tabular summary of model metrics for quick comparison."""

    rows = []
    for result in results:
        row = {
            "model": result.spec.name,
            "features": ", ".join(result.spec.features),
            "adj_r2": result.train_metrics["adj_r2"],
            "aic": result.train_metrics["aic"],
            "bic": result.train_metrics["bic"],
            "test_r2": result.test_metrics["r2"],
            "test_rmse": result.test_metrics["rmse"],
            "test_mae": result.test_metrics["mae"],
            "test_mape": result.test_metrics["mape"],
            "notes": result.spec.notes or "",
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values("aic").reset_index(drop=True)


def run_pipeline() -> Tuple[pd.DataFrame, List[ModelResult]]:
    """Execute the full modelling workflow and return summary and detailed results."""

    df = load_panel_data()
    train_df, test_df = stratified_train_test_split(df)
    specs = build_candidate_model_specs(train_df)
    results = [evaluate_spec(train_df, test_df, spec) for spec in specs]
    summary = summarise_results(results)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    for result in results:
        filename = ARTIFACTS_DIR / f"{result.spec.name}_test_predictions.csv"
        result.test_predictions.to_csv(filename, index=True)

    return summary, results
