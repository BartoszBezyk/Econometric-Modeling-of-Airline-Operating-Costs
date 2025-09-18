# Econometric Modeling of Airline Operating Costs

## Overview

This repository contains a Python, production-style reimplementation of the original notebook that modelled airline operating costs. The code base keeps the original econometric intent—feature selection with the Information Capacity Index (ICA) and correlation screening, linear modelling under different target transformations, and a comprehensive suite of residual diagnostics—while organising the workflow into reusable modules.

## Dataset

PanelData.csv stores six airlines observed annually for fifteen years. The raw column symbols (I, T, C, Q, PF, LF) are mapped to descriptive names (AirlineID, Year, TotalCost, Output, FuelPrice, LoadFactor) during loading. The target variable is TotalCost and the candidate explanatory variables mirror the original study.

## Project Structure

`
src/
  config.py                # Shared settings (paths, feature lists, seeds)
  data_loading.py          # CSV ingestion and column harmonisation
  preprocessing.py         # Stratified train/test split and X/y extraction
  feature_selection/
    information_capacity.py  # ICA scoring and subset ranking
    correlation_threshold.py # Correlation-based screening variants
  modeling/
    regression.py            # Ordinary least squares via scikit-learn
  evaluation/
    metrics.py               # Adjusted R^2, AIC, BIC, RMSE, MAE, MAPE
  diagnostics/
    residual_tests.py        # Shapiro-Wilk, runs, Durbin-Watson, Spearman, skewness
  pipeline.py              # End-to-end orchestration utilities (reused programmatically)
main.py                   # Entry point delegating to the linear workflow
workflow.py               # Step-by-step pipeline execution with rich logging
artifacts/                # Generated test-set prediction tables
`

Each public function includes a docstring describing both intent and relevant statistical background, making it straightforward to extend the project with additional selection methods or model classes.

## Workflow Summary

1. **Load & rename** the panel data (src.data_loading).
2. **Stratified split** per airline to maintain panel balance (src.preprocessing).
3. **Feature selection** using ICA (top three combinations) and two correlation-threshold strategies (src.feature_selection).
4. **Model fitting** with scikit-learn linear regression on raw, log-transformed, and square-root-transformed targets (src.modeling).
5. **Evaluation** via adjusted R^2, AIC, BIC, and test-set R^2/RMSE/MAE/MAPE alongside saved prediction tables (src.evaluation).
6. **Diagnostics pipeline** covering normality, randomness, autocorrelation, heteroscedasticity, and skewness with human-readable decisions (src.diagnostics).

## Running the Workflow

`
python workflow.py
`

or equivalently:

`
python main.py
`

The script prints a model-comparison table (sorted by AIC), detailed test metrics, residual-diagnostic interpretations, and stores per-model test predictions under rtifacts/.

## Extending the Project

- Add new feature selectors by creating modules under src/feature_selection and registering them in pipeline.build_candidate_model_specs.
- Introduce alternative model families (e.g., regularised regression) by extending src/modeling and updating the pipeline to emit additional ModelSpec entries.
- Expand diagnostic coverage inside src/diagnostics to keep the downstream reporting automatic.

Dependencies: pandas, 
umpy, scikit-learn, scipy, and statsmodels.
