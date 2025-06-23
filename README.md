# Econometric Modeling of Airline Operating Costs

## Project Overview

This project provides a simple production-style pipeline for modelling airline operating costs using Python. The goal is to predict the `TotalCost` for each airline based on historical operating information. A scikit-learn pipeline is used for preprocessing, linear regression modelling and evaluation.

## Data Source

The [dataset](https://www.kaggle.com/code/sandhyakrishnan02/econometric-analysis-of-panel-data-using-r) used in this analysis includes data from six airlines over fifteen years. The key variables are:

- **Year** – year of observation
- **AirlineID** – identifier of the airline
- **TotalCost** – operating cost
- **Output** – passenger miles
- **FuelPrice** – fuel price
- **LoadFactor** – fleet utilisation

The dataset should be saved locally in CSV format before running the pipeline.
By default, `run_pipeline.py` looks for `PanelData.csv` next to the script.

## Usage

Install the dependencies and run the pipeline script:

```bash
pip install -r requirements.txt
python run_pipeline.py --model model.joblib
```

The script trains the model, evaluates it on a hold-out set and prints RMSE and MAPE metrics. It also reports diagnostic statistics (Shapiro-Wilk, runs, Durbin-Watson, Spearman correlation and residual skewness). The trained model is saved to `model.joblib` by default.

## Implementation Notes

The pipeline is implemented in the `airline_model` package. Data loading and preprocessing logic is separated from model training, inference and evaluation to make the project easier to maintain and extend.
