"""CLI entry point for training and evaluating the airline cost model."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from airline_model.training import TrainConfig, train_model, split_features_target
from airline_model.evaluation import rmse, mape
from airline_model.inference import load_model
from airline_model.preprocessing import load_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate airline cost model")
    parser.add_argument("data", help="Path to CSV dataset")
    parser.add_argument("--model", default="model.joblib", help="Path to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    args = parser.parse_args()

    config = TrainConfig(data_path=args.data, model_path=args.model, test_size=args.test_size)
    model = train_model(config)

    df = load_data(args.data)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    preds = model.predict(X_test) ** 2  # inverse sqrt transform
    y_true = y_test ** 2

    print(f"RMSE: {rmse(y_true, preds):.4f}")
    print(f"MAPE: {mape(y_true, preds):.2f}%")


if __name__ == "__main__":
    main()
