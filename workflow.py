"""Linear workflow script executing the econometric modelling steps."""

from __future__ import annotations

from src.data_loading import load_panel_data
from src.preprocessing import stratified_train_test_split
from src.pipeline import (
    ARTIFACTS_DIR,
    build_candidate_model_specs,
    evaluate_spec,
    summarise_results,
)

# Human-readable notes for residual diagnostics.
DIAGNOSTIC_HYPOTHESES = {
    "shapiro_wilk": "H0: residuals follow a normal distribution",
    "runs_test": "H0: residual signs occur in random order",
    "spearman_rho": "H0: no monotonic relation between fitted values and residuals",
}

if __name__ == "__main__":
    print("[1/6] Loading dataset...")
    data = load_panel_data()
    print(f"    Loaded {len(data)} rows with columns: {list(data.columns)}")

    print("[2/6] Creating stratified train/test split...")
    train_df, test_df = stratified_train_test_split(data)
    print(f"    Train size: {train_df.shape}, Test size: {test_df.shape}")

    print("[3/6] Building model specifications...")
    specifications = build_candidate_model_specs(train_df)
    for spec in specifications:
        print(f"    - {spec.name}: features -> {spec.features}")

    print("[4/6] Evaluating specifications...")
    results = []
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    for spec in specifications:
        print(f"    > Fitting {spec.name}")
        result = evaluate_spec(train_df, test_df, spec)
        results.append(result)

        metrics = result.test_metrics
        print(
            "      Test metrics: "
            f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%"
        )

        head_df = result.test_predictions.head()
        print("      Prediction sample (first 5 rows):")
        print(head_df)

        diagnostics = result.diagnostics
        print("      Residual diagnostics:")
        for name, outcome in diagnostics.items():
            statistic = outcome["statistic"]
            p_value = outcome.get("p_value")
            if p_value is None:
                if name == "durbin_watson":
                    if statistic < 1.5:
                        message = "Possible positive autocorrelation (DW far below 2)."
                    elif statistic > 2.5:
                        message = "Possible negative autocorrelation (DW far above 2)."
                    else:
                        message = "DW close to 2 -> no strong autocorrelation signal."
                elif name == "skewness":
                    if abs(statistic) < 0.1:
                        message = "Skewness near zero -> residuals roughly symmetric."
                    else:
                        message = "Noticeable skewness -> consider transformation or robust errors."
                else:
                    message = "(No decision rule available.)"
            else:
                decision = "fail to reject H0" if p_value > 0.05 else "reject H0"
                hypothesis = DIAGNOSTIC_HYPOTHESES.get(name, "H0 statement not documented")
                message = f"{hypothesis} | decision: {decision} (p={p_value:.4f})."
            label = name.replace("_", " ").title()
            print(f"        {label}: {message}")

        output_path = ARTIFACTS_DIR / f"{spec.name}_test_predictions.csv"
        result.test_predictions.to_csv(output_path, index=True)
        print(f"      Saved test predictions to {output_path}")

    print("[5/6] Summarising model performance...")
    summary_table = summarise_results(results)
    print(summary_table.to_string(index=False))

    best_name = summary_table.iloc[0]["model"]
    print(f"[6/6] Best model by AIC: {best_name}")
