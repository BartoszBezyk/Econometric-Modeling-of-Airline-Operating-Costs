# Econometric Modeling of Airline Operating Costs

## Project Overview

This project aims to develop an econometric model to predict the operating costs (`TotalCost`) for various airlines over a 15-year period. By analyzing historical data and employing different statistical methods, the project seeks to identify key factors that influence airline costs and build a reliable predictive model.
Analysis was performed in R programming language.

## Data Source

The dataset used in this analysis includes data from six different airlines, covering a span of 15 years. The key variables in the dataset are:
- **Year**: The year of observation.
- **AirlineID**: Identifier for each airline.
- **TotalCost**: The total operating cost incurred by the airline.
- **Output**: Measured in passenger miles, representing the production output of the airline.
- **FuelPrice**: The price of fuel, a significant component of operating costs.
- **LoadFactor**: The utilization rate of the airline's fleet, indicating efficiency in passenger space usage.

## Statistical Methods Used

Several statistical methods were employed in this project to develop and refine the econometric models:

- **Information Capacity Analysis (ICA)**: A method used to assess the information capacity of different combinations of explanatory variables, helping to identify the most informative sets of variables for model building.

- **Correlation Coefficient Analysis**: This technique was applied to evaluate the strength and direction of the relationship between the dependent variable (`TotalCost`) and potential explanatory variables. It was used in two rounds to refine the selection of variables.

- **Data Transformation**: The `TotalCost` variable was transformed using square root and logarithmic functions to stabilize variance and improve the model's predictive accuracy.

- **Shapiro-Wilk Test**: A test for normality that was used to verify whether the residuals of the model followed a normal distribution.

- **Runs Test**: This test was employed to assess the randomness of the residuals, ensuring that they do not exhibit any systematic patterns.

- **Durbin-Watson Test**: A test used to detect the presence of autocorrelation in the residuals, which can indicate issues with model specification.

- **Spearman Rank Correlation Test**: This test was conducted to check for homoscedasticity, ensuring that the variance of the residuals is consistent across all levels of the fitted values.

## Model Selection

The final model, `model_ICA1_sqrt`, was selected based on the highest adjusted R-squared value and the lowest AIC and BIC values. The model includes the following variables: `FuelPrice`, `LoadFactor`, `Output`, `AirlineID`, and `Year`.

## Model Evaluation

The model was rigorously tested using several diagnostic tests:
- **RMSE (Root Mean Square Error)**: 63.3935
- **MAPE (Mean Absolute Percentage Error)**: 6.472811%

These metrics indicate that the model has a good predictive performance, with the MAPE suggesting an average prediction error of approximately 6.47%.

## Results Summary

The `model_ICA1_sqrt` was identified as the best-performing model, explaining approximately 98.32% of the variance in airline operating costs. The model passed several diagnostic checks, confirming that it meets the assumptions required for a reliable predictive model. The RMSE of 63.3935 and MAPE of 6.472811% indicate that the model is both accurate and robust, making it a valuable tool for forecasting airline costs.
