# Air Quality Trend Predictor — Linear Regression Only

A small, beginner-friendly Python project that predicts next-hour PM2.5 concentrations using the Beijing Multi-Site Air Quality dataset. It uses **Linear Regression as the only predictive model**.

## Run it on Windows (minimal steps)

1. Extract this ZIP file to a normal folder, such as `Desktop\Air_Quality_Trend_Predictor`.
2. Install Python 3.10 or newer from <https://www.python.org/downloads/>. During setup, check **Add Python to PATH**.
3. Double-click `run_project.bat`.
4. The first run creates a local virtual environment and installs the required packages. An internet connection is needed for package installation.
5. When it finishes, open the `outputs` folder. The graphs, evaluation metrics, predictions, and saved model will be there.

No notebook, browser, dashboard, or manual data upload is required. You can also run it from a terminal using `python main.py` after installing `requirements.txt`.

## What the project does

1. Loads all 12 station CSV files included in `data/`.
2. Combines the hourly observations and builds a timestamp.
3. Treats `-200` sensor readings as missing values, removes duplicate station-hour rows, and reports missingness.
4. Saves a PM2.5 trend graph, missing-value graph, and station-average comparison.
5. Uses current pollutant/weather measurements, station and wind-direction categories, hour/month, and PM2.5 readings from 1, 3, and 6 hours earlier as inputs.
6. Predicts the same station's PM2.5 concentration one hour ahead.
7. Splits by chronological target timestamp (first 80% for training, later 20% for testing) instead of random splitting.
8. Learns missing-value imputation from the training set only and uses a single **Linear Regression** estimator.
9. Evaluates with MAE, and also reports RMSE and R². It saves actual-versus-predicted graphs and test predictions.
10. Produces an illustrative next-hour estimate for each station using the latest available historical observations.

## Main outputs

- `outputs/evaluation_metrics.txt` — MAE, RMSE, R², and split details.
- `outputs/data_cleaning_summary.txt` — dataset size, dates, duplicates, and missingness summary.
- `outputs/daily_pm25_trend.png` — pollution trend and 30-day moving average.
- `outputs/missing_values.png` — missing sensor readings by column.
- `outputs/station_comparison.png` — mean PM2.5 by station.
- `outputs/actual_vs_predicted.png` — average actual and predicted PM2.5 over the final test week.
- `outputs/actual_vs_predicted_scatter.png` — actual versus predicted test values.
- `outputs/prediction_errors.png` — distribution of prediction errors.
- `outputs/test_predictions.csv` — every test prediction, actual value, and absolute error.
- `outputs/next_hour_forecast_by_station.csv` — one illustrative next-hour estimate per station.
- `outputs/linear_regression_pipeline.joblib` — trained preprocessing + Linear Regression pipeline.
- `outputs/linear_regression_coefficients.csv` — fitted linear coefficients for interpretation.

The `outputs` folder is generated automatically and is not required to be present in the ZIP.

## Dataset and limitations

Dataset: **Beijing Multi-Site Air Quality**, UCI Machine Learning Repository, dataset 501. It contains hourly pollution and weather observations from 12 Beijing monitoring stations, from March 2013 to February 2017.

Official source: <https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data>

Citation: Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5RK5G>

Dataset license: **CC BY 4.0**. Credit the UCI repository and dataset creator when reusing the data.

The included dataset is historical. The CSV named `next_hour_forecast_by_station.csv` demonstrates the prediction process after the historical final timestamp; it is **not a live forecast for current air quality**. It uses the latest observed weather and pollutant values because future sensor/weather inputs are not available in this dataset.

## Concepts in plain language

- **PM2.5:** tiny airborne particles; the project predicts their measured concentration.
- **Feature/input:** information the model can use to make a prediction, such as recent PM2.5 or temperature.
- **Target/label:** the value to predict: PM2.5 one hour after the input timestamp.
- **Lag:** a previous reading. For example, `PM2.5_lag_3h` is PM2.5 recorded three hours earlier at the same station.
- **Chronological split:** train using earlier dates and evaluate on later dates, closer to how a forecast is used in practice.
- **Missing-value imputation:** fill missing input fields with representative values learned from the training set. Rows without a known future PM2.5 value cannot be used as training/evaluation labels.
- **MAE:** average absolute prediction error in the same units as PM2.5. Lower is generally better on the same test set.
