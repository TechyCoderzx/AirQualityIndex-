# Air Quality Trend Predictor

**From historical air-quality readings to next-hour PM2.5 estimates.**

A compact, beginner-friendly machine learning project that uses the **Beijing Multi-Site Air Quality dataset** to predict PM2.5 concentrations one hour ahead. The project focuses on data cleaning, time-based feature engineering, visualization, and model evaluation using **Linear Regression as its only predictive model**.

---

## Run the project on Windows

You do not need Jupyter Notebook, a browser, a dashboard, or manual data uploads.

1. Extract the project ZIP to a regular folder, for example `Desktop\Air_Quality_Trend_Predictor`.
2. Install [Python 3.10 or newer](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
3. Double-click **`run_project.bat`**.
4. On its first run, the script creates a local virtual environment and installs the required packages. An internet connection is needed for this setup step.
5. Once execution finishes, open the **`outputs/`** folder to view the generated graphs, evaluation metrics, predictions, and saved model.

### Run it from a terminal instead

Open a terminal in the project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## What happens inside the project?

The project follows a time-aware machine learning workflow:

1. **Load the data** — reads all 12 station CSV files included in `data/` and combines their hourly observations.
2. **Prepare timestamps** — builds a timestamp for each observation and removes duplicate station-hour records.
3. **Clean sensor readings** — treats `-200` as a missing-value marker and summarizes missing data.
4. **Explore the dataset** — generates visualizations of PM2.5 trends, missing readings, and average pollution by station.
5. **Build input features** — uses current pollutant and weather measurements, station and wind-direction categories, hour and month, and same-station PM2.5 readings from 1, 3, and 6 hours earlier.
6. **Define the prediction target** — predicts the same station's PM2.5 concentration one hour after the input timestamp.
7. **Split by time** — uses the earlier 80% of target timestamps for training and the later 20% for testing, rather than randomly shuffling observations.
8. **Train the model** — learns missing-value imputation from the training data only, then fits a single **Linear Regression** estimator.
9. **Evaluate predictions** — reports MAE, RMSE, and R², and saves actual-versus-predicted visualizations and test predictions.
10. **Generate example forecasts** — produces an illustrative next-hour estimate for each station using its latest available historical observations.

### Workflow at a glance

`12 station CSVs` → `Data cleaning` → `Exploratory graphs` → `Time-based features` → `Chronological split` → `Linear Regression` → `Evaluation & forecasts`

---

## Generated outputs

All generated files are saved in `outputs/`.

| File | What it contains |
|---|---|
| `evaluation_metrics.txt` | MAE, RMSE, R², and chronological split details. |
| `data_cleaning_summary.txt` | Dataset size, date range, duplicate counts, and missingness summary. |
| `daily_pm25_trend.png` | Daily PM2.5 trend with a 30-day moving average. |
| `missing_values.png` | Missing sensor readings by column. |
| `station_comparison.png` | Mean PM2.5 concentration across the 12 stations. |
| `actual_vs_predicted.png` | Average actual and predicted PM2.5 over the final test week. |
| `actual_vs_predicted_scatter.png` | Scatter plot comparing actual and predicted test values. |
| `prediction_errors.png` | Distribution of prediction errors. |
| `test_predictions.csv` | Test-set predictions, actual values, and absolute errors. |
| `next_hour_forecast_by_station.csv` | One illustrative next-hour estimate for each station. |
| `linear_regression_pipeline.joblib` | Saved preprocessing and Linear Regression pipeline. |
| `linear_regression_coefficients.csv` | Fitted linear coefficients for interpretation. |

**Note:** The `outputs/` folder is generated automatically when the project runs, so it does not need to be included in the ZIP file.

---

## Dataset

**Beijing Multi-Site Air Quality** — UCI Machine Learning Repository, dataset 501.

The dataset contains hourly air-pollution and weather observations collected at **12 monitoring stations in Beijing**, covering **March 2013 to February 2017**.

- **Official dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **Dataset DOI:** [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G)
- **Citation:** Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit the UCI repository and dataset creator when reusing the data.

---

## Key concepts

- **PM2.5:** Fine particulate matter with a diameter of 2.5 micrometres or less. This is the pollutant concentration the model predicts.
- **Feature:** An input the model uses to make a prediction, such as temperature or a recent PM2.5 reading.
- **Target:** The value the model is trained to predict — PM2.5 one hour after the input timestamp.
- **Lag feature:** A previous reading used as an input. For example, `PM2.5_lag_3h` represents the same station's PM2.5 reading from three hours earlier.
- **Chronological split:** Training on earlier observations and testing on later observations. This better reflects a forecasting use case than a random split.
- **Missing-value imputation:** Filling missing input values with representative values learned from the training data. Rows without a known future PM2.5 target cannot be used for training or evaluation.
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual PM2.5 values, in the same units as the target. Lower MAE indicates smaller average errors on the same test set.
- **RMSE and R²:** Additional evaluation metrics that describe prediction error magnitude and the proportion of target variation explained by the model.

---

## Scope and limitations

- **Historical data, not live monitoring:** The dataset ends in February 2017. The project does not retrieve current air-quality readings or provide a live forecast.
- **Illustrative next-hour estimates:** `next_hour_forecast_by_station.csv` demonstrates how the trained model can generate a next-hour estimate after the historical final timestamp. It reuses the latest available pollutant and weather measurements because future sensor and weather inputs are not available in this dataset.
- **One predictive model:** Linear Regression is the only model used. Random Forest is not included because the project is intentionally scoped around a single, interpretable baseline model.
- **Forecasting assumptions:** The estimates depend on historical patterns and the available input features; they should not be treated as official air-quality measurements or public-health guidance.

---

## Project objective

Build a clear, reproducible machine learning workflow that turns historical air-quality and weather observations into **next-hour PM2.5 estimates**, while making the data preparation, time-based evaluation, and prediction results easy to inspect.
