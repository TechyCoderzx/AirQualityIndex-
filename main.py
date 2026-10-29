"""Air Quality Trend Predictor

A beginner-friendly, end-to-end machine-learning project using Linear Regression only.
The script loads the 12 Beijing station CSV files, cleans the data, explores trends,
creates one-hour-ahead features, trains a chronological train/test model, evaluates it,
and saves plots, metrics, predictions, and the trained pipeline to ./outputs.

Run: python main.py
"""
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Save graphs to files; no plot windows/UI are needed.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. PROJECT SETTINGS
# ---------------------------------------------------------------------------
# You can change these values if needed. The defaults are deliberately simple:
# predict one hour ahead and reserve the latest 20% of time for testing.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TRAIN_FRACTION = 0.80
FORECAST_HOURS = 1
RANDOM_STATE = 42  # Used only to make plotting samples repeatable, not the model.

# The six pollutant measurements and five meteorological variables supplied by UCI.
POLLUTANT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_COLUMNS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_SENSOR_COLUMNS = POLLUTANT_COLUMNS + WEATHER_COLUMNS
TIME_FEATURE_COLUMNS = ["hour", "month"]
LAG_COLUMNS = ["PM2.5_lag_1h", "PM2.5_lag_3h", "PM2.5_lag_6h"]
NUMERIC_FEATURES = NUMERIC_SENSOR_COLUMNS + TIME_FEATURE_COLUMNS + LAG_COLUMNS
CATEGORICAL_FEATURES = ["station", "wd"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "PM2.5_future"


def print_section(title: str) -> None:
    """Print an easy-to-follow section heading in the terminal."""
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def load_and_clean_data() -> tuple[pd.DataFrame, dict]:
    """Load all station CSVs and perform basic, transparent data cleaning."""
    print_section("STEP 1 — LOAD AND CLEAN THE DATA")
    csv_files = sorted(DATA_DIR.glob("PRSA_Data_*.csv"))
    if len(csv_files) != 12:
        raise FileNotFoundError(
            f"Expected the 12 station CSV files in:\n  {DATA_DIR}\n"
            f"Found {len(csv_files)} files. Re-extract the provided dataset into the data folder."
        )

    station_frames = []
    for file_path in csv_files:
        station = file_path.name.replace("PRSA_Data_", "").replace("_20130301-20170228.csv", "")
        frame = pd.read_csv(file_path)
        frame["station"] = station  # Explicit station label helps the model distinguish locations.
        station_frames.append(frame)
        print(f"Loaded {station:<16} {len(frame):,} rows")

    data = pd.concat(station_frames, ignore_index=True)
    original_rows = len(data)

    # The dataset contains separate year/month/day/hour columns. Combine them into a timestamp.
    timestamp_parts = {
        "year": pd.to_numeric(data["year"], errors="coerce"),
        "month": pd.to_numeric(data["month"], errors="coerce"),
        "day": pd.to_numeric(data["day"], errors="coerce"),
        "hour": pd.to_numeric(data["hour"], errors="coerce"),
    }
    data["datetime"] = pd.to_datetime(timestamp_parts, errors="coerce")
    invalid_dates = int(data["datetime"].isna().sum())
    data = data.dropna(subset=["datetime"]).copy()

    # Convert sensor measurements into numbers. Invalid text becomes NaN (missing).
    for column in NUMERIC_SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # In this dataset, -200 is a sentinel for a missing sensor reading, not a real measurement.
    # Replace it with NaN so it will not mislead the model or charts.
    data[NUMERIC_SENSOR_COLUMNS] = data[NUMERIC_SENSOR_COLUMNS].replace(-200, np.nan)

    # Wind direction is categorical (e.g. NW, SE), so keep it as text.
    if "wd" not in data.columns:
        data["wd"] = np.nan
    # Use ordinary NumPy NaN rather than pandas' nullable sentinel so scikit-learn
    # imputers and category encoders can process this column consistently.
    data["wd"] = data["wd"].astype(str).str.strip()
    data["wd"] = data["wd"].replace({
        "nan": np.nan, "NaN": np.nan, "<NA>": np.nan, "-200": np.nan, "": np.nan
    })

    # A station should normally have one record per hour. Remove repeated station-hour rows.
    data = data.sort_values(["station", "datetime"]).drop_duplicates(
        subset=["station", "datetime"], keep="first"
    )
    data = data.reset_index(drop=True)

    missing_before_model = data[NUMERIC_SENSOR_COLUMNS + ["wd"]].isna().sum()
    target_missing = int(data["PM2.5"].isna().sum())
    summary = {
        "station_csv_files": len(csv_files),
        "rows_loaded": original_rows,
        "rows_after_cleaning": len(data),
        "invalid_timestamps_removed": invalid_dates,
        "duplicate_station_hours_removed": original_rows - invalid_dates - len(data),
        "missing_pm25_readings": target_missing,
        "start_timestamp": data["datetime"].min(),
        "end_timestamp": data["datetime"].max(),
        "missing_counts": missing_before_model,
    }

    print(f"\nCombined rows: {original_rows:,}")
    print(f"Rows after basic cleaning: {len(data):,}")
    print(f"Monitoring stations: {data['station'].nunique()}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"PM2.5 readings missing: {target_missing:,}")
    print("Missing input readings will be imputed using medians/modes learned from TRAINING data only.")
    print("Rows with no known future PM2.5 target cannot be used to calculate training/evaluation error.")

    # Save a transparent record of what was cleaned and how many missing values remain.
    lines = [
        "DATA CLEANING SUMMARY",
        f"Station CSV files loaded: {summary['station_csv_files']}",
        f"Rows loaded: {summary['rows_loaded']:,}",
        f"Rows after cleaning: {summary['rows_after_cleaning']:,}",
        f"Invalid timestamps removed: {summary['invalid_timestamps_removed']:,}",
        f"Duplicate station-hour rows removed: {summary['duplicate_station_hours_removed']:,}",
        f"Missing PM2.5 readings: {summary['missing_pm25_readings']:,}",
        f"Start: {summary['start_timestamp']}",
        f"End: {summary['end_timestamp']}",
        "",
        "Missing value counts after converting -200 to missing:",
        missing_before_model.sort_values(ascending=False).to_string(),
        "",
        "Note: missing feature values are imputed inside the model pipeline using training data only.",
        "Rows without a valid next-hour PM2.5 target are excluded from supervised training/evaluation.",
    ]
    (OUTPUT_DIR / "data_cleaning_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return data, summary


def create_exploration_plots(data: pd.DataFrame) -> None:
    """Create simple plots that show what the original data looks like."""
    print_section("STEP 2 — EXPLORE POLLUTION AND MISSING VALUES")

    # Plot 1: daily mean PM2.5 averaged over the station observations available that day.
    daily = data.dropna(subset=["PM2.5"]).set_index("datetime")["PM2.5"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily.values, linewidth=0.8, label="Daily mean PM2.5")
    ax.plot(daily.index, daily.rolling(30, min_periods=1).mean(), linewidth=1.8,
            label="30-day moving average")
    ax.set_title("Beijing PM2.5 Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 concentration (µg/m³)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_pm25_trend.png", dpi=150)
    plt.close(fig)

    # Plot 2: how much data is missing in each sensor field after -200 has been handled.
    missing_percent = (data[NUMERIC_SENSOR_COLUMNS + ["wd"]].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_percent.plot(kind="barh", ax=ax)
    ax.set_title("Missing Sensor Readings by Column")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Sensor / variable")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    # Plot 3: station-level averages. This is descriptive analysis, not a second model.
    station_means = data.groupby("station")["PM2.5"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    station_means.plot(kind="barh", ax=ax)
    ax.set_title("Average PM2.5 by Monitoring Station")
    ax.set_xlabel("Mean PM2.5 concentration (µg/m³)")
    ax.set_ylabel("Monitoring station")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "station_comparison.png", dpi=150)
    plt.close(fig)

    print("Saved: daily_pm25_trend.png — long-term pollution trend and 30-day average")
    print("Saved: missing_values.png — missing sensor readings before model imputation")
    print("Saved: station_comparison.png — average PM2.5 at each of the 12 stations")


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features available at time t to estimate PM2.5 at time t + 1 hour."""
    print_section("STEP 3 — CREATE INPUT FEATURES AND THE FUTURE TARGET")
    features = data.sort_values(["station", "datetime"]).copy()
    station_groups = features.groupby("station", sort=False)

    # Historical PM2.5 values: lag_1h is the previous recorded hour, and so on.
    # The raw data is hourly; shifting within each station prevents mixing locations.
    for lag in (1, 3, 6):
        features[f"PM2.5_lag_{lag}h"] = station_groups["PM2.5"].shift(lag)

    # The label is the next record's PM2.5 for the SAME station. It is not an input feature.
    features[TARGET_COLUMN] = station_groups["PM2.5"].shift(-FORECAST_HOURS)
    features["target_time"] = station_groups["datetime"].shift(-FORECAST_HOURS)
    features["hour"] = features["datetime"].dt.hour
    features["month"] = features["datetime"].dt.month

    print("Each training example uses information known at the current hour:")
    print("  • current pollution measurements (PM2.5, PM10, SO2, NO2, CO, O3)")
    print("  • weather measurements (temperature, pressure, dew point, rain, wind speed)")
    print("  • PM2.5 from 1, 3, and 6 hours earlier")
    print("  • hour, month, station identity, and wind direction")
    print("The answer the model learns to predict is PM2.5 at the NEXT hour.")
    return features


def train_evaluate_and_forecast(features: pd.DataFrame) -> None:
    """Fit only Linear Regression, evaluate on later times, and forecast one future hour."""
    print_section("STEP 4 — CHRONOLOGICAL TRAIN/TEST SPLIT")

    # Only rows with a known next-hour target can be used for supervised learning.
    labeled = features.dropna(subset=[TARGET_COLUMN, "target_time"]).copy()
    all_target_times = np.array(sorted(labeled["target_time"].unique()))
    cutoff_index = int(len(all_target_times) * TRAIN_FRACTION)
    cutoff_index = min(max(cutoff_index, 1), len(all_target_times) - 1)
    cutoff_time = pd.Timestamp(all_target_times[cutoff_index])

    # Split by time, NOT randomly. The model is trained on earlier observations and tested on later ones.
    train_data = labeled[labeled["target_time"] < cutoff_time].copy()
    test_data = labeled[labeled["target_time"] >= cutoff_time].copy()
    if train_data.empty or test_data.empty:
        raise ValueError("Chronological split produced an empty train or test set. Check the dataset.")

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    print(f"Training period targets: before {cutoff_time}")
    print(f"Testing period targets:  {cutoff_time} onward")
    print(f"Training examples: {len(train_data):,}")
    print(f"Testing examples:  {len(test_data):,}")
    print(f"Training fraction by timestamp: {TRAIN_FRACTION:.0%}")

    print_section("STEP 5 — TRAIN LINEAR REGRESSION")
    print("Missing numeric inputs are filled using training-set medians.")
    print("Missing categories are filled using the most frequent training category.")
    print("Station and wind-direction names are converted to indicator columns.")
    print("All preprocessing is fitted on training data only to reduce data leakage.")

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    # The only predictive model in this project is LinearRegression.
    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("linear_regression", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # MAE is the main assignment metric. RMSE and R² provide additional context.
    mae = mean_absolute_error(y_test, predicted)
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    r2 = r2_score(y_test, predicted)

    print_section("STEP 6 — EVALUATE THE MODEL")
    print("Model: Linear Regression (no Random Forest or other predictive model is used)")
    print(f"MAE:  {mae:.2f} µg/m³")
    print(f"RMSE: {rmse:.2f} µg/m³")
    print(f"R²:   {r2:.4f}")
    print("Interpretation: MAE is the average absolute difference between predicted and actual PM2.5.")
    print("For example, an MAE of 10 means predictions differ from actual values by 10 µg/m³ on average.")

    predictions = test_data[["station", "datetime", "target_time", TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={
        "datetime": "feature_time",
        "target_time": "forecast_for",
        TARGET_COLUMN: "actual_PM2.5",
    })
    predictions["predicted_PM2.5"] = predicted
    predictions["absolute_error"] = np.abs(predictions["actual_PM2.5"] - predictions["predicted_PM2.5"])
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metrics = [
        "AIR QUALITY TREND PREDICTOR — EVALUATION RESULTS",
        "Model: Linear Regression only",
        f"Forecast horizon: {FORECAST_HOURS} hour(s)",
        f"Train fraction by time: {TRAIN_FRACTION:.0%}",
        f"Training examples: {len(train_data):,}",
        f"Testing examples: {len(test_data):,}",
        f"Test period begins: {cutoff_time}",
        f"MAE (µg/m³): {mae:.4f}",
        f"RMSE (µg/m³): {rmse:.4f}",
        f"R²: {r2:.6f}",
        "",
        "MAE is the mean absolute error, the main metric required by the assignment.",
        "The test set contains later timestamps than the training set.",
    ]
    (OUTPUT_DIR / "evaluation_metrics.txt").write_text("\n".join(metrics), encoding="utf-8")

    # Plot the actual vs predicted average across all stations during the last week of the test set.
    hourly_comparison = predictions.groupby("forecast_for").agg(
        actual_PM2_5=("actual_PM2.5", "mean"),
        predicted_PM2_5=("predicted_PM2.5", "mean"),
    ).sort_index()
    last_time = hourly_comparison.index.max()
    first_time = last_time - pd.Timedelta(days=7)
    recent = hourly_comparison[hourly_comparison.index >= first_time]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent["actual_PM2_5"], label="Actual mean PM2.5", linewidth=1.3)
    ax.plot(recent.index, recent["predicted_PM2_5"], label="Predicted mean PM2.5", linewidth=1.3)
    ax.set_title("Actual vs Predicted PM2.5 — Final Week of Test Period")
    ax.set_xlabel("Forecast timestamp")
    ax.set_ylabel("Mean PM2.5 concentration (µg/m³)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Scatter plot: a point close to the diagonal means predicted and actual values are similar.
    if len(predictions) > 5000:
        scatter_data = predictions.sample(5000, random_state=RANDOM_STATE)
    else:
        scatter_data = predictions
    low = min(scatter_data["actual_PM2.5"].min(), scatter_data["predicted_PM2.5"].min())
    high = max(scatter_data["actual_PM2.5"].max(), scatter_data["predicted_PM2.5"].max())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(scatter_data["actual_PM2.5"], scatter_data["predicted_PM2.5"], alpha=0.25, s=10)
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_title("Actual vs Predicted PM2.5 (Test Sample)")
    ax.set_xlabel("Actual PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close(fig)

    # Prediction errors help show if the model often overpredicts or underpredicts.
    errors = predictions["predicted_PM2.5"] - predictions["actual_PM2.5"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=60, edgecolor="black", linewidth=0.25)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Predicted minus actual PM2.5 (µg/m³)")
    ax.set_ylabel("Number of test examples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_errors.png", dpi=150)
    plt.close(fig)

    # Save fitted model for reuse without retraining. joblib is part of scikit-learn's ecosystem.
    joblib.dump(model, OUTPUT_DIR / "linear_regression_pipeline.joblib")

    # Write interpretable Linear Regression coefficients, including encoded station/category features.
    try:
        transformed_names = model.named_steps["preprocessing"].get_feature_names_out()
        coefficients = model.named_steps["linear_regression"].coef_
        coefficient_table = pd.DataFrame({
            "feature": transformed_names,
            "coefficient": coefficients,
        })
        coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
        coefficient_table.sort_values("absolute_coefficient", ascending=False).to_csv(
            OUTPUT_DIR / "linear_regression_coefficients.csv", index=False
        )
    except (AttributeError, ValueError):
        print("Note: this scikit-learn version could not export coefficient names; the model was still saved.")

    print("Saved: evaluation_metrics.txt")
    print("Saved: test_predictions.csv")
    print("Saved: actual_vs_predicted.png")
    print("Saved: actual_vs_predicted_scatter.png")
    print("Saved: prediction_errors.png")
    print("Saved: linear_regression_pipeline.joblib")

    print_section("STEP 7 — MAKE A NEXT-HOUR DEMONSTRATION FORECAST")
    # The last row for each station has known current-time inputs but no recorded next-hour target.
    # This is a demonstration forecast based on the last observations in the 2013–2017 dataset.
    latest_rows = features.sort_values(["station", "datetime"]).groupby("station", as_index=False).tail(1).copy()
    forecast_values = model.predict(latest_rows[FEATURE_COLUMNS])
    future_forecast = pd.DataFrame({
        "station": latest_rows["station"].values,
        "last_observation_time": latest_rows["datetime"].values,
        "forecast_for": (latest_rows["datetime"] + pd.Timedelta(hours=FORECAST_HOURS)).values,
        "last_observed_PM2.5": latest_rows["PM2.5"].values,
        "predicted_PM2.5": forecast_values,
    }).sort_values("station")
    future_forecast.to_csv(OUTPUT_DIR / "next_hour_forecast_by_station.csv", index=False)
    print(future_forecast.to_string(index=False, formatters={"predicted_PM2.5": "{:.2f}".format}))
    print("\nIMPORTANT: This is a demonstration forecast after the dataset's historical end date (2017).")
    print("It is not a live forecast for today's weather or current air quality.")
    print("Weather and pollutant inputs for the next hour are unavailable, so the model uses the latest observed inputs.")
    print("Saved: next_hour_forecast_by_station.csv")

    print_section("PROJECT COMPLETE")
    print(f"All results are saved in:\n{OUTPUT_DIR}")
    print("Open the PNG files to view graphs; open evaluation_metrics.txt to see the MAE result.")


def main() -> int:
    """Run the complete project without requiring interactive UI steps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("AIR QUALITY TREND PREDICTOR")
    print("Target: PM2.5 one hour ahead | Model: Linear Regression only")
    print("Dataset: Beijing Multi-Site Air Quality, 12 monitoring stations")
    try:
        data, _ = load_and_clean_data()
        create_exploration_plots(data)
        feature_frame = make_features(data)
        train_evaluate_and_forecast(feature_frame)
        return 0
    except Exception as exc:
        print("\nThe project stopped because of an error:")
        print(f"{type(exc).__name__}: {exc}")
        print("Check README.md for setup instructions and confirm the data folder contains 12 CSV files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
