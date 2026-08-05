from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/karachi_aqi_features.csv"
)

METRICS_FOLDER = Path("outputs/metrics")
METRICS_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    METRICS_FOLDER
    / "xgboost_hourly_1_to_72_metrics.csv"
)

GRAPHS_FOLDER = Path("outputs/graphs")

MODELS_FOLDER = Path("models/xgboost_hourly")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS_FILE = (
    MODELS_FOLDER
    / "feature_columns.json"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["time"],
)

df = df.sort_values("time").reset_index(drop=True)

print("\nDataset shape:")
print(df.shape)

print("\nDataset range:")
print(df["time"].min(), "to", df["time"].max())


# ---------------------------------------------------------
# Base features available at prediction time
# ---------------------------------------------------------

feature_columns = [
    # Current pollutant values
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",

    # Current weather values
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    # Current time features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # AQI lag features
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",

    # PM2.5 lag features
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_12h",
    "pm2_5_lag_24h",
    "pm2_5_lag_48h",
    "pm2_5_lag_72h",

    # PM10 lag features
    "pm10_lag_1h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",
    "pm10_lag_48h",
    "pm10_lag_72h",

    # AQI rolling means
    "aqi_mean_3h",
    "aqi_mean_6h",
    "aqi_mean_12h",
    "aqi_mean_24h",
    "aqi_mean_48h",
    "aqi_mean_72h",

    # PM2.5 rolling means
    "pm2_5_mean_3h",
    "pm2_5_mean_6h",
    "pm2_5_mean_12h",
    "pm2_5_mean_24h",
    "pm2_5_mean_48h",
    "pm2_5_mean_72h",

    # PM10 rolling means
    "pm10_mean_3h",
    "pm10_mean_6h",
    "pm10_mean_12h",
    "pm10_mean_24h",
    "pm10_mean_48h",
    "pm10_mean_72h",

    # AQI change features
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_12h",
    "aqi_change_24h",
    "aqi_change_48h",
    "aqi_change_72h",

    # AQI volatility
    "aqi_std_24h",
    "aqi_std_72h",
]


# ---------------------------------------------------------
# Horizon-specific feature names
# ---------------------------------------------------------

future_feature_columns = [
    "future_temperature",
    "future_humidity",
    "future_dew_point",
    "future_pressure",
    "future_precipitation",
    "future_cloud_cover",
    "future_wind_speed",
    "future_wind_gusts",
    "future_wind_direction_sin",
    "future_wind_direction_cos",
    "target_hour",
    "target_day_of_week",
    "target_month",
    "target_is_weekend",
]

model_feature_columns = (
    feature_columns
    + future_feature_columns
)

with open(
    FEATURE_COLUMNS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        model_feature_columns,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Fixed test boundary
# ---------------------------------------------------------

TEST_START = pd.Timestamp(
    "2025-09-17 20:00:00"
)


# ---------------------------------------------------------
# Store metrics
# ---------------------------------------------------------

metrics_results = []


# ---------------------------------------------------------
# Train one model for each horizon
# ---------------------------------------------------------

for horizon in range(1, 73):

    print("\n" + "=" * 60)
    print(f"Training model for {horizon}-hour forecast")
    print("=" * 60)

    horizon_df = df.copy()


    # -----------------------------------------------------
    # Create AQI target
    # -----------------------------------------------------

    horizon_df["target_aqi"] = (
        horizon_df["us_aqi"]
        .shift(-horizon)
    )


    # -----------------------------------------------------
    # Create future-weather features
    # -----------------------------------------------------

    horizon_df["future_temperature"] = (
        horizon_df["temperature_2m"]
        .shift(-horizon)
    )

    horizon_df["future_humidity"] = (
        horizon_df["relative_humidity_2m"]
        .shift(-horizon)
    )

    horizon_df["future_dew_point"] = (
        horizon_df["dew_point_2m"]
        .shift(-horizon)
    )

    horizon_df["future_pressure"] = (
        horizon_df["surface_pressure"]
        .shift(-horizon)
    )

    horizon_df["future_precipitation"] = (
        horizon_df["precipitation"]
        .shift(-horizon)
    )

    horizon_df["future_cloud_cover"] = (
        horizon_df["cloud_cover"]
        .shift(-horizon)
    )

    horizon_df["future_wind_speed"] = (
        horizon_df["wind_speed_10m"]
        .shift(-horizon)
    )

    horizon_df["future_wind_gusts"] = (
        horizon_df["wind_gusts_10m"]
        .shift(-horizon)
    )


    # -----------------------------------------------------
    # Encode future wind direction
    # -----------------------------------------------------

    future_wind_direction = (
        horizon_df["wind_direction_10m"]
        .shift(-horizon)
    )

    future_wind_radians = np.radians(
        future_wind_direction
    )

    horizon_df["future_wind_direction_sin"] = (
        np.sin(future_wind_radians)
    )

    horizon_df["future_wind_direction_cos"] = (
        np.cos(future_wind_radians)
    )


    # -----------------------------------------------------
    # Create target-time features
    # -----------------------------------------------------

    target_time = (
        horizon_df["time"]
        + pd.Timedelta(hours=horizon)
    )

    horizon_df["target_hour"] = (
        target_time.dt.hour
    )

    horizon_df["target_day_of_week"] = (
        target_time.dt.dayofweek
    )

    horizon_df["target_month"] = (
        target_time.dt.month
    )

    horizon_df["target_is_weekend"] = (
        horizon_df["target_day_of_week"] >= 5
    ).astype(int)


    # -----------------------------------------------------
    # Remove final rows without future target/weather
    # -----------------------------------------------------

    horizon_df = horizon_df.dropna(
        subset=(
            model_feature_columns
            + ["target_aqi"]
        )
    ).reset_index(drop=True)


    # -----------------------------------------------------
    # Fixed chronological train-test split
    # -----------------------------------------------------

    train_df = horizon_df[
        horizon_df["time"] < TEST_START
    ].copy()

    test_df = horizon_df[
        horizon_df["time"] >= TEST_START
    ].copy()


    X_train = train_df[
        model_feature_columns
    ]

    y_train = train_df["target_aqi"]

    X_test = test_df[
        model_feature_columns
    ]

    y_test = test_df["target_aqi"]


    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))


    # -----------------------------------------------------
    # Build XGBoost model
    # -----------------------------------------------------

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.02,
        max_depth=2,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=20.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # -----------------------------------------------------
    # Predict test values
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )


    # -----------------------------------------------------
    # Display horizon result
    # -----------------------------------------------------

    print(f"\nHorizon: {horizon} hours")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    model_output_file = (
    MODELS_FOLDER
    / f"xgboost_{horizon:02d}h.joblib"
    )

    joblib.dump(
        model,
        model_output_file,
    )


    # -----------------------------------------------------
    # Store horizon metrics
    # -----------------------------------------------------

    metrics_results.append(
        {
            "model": "XGBoost",
            "forecast_horizon_hours": horizon,
            "training_rows": len(train_df),
            "test_rows": len(test_df),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_file": str(model_output_file),
        }
    )


# ---------------------------------------------------------
# Save all metrics
# ---------------------------------------------------------

metrics_df = pd.DataFrame(
    metrics_results
)

metrics_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("All hourly models completed")
print("=" * 60)

print("\nMetrics:")
print(metrics_df)

print("\nMetrics saved to:")
print(OUTPUT_FILE)


print("\nAverage performance across all horizons:")
print(
    metrics_df[
        ["mae", "rmse", "r2"]
    ].mean()
)


print("\nSelected horizon results:")
print(
    metrics_df[
        metrics_df[
            "forecast_horizon_hours"
        ].isin(
            [1, 6, 12, 24, 48, 72]
        )
    ]
)

plt.figure(figsize=(11, 6))

plt.plot(
    metrics_df["forecast_horizon_hours"],
    metrics_df["mae"],
    marker="o",
    markersize=3,
)

plt.xlabel("Forecast Horizon (Hours)")
plt.ylabel("Mean Absolute Error")
plt.title("XGBoost MAE from 1 to 72 Forecast Hours")
plt.grid(True, alpha=0.3)
plt.tight_layout()

mae_output = GRAPHS_FOLDER / "xgboost_mae_by_horizon.png"

plt.savefig(
    mae_output,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ---------------------------------------------------------
# RMSE graph
# ---------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    metrics_df["forecast_horizon_hours"],
    metrics_df["rmse"],
    marker="o",
    markersize=3,
)

plt.xlabel("Forecast Horizon (Hours)")
plt.ylabel("Root Mean Squared Error")
plt.title("XGBoost RMSE from 1 to 72 Forecast Hours")
plt.grid(True, alpha=0.3)
plt.tight_layout()

rmse_output = GRAPHS_FOLDER / "xgboost_rmse_by_horizon.png"

plt.savefig(
    rmse_output,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ---------------------------------------------------------
# R² graph
# ---------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    metrics_df["forecast_horizon_hours"],
    metrics_df["r2"],
    marker="o",
    markersize=3,
    label="XGBoost R²",
)

plt.axhline(
    y=0.7,
    linestyle="--",
    label="Target R² = 0.70",
)

plt.xlabel("Forecast Horizon (Hours)")
plt.ylabel("R² Score")
plt.title("XGBoost R² from 1 to 72 Forecast Hours")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

r2_output = GRAPHS_FOLDER / "xgboost_r2_by_horizon.png"

plt.savefig(
    r2_output,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()