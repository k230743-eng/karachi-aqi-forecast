from pathlib import Path
import json
import joblib
import os

import hopsworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ---------------------------------------------------------
# Environment / Hopsworks setup
# ---------------------------------------------------------

RUNNING_IN_GITHUB = (
    os.getenv("GITHUB_ACTIONS") == "true"
)


if not RUNNING_IN_GITHUB:

    Path(r"D:\tmp").mkdir(
        parents=True,
        exist_ok=True
    )

    CERT_FOLDER = Path(
        r"D:\karachi-aqi-forecast\.hopsworks_certs"
    )

    CERT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Output folders
# ---------------------------------------------------------

MODELS_FOLDER = Path(
    "models/xgboost_hourly_hopsworks"
)

MODELS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


METRICS_FOLDER = Path(
    "outputs/metrics"
)

METRICS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


GRAPHS_FOLDER = Path(
    "outputs/graphs/hopsworks"
)

GRAPHS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


METRICS_FILE = (
    METRICS_FOLDER
    / "xgboost_hourly_hopsworks_1_to_72_metrics.csv"
)


FEATURE_COLUMNS_FILE = (
    MODELS_FOLDER
    / "feature_columns.json"
)


# ---------------------------------------------------------
# Connect to Hopsworks
# ---------------------------------------------------------

print("\nConnecting to Hopsworks...")


if RUNNING_IN_GITHUB:

    print("Running inside GitHub Actions.")

    project = hopsworks.login(
        host=os.environ[
            "HOPSWORKS_HOST"
        ],
        project=os.environ[
            "HOPSWORKS_PROJECT"
        ],
        api_key_value=os.environ[
            "HOPSWORKS_API_KEY"
        ],
    )

else:

    print("Running locally.")

    project = hopsworks.login(
        cert_folder=str(CERT_FOLDER)
    )


fs = project.get_feature_store()

print("\nConnected to Feature Store.")


# =========================================================
# Read latest features and targets directly
# =========================================================

print(
    "\nReading latest features "
    "from Hopsworks..."
)


feature_group = fs.get_feature_group(
    name="karachi_aqi_features",
    version=2,
)


target_group = fs.get_feature_group(
    name="karachi_aqi_targets",
    version=1,
)


features_df = feature_group.read()

targets_df = target_group.read()


if features_df.empty:

    raise RuntimeError(
        "Feature group is empty."
    )


if targets_df.empty:

    raise RuntimeError(
        "Target group is empty."
    )


print(
    "\nFeatures retrieved:",
    features_df.shape
)

print(
    "Targets retrieved:",
    targets_df.shape
)


# =========================================================
# Fix timestamps
# =========================================================

features_df["time"] = pd.to_datetime(
    features_df["time"]
)

targets_df["time"] = pd.to_datetime(
    targets_df["time"]
)


if features_df["time"].dt.tz is not None:

    features_df["time"] = (
        features_df["time"]
        .dt.tz_localize(None)
    )


if targets_df["time"].dt.tz is not None:

    targets_df["time"] = (
        targets_df["time"]
        .dt.tz_localize(None)
    )


# =========================================================
# Sort / deduplicate
# =========================================================

features_df = (
    features_df
    .sort_values("time")
    .drop_duplicates(
        subset=["time"],
        keep="last"
    )
    .reset_index(drop=True)
)


targets_df = (
    targets_df
    .sort_values("time")
    .drop_duplicates(
        subset=["time"],
        keep="last"
    )
    .reset_index(drop=True)
)


# =========================================================
# Join features + targets
# =========================================================

df = pd.merge(
    features_df,
    targets_df,
    on="time",
    how="inner",
)


df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)


if df.empty:

    raise RuntimeError(
        "No matching feature/target "
        "rows were found."
    )


print(
    "\nCombined dataset shape:"
)

print(
    df.shape
)


print(
    "\nDataset range:"
)

print(
    df["time"].min(),
    "to",
    df["time"].max(),
)


# ---------------------------------------------------------
# Base feature columns
#
# Same feature set used in the original final models
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
# Horizon-specific model features
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


print("\nNumber of model features:")
print(len(model_feature_columns))


# ---------------------------------------------------------
# Save exact feature order
# ---------------------------------------------------------

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
# Rolling chronological train/test split
#
# Keep the newest 20% for evaluation.
# ---------------------------------------------------------

TEST_START = df[
    "time"
].quantile(
    0.80
)


print(
    "\nAutomatic test boundary:"
)

print(
    TEST_START
)


# ---------------------------------------------------------
# Store results from all horizons
# ---------------------------------------------------------

metrics_results = []


# =========================================================
# Train 72 direct forecasting models
# =========================================================

for horizon in range(1, 73):

    print("\n" + "=" * 70)

    print(
        f"Training XGBoost model for "
        f"{horizon}-hour forecast"
    )

    print("=" * 70)


    # -----------------------------------------------------
    # Copy data for this horizon
    # -----------------------------------------------------

    horizon_df = df.copy()


    # -----------------------------------------------------
    # Target
    #
    # IMPORTANT:
    # We no longer create the target using shift().
    #
    # The target is being fetched directly from the
    # Hopsworks Feature Store.
    # -----------------------------------------------------

    target_column = (
        f"aqi_target_{horizon}h"
    )


    if target_column not in horizon_df.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )


    # -----------------------------------------------------
    # Future weather features
    #
    # Historical weather is shifted to represent the
    # weather conditions at the target forecast hour.
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
    # Future wind direction
    # -----------------------------------------------------

    future_wind_direction = (
        horizon_df["wind_direction_10m"]
        .shift(-horizon)
    )


    future_wind_radians = np.radians(
        future_wind_direction
    )


    horizon_df[
        "future_wind_direction_sin"
    ] = np.sin(
        future_wind_radians
    )


    horizon_df[
        "future_wind_direction_cos"
    ] = np.cos(
        future_wind_radians
    )


    # -----------------------------------------------------
    # Time information for forecast target
    # -----------------------------------------------------

    target_time = (
        horizon_df["time"]
        + pd.Timedelta(
            hours=horizon
        )
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
        horizon_df[
            "target_day_of_week"
        ] >= 5
    ).astype(int)


    # -----------------------------------------------------
    # Remove rows without required data
    # -----------------------------------------------------

    required_columns = (
        model_feature_columns
        + [target_column]
    )


    horizon_df = (
        horizon_df
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # Chronological train/test split
    # -----------------------------------------------------

    train_df = horizon_df[
        horizon_df["time"] < TEST_START
    ].copy()


    test_df = horizon_df[
        horizon_df["time"] >= TEST_START
    ].copy()


    # -----------------------------------------------------
    # X and y
    # -----------------------------------------------------

    X_train = train_df[
        model_feature_columns
    ]


    y_train = train_df[
        target_column
    ]


    X_test = test_df[
        model_feature_columns
    ]


    y_test = test_df[
        target_column
    ]


    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "Test rows:",
        len(test_df)
    )


    # -----------------------------------------------------
    # Build XGBoost model
    #
    # Same configuration as the previous final pipeline
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
    # Train
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )


    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )


    r2 = r2_score(
        y_test,
        predictions,
    )


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        f"\nHorizon: "
        f"{horizon} hours"
    )

    print(
        f"MAE:  {mae:.3f}"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"R²:   {r2:.3f}"
    )


    # -----------------------------------------------------
    # Save model locally
    # -----------------------------------------------------

    model_output_file = (
        MODELS_FOLDER
        / f"xgboost_{horizon:02d}h.joblib"
    )


    joblib.dump(
        model,
        model_output_file,
    )


    # -----------------------------------------------------
    # Store metrics
    # -----------------------------------------------------

    metrics_results.append(
        {
            "model": "XGBoost",
            "data_source": "Hopsworks Feature Store",
            "forecast_horizon_hours": horizon,
            "training_rows": len(train_df),
            "test_rows": len(test_df),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_file": str(
                model_output_file
            ),
        }
    )


# =========================================================
# Save metrics
# =========================================================

metrics_df = pd.DataFrame(
    metrics_results
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)


print("\n" + "=" * 70)

print(
    "All 72 Hopsworks-based "
    "XGBoost models completed"
)

print("=" * 70)


print("\nMetrics saved to:")

print(
    METRICS_FILE
)


# ---------------------------------------------------------
# Average metrics
# ---------------------------------------------------------

print(
    "\nAverage performance "
    "across all horizons:"
)


print(
    metrics_df[
        [
            "mae",
            "rmse",
            "r2",
        ]
    ].mean()
)


# ---------------------------------------------------------
# Selected horizons
# ---------------------------------------------------------

selected_horizons = [
    1,
    6,
    12,
    24,
    48,
    72,
]


print(
    "\nSelected horizon results:"
)


print(
    metrics_df[
        metrics_df[
            "forecast_horizon_hours"
        ].isin(
            selected_horizons
        )
    ][
        [
            "forecast_horizon_hours",
            "mae",
            "rmse",
            "r2",
        ]
    ]
)


# =========================================================
# MAE graph
# =========================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    metrics_df[
        "forecast_horizon_hours"
    ],

    metrics_df["mae"],

    marker="o",

    markersize=3,
)


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "Mean Absolute Error"
)

plt.title(
    "Hopsworks XGBoost MAE "
    "from 1 to 72 Forecast Hours"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


mae_output = (
    GRAPHS_FOLDER
    / "xgboost_hopsworks_mae_by_horizon.png"
)


plt.savefig(
    mae_output,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# RMSE graph
# =========================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    metrics_df[
        "forecast_horizon_hours"
    ],

    metrics_df["rmse"],

    marker="o",

    markersize=3,
)


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "Root Mean Squared Error"
)

plt.title(
    "Hopsworks XGBoost RMSE "
    "from 1 to 72 Forecast Hours"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


rmse_output = (
    GRAPHS_FOLDER
    / "xgboost_hopsworks_rmse_by_horizon.png"
)


plt.savefig(
    rmse_output,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# R² graph
# =========================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    metrics_df[
        "forecast_horizon_hours"
    ],

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


plt.xlabel(
    "Forecast Horizon (Hours)"
)

plt.ylabel(
    "R² Score"
)

plt.title(
    "Hopsworks XGBoost R² "
    "from 1 to 72 Forecast Hours"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


r2_output = (
    GRAPHS_FOLDER
    / "xgboost_hopsworks_r2_by_horizon.png"
)


plt.savefig(
    r2_output,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\nModels saved to:")
print(MODELS_FOLDER)

print("\nFeature column order saved to:")
print(FEATURE_COLUMNS_FILE)

print("\nGraphs saved to:")
print(GRAPHS_FOLDER)

print(
    "\nHopsworks-based training "
    "pipeline completed successfully."
)