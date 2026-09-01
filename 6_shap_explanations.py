from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/karachi_aqi_features.csv"
)

MODELS_FOLDER = Path(
    "models/xgboost_hourly"
)

FEATURE_COLUMNS_FILE = (
    MODELS_FOLDER
    / "feature_columns.json"
)

SHAP_FOLDER = Path(
    "outputs/shap"
)

SHAP_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Horizons to explain
# ---------------------------------------------------------

HORIZONS = [
    1,
    6,
    12,
    24,
    48,
    72,
]


# ---------------------------------------------------------
# Fixed test boundary
# Same as training script
# ---------------------------------------------------------

TEST_START = pd.Timestamp(
    "2025-09-17 20:00:00"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["time"],
)

df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)


print("\nDataset shape:")
print(df.shape)

print("\nDataset range:")
print(
    df["time"].min(),
    "to",
    df["time"].max()
)


# ---------------------------------------------------------
# Load exact feature order used during training
# ---------------------------------------------------------

with open(
    FEATURE_COLUMNS_FILE,
    "r",
    encoding="utf-8",
) as file:
    model_feature_columns = json.load(file)


print("\nNumber of model features:")
print(len(model_feature_columns))


# ---------------------------------------------------------
# Function to recreate features for one horizon
# ---------------------------------------------------------

def create_horizon_dataset(
    df,
    horizon,
):

    horizon_df = df.copy()


    # -----------------------------------------------------
    # Target AQI
    # -----------------------------------------------------

    horizon_df["target_aqi"] = (
        horizon_df["us_aqi"]
        .shift(-horizon)
    )


    # -----------------------------------------------------
    # Future weather
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
    # Target time information
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

    horizon_df[
        "target_day_of_week"
    ] = (
        target_time.dt.dayofweek
    )

    horizon_df["target_month"] = (
        target_time.dt.month
    )

    horizon_df[
        "target_is_weekend"
    ] = (
        horizon_df[
            "target_day_of_week"
        ] >= 5
    ).astype(int)


    # -----------------------------------------------------
    # Drop rows missing target or model features
    # -----------------------------------------------------

    horizon_df = (
        horizon_df
        .dropna(
            subset=(
                model_feature_columns
                + ["target_aqi"]
            )
        )
        .reset_index(drop=True)
    )

    return horizon_df


# ---------------------------------------------------------
# Process each selected horizon
# ---------------------------------------------------------

for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print(
        f"SHAP explanation for "
        f"{horizon}-hour model"
    )
    print("=" * 70)


    # -----------------------------------------------------
    # Recreate horizon-specific dataset
    # -----------------------------------------------------

    horizon_df = (
        create_horizon_dataset(
            df,
            horizon,
        )
    )


    # -----------------------------------------------------
    # Same chronological test split
    # -----------------------------------------------------

    test_df = horizon_df[
        horizon_df["time"]
        >= TEST_START
    ].copy()


    X_test = test_df[
        model_feature_columns
    ].copy()

    y_test = test_df[
        "target_aqi"
    ].copy()


    print(
        "Test rows:",
        len(X_test)
    )


    # -----------------------------------------------------
    # Load saved model
    # -----------------------------------------------------

    model_file = (
        MODELS_FOLDER
        / f"xgboost_{horizon:02d}h.joblib"
    )

    model = joblib.load(
        model_file
    )

    print(
        "Loaded:",
        model_file
    )

    X_shap = X_test.sample(
        n=len(X_test),
        random_state=42,
    )


    # -----------------------------------------------------
    # Create SHAP explainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_shap
    )


    # =====================================================
    # 1. GLOBAL BAR PLOT
    # =====================================================

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.title(
        f"Global SHAP Feature Importance "
        f"- {horizon}h Forecast"
    )

    plt.tight_layout()

    bar_output = (
        SHAP_FOLDER
        / f"shap_bar_{horizon:02d}h.png"
    )

    plt.savefig(
        bar_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # 2. BEESWARM PLOT
    # =====================================================

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.title(
        f"SHAP Summary "
        f"- {horizon}h Forecast"
    )

    plt.tight_layout()

    beeswarm_output = (
        SHAP_FOLDER
        / f"shap_beeswarm_{horizon:02d}h.png"
    )

    plt.savefig(
        beeswarm_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # 3. LOCAL WATERFALL EXPLANATION
    # =====================================================

    # Pick the middle test observation so that
    # we are not always explaining the first row.
    local_index = (
        len(X_test) // 2
    )

    local_row = X_test.iloc[
        local_index:
        local_index + 1
    ]

    local_time = test_df.iloc[
        local_index
    ]["time"]

    actual_aqi = y_test.iloc[
        local_index
    ]

    predicted_aqi = float(
        model.predict(
            local_row
        )[0]
    )


    local_shap_values = explainer(
        local_row
    )


    plt.figure()

    shap.plots.waterfall(
        local_shap_values[0],
        max_display=20,
        show=False,
    )

    plt.title(
        f"{horizon}h AQI Prediction\n"
        f"Time: {local_time}\n"
        f"Actual: {actual_aqi:.1f} | "
        f"Predicted: {predicted_aqi:.1f}"
    )

    plt.tight_layout()

    waterfall_output = (
        SHAP_FOLDER
        / f"shap_waterfall_{horizon:02d}h.png"
    )

    plt.savefig(
        waterfall_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "Saved:",
        bar_output
    )

    print(
        "Saved:",
        beeswarm_output
    )

    print(
        "Saved:",
        waterfall_output
    )


print("\n" + "=" * 70)
print("SHAP analysis completed")
print("=" * 70)

print("\nFiles saved to:")
print(SHAP_FOLDER)