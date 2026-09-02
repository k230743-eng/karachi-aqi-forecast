from pathlib import Path
import json
import shutil
import zipfile
import os

import hopsworks
import joblib
import numpy as np
import pandas as pd
import requests


# =========================================================
# Configuration
# =========================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ---------------------------------------------------------
# Hopsworks feature data
# ---------------------------------------------------------

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 2


# ---------------------------------------------------------
# Hopsworks prediction storage
# ---------------------------------------------------------

PREDICTION_FEATURE_GROUP_NAME = (
    "karachi_aqi_predictions"
)

PREDICTION_FEATURE_GROUP_VERSION = 1


# ---------------------------------------------------------
# Registered model
# ---------------------------------------------------------

MODEL_NAME = "karachi_aqi_xgboost_72h"


# ---------------------------------------------------------
# Detect environment
# ---------------------------------------------------------

RUNNING_IN_GITHUB = (
    os.getenv("GITHUB_ACTIONS") == "true"
)


# ---------------------------------------------------------
# Local Windows Hopsworks setup
# ---------------------------------------------------------

if not RUNNING_IN_GITHUB:

    CERT_FOLDER = Path(
        r"D:\karachi-aqi-forecast\.hopsworks_certs"
    )

    CERT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(r"D:\tmp").mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Local model cache
# ---------------------------------------------------------

MODEL_CACHE_FOLDER = Path(
    "models/live_model_cache"
)

MODEL_CACHE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


EXTRACTED_MODEL_FOLDER = (
    MODEL_CACHE_FOLDER
    / "extracted"
)


# ---------------------------------------------------------
# Prediction output
# ---------------------------------------------------------

PREDICTIONS_FOLDER = Path(
    "outputs/predictions"
)

PREDICTIONS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    PREDICTIONS_FOLDER
    / "latest_72h_forecast.csv"
)


# =========================================================
# Exact model feature columns
# =========================================================

BASE_FEATURE_COLUMNS = [

    # Current pollutants
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",

    # Current weather
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    # Current time
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # AQI lags
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",

    # PM2.5 lags
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_12h",
    "pm2_5_lag_24h",
    "pm2_5_lag_48h",
    "pm2_5_lag_72h",

    # PM10 lags
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

    # AQI changes
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


FUTURE_FEATURE_COLUMNS = [

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


EXPECTED_MODEL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + FUTURE_FEATURE_COLUMNS
)


# =========================================================
# Connect to Hopsworks
# =========================================================

print("\nConnecting to Hopsworks...")


if RUNNING_IN_GITHUB:

    print(
        "Running inside GitHub Actions."
    )

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

    print(
        "Running locally."
    )

    project = hopsworks.login(
        cert_folder=str(CERT_FOLDER)
    )


fs = project.get_feature_store()

mr = project.get_model_registry()


print("\nConnected successfully.")


# =========================================================
# Get latest historical feature row
# =========================================================

print(
    "\nReading latest feature data..."
)


feature_group = fs.get_feature_group(

    name=FEATURE_GROUP_NAME,

    version=FEATURE_GROUP_VERSION,
)


feature_df = feature_group.read()


if feature_df.empty:

    raise RuntimeError(
        "Feature group contains no data."
    )


feature_df["time"] = pd.to_datetime(
    feature_df["time"]
)


# Hopsworks may return our Karachi-local clock values
# with a timezone attached.
#
# Preserve the original local clock value.

if feature_df["time"].dt.tz is not None:

    feature_df["time"] = (
        feature_df["time"]
        .dt.tz_localize(None)
    )


feature_df = (
    feature_df
    .sort_values("time")
    .reset_index(drop=True)
)


latest_row = (
    feature_df
    .iloc[-1]
    .copy()
)


latest_time = latest_row["time"]


print(
    "\nLatest feature timestamp:"
)

print(
    latest_time
)


print(
    "\nCurrent AQI:"
)

print(
    latest_row["us_aqi"]
)


# =========================================================
# Validate base model features
# =========================================================

missing_base_features = [

    column

    for column in BASE_FEATURE_COLUMNS

    if column not in latest_row.index
]


if missing_base_features:

    raise RuntimeError(

        "Missing base model features:\n"
        f"{missing_base_features}"
    )


# =========================================================
# Fetch future weather from Open-Meteo
# =========================================================

print(
    "\nFetching 72-hour weather forecast "
    "from Open-Meteo..."
)


weather_variables = (

    "temperature_2m,"
    "relative_humidity_2m,"
    "dew_point_2m,"
    "surface_pressure,"
    "precipitation,"
    "cloud_cover,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "wind_gusts_10m"
)


weather_params = {

    "latitude": LATITUDE,

    "longitude": LONGITUDE,

    "hourly": weather_variables,

    "timezone": TIMEZONE,

    # Four days gives enough room for our
    # complete 72-hour forecast.
    "forecast_days": 4,
}


weather_response = requests.get(

    FORECAST_URL,

    params=weather_params,

    timeout=120,
)


weather_response.raise_for_status()


weather_json = weather_response.json()


if "hourly" not in weather_json:

    raise RuntimeError(

        "Open-Meteo did not return "
        "hourly forecast data."
    )


forecast_weather_df = pd.DataFrame(
    weather_json["hourly"]
)


forecast_weather_df["time"] = (
    pd.to_datetime(
        forecast_weather_df["time"]
    )
)


forecast_weather_df = (
    forecast_weather_df
    .sort_values("time")
    .reset_index(drop=True)
)


print(
    "\nOpen-Meteo forecast range:"
)

print(

    forecast_weather_df["time"].min(),

    "to",

    forecast_weather_df["time"].max()
)


# =========================================================
# Determine required weather timestamps
# =========================================================

required_times = [

    latest_time
    + pd.Timedelta(hours=horizon)

    for horizon in range(1, 73)
]


required_start = required_times[0]

required_end = required_times[-1]


print(
    "\nRequired prediction weather range:"
)

print(
    required_start,
    "to",
    required_end
)


# =========================================================
# Validate forecast coverage
# =========================================================

available_times = set(
    forecast_weather_df["time"]
)


missing_forecast_times = [

    timestamp

    for timestamp in required_times

    if timestamp not in available_times
]


if missing_forecast_times:

    print(
        "\nMissing required weather timestamps:"
    )


    for timestamp in missing_forecast_times[:10]:

        print(timestamp)


    raise RuntimeError(

        "Open-Meteo forecast does not cover "
        "all 72 required target hours."
    )


print(
    "\nAll required future weather "
    "timestamps are available."
)


# =========================================================
# Retrieve model from Model Registry
# =========================================================

print(
    "\nRetrieving registered forecasting model..."
)


available_models = mr.get_models(
    name=MODEL_NAME
)


if not available_models:

    raise RuntimeError(
        f"No registered models found "
        f"for {MODEL_NAME}."
    )


registered_model = max(
    available_models,
    key=lambda model: model.version
)


print(
    "\nUsing latest model version:"
)

print(
    registered_model.version
)


if registered_model is None:

    raise RuntimeError(

        f"Model {MODEL_NAME} "
        f"was not found."
    )


print(
    "\nRegistered model found:"
)

print(
    registered_model.name,
    "version",
    registered_model.version
)


# =========================================================
# Check local model cache
# =========================================================

print(
    "\nChecking local model cache..."
)


EXPECTED_FIRST_MODEL = (
    EXTRACTED_MODEL_FOLDER
    / "xgboost_01h.joblib"
)


EXPECTED_LAST_MODEL = (
    EXTRACTED_MODEL_FOLDER
    / "xgboost_72h.joblib"
)


models_ready = (
    EXPECTED_FIRST_MODEL.exists()
    and
    EXPECTED_LAST_MODEL.exists()
)


# GitHub Actions should always fetch the
# newest registered model version.
if RUNNING_IN_GITHUB:

    models_ready = False


# =========================================================
# Download registered model when cache is missing
# =========================================================

if not models_ready:

    print(
        "\nModels not found locally."
    )

    print(
        "Downloading registered model artifact..."
    )


    DOWNLOAD_FOLDER = (
        MODEL_CACHE_FOLDER
        / "download"
    )


    if DOWNLOAD_FOLDER.exists():

        shutil.rmtree(
            DOWNLOAD_FOLDER
        )


    DOWNLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


    downloaded_path = (
        registered_model.download(
            local_path=str(
                DOWNLOAD_FOLDER
            )
        )
    )


    print(
        "\nDownloaded model path:"
    )

    print(
        downloaded_path
    )


    # -----------------------------------------------------
    # Find downloaded ZIP
    # -----------------------------------------------------

    downloaded_path_obj = Path(
        downloaded_path
    )


    zip_candidates = []


    if (
        downloaded_path_obj.is_file()
        and
        downloaded_path_obj.suffix.lower()
        == ".zip"
    ):

        zip_candidates.append(
            downloaded_path_obj
        )


    if downloaded_path_obj.is_dir():

        zip_candidates.extend(

            downloaded_path_obj.rglob(
                "*.zip"
            )
        )


    zip_candidates.extend(

        DOWNLOAD_FOLDER.rglob(
            "*.zip"
        )
    )


    zip_candidates = list(
        dict.fromkeys(
            zip_candidates
        )
    )


    if not zip_candidates:

        raise RuntimeError(

            "Registered model was downloaded, "
            "but no ZIP artifact was found."
        )


    model_zip = zip_candidates[0]


    print(
        "\nModel ZIP located:"
    )

    print(
        model_zip
    )


    # -----------------------------------------------------
    # Extract models
    # -----------------------------------------------------

    if EXTRACTED_MODEL_FOLDER.exists():

        shutil.rmtree(
            EXTRACTED_MODEL_FOLDER
        )


    EXTRACTED_MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "\nExtracting registered models..."
    )


    with zipfile.ZipFile(
        model_zip,
        "r"
    ) as zip_file:

        zip_file.extractall(
            EXTRACTED_MODEL_FOLDER
        )


    print(
        "Extraction completed."
    )


else:

    print(
        "\nUsing cached models."
    )


# =========================================================
# Locate actual model directory
# =========================================================

model_files = list(

    EXTRACTED_MODEL_FOLDER.rglob(
        "xgboost_01h.joblib"
    )
)


if not model_files:

    raise RuntimeError(

        "xgboost_01h.joblib "
        "could not be found after extraction."
    )


MODEL_FILES_FOLDER = (
    model_files[0].parent
)


print(
    "\nModel files directory:"
)

print(
    MODEL_FILES_FOLDER
)


# =========================================================
# Load saved feature order
# =========================================================

feature_columns_path = (

    MODEL_FILES_FOLDER
    / "feature_columns.json"
)


if not feature_columns_path.exists():

    raise RuntimeError(

        "feature_columns.json "
        "was not found in model artifact."
    )


with open(
    feature_columns_path,
    "r",
    encoding="utf-8",
) as file:

    saved_feature_columns = (
        json.load(file)
    )


if (
    saved_feature_columns
    != EXPECTED_MODEL_FEATURE_COLUMNS
):

    print(
        "\nExpected:"
    )

    print(
        EXPECTED_MODEL_FEATURE_COLUMNS
    )


    print(
        "\nSaved:"
    )

    print(
        saved_feature_columns
    )


    raise RuntimeError(

        "Model feature column order "
        "does not match prediction code."
    )


print(
    "\nModel feature schema verified."
)


# =========================================================
# Generate 72 predictions
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "GENERATING 72-HOUR AQI FORECAST"
)

print(
    "=" * 70
)


prediction_results = []


for horizon in range(1, 73):

    target_time = (

        latest_time

        + pd.Timedelta(
            hours=horizon
        )
    )


    # -----------------------------------------------------
    # Weather forecast at target hour
    # -----------------------------------------------------

    weather_match = (

        forecast_weather_df[
            forecast_weather_df["time"]
            == target_time
        ]
    )


    if len(weather_match) != 1:

        raise RuntimeError(

            f"Expected exactly one weather row "
            f"for {target_time}, "
            f"found {len(weather_match)}."
        )


    weather_row = (
        weather_match.iloc[0]
    )


    # -----------------------------------------------------
    # Start with latest historical feature row
    # -----------------------------------------------------

    model_input = {}


    for column in BASE_FEATURE_COLUMNS:

        model_input[column] = (
            latest_row[column]
        )


    # -----------------------------------------------------
    # Future weather
    # -----------------------------------------------------

    model_input[
        "future_temperature"
    ] = weather_row[
        "temperature_2m"
    ]


    model_input[
        "future_humidity"
    ] = weather_row[
        "relative_humidity_2m"
    ]


    model_input[
        "future_dew_point"
    ] = weather_row[
        "dew_point_2m"
    ]


    model_input[
        "future_pressure"
    ] = weather_row[
        "surface_pressure"
    ]


    model_input[
        "future_precipitation"
    ] = weather_row[
        "precipitation"
    ]


    model_input[
        "future_cloud_cover"
    ] = weather_row[
        "cloud_cover"
    ]


    model_input[
        "future_wind_speed"
    ] = weather_row[
        "wind_speed_10m"
    ]


    model_input[
        "future_wind_gusts"
    ] = weather_row[
        "wind_gusts_10m"
    ]


    # -----------------------------------------------------
    # Circular encoding for future wind direction
    # -----------------------------------------------------

    future_wind_direction = (
        weather_row[
            "wind_direction_10m"
        ]
    )


    future_wind_radians = (
        np.radians(
            future_wind_direction
        )
    )


    model_input[
        "future_wind_direction_sin"
    ] = np.sin(
        future_wind_radians
    )


    model_input[
        "future_wind_direction_cos"
    ] = np.cos(
        future_wind_radians
    )


    # -----------------------------------------------------
    # Target-time features
    # -----------------------------------------------------

    model_input[
        "target_hour"
    ] = target_time.hour


    model_input[
        "target_day_of_week"
    ] = target_time.dayofweek


    model_input[
        "target_month"
    ] = target_time.month


    model_input[
        "target_is_weekend"
    ] = int(
        target_time.dayofweek >= 5
    )


    # -----------------------------------------------------
    # Convert to exact model input format
    # -----------------------------------------------------

    model_input_df = pd.DataFrame(
        [model_input]
    )


    model_input_df = (
        model_input_df[
            saved_feature_columns
        ]
    )


    # -----------------------------------------------------
    # Validate model input
    # -----------------------------------------------------

    if (
        model_input_df
        .isnull()
        .any()
        .any()
    ):

        missing_values = (

            model_input_df
            .isnull()
            .sum()
        )


        missing_values = (

            missing_values[
                missing_values > 0
            ]
        )


        raise RuntimeError(

            f"Missing prediction features "
            f"for horizon {horizon}:\n"
            f"{missing_values}"
        )


    # -----------------------------------------------------
    # Load horizon-specific model
    # -----------------------------------------------------

    model_file = (

        MODEL_FILES_FOLDER

        / f"xgboost_{horizon:02d}h.joblib"
    )


    if not model_file.exists():

        raise FileNotFoundError(

            f"Missing model file: "
            f"{model_file}"
        )


    model = joblib.load(
        model_file
    )


    # -----------------------------------------------------
    # Predict AQI
    # -----------------------------------------------------

    prediction = model.predict(
        model_input_df
    )[0]


    prediction = float(
        prediction
    )


    # AQI cannot be negative
    prediction = max(
        0.0,
        prediction
    )


    # -----------------------------------------------------
    # Save prediction
    # -----------------------------------------------------

    prediction_results.append(
        {

            "forecast_generated_at":
                latest_time,

            "forecast_time":
                target_time,

            "horizon_hours":
                horizon,

            "predicted_aqi":
                prediction,

            "forecast_temperature":
                float(
                    weather_row[
                        "temperature_2m"
                    ]
                ),

            "forecast_humidity":
                float(
                    weather_row[
                        "relative_humidity_2m"
                    ]
                ),

            "forecast_precipitation":
                float(
                    weather_row[
                        "precipitation"
                    ]
                ),

            "forecast_wind_speed":
                float(
                    weather_row[
                        "wind_speed_10m"
                    ]
                ),
        }
    )


    print(

        f"{horizon:02d}h | "
        f"{target_time} | "
        f"AQI = {prediction:.2f}"
    )


# =========================================================
# Create final prediction dataframe
# =========================================================

predictions_df = pd.DataFrame(
    prediction_results
)


# =========================================================
# AQI category
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:

        return "Good"

    elif aqi <= 100:

        return "Moderate"

    elif aqi <= 150:

        return (
            "Unhealthy for Sensitive Groups"
        )

    elif aqi <= 200:

        return "Unhealthy"

    elif aqi <= 300:

        return "Very Unhealthy"

    else:

        return "Hazardous"


predictions_df[
    "aqi_category"
] = predictions_df[
    "predicted_aqi"
].apply(
    get_aqi_category
)


# =========================================================
# Explicit data types for Hopsworks
# =========================================================

predictions_df[
    "horizon_hours"
] = predictions_df[
    "horizon_hours"
].astype("int64")


predictions_df[
    "predicted_aqi"
] = predictions_df[
    "predicted_aqi"
].astype("float64")


predictions_df[
    "forecast_temperature"
] = predictions_df[
    "forecast_temperature"
].astype("float64")


predictions_df[
    "forecast_humidity"
] = predictions_df[
    "forecast_humidity"
].astype("float64")


predictions_df[
    "forecast_precipitation"
] = predictions_df[
    "forecast_precipitation"
].astype("float64")


predictions_df[
    "forecast_wind_speed"
] = predictions_df[
    "forecast_wind_speed"
].astype("float64")


# =========================================================
# Validate prediction dataframe
# =========================================================

if len(predictions_df) != 72:

    raise RuntimeError(

        f"Expected 72 predictions, "
        f"but generated {len(predictions_df)}."
    )


if predictions_df.isnull().any().any():

    raise RuntimeError(

        "Missing values detected "
        "in prediction output."
    )


if predictions_df[
    "forecast_time"
].duplicated().any():

    raise RuntimeError(

        "Duplicate forecast timestamps "
        "detected."
    )


print(
    "\nPrediction dataframe validation passed."
)


# =========================================================
# Save latest forecast locally
# =========================================================

predictions_df.to_csv(

    OUTPUT_FILE,

    index=False,
)


print(
    "\nLocal forecast saved to:"
)

print(
    OUTPUT_FILE
)


# =========================================================
# Create / retrieve prediction Feature Group
# =========================================================

print(
    "\nPreparing Hopsworks prediction Feature Group..."
)


prediction_feature_group = (
    fs.get_or_create_feature_group(

        name=(
            PREDICTION_FEATURE_GROUP_NAME
        ),

        version=(
            PREDICTION_FEATURE_GROUP_VERSION
        ),

        description=(
            "Latest hourly Karachi AQI forecasts "
            "generated by the 1-to-72-hour "
            "XGBoost forecasting system."
        ),

        primary_key=[
            "forecast_time"
        ],

        event_time=(
            "forecast_time"
        ),

        online_enabled=True,

        time_travel_format="HUDI",
    )
)


print(
    "\nPrediction Feature Group ready:"
)

print(
    f"{PREDICTION_FEATURE_GROUP_NAME} "
    f"v{PREDICTION_FEATURE_GROUP_VERSION}"
)


# =========================================================
# Upload / upsert latest forecast
# =========================================================

print(
    "\nUploading latest forecast "
    "to Hopsworks..."
)


prediction_feature_group.insert(

    predictions_df,

    write_options={
        "wait_for_job": True
    },
)


print(
    "\nPrediction upload completed."
)


# =========================================================
# Final summary
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "LIVE AQI FORECAST COMPLETED"
)

print(
    "=" * 70
)


print(
    "\nForecast based on feature timestamp:"
)

print(
    latest_time
)


print(
    "\nForecast range:"
)

print(

    predictions_df[
        "forecast_time"
    ].min(),

    "to",

    predictions_df[
        "forecast_time"
    ].max()
)


print(
    "\nForecast rows:"
)

print(
    len(predictions_df)
)


print(
    "\nMinimum predicted AQI:"
)

print(
    predictions_df[
        "predicted_aqi"
    ].min()
)


print(
    "\nMaximum predicted AQI:"
)

print(
    predictions_df[
        "predicted_aqi"
    ].max()
)


print(
    "\nForecast stored locally at:"
)

print(
    OUTPUT_FILE
)


print(
    "\nForecast stored in Hopsworks:"
)

print(
    f"{PREDICTION_FEATURE_GROUP_NAME} "
    f"v{PREDICTION_FEATURE_GROUP_VERSION}"
)


# =========================================================
# Selected forecast horizons
# =========================================================

print(
    "\nSelected forecast horizons:"
)


print(

    predictions_df[

        predictions_df[
            "horizon_hours"
        ].isin(
            [
                1,
                6,
                12,
                24,
                48,
                72,
            ]
        )

    ][

        [
            "horizon_hours",
            "forecast_time",
            "predicted_aqi",
            "aqi_category",
        ]

    ].to_string(
        index=False
    )
)