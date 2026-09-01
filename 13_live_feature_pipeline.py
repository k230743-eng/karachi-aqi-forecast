from pathlib import Path
from datetime import timedelta
import os

import hopsworks
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# Configuration
# =========================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ---------------------------------------------------------
# Hopsworks configuration
# ---------------------------------------------------------

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 2

RUNNING_IN_GITHUB = (
    os.getenv("GITHUB_ACTIONS") == "true"
)


#local winwos hopsworks setup
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
# Pipeline configuration
# ---------------------------------------------------------

# Largest feature lag is 168 hours.
#
# We fetch 192 hours to give ourselves a 24-hour buffer.
LOOKBACK_HOURS = 192


# Recalculate and upsert the latest 48 hours each run.
#
# This is useful because very recent API/model data can
# sometimes be revised after it first becomes available.
REFRESH_HOURS = 48


# Split large API downloads into smaller requests.
#
# This means that if the pipeline has not run for several
# months, it will not try to retrieve everything with one
# giant API request.
API_CHUNK_DAYS = 30


# ---------------------------------------------------------
# Optional local cache
# ---------------------------------------------------------

CACHE_FOLDER = Path(
    "data/live_cache"
)

CACHE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Exact Hopsworks feature schema
# =========================================================

EXPECTED_COLUMNS = [

    # Timestamp
    "time",

    # -----------------------------------------------------
    # Current pollutant values
    # -----------------------------------------------------

    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",

    # -----------------------------------------------------
    # Current weather values
    # -----------------------------------------------------

    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    # -----------------------------------------------------
    # Current time features
    # -----------------------------------------------------

    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # -----------------------------------------------------
    # AQI lag features
    # -----------------------------------------------------

    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",

    # -----------------------------------------------------
    # PM2.5 lag features
    # -----------------------------------------------------

    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_12h",
    "pm2_5_lag_24h",
    "pm2_5_lag_48h",
    "pm2_5_lag_72h",

    # -----------------------------------------------------
    # PM10 lag features
    # -----------------------------------------------------

    "pm10_lag_1h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",
    "pm10_lag_48h",
    "pm10_lag_72h",

    # -----------------------------------------------------
    # AQI rolling means
    # -----------------------------------------------------

    "aqi_mean_3h",
    "aqi_mean_6h",
    "aqi_mean_12h",
    "aqi_mean_24h",
    "aqi_mean_48h",
    "aqi_mean_72h",

    # -----------------------------------------------------
    # PM2.5 rolling means
    # -----------------------------------------------------

    "pm2_5_mean_3h",
    "pm2_5_mean_6h",
    "pm2_5_mean_12h",
    "pm2_5_mean_24h",
    "pm2_5_mean_48h",
    "pm2_5_mean_72h",

    # -----------------------------------------------------
    # PM10 rolling means
    # -----------------------------------------------------

    "pm10_mean_3h",
    "pm10_mean_6h",
    "pm10_mean_12h",
    "pm10_mean_24h",
    "pm10_mean_48h",
    "pm10_mean_72h",

    # -----------------------------------------------------
    # AQI change features
    # -----------------------------------------------------

    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_12h",
    "aqi_change_24h",
    "aqi_change_48h",
    "aqi_change_72h",

    # -----------------------------------------------------
    # AQI volatility
    # -----------------------------------------------------

    "aqi_std_24h",
    "aqi_std_72h",
]


# =========================================================
# Open-Meteo variable lists
# =========================================================

AIR_QUALITY_VARIABLES = (
    "pm10,"
    "pm2_5,"
    "carbon_monoxide,"
    "nitrogen_dioxide,"
    "sulphur_dioxide,"
    "ozone,"
    "dust,"
    "aerosol_optical_depth,"
    "us_aqi"
)


WEATHER_VARIABLES = (
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


# =========================================================
# HTTP session with automatic retries
# =========================================================

def create_http_session():

    retry_strategy = Retry(

        total=5,

        connect=5,

        read=5,

        status=5,

        backoff_factor=2,

        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],

        allowed_methods=frozenset(
            ["GET"]
        ),
    )


    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )


    session = requests.Session()

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )


    return session


# =========================================================
# Fetch one API chunk
# =========================================================

def fetch_api_chunk(
    session,
    url,
    variables,
    start_date,
    end_date,
    label,
):

    params = {

        "latitude": LATITUDE,

        "longitude": LONGITUDE,

        "start_date": start_date,

        "end_date": end_date,

        "hourly": variables,

        "timezone": TIMEZONE,
    }


    print(
        f"Fetching {label}: "
        f"{start_date} -> {end_date}"
    )


    response = session.get(
        url,
        params=params,
        timeout=120,
    )


    response.raise_for_status()


    data = response.json()


    if "hourly" not in data:

        raise RuntimeError(
            f"No hourly data returned for {label}.\n"
            f"Response: {data}"
        )


    df = pd.DataFrame(
        data["hourly"]
    )


    if df.empty:

        raise RuntimeError(
            f"Open-Meteo returned an empty "
            f"{label} dataframe."
        )


    df["time"] = pd.to_datetime(
        df["time"]
    )


    return df


# =========================================================
# Fetch date range in chunks
# =========================================================

def fetch_in_chunks(
    session,
    url,
    variables,
    start_date,
    end_date,
    label,
):

    all_chunks = []


    current_start = start_date


    while current_start <= end_date:

        current_end = min(
            current_start
            + timedelta(
                days=API_CHUNK_DAYS - 1
            ),

            end_date,
        )


        chunk_df = fetch_api_chunk(

            session=session,

            url=url,

            variables=variables,

            start_date=(
                current_start.isoformat()
            ),

            end_date=(
                current_end.isoformat()
            ),

            label=label,
        )


        all_chunks.append(
            chunk_df
        )


        current_start = (
            current_end
            + timedelta(days=1)
        )


    final_df = pd.concat(
        all_chunks,
        ignore_index=True,
    )


    final_df = (
        final_df
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


    return final_df


# =========================================================
# Feature engineering
#
# IMPORTANT:
# These formulas intentionally match 2_prepare_data.py.
# =========================================================

def engineer_features(
    air_quality_df,
    weather_df,
):

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    merged_df = pd.merge(

        air_quality_df,

        weather_df,

        on="time",

        how="inner",
    )


    merged_df["time"] = pd.to_datetime(
        merged_df["time"]
    )


    merged_df = (
        merged_df
        .sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )


    print(
        "\nMerged raw data shape:",
        merged_df.shape
    )


    # -----------------------------------------------------
    # Essential air-quality fields
    # -----------------------------------------------------

    essential_air_quality_columns = [

        "pm10",

        "pm2_5",

        "carbon_monoxide",

        "nitrogen_dioxide",

        "sulphur_dioxide",

        "ozone",

        "us_aqi",
    ]


    merged_df = (
        merged_df
        .dropna(
            subset=(
                essential_air_quality_columns
            )
        )
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # IMPORTANT:
    # shift(24) means 24 ROWS, not automatically 24 hours.
    #
    # Therefore we make sure there are no missing hours.
    # -----------------------------------------------------

    time_differences = (
        merged_df["time"]
        .diff()
    )


    missing_hour_gaps = (
        time_differences[
            time_differences
            > pd.Timedelta(hours=1)
        ]
    )


    if not missing_hour_gaps.empty:

        print(
            "\nERROR: Missing hourly gaps detected:"
        )

        print(
            missing_hour_gaps
        )


        raise RuntimeError(

            "Hourly gaps exist in the fetched dataset. "
            "Feature engineering has been stopped because "
            "using shift() with missing hours would produce "
            "incorrect lag features."
        )


    print(
        "No hourly gaps detected."
    )


    # =====================================================
    # Time features
    # =====================================================

    merged_df["hour"] = (
        merged_df["time"].dt.hour
    )


    merged_df["day_of_week"] = (
        merged_df["time"].dt.dayofweek
    )


    merged_df["month"] = (
        merged_df["time"].dt.month
    )


    merged_df["is_weekend"] = (
        merged_df["day_of_week"] >= 5
    ).astype(int)


    # =====================================================
    # Lag features
    # =====================================================

    # AQI

    merged_df["aqi_lag_1h"] = (
        merged_df["us_aqi"].shift(1)
    )

    merged_df["aqi_lag_3h"] = (
        merged_df["us_aqi"].shift(3)
    )

    merged_df["aqi_lag_6h"] = (
        merged_df["us_aqi"].shift(6)
    )

    merged_df["aqi_lag_12h"] = (
        merged_df["us_aqi"].shift(12)
    )

    merged_df["aqi_lag_24h"] = (
        merged_df["us_aqi"].shift(24)
    )

    merged_df["aqi_lag_48h"] = (
        merged_df["us_aqi"].shift(48)
    )

    merged_df["aqi_lag_72h"] = (
        merged_df["us_aqi"].shift(72)
    )

    merged_df["aqi_lag_168h"] = (
        merged_df["us_aqi"].shift(168)
    )


    # -----------------------------------------------------
    # PM2.5
    # -----------------------------------------------------

    merged_df["pm2_5_lag_1h"] = (
        merged_df["pm2_5"].shift(1)
    )

    merged_df["pm2_5_lag_3h"] = (
        merged_df["pm2_5"].shift(3)
    )

    merged_df["pm2_5_lag_6h"] = (
        merged_df["pm2_5"].shift(6)
    )

    merged_df["pm2_5_lag_12h"] = (
        merged_df["pm2_5"].shift(12)
    )

    merged_df["pm2_5_lag_24h"] = (
        merged_df["pm2_5"].shift(24)
    )

    merged_df["pm2_5_lag_48h"] = (
        merged_df["pm2_5"].shift(48)
    )

    merged_df["pm2_5_lag_72h"] = (
        merged_df["pm2_5"].shift(72)
    )


    # -----------------------------------------------------
    # PM10
    # -----------------------------------------------------

    merged_df["pm10_lag_1h"] = (
        merged_df["pm10"].shift(1)
    )

    merged_df["pm10_lag_3h"] = (
        merged_df["pm10"].shift(3)
    )

    merged_df["pm10_lag_6h"] = (
        merged_df["pm10"].shift(6)
    )

    merged_df["pm10_lag_12h"] = (
        merged_df["pm10"].shift(12)
    )

    merged_df["pm10_lag_24h"] = (
        merged_df["pm10"].shift(24)
    )

    merged_df["pm10_lag_48h"] = (
        merged_df["pm10"].shift(48)
    )

    merged_df["pm10_lag_72h"] = (
        merged_df["pm10"].shift(72)
    )


    # =====================================================
    # Rolling averages
    # =====================================================

    # AQI

    merged_df["aqi_mean_3h"] = (
        merged_df["us_aqi"]
        .rolling(window=3)
        .mean()
    )

    merged_df["aqi_mean_6h"] = (
        merged_df["us_aqi"]
        .rolling(window=6)
        .mean()
    )

    merged_df["aqi_mean_12h"] = (
        merged_df["us_aqi"]
        .rolling(window=12)
        .mean()
    )

    merged_df["aqi_mean_24h"] = (
        merged_df["us_aqi"]
        .rolling(window=24)
        .mean()
    )

    merged_df["aqi_mean_48h"] = (
        merged_df["us_aqi"]
        .rolling(window=48)
        .mean()
    )

    merged_df["aqi_mean_72h"] = (
        merged_df["us_aqi"]
        .rolling(window=72)
        .mean()
    )


    # -----------------------------------------------------
    # PM2.5
    # -----------------------------------------------------

    merged_df["pm2_5_mean_3h"] = (
        merged_df["pm2_5"]
        .rolling(window=3)
        .mean()
    )

    merged_df["pm2_5_mean_6h"] = (
        merged_df["pm2_5"]
        .rolling(window=6)
        .mean()
    )

    merged_df["pm2_5_mean_12h"] = (
        merged_df["pm2_5"]
        .rolling(window=12)
        .mean()
    )

    merged_df["pm2_5_mean_24h"] = (
        merged_df["pm2_5"]
        .rolling(window=24)
        .mean()
    )

    merged_df["pm2_5_mean_48h"] = (
        merged_df["pm2_5"]
        .rolling(window=48)
        .mean()
    )

    merged_df["pm2_5_mean_72h"] = (
        merged_df["pm2_5"]
        .rolling(window=72)
        .mean()
    )


    # -----------------------------------------------------
    # PM10
    # -----------------------------------------------------

    merged_df["pm10_mean_3h"] = (
        merged_df["pm10"]
        .rolling(window=3)
        .mean()
    )

    merged_df["pm10_mean_6h"] = (
        merged_df["pm10"]
        .rolling(window=6)
        .mean()
    )

    merged_df["pm10_mean_12h"] = (
        merged_df["pm10"]
        .rolling(window=12)
        .mean()
    )

    merged_df["pm10_mean_24h"] = (
        merged_df["pm10"]
        .rolling(window=24)
        .mean()
    )

    merged_df["pm10_mean_48h"] = (
        merged_df["pm10"]
        .rolling(window=48)
        .mean()
    )

    merged_df["pm10_mean_72h"] = (
        merged_df["pm10"]
        .rolling(window=72)
        .mean()
    )


    # =====================================================
    # AQI change features
    # =====================================================

    merged_df["aqi_change_1h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_1h"]
    )

    merged_df["aqi_change_3h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_3h"]
    )

    merged_df["aqi_change_6h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_6h"]
    )

    merged_df["aqi_change_12h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_12h"]
    )

    merged_df["aqi_change_24h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_24h"]
    )

    merged_df["aqi_change_48h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_48h"]
    )

    merged_df["aqi_change_72h"] = (
        merged_df["us_aqi"]
        - merged_df["aqi_lag_72h"]
    )


    # =====================================================
    # AQI volatility
    # =====================================================

    merged_df["aqi_std_24h"] = (
        merged_df["us_aqi"]
        .rolling(window=24)
        .std()
    )

    merged_df["aqi_std_72h"] = (
        merged_df["us_aqi"]
        .rolling(window=72)
        .std()
    )


    # =====================================================
    # Drop rows missing engineered values
    # =====================================================

    merged_df = (
        merged_df
        .dropna()
        .reset_index(drop=True)
    )


    # =====================================================
    # Exact schema/order
    # =====================================================

    missing_columns = [

        column

        for column in EXPECTED_COLUMNS

        if column not in merged_df.columns
    ]


    if missing_columns:

        raise RuntimeError(

            "The following expected features "
            "were not generated:\n"
            f"{missing_columns}"
        )


    # Keep ONLY the exact columns expected by Hopsworks.
    merged_df = merged_df[
        EXPECTED_COLUMNS
    ].copy()


    if list(merged_df.columns) != EXPECTED_COLUMNS:

        raise RuntimeError(
            "Feature column order mismatch."
        )


    print(
        "\nFeature engineering successful."
    )

    print(
        "Feature shape:",
        merged_df.shape
    )


    return merged_df


# =========================================================
# Connect to Hopsworks
# =========================================================

print(
    "\nConnecting to Hopsworks..."
)


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


feature_group = fs.get_feature_group(

    name=FEATURE_GROUP_NAME,

    version=FEATURE_GROUP_VERSION,
)


print(
    "\nConnected to feature group:"
)

print(
    f"{FEATURE_GROUP_NAME} "
    f"v{FEATURE_GROUP_VERSION}"
)


# =========================================================
# Determine latest stored timestamp
# =========================================================

print(
    "\nChecking latest timestamp "
    "already stored in Hopsworks..."
)


stored_df = feature_group.read()


if stored_df.empty:

    raise RuntimeError(

        "The feature group is empty. "
        "This script expects the historical "
        "backfill to already exist."
    )


stored_df["time"] = pd.to_datetime(
    stored_df["time"]
)


# Hopsworks may return our original local timestamps
# with a UTC timezone tag.
#
# Remove the tag WITHOUT changing the clock value because
# the original dataset was stored using Karachi-local
# naive timestamps.

if stored_df["time"].dt.tz is not None:

    stored_df["time"] = (
        stored_df["time"]
        .dt.tz_localize(None)
    )


latest_stored_time = (
    stored_df["time"].max()
)


print(
    "\nLatest stored timestamp:"
)

print(
    latest_stored_time
)


# =========================================================
# Determine latest hour we want to store
# =========================================================

karachi_now = pd.Timestamp.now(
    tz=TIMEZONE
)


# Use the most recently COMPLETED hour.
#
# Example:
# current time = 20:34
# latest allowed = 19:00
#
# This avoids depending on a partially completed hour.

latest_allowed_time = (
    karachi_now
    .floor("h")
    - pd.Timedelta(hours=1)
)


# Remove timezone while preserving local Karachi clock time.

latest_allowed_time = (
    latest_allowed_time
    .tz_localize(None)
)


print(
    "\nCurrent Karachi time:"
)

print(
    karachi_now
)


print(
    "\nLatest completed hour:"
)

print(
    latest_allowed_time
)


# =========================================================
# Determine what needs to be rebuilt
# =========================================================

next_missing_time = (
    latest_stored_time
    + pd.Timedelta(hours=1)
)


refresh_start_time = (
    latest_allowed_time
    - pd.Timedelta(
        hours=REFRESH_HOURS - 1
    )
)


# Whichever is earlier:
#
# 1. the beginning of the missing gap
# 2. the beginning of our refresh window

recompute_start_time = min(
    next_missing_time,
    refresh_start_time,
)


# We need historical context BEFORE the first row
# we intend to calculate.

fetch_start_time = (
    recompute_start_time
    - pd.Timedelta(
        hours=LOOKBACK_HOURS
    )
)


print(
    "\nNext missing timestamp:"
)

print(
    next_missing_time
)


print(
    "\nRefresh window begins:"
)

print(
    refresh_start_time
)


print(
    "\nFeatures will be recalculated from:"
)

print(
    recompute_start_time
)


print(
    "\nRaw API data will be fetched from:"
)

print(
    fetch_start_time
)


# =========================================================
# Check whether there is anything to do
# =========================================================

if latest_stored_time >= latest_allowed_time:

    print(
        "\nNo new hourly data is missing."
    )

    print(
        "The latest 48 hours will still be "
        "refreshed."
    )


# =========================================================
# API dates
# =========================================================

api_start_date = (
    fetch_start_time.date()
)


api_end_date = (
    latest_allowed_time.date()
)


print(
    "\nAPI date range:"
)

print(
    api_start_date,
    "to",
    api_end_date
)


# =========================================================
# Fetch Open-Meteo data
# =========================================================

session = create_http_session()


print(
    "\n" + "=" * 70
)

print(
    "FETCHING AIR QUALITY DATA"
)

print(
    "=" * 70
)


air_quality_df = fetch_in_chunks(

    session=session,

    url=AIR_QUALITY_URL,

    variables=AIR_QUALITY_VARIABLES,

    start_date=api_start_date,

    end_date=api_end_date,

    label="air-quality data",
)


print(
    "\n" + "=" * 70
)

print(
    "FETCHING WEATHER DATA"
)

print(
    "=" * 70
)


weather_df = fetch_in_chunks(

    session=session,

    url=WEATHER_URL,

    variables=WEATHER_VARIABLES,

    start_date=api_start_date,

    end_date=api_end_date,

    label="weather data",
)


# =========================================================
# Remove anything beyond latest completed hour
# =========================================================

air_quality_df = air_quality_df[
    air_quality_df["time"]
    <= latest_allowed_time
].copy()


weather_df = weather_df[
    weather_df["time"]
    <= latest_allowed_time
].copy()


air_quality_df = (
    air_quality_df
    .sort_values("time")
    .reset_index(drop=True)
)


weather_df = (
    weather_df
    .sort_values("time")
    .reset_index(drop=True)
)


print(
    "\nAir-quality rows fetched:",
    len(air_quality_df)
)

print(
    "Air-quality range:",
    air_quality_df["time"].min(),
    "to",
    air_quality_df["time"].max()
)


print(
    "\nWeather rows fetched:",
    len(weather_df)
)

print(
    "Weather range:",
    weather_df["time"].min(),
    "to",
    weather_df["time"].max()
)


# =========================================================
# Save local cache for debugging
# =========================================================

air_quality_df.to_csv(

    CACHE_FOLDER
    / "latest_air_quality_fetch.csv",

    index=False,
)


weather_df.to_csv(

    CACHE_FOLDER
    / "latest_weather_fetch.csv",

    index=False,
)


# =========================================================
# Generate engineered features
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "ENGINEERING FEATURES"
)

print(
    "=" * 70
)


features_df = engineer_features(

    air_quality_df,

    weather_df,
)


# =========================================================
# Keep only rows that need inserting/updating
# =========================================================

rows_to_upload = features_df[

    (
        features_df["time"]
        >= recompute_start_time
    )

    &

    (
        features_df["time"]
        <= latest_allowed_time
    )

].copy()


rows_to_upload = (
    rows_to_upload
    .sort_values("time")
    .reset_index(drop=True)
)

# Match Hopsworks schema

rows_to_upload["us_aqi"] = (
    rows_to_upload["us_aqi"].astype("float64")
)

rows_to_upload["hour"] = (
    rows_to_upload["hour"].astype("int64")
)

rows_to_upload["day_of_week"] = (
    rows_to_upload["day_of_week"].astype("int64")
)

rows_to_upload["month"] = (
    rows_to_upload["month"].astype("int64")
)

rows_to_upload["is_weekend"] = (
    rows_to_upload["is_weekend"].astype("int64")
)


print(
    "\nRows prepared for Hopsworks:",
    len(rows_to_upload)
)


if not rows_to_upload.empty:

    print(
        "\nUpload range:"
    )

    print(
        rows_to_upload["time"].min(),
        "to",
        rows_to_upload["time"].max()
    )


# =========================================================
# Count genuinely new rows
# =========================================================

new_rows = rows_to_upload[

    rows_to_upload["time"]
    > latest_stored_time

]


refresh_rows = rows_to_upload[

    rows_to_upload["time"]
    <= latest_stored_time

]


print(
    "\nNew missing rows:",
    len(new_rows)
)


print(
    "Existing recent rows being refreshed:",
    len(refresh_rows)
)


# =========================================================
# Final validation
# =========================================================

if rows_to_upload.empty:

    print(
        "\nThere are no rows to upload."
    )

    print(
        "Pipeline finished successfully."
    )

    raise SystemExit(0)


if list(
    rows_to_upload.columns
) != EXPECTED_COLUMNS:

    raise RuntimeError(
        "Final feature schema does not "
        "match Hopsworks."
    )


if rows_to_upload.isnull().any().any():

    null_counts = (
        rows_to_upload
        .isnull()
        .sum()
    )

    null_counts = null_counts[
        null_counts > 0
    ]

    raise RuntimeError(

        "Missing values detected before upload:\n"
        f"{null_counts}"
    )


if rows_to_upload[
    "time"
].duplicated().any():

    raise RuntimeError(
        "Duplicate timestamps detected "
        "before upload."
    )


print(
    "\nFinal validation passed."
)


# =========================================================
# Upload / upsert into Hopsworks
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "UPLOADING FEATURES TO HOPSWORKS"
)

print(
    "=" * 70
)


feature_group.insert(

    rows_to_upload,

    write_options={
        "wait_for_job": True
    },
)


# =========================================================
# Verify result
# =========================================================

print(
    "\nUpload completed."
)


print(
    "\nExpected newest timestamp:"
)

print(
    rows_to_upload["time"].max()
)


print(
    "\n" + "=" * 70
)

print(
    "LIVE FEATURE PIPELINE COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nPrevious latest timestamp:"
)

print(
    latest_stored_time
)


print(
    "\nNew rows added:"
)

print(
    len(new_rows)
)


print(
    "\nRecent rows refreshed:"
)

print(
    len(refresh_rows)
)


print(
    "\nNewest timestamp processed:"
)

print(
    rows_to_upload["time"].max()
)