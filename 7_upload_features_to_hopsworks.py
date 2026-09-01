import os
from pathlib import Path

TEMP_FOLDER = Path(
    r"D:\karachi-aqi-forecast\.hopsworks_tmp"
)

TEMP_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

os.environ["TMP"] = str(TEMP_FOLDER)
os.environ["TEMP"] = str(TEMP_FOLDER)
os.environ["TMPDIR"] = str(TEMP_FOLDER)

import pandas as pd
import hopsworks


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/karachi_aqi_features.csv"
)

CERT_FOLDER = Path(".hopsworks_certs")
CERT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load historical feature data
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
    df["time"].max(),
)


# ---------------------------------------------------------
# Convert timestamp
#
# Hopsworks will use this as:
# - primary key
# - event time
# ---------------------------------------------------------

df["time"] = pd.to_datetime(
    df["time"]
)


# ---------------------------------------------------------
# Connect to Hopsworks
# ---------------------------------------------------------

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    cert_folder=str(CERT_FOLDER)
)

fs = project.get_feature_store()

print("\nConnected to Feature Store.")


# ---------------------------------------------------------
# Create or retrieve Feature Group
# ---------------------------------------------------------

feature_group = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=2,
    description=(
        "Historical hourly Karachi air-quality, "
        "weather and engineered AQI forecasting features."
    ),
    primary_key=["time"],
    event_time="time",
    online_enabled=True,
    time_travel_format="HUDI",
)


# ---------------------------------------------------------
# Upload historical features
# ---------------------------------------------------------

print("\nUploading historical features...")

feature_group.insert(
    df,
    wait=True,
)


print("\nUpload completed.")

print("\nFeature Group:")
print("karachi_aqi_features")

print("\nRows uploaded:")
print(len(df))