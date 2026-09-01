import os
from pathlib import Path

import pandas as pd
import hopsworks


# ---------------------------------------------------------
# Windows/Hopsworks temp folder workaround
# ---------------------------------------------------------

Path(r"D:\tmp").mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/karachi_aqi_features.csv"
)

CERT_FOLDER = Path(
    r"D:\karachi-aqi-forecast\.hopsworks_certs"
)

CERT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Load processed historical dataset
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
# Create target dataframe
# ---------------------------------------------------------

targets_df = df[
    [
        "time",
        "us_aqi",
    ]
].copy()


# ---------------------------------------------------------
# Create targets from 1h to 72h
# ---------------------------------------------------------

for horizon in range(1, 73):

    targets_df[
        f"aqi_target_{horizon}h"
    ] = (
        targets_df["us_aqi"]
        .shift(-horizon)
    )


# ---------------------------------------------------------
# us_aqi was only needed to create targets
# ---------------------------------------------------------

targets_df = targets_df.drop(
    columns=["us_aqi"]
)


# ---------------------------------------------------------
# Drop rows where future targets do not exist
#
# Since we need ALL 72 targets in each row,
# the final 72 rows will be removed.
# ---------------------------------------------------------

targets_df = (
    targets_df
    .dropna()
    .reset_index(drop=True)
)


print("\nTarget dataset shape:")
print(targets_df.shape)

print("\nTarget dataset range:")
print(
    targets_df["time"].min(),
    "to",
    targets_df["time"].max(),
)

print("\nNumber of target columns:")
print(
    len(targets_df.columns) - 1
)

print("\nFirst few target columns:")
print(
    targets_df[
        [
            "time",
            "aqi_target_1h",
            "aqi_target_6h",
            "aqi_target_12h",
            "aqi_target_24h",
            "aqi_target_48h",
            "aqi_target_72h",
        ]
    ].head()
)

check_row = 500

original_index = (
    df.index[
        df["time"]
        == targets_df.loc[
            check_row,
            "time"
        ]
    ][0]
)

assert (
    targets_df.loc[
        check_row,
        "aqi_target_1h"
    ]
    ==
    df.loc[
        original_index + 1,
        "us_aqi"
    ]
)

assert (
    targets_df.loc[
        check_row,
        "aqi_target_24h"
    ]
    ==
    df.loc[
        original_index + 24,
        "us_aqi"
    ]
)

assert (
    targets_df.loc[
        check_row,
        "aqi_target_72h"
    ]
    ==
    df.loc[
        original_index + 72,
        "us_aqi"
    ]
)

print(
    "\nTarget alignment checks passed."
)


# ---------------------------------------------------------
# Connection to Hopsworks
# ---------------------------------------------------------

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    cert_folder=str(CERT_FOLDER)
)

fs = project.get_feature_store()

print("\nConnected to Feature Store.")


# ---------------------------------------------------------
# Create targets Feature Group
# ---------------------------------------------------------

targets_feature_group = (
    fs.get_or_create_feature_group(
        name="karachi_aqi_targets",
        version=1,

        description=(
            "Hourly AQI forecasting targets for "
            "Karachi from 1 to 72 hours ahead."
        ),

        primary_key=["time"],

        event_time="time",

        online_enabled=False,

        time_travel_format="HUDI",
    )
)


# ---------------------------------------------------------
# Upload historical targets
# ---------------------------------------------------------

print("\nUploading AQI targets...")

targets_feature_group.insert(
    targets_df,
    wait=True,
)


print("\nTarget upload completed.")

print("\nFeature Group:")
print("karachi_aqi_targets")

print("\nRows uploaded:")
print(len(targets_df))

print("\nNumber of AQI targets:")
print(72)