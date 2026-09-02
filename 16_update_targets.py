from pathlib import Path
import os

import hopsworks
import pandas as pd


# =========================================================
# Configuration
# =========================================================

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 2

TARGET_GROUP_NAME = "karachi_aqi_targets"
TARGET_GROUP_VERSION = 1

MAX_HORIZON = 72


# =========================================================
# Environment
# =========================================================

RUNNING_IN_GITHUB = (
    os.getenv("GITHUB_ACTIONS") == "true"
)


if not RUNNING_IN_GITHUB:

    CERT_FOLDER = Path(
        r"D:\karachi-aqi-forecast\.hopsworks_certs"
    )

    CERT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(r"D:\tmp").mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# Connect to Hopsworks
# =========================================================

print("\nConnecting to Hopsworks...")


if RUNNING_IN_GITHUB:

    print("Running inside GitHub Actions.")

    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
    )

else:

    print("Running locally.")

    project = hopsworks.login(
        cert_folder=str(CERT_FOLDER)
    )


fs = project.get_feature_store()


feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
)


target_group = fs.get_feature_group(
    name=TARGET_GROUP_NAME,
    version=TARGET_GROUP_VERSION,
)


print("\nConnected successfully.")


# =========================================================
# Read AQI history
# =========================================================

print("\nReading AQI history...")


feature_df = feature_group.select(
    [
        "time",
        "us_aqi",
    ]
).read()


if feature_df.empty:

    raise RuntimeError(
        "Feature group is empty."
    )


feature_df["time"] = pd.to_datetime(
    feature_df["time"]
)


# Preserve original Karachi-local clock values.
if feature_df["time"].dt.tz is not None:

    feature_df["time"] = (
        feature_df["time"]
        .dt.tz_localize(None)
    )


feature_df = (
    feature_df
    .sort_values("time")
    .drop_duplicates(
        subset=["time"],
        keep="last",
    )
    .reset_index(drop=True)
)


print(
    "Feature rows:",
    len(feature_df),
)

print(
    "Feature range:",
    feature_df["time"].min(),
    "to",
    feature_df["time"].max(),
)


# =========================================================
# Validate hourly continuity
# =========================================================

expected_times = pd.date_range(
    start=feature_df["time"].min(),
    end=feature_df["time"].max(),
    freq="h",
)


actual_times = pd.DatetimeIndex(
    feature_df["time"]
)


missing_times = (
    expected_times
    .difference(actual_times)
)


if len(missing_times) > 0:

    print(
        "\nMissing timestamps:"
    )

    print(
        missing_times[:20]
    )

    raise RuntimeError(
        "Hourly gaps exist in feature history. "
        "Target generation stopped."
    )


print(
    "\nNo hourly gaps detected."
)


# =========================================================
# Determine newest fully mature training timestamp
# =========================================================

latest_feature_time = (
    feature_df["time"].max()
)


latest_mature_target_time = (
    latest_feature_time
    - pd.Timedelta(
        hours=MAX_HORIZON
    )
)


print(
    "\nLatest feature timestamp:"
)

print(
    latest_feature_time
)


print(
    "\nLatest timestamp for which all "
    "72 targets are now known:"
)

print(
    latest_mature_target_time
)


# =========================================================
# Read existing target group
# =========================================================

print(
    "\nReading existing targets..."
)


existing_targets_df = (
    target_group
    .select(["time"])
    .read()
)


if existing_targets_df.empty:

    latest_existing_target_time = None

else:

    existing_targets_df[
        "time"
    ] = pd.to_datetime(
        existing_targets_df["time"]
    )


    if (
        existing_targets_df[
            "time"
        ].dt.tz
        is not None
    ):

        existing_targets_df[
            "time"
        ] = (
            existing_targets_df[
                "time"
            ]
            .dt.tz_localize(None)
        )


    latest_existing_target_time = (
        existing_targets_df[
            "time"
        ].max()
    )


print(
    "\nLatest existing target timestamp:"
)

print(
    latest_existing_target_time
)


# =========================================================
# Determine rows that need targets
# =========================================================

if latest_existing_target_time is None:

    target_start_time = (
        feature_df["time"].min()
    )

else:

    target_start_time = (
        latest_existing_target_time
        + pd.Timedelta(hours=1)
    )


if target_start_time > latest_mature_target_time:

    print(
        "\nNo new fully mature target rows "
        "are available."
    )

    print(
        "Target pipeline completed successfully."
    )

    raise SystemExit(0)


print(
    "\nNew target range:"
)

print(
    target_start_time,
    "to",
    latest_mature_target_time,
)


# =========================================================
# Create timestamp-indexed AQI series
# =========================================================

aqi_series = (
    feature_df
    .set_index("time")[
        "us_aqi"
    ]
)


base_times = pd.date_range(
    start=target_start_time,
    end=latest_mature_target_time,
    freq="h",
)


targets_df = pd.DataFrame(
    {
        "time": base_times
    }
)


# =========================================================
# Generate 1h -> 72h targets
# =========================================================

print(
    "\nGenerating targets..."
)


for horizon in range(
    1,
    MAX_HORIZON + 1,
):

    future_times = (
        targets_df["time"]
        + pd.Timedelta(
            hours=horizon
        )
    )


    targets_df[
        f"aqi_target_{horizon}h"
    ] = (
        future_times.map(
            aqi_series
        )
    )


print(
    "Target generation completed."
)


# =========================================================
# Exact column order
# =========================================================

TARGET_COLUMNS = [

    "time",

    *[
        f"aqi_target_{horizon}h"

        for horizon in range(
            1,
            MAX_HORIZON + 1
        )
    ],
]


targets_df = targets_df[
    TARGET_COLUMNS
].copy()


# =========================================================
# Validate targets
# =========================================================

if targets_df.empty:

    raise RuntimeError(
        "No target rows were generated."
    )


if targets_df.isnull().any().any():

    null_counts = (
        targets_df
        .isnull()
        .sum()
    )

    null_counts = (
        null_counts[
            null_counts > 0
        ]
    )

    raise RuntimeError(
        "Missing values detected in "
        f"generated targets:\n{null_counts}"
    )


if targets_df[
    "time"
].duplicated().any():

    raise RuntimeError(
        "Duplicate target timestamps detected."
    )


# Explicit Hopsworks schema

for horizon in range(
    1,
    MAX_HORIZON + 1,
):

    column = (
        f"aqi_target_{horizon}h"
    )

    targets_df[column] = (
        targets_df[column]
        .astype("float64")
    )


print(
    "\nTarget validation passed."
)


print(
    "\nRows to upload:",
    len(targets_df),
)


print(
    "Upload range:",
    targets_df["time"].min(),
    "to",
    targets_df["time"].max(),
)


# =========================================================
# Upload to Hopsworks
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "UPLOADING TARGETS TO HOPSWORKS"
)

print(
    "=" * 70
)


target_group.insert(

    targets_df,

    write_options={
        "wait_for_job": True
    },
)


# =========================================================
# Summary
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TARGET UPDATE COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nNew target rows added:"
)

print(
    len(targets_df)
)


print(
    "\nNewest complete training timestamp:"
)

print(
    targets_df["time"].max()
)


print(
    "\nAll rows include targets "
    "from 1h through 72h."
)