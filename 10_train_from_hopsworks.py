from pathlib import Path

import hopsworks
import pandas as pd


# ---------------------------------------------------------
# Windows/Hopsworks setup
# ---------------------------------------------------------

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
# Connect to Hopsworks
# ---------------------------------------------------------

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    cert_folder=str(CERT_FOLDER)
)

fs = project.get_feature_store()

print("\nConnected to Feature Store.")


# ---------------------------------------------------------
# Load Feature View
# ---------------------------------------------------------

feature_view = fs.get_feature_view(
    name="karachi_aqi_training_view",
    version=1,
)

print("\nFeature View loaded.")


# ---------------------------------------------------------
# Load training dataset
# ---------------------------------------------------------

X, y = feature_view.get_training_data(
    training_dataset_version=1
)

print("\nFeatures shape:")
print(X.shape)

print("\nLabels shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget columns:")
print(y.columns.tolist())


# ---------------------------------------------------------
# Restore time ordering
# ---------------------------------------------------------

X["time"] = pd.to_datetime(X["time"])

X["time"] = (X["time"].dt.tz_convert("Asia/Karachi"))

combined_df = pd.concat(
    [
        X.reset_index(drop=True),
        y.reset_index(drop=True),
    ],
    axis=1,
)

combined_df = (
    combined_df
    .sort_values("time")
    .reset_index(drop=True)
)

print("\nCombined dataset shape:")
print(combined_df.shape)

print("\nCombined dataset range:")
print(
    combined_df["time"].min(),
    "to",
    combined_df["time"].max(),
)


# ---------------------------------------------------------
# Fixed chronological split
# ---------------------------------------------------------

TEST_START = pd.Timestamp(
    "2025-09-17 20:00:00",
    tz="Asia/Karachi"
)

train_df = combined_df[
    combined_df["time"] < TEST_START
].copy()

test_df = combined_df[
    combined_df["time"] >= TEST_START
].copy()


print("\nTraining period:")
print(
    train_df["time"].min(),
    "to",
    train_df["time"].max(),
)

print("\nTest period:")
print(
    test_df["time"].min(),
    "to",
    test_df["time"].max(),
)

print("\nTraining rows:")
print(len(train_df))

print("\nTest rows:")
print(len(test_df))