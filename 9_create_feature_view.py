from pathlib import Path

import hopsworks


# ---------------------------------------------------------
# Windows certificate/temp workaround
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
# Get existing Feature Groups
# ---------------------------------------------------------

features_fg = fs.get_feature_group(
    name="karachi_aqi_features",
    version=2,   # use the version that successfully uploaded
)

targets_fg = fs.get_feature_group(
    name="karachi_aqi_targets",
    version=1,
)

print("\nFeature Groups loaded.")


# ---------------------------------------------------------
# Build queries
# ---------------------------------------------------------

features_query = features_fg.select_all()

targets_query = targets_fg.select_features()


# ---------------------------------------------------------
# Join on time
# ---------------------------------------------------------

joined_query = features_query.join(
    targets_query,
    left_on=["time"],
    right_on=["time"],
)


# ---------------------------------------------------------
# Inspect the joined result
# ---------------------------------------------------------

print("\nJoined query preview:")

print(joined_query.show(5))


# ---------------------------------------------------------
# Create Feature View
# ---------------------------------------------------------

label_columns = [
    f"aqi_target_{horizon}h"
    for horizon in range(1, 73)
]

feature_view = fs.get_or_create_feature_view(
    name="karachi_aqi_training_view",
    version=1,

    description=(
        "Karachi AQI training view containing historical "
        "weather, pollutant and engineered features joined "
        "with AQI targets from 1 to 72 hours ahead."
    ),

    query=joined_query,
    labels=label_columns,
)


print("\nFeature View created successfully.")

print("Name: karachi_aqi_training_view")
print("Version: 1")

print("\nCreating training dataset...")

training_dataset_version, job = (
    feature_view.create_training_data(
        description=(
            "Historical Karachi AQI training dataset "
            "for 1-to-72-hour forecasting."
        ),

        data_format="parquet",

        write_options={
            "wait_for_job": True
        },
    )
)

print("\nTraining dataset created.")

print(
    "Training dataset version:",
    training_dataset_version,
)

print("\nPreviewing training data...")

features_df, labels_df = (
    feature_view.get_training_data(
        training_dataset_version=training_dataset_version
    )
)

print("\nFeatures shape:")
print(features_df.shape)

print("\nFeature columns:")
print(features_df.columns.tolist())

print("\nLabels shape:")
print(labels_df.shape)

print("\nLabel columns:")
print(labels_df.columns.tolist())

print("\nLabels preview:")
print(labels_df.head())