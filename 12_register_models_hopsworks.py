from pathlib import Path
import shutil
import json

import hopsworks
import pandas as pd


# ---------------------------------------------------------
# Windows / Hopsworks setup
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
# Paths
# ---------------------------------------------------------

MODELS_FOLDER = Path(
    "models/xgboost_hourly_hopsworks"
)

METRICS_FILE = Path(
    "outputs/metrics/"
    "xgboost_hourly_hopsworks_1_to_72_metrics.csv"
)

ARTIFACT_FOLDER = Path(
    "models/hopsworks_registry_artifact"
)

ZIP_BASE = Path(
    "models/karachi_aqi_xgboost_72h"
)


# ---------------------------------------------------------
# Rebuild artifact folder from scratch
# ---------------------------------------------------------

if ARTIFACT_FOLDER.exists():
    shutil.rmtree(
        ARTIFACT_FOLDER
    )

ARTIFACT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPreparing registry artifact...")


# ---------------------------------------------------------
# Copy all 72 models
# ---------------------------------------------------------

for horizon in range(1, 73):

    source_file = (
        MODELS_FOLDER
        / f"xgboost_{horizon:02d}h.joblib"
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Missing model file: {source_file}"
        )

    destination_file = (
        ARTIFACT_FOLDER
        / source_file.name
    )

    shutil.copy2(
        source_file,
        destination_file
    )


print("72 XGBoost models copied.")


# ---------------------------------------------------------
# Copy feature column configuration
# ---------------------------------------------------------

FEATURE_COLUMNS_FILE = (
    MODELS_FOLDER
    / "feature_columns.json"
)

if not FEATURE_COLUMNS_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {FEATURE_COLUMNS_FILE}"
    )

shutil.copy2(
    FEATURE_COLUMNS_FILE,
    ARTIFACT_FOLDER
    / "feature_columns.json"
)


# ---------------------------------------------------------
# Copy metrics
# ---------------------------------------------------------

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Missing metrics file: {METRICS_FILE}"
    )

shutil.copy2(
    METRICS_FILE,
    ARTIFACT_FOLDER
    / "metrics.csv"
)


# ---------------------------------------------------------
# Load metrics
# ---------------------------------------------------------

metrics_df = pd.read_csv(
    METRICS_FILE
)


metrics_1h = metrics_df[
    metrics_df[
        "forecast_horizon_hours"
    ] == 1
].iloc[0]


metrics_24h = metrics_df[
    metrics_df[
        "forecast_horizon_hours"
    ] == 24
].iloc[0]


metrics_72h = metrics_df[
    metrics_df[
        "forecast_horizon_hours"
    ] == 72
].iloc[0]


# ---------------------------------------------------------
# Create metadata file
# ---------------------------------------------------------

metadata = {

    "model_name":
        "karachi_aqi_xgboost_72h",

    "model_type":
        "XGBoost",

    "forecast_strategy":
        "Direct multi-horizon forecasting",

    "city":
        "Karachi",

    "forecast_horizons":
        "1 to 72 hours",

    "number_of_models":
        72,

    "training_source":
        "Hopsworks Feature Store",

    "feature_view":
        "karachi_aqi_training_view",

    "feature_view_version":
        1,

    "training_dataset_version":
        1,

    "test_start":
        "2025-09-17 20:00:00",
}


with open(
    ARTIFACT_FOLDER
    / "metadata.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Verify artifact folder
# ---------------------------------------------------------

artifact_files = list(
    ARTIFACT_FOLDER.iterdir()
)

print(
    "\nFiles inside artifact folder:",
    len(artifact_files)
)


expected_count = 75

# 72 joblib files
# + feature_columns.json
# + metrics.csv
# + metadata.json

if len(artifact_files) != expected_count:

    raise RuntimeError(
        f"Expected {expected_count} files "
        f"but found {len(artifact_files)}"
    )


print(
    "Artifact folder verification successful."
)


# ---------------------------------------------------------
# Remove old ZIP if one already exists
# ---------------------------------------------------------

ZIP_FILE = Path(
    str(ZIP_BASE) + ".zip"
)

if ZIP_FILE.exists():

    ZIP_FILE.unlink()


# ---------------------------------------------------------
# Create ZIP
# ---------------------------------------------------------

print(
    "\nCreating ZIP archive..."
)


zip_path = shutil.make_archive(
    base_name=str(ZIP_BASE),
    format="zip",
    root_dir=ARTIFACT_FOLDER,
)


ZIP_FILE = Path(
    zip_path
)


print(
    "ZIP created:"
)

print(
    ZIP_FILE
)


print(
    "\nZIP size:"
)

print(
    f"{ZIP_FILE.stat().st_size / (1024 * 1024):.2f} MB"
)


# ---------------------------------------------------------
# Connect to Hopsworks
# ---------------------------------------------------------

print(
    "\nConnecting to Hopsworks..."
)


project = hopsworks.login(
    cert_folder=str(CERT_FOLDER)
)


mr = project.get_model_registry()


print(
    "\nConnected to Model Registry."
)


# ---------------------------------------------------------
# Optional Feature View provenance
# ---------------------------------------------------------

fs = project.get_feature_store()


feature_view = fs.get_feature_view(
    name="karachi_aqi_training_view",
    version=1,
)


# ---------------------------------------------------------
# Create registry model metadata
# ---------------------------------------------------------

model = mr.python.create_model(

    name="karachi_aqi_xgboost_72h",

    description=(
        "Karachi AQI direct multi-horizon forecasting "
        "system containing 72 XGBoost models for hourly "
        "AQI predictions from 1 to 72 hours ahead."
    ),

    metrics={

        "mae_1h": float(
            metrics_1h["mae"]
        ),

        "rmse_1h": float(
            metrics_1h["rmse"]
        ),

        "r2_1h": float(
            metrics_1h["r2"]
        ),

        "mae_24h": float(
            metrics_24h["mae"]
        ),

        "rmse_24h": float(
            metrics_24h["rmse"]
        ),

        "r2_24h": float(
            metrics_24h["r2"]
        ),

        "mae_72h": float(
            metrics_72h["mae"]
        ),

        "rmse_72h": float(
            metrics_72h["rmse"]
        ),

        "r2_72h": float(
            metrics_72h["r2"]
        ),
    },

    feature_view=feature_view,

    training_dataset_version=1,
)


# ---------------------------------------------------------
# Upload ZIP as the model artifact
# ---------------------------------------------------------

print(
    "\nUploading ZIP artifact to "
    "Hopsworks Model Registry..."
)


registered_model = model.save(
    str(ZIP_FILE)
)


# ---------------------------------------------------------
# Success
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "MODEL REGISTERED SUCCESSFULLY"
)

print(
    "=" * 60
)


print(
    "\nModel name:"
)

print(
    registered_model.name
)


print(
    "\nModel version:"
)

print(
    registered_model.version
)


print(
    "\nRegistry URL:"
)

print(
    registered_model.get_url()
)


print(
    "\nRegistered artifact:"
)

print(
    ZIP_FILE.name
)


print(
    "\nThe ZIP contains:"
)

print(
    "72 XGBoost models"
)

print(
    "feature_columns.json"
)

print(
    "metrics.csv"
)

print(
    "metadata.json"
)