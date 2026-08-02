from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score,)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# File paths
DATA_FILE = Path("data/processed/karachi_aqi_features.csv")

METRICS_FOLDER = Path("outputs/metrics")
GRAPHS_FOLDER = Path("outputs/graphs")
MODELS_FOLDER = Path("models")

METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)


# Load and sort data
df = pd.read_csv(DATA_FILE,parse_dates=["time"],)
df = df.sort_values("time").reset_index(drop=True)

# Chronological train-test split
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nComplete dataset:")
print(df.shape)

print("\nTraining set:")
print(train_df.shape)
print(train_df["time"].min(), "to", train_df["time"].max(),)

print("\nTest set:")
print(test_df.shape)
print(test_df["time"].min(), "to", test_df["time"].max(),)


# Metric function
def calculate_metrics(actual_values, predicted_values,):

    mae = mean_absolute_error(actual_values,predicted_values,)
    rmse = mean_squared_error(actual_values,predicted_values,) ** 0.5
    r2 = r2_score(actual_values,predicted_values,)

    return mae, rmse, r2


# Persistence baselines
target_columns = {
    "1_hour": "aqi_target_1h",
    "6_hours": "aqi_target_6h",
    "24_hours": "aqi_target_24h",
    "48_hours": "aqi_target_48h",
    "72_hours": "aqi_target_72h",
}

baseline_results = []

# Persistence baseline
for horizon_name, target_column in target_columns.items():
    actual = test_df[target_column]
    predicted = test_df["us_aqi"]

    mae, rmse, r2 = calculate_metrics(actual, predicted,)

    baseline_results.append(
        {
            "model": "Persistence baseline",
            "forecast_horizon": horizon_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    print(f"\nPersistence baseline — {horizon_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")


# Training mean baseline
for horizon_name, target_column in target_columns.items():
    actual = test_df[target_column]

    training_target_mean = train_df[target_column].mean()

    predicted = pd.Series(training_target_mean, index=actual.index,)

    mae, rmse, r2 = calculate_metrics(actual, predicted,)

    baseline_results.append(
        {
            "model": "Training mean baseline",
            "forecast_horizon": horizon_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    print(f"\nTraining mean baseline — {horizon_name}")
    print(f"Training target mean: {training_target_mean:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

# Save baseline metrics
baseline_results_df = pd.DataFrame(baseline_results)

METRICS_FILE = (METRICS_FOLDER/ "baseline_metrics.csv")

baseline_results_df.to_csv(METRICS_FILE,index=False,)

print("\nBaseline metrics:")
print(baseline_results_df)

print("\nMetrics saved to:")
print(METRICS_FILE)


# Plot baseline MAE across forecast horizons
plt.figure(figsize=(10, 6))

for model_name, model_results in baseline_results_df.groupby("model"):

    plt.plot(
        model_results["forecast_horizon"],
        model_results["mae"],
        marker="o",
        label=model_name,
    )

plt.title("Baseline MAE by Forecast Horizon")
plt.xlabel("Forecast horizon")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

MAE_GRAPH_FILE = (
    GRAPHS_FOLDER
    / "baseline_mae_by_horizon.png"
)

plt.savefig(
    MAE_GRAPH_FILE,
    dpi=300,
)

plt.show()
plt.close()

print("\nBaseline MAE graph saved to:")
print(MAE_GRAPH_FILE)


# Plot baseline RMSE across forecast horizons
plt.figure(figsize=(10, 6))

for model_name, model_results in baseline_results_df.groupby("model"):

    plt.plot(
        model_results["forecast_horizon"],
        model_results["rmse"],
        marker="o",
        label=model_name,
    )

plt.title("Baseline RMSE by Forecast Horizon")
plt.xlabel("Forecast horizon")
plt.ylabel("Root Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

RMSE_GRAPH_FILE = (
    GRAPHS_FOLDER
    / "baseline_rmse_by_horizon.png"
)

plt.savefig(
    RMSE_GRAPH_FILE,
    dpi=300,
)

plt.show()
plt.close()

print("\nBaseline RMSE graph saved to:")
print(RMSE_GRAPH_FILE)

#Feature columns
feature_columns = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "pm2_5_lag_1h",
    "pm2_5_lag_24h",
    "pm10_lag_1h",
    "pm10_lag_24h",
    "aqi_mean_3h",
    "aqi_mean_6h",
    "aqi_mean_12h",
    "aqi_mean_24h",
    "pm2_5_mean_24h",
    "pm10_mean_24h",
    "aqi_change_1h",
    "aqi_change_3h",
]

#Ridge Regression Models
ridge_results = []

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

for horizon_name, target_column in target_columns.items():

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    ridge_model.fit(X_train, y_train)

    predictions = ridge_model.predict(X_test)

    mae, rmse, r2 = calculate_metrics(y_test, predictions,)

    ridge_results.append(
        {
            "model": "Ridge Regression",
            "forecast_horizon": horizon_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    print(f"\nRidge Regression — {horizon_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    joblib.dump(ridge_model,MODELS_FOLDER / f"ridge_{horizon_name}.joblib",)


# Random Forest models
random_forest_results = []

for horizon_name, target_column in target_columns.items():

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    random_forest_model.fit(X_train,y_train,)

    predictions = random_forest_model.predict(X_test)

    mae, rmse, r2 = calculate_metrics(y_test, predictions,)

    random_forest_results.append(
        {
            "model": "Random Forest",
            "forecast_horizon": horizon_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    print(f"\nRandom Forest — {horizon_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    joblib.dump(random_forest_model,MODELS_FOLDER / f"random_forest_{horizon_name}.joblib",)

joblib.dump(feature_columns,MODELS_FOLDER / "feature_columns.joblib",)

ridge_results_df = pd.DataFrame(ridge_results)

random_forest_results_df = pd.DataFrame(random_forest_results)

all_results_df = pd.concat(
    [
        baseline_results_df,
        ridge_results_df,
        random_forest_results_df,
    ],
    ignore_index=True,
)

ALL_METRICS_FILE = (METRICS_FOLDER / "model_comparison_metrics.csv")

all_results_df.to_csv(ALL_METRICS_FILE,index=False,)

print("\nAll model comparison metrics:")
print(all_results_df)

print("\nComparison metrics saved to:")
print(ALL_METRICS_FILE)

#Loading Random Forest 1 hour and 72 hour model to see which features are most important
rf_1h_model = joblib.load(MODELS_FOLDER / "random_forest_1_hour.joblib")
feature_columns = joblib.load(MODELS_FOLDER / "feature_columns.joblib")

feature_importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": rf_1h_model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\nTop Random Forest features for 1 hour:")
print(feature_importance_df.head(10))

rf_72h_model = joblib.load(MODELS_FOLDER / "random_forest_72_hours.joblib")
feature_importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": rf_72h_model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\nTop Random Forest features for 72 hour:")
print(feature_importance_df.head(10))