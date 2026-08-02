from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# File paths
DATA_FILE = Path("data/processed/karachi_aqi_features.csv")

METRICS_FOLDER = Path("outputs/metrics")
GRAPHS_FOLDER = Path("outputs/graphs")

METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)


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