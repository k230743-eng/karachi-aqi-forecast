from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


# Reproducibility
tf.keras.utils.set_random_seed(42)

# File paths
DATA_FILE = Path("data/processed/karachi_aqi_features.csv")
MODELS_FOLDER = Path("models")
METRICS_FOLDER = Path("outputs/metrics")
GRAPHS_FOLDER = Path("outputs/graphs")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_FILE,parse_dates=["time"],)
df = df.sort_values("time").reset_index(drop=True)

# Loading same feature list used previously for 3_train_models.py
feature_columns = joblib.load(MODELS_FOLDER / "feature_columns.joblib")
target_column = "aqi_target_72h"

# Chronological train-test split
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train_full = train_df[feature_columns]
y_train_full = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


# Chronological training-validation split
validation_split_index = int(len(X_train_full) * 0.8)
X_train = X_train_full.iloc[:validation_split_index]
y_train = y_train_full.iloc[:validation_split_index]
X_validation = X_train_full.iloc[validation_split_index:]
y_validation = y_train_full.iloc[validation_split_index:]

print("\nTraining set:")
print(X_train.shape)

print("\nValidation set:")
print(X_validation.shape)

print("\nTest set:")
print(X_test.shape)


# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler,MODELS_FOLDER / "tensorflow_scaler_72_hours.joblib",)


#Neural Network
model = Sequential(
    [
        tf.keras.Input(
            shape=(len(feature_columns),)
        ),

        Dense(
            64,
            activation="relu",
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu",
        ),

        Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,)


# Train model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(
        X_validation_scaled,
        y_validation,
    ),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1,
)


# Evaluate on test data
predictions = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test,predictions,)

rmse = mean_squared_error(y_test,predictions,) ** 0.5

r2 = r2_score(y_test,predictions,)

print("\nTensorFlow — 72 hours")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")



# Save model
MODEL_FILE = (MODELS_FOLDER/ "tensorflow_72_hours.keras")
model.save(MODEL_FILE)
print("\nTensorFlow model saved to:")
print(MODEL_FILE)


# Save metrics
tensorflow_metrics_df = pd.DataFrame(
    [
        {
            "model": "TensorFlow Dense Neural Network",
            "forecast_horizon": "72_hours",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    ]
)

METRICS_FILE = (METRICS_FOLDER/ "tensorflow_72h_metrics.csv")

tensorflow_metrics_df.to_csv(METRICS_FILE,index=False,)
print("\nMetrics saved to:")
print(METRICS_FILE)



# Training history plot
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"],label="Training loss",)

plt.plot(history.history["val_loss"],label="Validation loss",)

plt.title("TensorFlow Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

GRAPH_FILE = (GRAPHS_FOLDER/ "tensorflow_training_history_72h.png")

plt.savefig(GRAPH_FILE,dpi=300,)

plt.show()
plt.close()

print("\nTraining history graph saved to:")
print(GRAPH_FILE)