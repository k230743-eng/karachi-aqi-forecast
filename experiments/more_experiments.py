from pathlib import Path

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

DATA_FILE = Path("data/processed/karachi_aqi_features.csv")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["time"]
)

df = df.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------
# Original features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Longer-history features
# ---------------------------------------------------------

long_history_features = [
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",

    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_12h",
    "pm2_5_lag_48h",
    "pm2_5_lag_72h",

    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_48h",
    "pm10_lag_72h",

    "aqi_mean_48h",
    "aqi_mean_72h",

    "pm2_5_mean_3h",
    "pm2_5_mean_6h",
    "pm2_5_mean_12h",
    "pm2_5_mean_48h",
    "pm2_5_mean_72h",

    "pm10_mean_3h",
    "pm10_mean_6h",
    "pm10_mean_12h",
    "pm10_mean_48h",
    "pm10_mean_72h",

    "aqi_change_6h",
    "aqi_change_12h",
    "aqi_change_24h",
    "aqi_change_48h",
    "aqi_change_72h",

    "aqi_std_24h",
    "aqi_std_72h",
]


# ---------------------------------------------------------
# Future-weather and target-time features
# ---------------------------------------------------------

future_weather_features = [
    "future_temperature_72h",
    "future_humidity_72h",
    "future_dew_point_72h",
    "future_pressure_72h",
    "future_precipitation_72h",
    "future_cloud_cover_72h",
    "future_wind_speed_72h",
    "future_wind_gusts_72h",
    "future_wind_direction_sin_72h",
    "future_wind_direction_cos_72h",

    "target_hour_sin_72h",
    "target_hour_cos_72h",
    "target_day_sin_72h",
    "target_day_cos_72h",
    "target_month_sin_72h",
    "target_month_cos_72h",
    "target_is_weekend_72h",
]

reduced_feature_columns = [
    # Current pollution
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",

    # Current weather
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    # Basic time
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # Selected AQI history
    "aqi_lag_6h",
    "aqi_lag_24h",
    "aqi_lag_72h",
    "aqi_lag_168h",
    "aqi_mean_24h",
    "aqi_mean_72h",
    "aqi_std_24h",
    "aqi_std_72h",
    "aqi_change_24h",
    "aqi_change_72h",

    # Selected PM2.5 history
    "pm2_5_lag_24h",
    "pm2_5_lag_72h",
    "pm2_5_mean_24h",
    "pm2_5_mean_72h",

    # Selected PM10 history
    "pm10_lag_24h",
    "pm10_lag_72h",
    "pm10_mean_24h",
    "pm10_mean_72h",

    # Future weather
    "future_temperature_72h",
    "future_humidity_72h",
    "future_dew_point_72h",
    "future_pressure_72h",
    "future_precipitation_72h",
    "future_cloud_cover_72h",
    "future_wind_speed_72h",
    "future_wind_gusts_72h",
    "future_wind_direction_sin_72h",
    "future_wind_direction_cos_72h",

    # Target-time information
    "target_hour_sin_72h",
    "target_hour_cos_72h",
    "target_day_sin_72h",
    "target_day_cos_72h",
    "target_month_sin_72h",
    "target_month_cos_72h",
    "target_is_weekend_72h",
]


# Use all feature groups
feature_columns = reduced_feature_columns

target_column = "aqi_target_72h"


# ---------------------------------------------------------
# Chronological 80/20 train-test split
# ---------------------------------------------------------

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


# ---------------------------------------------------------
# Display split information
# ---------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nDataset range:")
print(
    df["time"].min(),
    "to",
    df["time"].max()
)

print("\nTraining period:")
print(
    train_df["time"].min(),
    "to",
    train_df["time"].max()
)

print("\nTest period:")
print(
    test_df["time"].min(),
    "to",
    test_df["time"].max()
)

print("\nTraining rows:")
print(len(train_df))

print("\nTest rows:")
print(len(test_df))

print("\nNumber of features:")
print(len(feature_columns))


# ---------------------------------------------------------
# Build Ridge Regression pipeline
# ---------------------------------------------------------

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1000)
        ),
    ]
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------

test_predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nFinal 72-hour Ridge result:")
print(f"MAE:  {test_mae:.3f}")
print(f"RMSE: {test_rmse:.3f}")
print(f"R²:   {test_r2:.3f}")

# ---------------------------------------------------------
# HistGradientBoostingRegressor
# ---------------------------------------------------------

hist_gradient_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=20,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

hist_gradient_model.fit(
    X_train,
    y_train
)

hist_predictions = hist_gradient_model.predict(
    X_test
)

hist_mae = mean_absolute_error(
    y_test,
    hist_predictions
)

hist_rmse = mean_squared_error(
    y_test,
    hist_predictions
) ** 0.5

hist_r2 = r2_score(
    y_test,
    hist_predictions
)

print("\nFinal 72-hour HistGradientBoosting result:")
print(f"MAE:  {hist_mae:.3f}")
print(f"RMSE: {hist_rmse:.3f}")
print(f"R²:   {hist_r2:.3f}")

# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------

xgboost_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=800,
    learning_rate=0.02,
    max_depth=2,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=20.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgboost_model.fit(
    X_train,
    y_train
)

xgboost_predictions = xgboost_model.predict(
    X_test
)

xgboost_mae = mean_absolute_error(
    y_test,
    xgboost_predictions
)

xgboost_rmse = mean_squared_error(
    y_test,
    xgboost_predictions
) ** 0.5

xgboost_r2 = r2_score(
    y_test,
    xgboost_predictions
)

print("\nFinal 72-hour XGBoost result:")
print(f"MAE:  {xgboost_mae:.3f}")
print(f"RMSE: {xgboost_rmse:.3f}")
print(f"R²:   {xgboost_r2:.3f}")