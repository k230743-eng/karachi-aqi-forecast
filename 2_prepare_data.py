#importing libraries
from pathlib import Path
import pandas as pd
import numpy as np

#paths to raw data and processed data folders
RAW_DATA_FOLDER = Path("data/raw")
PROCESSED_DATA_FOLDER = Path("data/processed")

#specifying file locations
AIR_QUALITY_FILE = RAW_DATA_FOLDER / "karachi_air_quality_raw.csv"
WEATHER_FILE = RAW_DATA_FOLDER / "karachi_weather_raw.csv"

#OUTPUT_FILE = PROCESSED_DATA_FOLDER / "karachi_merged_clean.csv"

#reading raw data csvs collected by collect_data.py
air_quality_df = pd.read_csv(AIR_QUALITY_FILE)
weather_df = pd.read_csv(WEATHER_FILE)

#merging the csvs on time
merged_df = pd.merge(air_quality_df,weather_df,on="time", how="inner")
merged_df["time"] = pd.to_datetime(merged_df["time"])

merged_df = (merged_df.sort_values("time").reset_index(drop=True))

#routine check
print("\nMerged df:")
print(merged_df.shape)
print(merged_df.head())

#essential columns that must be present for each row
essential_air_quality_columns = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

#dropping missing data rows
merged_df = merged_df.dropna(subset=essential_air_quality_columns).reset_index(drop=True)
print("Merged df:")
print(merged_df.shape)

#checking date range
print("\nDate range:")
print("Start:", merged_df["time"].min())
print("End:  ", merged_df["time"].max())

#checking missing values
print("\nMissing values:")
print(merged_df.isnull().sum())

#checking duplicated rows
print("\nDuplicate timestamps:")
print(merged_df["time"].duplicated().sum())

#checking if any row contains a difference of more than 1 hr to make sure below implementation of shifting is correct
time_differences = merged_df["time"].diff()

missing_hour_gaps = time_differences[
    time_differences > pd.Timedelta(hours=1)
]

print("\nGaps larger than one hour:")
print(missing_hour_gaps)

#routine check
print("Merged df first 5 rows:")
print(merged_df.head())

#merged_df.to_csv(OUTPUT_FILE,index=False,)
#print("\nClean merged dataset saved to:")
#print(OUTPUT_FILE)

#Time based features
merged_df["hour"] = merged_df["time"].dt.hour
merged_df["day_of_week"] = merged_df["time"].dt.dayofweek
merged_df["month"] = merged_df["time"].dt.month
merged_df["is_weekend"] = (merged_df["day_of_week"] >= 5).astype(int)

#Cyclical Time Features
#merged_df["hour_sin"] = np.sin(2*np.pi*merged_df["hour"] / 24)
#merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour"] / 24)
#merged_df["day_of_week_sin"] = np.sin(2 * np.pi * merged_df["day_of_week"] / 7)
#merged_df["day_of_week_cos"] = np.cos(2 * np.pi * merged_df["day_of_week"] / 7)
#merged_df["month_sin"] = np.sin(2 * np.pi * (merged_df["month"] - 1) / 12)
#merged_df["month_cos"] = np.cos(2 * np.pi * (merged_df["month"] - 1) / 12)

#Encoding Wind direction in a more meaningful way
#wind_radians = np.radians(merged_df["wind_direction_10m"])
#merged_df["wind_direction_sin"] = np.sin(wind_radians)
#merged_df["wind_direction_cos"] = np.cos(wind_radians)

#Lag Features
merged_df["aqi_lag_1h"] = merged_df["us_aqi"].shift(1)
merged_df["aqi_lag_3h"] = merged_df["us_aqi"].shift(3)
merged_df["aqi_lag_6h"] = merged_df["us_aqi"].shift(6)
merged_df["aqi_lag_12h"] = merged_df["us_aqi"].shift(12)
merged_df["aqi_lag_24h"] = merged_df["us_aqi"].shift(24)
merged_df["aqi_lag_48h"] = merged_df["us_aqi"].shift(48)
merged_df["aqi_lag_72h"] = merged_df["us_aqi"].shift(72)
merged_df["aqi_lag_168h"] = merged_df["us_aqi"].shift(168)

merged_df["pm2_5_lag_1h"] = merged_df["pm2_5"].shift(1)
merged_df["pm2_5_lag_3h"] = merged_df["pm2_5"].shift(3)
merged_df["pm2_5_lag_6h"] = merged_df["pm2_5"].shift(6)
merged_df["pm2_5_lag_12h"] = merged_df["pm2_5"].shift(12)
merged_df["pm2_5_lag_24h"] = merged_df["pm2_5"].shift(24)
merged_df["pm2_5_lag_48h"] = merged_df["pm2_5"].shift(48)
merged_df["pm2_5_lag_72h"] = merged_df["pm2_5"].shift(72)

merged_df["pm10_lag_1h"] = merged_df["pm10"].shift(1)
merged_df["pm10_lag_3h"] = merged_df["pm10"].shift(3)
merged_df["pm10_lag_6h"] = merged_df["pm10"].shift(6)
merged_df["pm10_lag_12h"] = merged_df["pm10"].shift(12)
merged_df["pm10_lag_24h"] = merged_df["pm10"].shift(24)
merged_df["pm10_lag_48h"] = merged_df["pm10"].shift(48)
merged_df["pm10_lag_72h"] = merged_df["pm10"].shift(72)


#Rolling averages
merged_df["aqi_mean_3h"] = (merged_df["us_aqi"].rolling(window=3).mean())
merged_df["aqi_mean_6h"] = (merged_df["us_aqi"].rolling(window=6).mean())
merged_df["aqi_mean_12h"] = (merged_df["us_aqi"].rolling(window=12).mean())
merged_df["aqi_mean_24h"] = (merged_df["us_aqi"].rolling(window=24).mean())
merged_df["aqi_mean_48h"] = (merged_df["us_aqi"].rolling(window=48).mean())
merged_df["aqi_mean_72h"] = (merged_df["us_aqi"].rolling(window=72).mean())

merged_df["pm2_5_mean_3h"] = (merged_df["pm2_5"].rolling(window=3).mean())
merged_df["pm2_5_mean_6h"] = (merged_df["pm2_5"].rolling(window=6).mean())
merged_df["pm2_5_mean_12h"] = (merged_df["pm2_5"].rolling(window=12).mean())
merged_df["pm2_5_mean_24h"] = (merged_df["pm2_5"].rolling(window=24).mean())
merged_df["pm2_5_mean_48h"] = (merged_df["pm2_5"].rolling(window=48).mean())
merged_df["pm2_5_mean_72h"] = (merged_df["pm2_5"].rolling(window=72).mean())

merged_df["pm10_mean_3h"] = (merged_df["pm10"].rolling(window=3).mean())
merged_df["pm10_mean_6h"] = (merged_df["pm10"].rolling(window=6).mean())
merged_df["pm10_mean_12h"] = (merged_df["pm10"].rolling(window=12).mean())
merged_df["pm10_mean_24h"] = (merged_df["pm10"].rolling(window=24).mean())
merged_df["pm10_mean_48h"] = (merged_df["pm10"].rolling(window=48).mean())
merged_df["pm10_mean_72h"] = (merged_df["pm10"].rolling(window=72).mean())

#Change Features
merged_df["aqi_change_1h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_1h"])
merged_df["aqi_change_3h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_3h"])
merged_df["aqi_change_6h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_6h"])
merged_df["aqi_change_12h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_12h"])
merged_df["aqi_change_24h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_24h"])
merged_df["aqi_change_48h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_48h"])
merged_df["aqi_change_72h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_72h"])

#AQI volatility features
merged_df["aqi_std_24h"] = (merged_df["us_aqi"].rolling(window=24).std())
merged_df["aqi_std_72h"] = (merged_df["us_aqi"].rolling(window=72).std())

# Weather conditions at the 72-hour target time
merged_df["future_temperature_72h"] = (merged_df["temperature_2m"].shift(-72))
merged_df["future_humidity_72h"] = (merged_df["relative_humidity_2m"].shift(-72))
merged_df["future_dew_point_72h"] = (merged_df["dew_point_2m"].shift(-72))
merged_df["future_pressure_72h"] = (merged_df["surface_pressure"].shift(-72))
merged_df["future_precipitation_72h"] = (merged_df["precipitation"].shift(-72))
merged_df["future_cloud_cover_72h"] = (merged_df["cloud_cover"].shift(-72))
merged_df["future_wind_speed_72h"] = (merged_df["wind_speed_10m"].shift(-72))
merged_df["future_wind_gusts_72h"] = (merged_df["wind_gusts_10m"].shift(-72))
future_wind_direction = (merged_df["wind_direction_10m"].shift(-72))
future_wind_radians = np.radians(future_wind_direction)
merged_df["future_wind_direction_sin_72h"] = (np.sin(future_wind_radians))
merged_df["future_wind_direction_cos_72h"] = (np.cos(future_wind_radians))

# Time information for the 72-hour target
target_time_72h = (merged_df["time"]+ pd.Timedelta(hours=72))
target_hour_72h = target_time_72h.dt.hour
target_day_72h = target_time_72h.dt.dayofweek
target_month_72h = target_time_72h.dt.month

merged_df["target_hour_sin_72h"] = np.sin(2 * np.pi * target_hour_72h / 24)
merged_df["target_hour_cos_72h"] = np.cos(2 * np.pi * target_hour_72h / 24)
merged_df["target_day_sin_72h"] = np.sin(2 * np.pi * target_day_72h / 7)
merged_df["target_day_cos_72h"] = np.cos(2 * np.pi * target_day_72h / 7)
merged_df["target_month_sin_72h"] = np.sin(2 * np.pi * (target_month_72h - 1) / 12)
merged_df["target_month_cos_72h"] = np.cos(2 * np.pi * (target_month_72h - 1) / 12)
merged_df["target_is_weekend_72h"] = (target_day_72h >= 5).astype(int)

#Forecast targets
merged_df["aqi_target_1h"] = merged_df["us_aqi"].shift(-1)
merged_df["aqi_target_6h"] = merged_df["us_aqi"].shift(-6)
merged_df["aqi_target_24h"] = merged_df["us_aqi"].shift(-24)
merged_df["aqi_target_48h"] = merged_df["us_aqi"].shift(-48)
merged_df["aqi_target_72h"] = merged_df["us_aqi"].shift(-72)

#dropping rows with missing values created due to lag features created
merged_df = merged_df.dropna().reset_index(drop=True)

#specifying path for new csv
OUTPUT_FILE = (PROCESSED_DATA_FOLDER/ "karachi_aqi_features.csv")

merged_df.to_csv(OUTPUT_FILE,index=False,)

print("\nFinal feature dataset shape:")
print(merged_df.shape)

print("\nFinal date range:")
print("Start:", merged_df["time"].min())
print("End:  ", merged_df["time"].max())

print("\nRemaining missing values:")
print(merged_df.isnull().sum().sum())

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)

