#importing libraries
from pathlib import Path
import pandas as pd

#paths to raw data and processed data folders
RAW_DATA_FOLDER = Path("data/raw")
PROCESSED_DATA_FOLDER = Path("data/processed")

#specifying file locations
AIR_QUALITY_FILE = RAW_DATA_FOLDER / "karachi_air_quality_raw.csv"
WEATHER_FILE = RAW_DATA_FOLDER / "karachi_weather_raw.csv"

OUTPUT_FILE = PROCESSED_DATA_FOLDER / "karachi_merged_clean.csv"

#reading raw data csvs collected by collect_data.py
air_quality_df = pd.read_csv(AIR_QUALITY_FILE)
weather_df = pd.read_csv(WEATHER_FILE)

#merging the csvs on time
merged_df = pd.merge(air_quality_df,weather_df,on="time", how="inner")

#routine check
print("\nMerged df:")
print(merged_df.shape)
print(merged_df.head)

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

#routine check
print("Merged df first 5 rows:")
print(merged_df.head)

#merged_df.to_csv(OUTPUT_FILE,index=False,)
#print("\nClean merged dataset saved to:")
#print(OUTPUT_FILE)

#Time based features
merged_df["hour"] = merged_df["time"].dt.hour
merged_df["day_of_week"] = merged_df["time"].dt.dayofweek
merged_df["month"] = merged_df["time"].dt.month
merged_df["is_weekend"] = (merged_df["day_of_week"] >= 5).astype(int)

#Lag Features
merged_df["aqi_lag_1h"] = merged_df["us_aqi"].shift(1)
merged_df["aqi_lag_3h"] = merged_df["us_aqi"].shift(3)
merged_df["aqi_lag_6h"] = merged_df["us_aqi"].shift(6)
merged_df["aqi_lag_12h"] = merged_df["us_aqi"].shift(12)
merged_df["aqi_lag_24h"] = merged_df["us_aqi"].shift(24)

merged_df["pm2_5_lag_1h"] = merged_df["pm2_5"].shift(1)
merged_df["pm2_5_lag_24h"] = merged_df["pm2_5"].shift(24)

merged_df["pm10_lag_1h"] = merged_df["pm10"].shift(1)
merged_df["pm10_lag_24h"] = merged_df["pm10"].shift(24)

#Rolling averages
merged_df["aqi_mean_3h"] = (merged_df["us_aqi"].rolling(window=3).mean())
merged_df["aqi_mean_6h"] = (merged_df["us_aqi"].rolling(window=6).mean())
merged_df["aqi_mean_12h"] = (merged_df["us_aqi"].rolling(window=12).mean())
merged_df["aqi_mean_24h"] = (merged_df["us_aqi"].rolling(window=24).mean())

merged_df["pm2_5_mean_24h"] = (merged_df["pm2_5"].rolling(window=24).mean())

merged_df["pm10_mean_24h"] = (merged_df["pm10"].rolling(window=24).mean())

#Change Features
merged_df["aqi_change_1h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_1h"])

merged_df["aqi_change_3h"] = (merged_df["us_aqi"] - merged_df["aqi_lag_3h"])

#Forecast targets
merged_df["aqi_target_1h"] = merged_df["us_aqi"].shift(-1)
merged_df["aqi_target_6h"] = merged_df["us_aqi"].shift(-6)
merged_df["aqi_target_24h"] = merged_df["us_aqi"].shift(-24)
merged_df["aqi_target_48h"] = merged_df["us_aqi"].shift(-48)
merged_df["aqi_target_72h"] = merged_df["us_aqi"].shift(-72)


