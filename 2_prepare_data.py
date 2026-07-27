from pathlib import Path
import pandas as pd

RAW_DATA_FOLDER = Path("data/raw")
PROCESSED_DATA_FOLDER = Path("data/processed")

AIR_QUALITY_FILE = RAW_DATA_FOLDER / "karachi_air_quality_raw.csv"
WEATHER_FILE = RAW_DATA_FOLDER / "karachi_weather_raw.csv"

OUTPUT_FILE = PROCESSED_DATA_FOLDER / "karachi_merged_clean.csv"

air_quality_df = pd.read_csv(AIR_QUALITY_FILE)
weather_df = pd.read_csv(WEATHER_FILE)

merged_df = pd.merge(air_quality_df,weather_df,on="time", how="inner")

print("\nMerged df:")
print(merged_df.shape)
print(merged_df.head)

essential_air_quality_columns = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

merged_df = merged_df.dropna(subset=essential_air_quality_columns).reset_index(drop=True)
print("Merged df:")
print(merged_df.shape)

print("\nDate range:")
print("Start:", merged_df["time"].min())
print("End:  ", merged_df["time"].max())

print("\nMissing values:")
print(merged_df.isnull().sum())

print("\nDuplicate timestamps:")
print(merged_df["time"].duplicated().sum())

print("Merged df first 5 rows:")
print(merged_df.head)

merged_df.to_csv(OUTPUT_FILE,index=False,)

print("\nClean merged dataset saved to:")
print(OUTPUT_FILE)





