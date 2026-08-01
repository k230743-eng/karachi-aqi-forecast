# Importing Libraries

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

#Karachi Latitude and Longitude
#Timezome and Start and End Dates

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

START_DATE = "2022-08-01"
END_DATE = "2026-06-30"

#URLs to fetch data from
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

#raw data folder path to save fetched data to
RAW_DATA_FOLDER = Path("data/raw")

#specifying parameters for api
air_quality_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": (
        "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        "sulphur_dioxide,ozone,dust,aerosol_optical_depth,us_aqi"
    ),
    "timezone": TIMEZONE,
}

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": (
        "temperature_2m,relative_humidity_2m,dew_point_2m,"
        "surface_pressure,precipitation,cloud_cover,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
    ),
    "timezone": TIMEZONE,
}


print("Fetching historical air-quality data...")

#fetching data with initialized parameters
air_quality_response = requests.get(
    AIR_QUALITY_URL,
    params=air_quality_params,
    timeout=120,
)

#to see if any error occured
air_quality_response.raise_for_status()

#converting to dictionary and generating pandas data frame and formatting date and time column
air_quality_data = air_quality_response.json()
air_quality_df = pd.DataFrame(air_quality_data["hourly"])
air_quality_df["time"] = pd.to_datetime(air_quality_df["time"])


print("Fetching historical weather data...")

weather_response = requests.get(
    WEATHER_URL,
    params=weather_params,
    timeout=120,
)

weather_response.raise_for_status()

weather_data = weather_response.json()
weather_df = pd.DataFrame(weather_data["hourly"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

#routine checks
print("\nAir-quality data:")
print("Shape:", air_quality_df.shape)
print(air_quality_df.head())

print("\nWeather data:")
print("Shape:", weather_df.shape)
print(weather_df.head())

#saving weather and air quality data separately to csvs
air_quality_df.to_csv(
    RAW_DATA_FOLDER / "karachi_air_quality_raw.csv",
    index=False,
)

weather_df.to_csv(
    RAW_DATA_FOLDER / "karachi_weather_raw.csv",
    index=False,
)


print("\nFiles saved successfully.")
print("Air-quality missing values:")
print(air_quality_df.isnull().sum())

print("\nWeather missing values:")
print(weather_df.isnull().sum())