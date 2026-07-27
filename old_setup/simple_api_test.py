import requests
import pandas as pd
import matplotlib.pyplot as plt

latitude = 24.8607
longitude = 67.0011

url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
    "timezone": "Asia/Karachi",
    "forecast_days": 3,
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)

data = response.json()

df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

plt.plot(df["time"], df["us_aqi"])

plt.title("Karachi AQI Forecast")
plt.xlabel("Time")
plt.ylabel("US AQI")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()