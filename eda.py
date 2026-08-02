from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# File paths
PROCESSED_DATA_FOLDER = Path("data/processed")
GRAPHS_FOLDER = Path("outputs/graphs")

FEATURES_FILE = (PROCESSED_DATA_FOLDER / "karachi_aqi_features.csv")

GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True,)


# Load processed feature dataset
df = pd.read_csv(FEATURES_FILE, parse_dates=["time"],)
df = df.sort_values("time").reset_index(drop=True)


# Basic dataset information
print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print("Start:", df["time"].min())
print("End:  ", df["time"].max())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate timestamps:")
print(df["time"].duplicated().sum())


# AQI descriptive statistics
print("\nAQI descriptive statistics:")
print(df["us_aqi"].describe())

print("\nHighest AQI observation:")
highest_aqi_row = df.loc[df["us_aqi"].idxmax()]

print(
    highest_aqi_row[
        [
            "time",
            "us_aqi",
            "pm2_5",
            "pm10",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]
    ]
)

print("\nLowest AQI observation:")
lowest_aqi_row = df.loc[df["us_aqi"].idxmin()]

print(
    lowest_aqi_row[
        [
            "time",
            "us_aqi",
            "pm2_5",
            "pm10",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]
    ]
)



# AQI over complete time range
plt.figure(figsize=(15, 6))

plt.plot(
    df["time"],
    df["us_aqi"],
    linewidth=0.7,
)

plt.title("Karachi AQI Over Time")
plt.xlabel("Time")
plt.ylabel("US AQI")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "aqi_over_time.png",
    dpi=300,
)

plt.show()
plt.close()


# AQI distribution
plt.figure(figsize=(10, 6))

plt.hist(
    df["us_aqi"],
    bins=40,
    edgecolor="black",
)

plt.axvline(
    df["us_aqi"].mean(),
    linestyle="--",
    label=f'Mean: {df["us_aqi"].mean():.2f}',
)

plt.axvline(
    df["us_aqi"].median(),
    linestyle=":",
    label=f'Median: {df["us_aqi"].median():.2f}',
)

plt.title("Distribution of Karachi AQI")
plt.xlabel("US AQI")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "aqi_distribution.png",
    dpi=300,
)

plt.show()
plt.close()


# Average AQI by hour
hourly_aqi = (
    df.groupby("hour")["us_aqi"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_aqi.index,
    hourly_aqi.values,
    marker="o",
)

plt.title("Average AQI by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average US AQI")
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "average_aqi_by_hour.png",
    dpi=300,
)

plt.show()
plt.close()


# Average AQI by day of week
day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

weekday_aqi = (
    df.groupby("day_of_week")["us_aqi"]
    .mean()
    .rename(index=day_names)
)

plt.figure(figsize=(10, 6))

plt.bar(
    weekday_aqi.index,
    weekday_aqi.values,
)

plt.title("Average AQI by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average US AQI")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "average_aqi_by_weekday.png",
    dpi=300,
)

plt.show()
plt.close()


# Average AQI by month
month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

monthly_aqi = (
    df.groupby("month")["us_aqi"]
    .mean()
    .rename(index=month_names)
)

plt.figure(figsize=(10, 6))

plt.bar(
    monthly_aqi.index,
    monthly_aqi.values,
)

plt.title("Average AQI by Month")
plt.xlabel("Month")
plt.ylabel("Average US AQI")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "average_aqi_by_month.png",
    dpi=300,
)

plt.show()
plt.close()


# Correlation analysis
correlation_columns = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

print("\nCorrelation of variables with AQI:")

aqi_correlations = (
    correlation_matrix["us_aqi"]
    .sort_values(ascending=False)
)

print(aqi_correlations)


plt.figure(figsize=(14, 11))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    annot=False,
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "correlation_heatmap.png",
    dpi=300,
)

plt.show()
plt.close()


# Main pollutants against AQI
plt.figure(figsize=(10, 6))

plt.scatter(
    df["pm2_5"],
    df["us_aqi"],
    alpha=0.2,
    s=10,
)

plt.title("PM2.5 Against AQI")
plt.xlabel("PM2.5")
plt.ylabel("US AQI")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "pm2_5_vs_aqi.png",
    dpi=300,
)

plt.show()
plt.close()


plt.figure(figsize=(10, 6))

plt.scatter(
    df["pm10"],
    df["us_aqi"],
    alpha=0.2,
    s=10,
)

plt.title("PM10 Against AQI")
plt.xlabel("PM10")
plt.ylabel("US AQI")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "pm10_vs_aqi.png",
    dpi=300,
)

plt.show()
plt.close()


# Weather conditions against AQI
plt.figure(figsize=(10, 6))

plt.scatter(
    df["wind_speed_10m"],
    df["us_aqi"],
    alpha=0.2,
    s=10,
)

plt.title("Wind Speed Against AQI")
plt.xlabel("Wind Speed at 10 m")
plt.ylabel("US AQI")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "wind_speed_vs_aqi.png",
    dpi=300,
)

plt.show()
plt.close()


plt.figure(figsize=(10, 6))

plt.scatter(
    df["relative_humidity_2m"],
    df["us_aqi"],
    alpha=0.2,
    s=10,
)

plt.title("Relative Humidity Against AQI")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("US AQI")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPHS_FOLDER / "humidity_vs_aqi.png",
    dpi=300,
)

plt.show()
plt.close()


print("\nEDA completed successfully.")
print("Graphs saved to:", GRAPHS_FOLDER)