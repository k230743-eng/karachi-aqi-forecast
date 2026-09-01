from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

PREDICTION_FILE = Path(
    "outputs/predictions/latest_72h_forecast.csv"
)


# =========================================================
# Page setup
# =========================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)


st.title("Karachi AQI Predictor")

st.caption(
    "Live 72-hour air quality forecast for Karachi"
)


# =========================================================
# AQI helper functions
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


def get_health_message(aqi):

    if aqi <= 50:

        return (
            "Air quality is satisfactory."
        )

    elif aqi <= 100:

        return (
            "Air quality is acceptable for most people."
        )

    elif aqi <= 150:

        return (
            "Sensitive groups may experience health effects."
        )

    elif aqi <= 200:

        return (
            "Everyone may begin to experience health effects."
        )

    elif aqi <= 300:

        return (
            "Health alert: increased risk of health effects."
        )

    else:

        return (
            "Health warning: emergency conditions possible."
        )


# =========================================================
# Load forecast
# =========================================================

if not PREDICTION_FILE.exists():

    st.error(
        "No live forecast file was found."
    )

    st.info(
        "Run 14_predict_live.py first to generate "
        "the latest 72-hour AQI forecast."
    )

    st.stop()


forecast_df = pd.read_csv(
    PREDICTION_FILE,
    parse_dates=[
        "forecast_generated_at",
        "forecast_time",
    ],
)


forecast_df = (
    forecast_df
    .sort_values("forecast_time")
    .reset_index(drop=True)
)


# =========================================================
# Validation
# =========================================================

required_columns = [

    "forecast_generated_at",
    "forecast_time",
    "horizon_hours",
    "predicted_aqi",
    "aqi_category",
    "forecast_temperature",
    "forecast_humidity",
    "forecast_precipitation",
    "forecast_wind_speed",
]


missing_columns = [

    column

    for column in required_columns

    if column not in forecast_df.columns
]


if missing_columns:

    st.error(
        "The forecast file is missing required columns:"
    )

    st.write(
        missing_columns
    )

    st.stop()


if forecast_df.empty:

    st.error(
        "The forecast file contains no rows."
    )

    st.stop()


# =========================================================
# Summary values
# =========================================================

generated_at = (
    forecast_df[
        "forecast_generated_at"
    ].iloc[0]
)


first_forecast = (
    forecast_df.iloc[0]
)


maximum_row = (
    forecast_df.loc[
        forecast_df[
            "predicted_aqi"
        ].idxmax()
    ]
)


minimum_row = (
    forecast_df.loc[
        forecast_df[
            "predicted_aqi"
        ].idxmin()
    ]
)


average_aqi = (
    forecast_df[
        "predicted_aqi"
    ].mean()
)


next_24h = forecast_df[
    forecast_df[
        "horizon_hours"
    ] <= 24
]


next_48h = forecast_df[
    forecast_df[
        "horizon_hours"
    ] <= 48
]


# =========================================================
# Forecast timestamp information
# =========================================================

st.write(
    f"Forecast generated from latest feature data at "
    f"**{generated_at}**"
)


# =========================================================
# Top summary cards
# =========================================================

st.subheader(
    "Forecast Summary"
)


column1, column2, column3, column4 = (
    st.columns(4)
)


with column1:

    st.metric(
        "Next Hour AQI",
        f"{first_forecast['predicted_aqi']:.1f}",
    )

    st.caption(
        first_forecast[
            "aqi_category"
        ]
    )


with column2:

    st.metric(
        "72h Average AQI",
        f"{average_aqi:.1f}",
    )

    st.caption(
        get_aqi_category(
            average_aqi
        )
    )


with column3:

    st.metric(
        "Highest Forecast AQI",
        f"{maximum_row['predicted_aqi']:.1f}",
    )

    st.caption(
        str(
            maximum_row[
                "forecast_time"
            ]
        )
    )


with column4:

    st.metric(
        "Lowest Forecast AQI",
        f"{minimum_row['predicted_aqi']:.1f}",
    )

    st.caption(
        str(
            minimum_row[
                "forecast_time"
            ]
        )
    )


# =========================================================
# Warning section
# =========================================================

st.subheader(
    "Air Quality Alert"
)


maximum_aqi = (
    maximum_row[
        "predicted_aqi"
    ]
)


maximum_category = (
    get_aqi_category(
        maximum_aqi
    )
)


if maximum_aqi > 300:

    st.error(
        f"Hazardous AQI forecast detected. "
        f"Maximum predicted AQI: {maximum_aqi:.1f}"
    )


elif maximum_aqi > 200:

    st.error(
        f"Very unhealthy AQI conditions are forecast. "
        f"Maximum predicted AQI: {maximum_aqi:.1f}"
    )


elif maximum_aqi > 150:

    st.warning(
        f"Unhealthy AQI conditions are forecast. "
        f"Maximum predicted AQI: {maximum_aqi:.1f}"
    )


elif maximum_aqi > 100:

    st.warning(
        f"Sensitive groups may experience unhealthy "
        f"conditions. Maximum predicted AQI: "
        f"{maximum_aqi:.1f}"
    )


else:

    st.success(
        "No unhealthy AQI conditions are forecast "
        "during the next 72 hours."
    )


st.write(
    f"Highest forecast category: "
    f"**{maximum_category}**"
)


st.write(
    get_health_message(
        maximum_aqi
    )
)


# =========================================================
# AQI forecast chart
# =========================================================

st.subheader(
    "72-Hour AQI Forecast"
)


aqi_chart_df = (
    forecast_df[
        [
            "forecast_time",
            "predicted_aqi",
        ]
    ]
    .set_index(
        "forecast_time"
    )
)


st.line_chart(
    aqi_chart_df
)


# =========================================================
# Selected horizon summary
# =========================================================

st.subheader(
    "Key Forecast Horizons"
)


selected_horizons = [

    1,
    6,
    12,
    24,
    48,
    72,
]


selected_df = forecast_df[

    forecast_df[
        "horizon_hours"
    ].isin(
        selected_horizons
    )

][

    [
        "horizon_hours",
        "forecast_time",
        "predicted_aqi",
        "aqi_category",
    ]

].copy()


selected_df["predicted_aqi"] = (
    selected_df[
        "predicted_aqi"
    ].round(1)
)


st.dataframe(
    selected_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# Period summaries
# =========================================================

st.subheader(
    "Short-Term Summary"
)


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


with summary_col1:

    st.metric(
        "Next 24h Average",
        f"{next_24h['predicted_aqi'].mean():.1f}",
    )


with summary_col2:

    st.metric(
        "Next 48h Average",
        f"{next_48h['predicted_aqi'].mean():.1f}",
    )


with summary_col3:

    st.metric(
        "Next 72h Average",
        f"{forecast_df['predicted_aqi'].mean():.1f}",
    )


# =========================================================
# Weather forecast
# =========================================================

st.subheader(
    "Weather Forecast"
)


weather_tabs = st.tabs(
    [
        "Temperature",
        "Humidity",
        "Precipitation",
        "Wind Speed",
    ]
)


with weather_tabs[0]:

    temperature_df = (
        forecast_df[
            [
                "forecast_time",
                "forecast_temperature",
            ]
        ]
        .set_index(
            "forecast_time"
        )
    )

    st.line_chart(
        temperature_df
    )


with weather_tabs[1]:

    humidity_df = (
        forecast_df[
            [
                "forecast_time",
                "forecast_humidity",
            ]
        ]
        .set_index(
            "forecast_time"
        )
    )

    st.line_chart(
        humidity_df
    )


with weather_tabs[2]:

    precipitation_df = (
        forecast_df[
            [
                "forecast_time",
                "forecast_precipitation",
            ]
        ]
        .set_index(
            "forecast_time"
        )
    )

    st.bar_chart(
        precipitation_df
    )


with weather_tabs[3]:

    wind_df = (
        forecast_df[
            [
                "forecast_time",
                "forecast_wind_speed",
            ]
        ]
        .set_index(
            "forecast_time"
        )
    )

    st.line_chart(
        wind_df
    )


# =========================================================
# Full hourly forecast table
# =========================================================

st.subheader(
    "Hourly Forecast"
)


display_df = forecast_df[

    [
        "forecast_time",
        "horizon_hours",
        "predicted_aqi",
        "aqi_category",
        "forecast_temperature",
        "forecast_humidity",
        "forecast_precipitation",
        "forecast_wind_speed",
    ]

].copy()


display_df = display_df.rename(
    columns={
        "forecast_time":
            "Time",

        "horizon_hours":
            "Horizon (h)",

        "predicted_aqi":
            "Predicted AQI",

        "aqi_category":
            "AQI Category",

        "forecast_temperature":
            "Temperature °C",

        "forecast_humidity":
            "Humidity %",

        "forecast_precipitation":
            "Precipitation mm",

        "forecast_wind_speed":
            "Wind Speed",
    }
)


display_df[
    "Predicted AQI"
] = display_df[
    "Predicted AQI"
].round(1)


display_df[
    "Temperature °C"
] = display_df[
    "Temperature °C"
].round(1)


display_df[
    "Humidity %"
] = display_df[
    "Humidity %"
].round(1)


display_df[
    "Precipitation mm"
] = display_df[
    "Precipitation mm"
].round(2)


display_df[
    "Wind Speed"
] = display_df[
    "Wind Speed"
].round(1)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# AQI scale explanation
# =========================================================

with st.expander(
    "AQI Categories"
):

    st.write(
        """
        **0–50:** Good

        **51–100:** Moderate

        **101–150:** Unhealthy for Sensitive Groups

        **151–200:** Unhealthy

        **201–300:** Very Unhealthy

        **301+:** Hazardous
        """
    )


# =========================================================
# Footer
# =========================================================

st.divider()


st.caption(
    "AQI predictions are generated using 72 direct "
    "XGBoost forecasting models with historical air "
    "quality features and Open-Meteo future weather data."
)