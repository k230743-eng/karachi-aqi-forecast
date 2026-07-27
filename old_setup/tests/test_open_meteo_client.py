import pandas as pd
import pytest

from src.data.open_meteo_client import (
    OpenMeteoAPIError,
    hourly_response_to_dataframe,
)


def test_hourly_response_to_dataframe() -> None:
    response_data = {
        "hourly": {
            "time": [
                "2026-07-27T00:00",
                "2026-07-27T01:00",
            ],
            "temperature_2m": [
                29.5,
                29.1,
            ],
        }
    }

    dataframe = hourly_response_to_dataframe(response_data)

    assert len(dataframe) == 2
    assert "time" in dataframe.columns
    assert "temperature_2m" in dataframe.columns
    assert pd.api.types.is_datetime64_any_dtype(dataframe["time"])


def test_missing_hourly_data_raises_error() -> None:
    with pytest.raises(
        OpenMeteoAPIError,
        match="does not contain hourly data",
    ):
        hourly_response_to_dataframe({})


def test_missing_time_column_raises_error() -> None:
    response_data = {
        "hourly": {
            "temperature_2m": [29.5, 29.1],
        }
    }

    with pytest.raises(
        OpenMeteoAPIError,
        match="does not contain timestamps",
    ):
        hourly_response_to_dataframe(response_data)