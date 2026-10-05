"""Tests for validating weather API responses."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def test_real_response_has_168_hours():
    # A plain relative path depends on the terminal's working directory.
    # Using this test file's location finds the JSON from any working directory.
    data_path = Path(__file__).parent.parent / "weather_raw.json"

    with data_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168


def small_response() -> dict:
    """Build fresh sample input for validation tests."""
    return {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [16.8, 15.0, 14.2],
            "precipitation": [0.0, 0.1, 0.0],
        },
    }


def test_invalid_latitude_raises():
    data = small_response()
    data["latitude"] = 200.0

    with pytest.raises(ValidationError, match="latitude"):
        WeatherResponse.model_validate(data)


def test_mismatched_lengths_raise():
    data = small_response()
    data["hourly"]["temperature_2m"].pop()

    with pytest.raises(ValidationError, match="same length"):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises():
    data = small_response()
    data["hourly"]["temperature_2m"][1] = None

    with pytest.raises(ValidationError, match="temperature_2m"):
        WeatherResponse.model_validate(data)