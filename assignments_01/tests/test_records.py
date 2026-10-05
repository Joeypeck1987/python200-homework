"""Tests for converting validated data into hourly records."""

import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings


@pytest.fixture
def sample_response() -> WeatherResponse:
    """Provide three observations with distinct measurement values."""
    return WeatherResponse.model_validate(
        {
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
                "precipitation": [0.0, 0.1, 0.2],
            },
        }
    )


def test_readings_preserve_count_and_order(sample_response):
    readings = to_readings(sample_response)

    assert len(readings) == 3
    assert [r.timestamp for r in readings] == sample_response.hourly.time
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


@pytest.mark.parametrize("i", [0, 1, 2])
def test_reading_values_match_input_index(sample_response, i):
    readings = to_readings(sample_response)
    hourly = sample_response.hourly

    assert readings[i].timestamp == hourly.time[i]
    assert readings[i].temperature_c == pytest.approx(
        hourly.temperature_2m[i]
    )
    assert readings[i].precipitation_mm == pytest.approx(
        hourly.precipitation[i]
    )


def test_identical_readings_compare_equal():
    first = HourlyReading("2026-04-08T00:00", 16.8, 0.0)
    second = HourlyReading("2026-04-08T00:00", 16.8, 0.0)

    assert first == second