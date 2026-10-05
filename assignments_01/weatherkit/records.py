"""Internal records for validated hourly weather data."""

from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """One hourly observation.

    timestamp is the API's timestamp string.
    temperature_c is in degrees Celsius.
    precipitation_mm is in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert parallel hourly lists into individual reading objects.

    Args:
        response: A weather response whose types and list lengths
            have already been validated.

    Returns:
        One HourlyReading per observation, preserving input order.
    """
    hourly = response.hourly
    readings = []

    for i in range(len(hourly.time)):
        reading = HourlyReading(
            timestamp=hourly.time[i],
            temperature_c=hourly.temperature_2m[i],
            precipitation_mm=hourly.precipitation[i],
        )
        readings.append(reading)

    return readings


# WeatherResponse validates outside data as it enters our program.
# HourlyReading represents internal data built from that validated response,
# so a dataclass bundles the values without repeating boundary validation.