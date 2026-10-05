"""Validation schemas for an hourly weather API response."""

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Parallel lists of hourly timestamps and weather measurements."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_lengths(self) -> "HourlyBlock":
        """Require all measurement lists to have the same length."""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError("Hourly lists must have the same length")
        return self


class WeatherResponse(BaseModel):
    """Location metadata and hourly observations from the weather API."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock