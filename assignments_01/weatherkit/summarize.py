"""Aggregate hourly weather observations into daily summaries."""

from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Daily temperatures in Celsius and total precipitation in millimeters."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the daily high minus the daily low in Celsius."""
        return self.temp_max - self.temp_min
        # return self.temp_max + self.temp_min

class DailyAggregator:
    """Summarize days that meet a minimum observation count."""

    def __init__(self, min_hours: int = 24) -> None:
        """Set the minimum number of observations required per day."""
        self.min_hours = min_hours

    def _group_by_date(
        self, readings: list[HourlyReading]
    ) -> dict[str, list[HourlyReading]]:
        """Group readings by the date portion of their timestamps."""
        groups: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            groups.setdefault(date, []).append(reading)

        return groups

    def summarize(
        self, readings: list[HourlyReading]
    ) -> list[DailySummary]:
        """Return summaries sorted by date, excluding incomplete days."""
        groups = self._group_by_date(readings)
        summaries: list[DailySummary] = []

        for date in sorted(groups):
            day_readings = groups[date]

            if len(day_readings) < self.min_hours:
                continue

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(r.temperature_c for r in day_readings),
                    temp_min=min(r.temperature_c for r in day_readings),
                    precipitation_sum=sum(
                        r.precipitation_mm for r in day_readings
                    ),
                    hours_observed=len(day_readings),
                )
            )

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return sorted dates with fewer than the required observations."""
        groups = self._group_by_date(readings)

        return [
            date
            for date in sorted(groups)
            if len(groups[date]) < self.min_hours
        ]

    