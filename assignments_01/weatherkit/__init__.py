"""Tools for validating and summarizing weather observations."""

from .schemas import HourlyBlock, WeatherResponse
from .records import HourlyReading, to_readings
from .summarize import DailySummary, DailyAggregator