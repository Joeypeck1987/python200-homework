"""Tests for daily weather aggregation."""

import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def sample_readings() -> list[HourlyReading]:
    """Two dates: one with three observations and one with two."""
    # Deliberately place the later date first to test sorting.
    return [
        HourlyReading("2026-04-09T00:00", 12.0, 0.3),
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 20.0, 0.2),
        HourlyReading("2026-04-09T01:00", 18.0, 0.4),
        HourlyReading("2026-04-08T02:00", 15.0, 0.0),
    ]


def test_grouping_returns_two_sorted_dates(sample_readings):
    summaries = DailyAggregator(min_hours=2).summarize(sample_readings)

    assert [s.date for s in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert [s.hours_observed for s in summaries] == [3, 2]


def test_daily_high_and_low(sample_readings):
    summaries = DailyAggregator(min_hours=2).summarize(sample_readings)

    assert summaries[0].temp_max == pytest.approx(20.0)
    assert summaries[0].temp_min == pytest.approx(10.0)
    assert summaries[1].temp_max == pytest.approx(18.0)
    assert summaries[1].temp_min == pytest.approx(12.0)


def test_precipitation_totals(sample_readings):
    summaries = DailyAggregator(min_hours=2).summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.3)
    assert summaries[1].precipitation_sum == pytest.approx(0.7)


def test_short_day_is_dropped_and_reported(sample_readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(sample_readings)

    assert [s.date for s in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(sample_readings) == ["2026-04-09"]


def test_lowering_min_hours_keeps_short_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert [s.date for s in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert aggregator.incomplete_days(sample_readings) == []


@pytest.mark.parametrize(
    "index, expected_range",
    [
        (0, 10.0),
        (1, 6.0),
    ],
)
def test_temperature_range(sample_readings, index, expected_range):
    summaries = DailyAggregator(min_hours=2).summarize(sample_readings)

    assert summaries[index].temp_range() == pytest.approx(expected_range)


# Deliberately changed DailySummary.temp_range() to add instead of subtract.
# test_temperature_range caught the change in both parametrized cases:
# the result was 30.0 instead of the expected 10.0 and 6.0.
# Restored subtraction after confirming the tests failed.