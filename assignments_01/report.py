"""Print daily summaries from the supplied weather API response."""

import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main() -> None:
    """Load, validate, convert, and summarize the weather file."""
    data_path = Path(__file__).parent / "weather_raw.json"

    with data_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)
    readings = to_readings(response)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)

    print(f"Daily weather report — timezone: {response.timezone}")
    print(
        f"{'Date':<12} {'High C':>8} {'Low C':>8} "
        f"{'Rain mm':>9} {'Range C':>9} {'Hours':>6}"
    )
    print("-" * 57)

    for summary in summaries:
        print(
            f"{summary.date:<12} "
            f"{summary.temp_max:>8.1f} "
            f"{summary.temp_min:>8.1f} "
            f"{summary.precipitation_sum:>9.1f} "
            f"{summary.temp_range():>9.1f} "
            f"{summary.hours_observed:>6}"
        )

    dropped_dates = aggregator.incomplete_days(readings)
    if dropped_dates:
        print(
            "\nWarning: incomplete days were dropped: "
            + ", ".join(dropped_dates)
        )


# Without this guard, an unconditional main() call would load the file
# and print the report whenever another program imported this module.
if __name__ == "__main__":
    main()

# --- Reflection ---
#
# 1. Missing temperatures:
# Rejecting the entire response is appropriate when a downstream model
# requires complete numeric inputs and missing data would invalidate its
# predictions. For a monitoring dashboard, I might prefer to display the
# available readings while clearly reporting gaps.
# To allow missing temperatures, I would change the schema field to
# temperature_2m: list[float | None].
# That schema change alone is not enough: conversion and aggregation
# would also need a policy for skipping or handling missing measurements.
#
# 2. Running at noon:
# At noon, the current day may have only about twelve observations.
# With min_hours=24, it is excluded even if every hour so far is present.
# incomplete_days() identifies the excluded date so the pipeline can
# explain that the day is unfinished and report it after more data arrives.
# Lowering min_hours would allow a partial summary, but its high, low,
# and precipitation total would not represent the whole day.
#
# 3. Benefits of a package:
# In Week 10, the pipeline can import WeatherResponse, to_readings,
# and DailyAggregator directly from weatherkit and reuse this logic.
# It does not need to copy the code or run the report script.