"""Classify four hypothetical days using the real saved Week 3 model."""
from pathlib import Path
from weather_model import WeatherClassifier

SAMPLE_DAYS = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
    {"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40},
    {"temperature_2m_max": 26, "temperature_2m_min": 15, "precipitation_sum": 2.9, "wind_speed_10m_max": 29},
    {"temperature_2m_max": 8, "temperature_2m_min": 0, "precipitation_sum": 0.0, "wind_speed_10m_max": 28},
]


def main() -> None:
    """Print inputs, labels, and probabilities of good weather."""
    path = Path(__file__).resolve().parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(path)
    for number, (day, result) in enumerate(zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)), 1):
        print(
            f"Day {number}: high {day['temperature_2m_max']}C, "
            f"low {day['temperature_2m_min']}C, rain {day['precipitation_sum']}mm, "
            f"wind {day['wind_speed_10m_max']}km/h -> {result.label} "
            f"(probability of good: {result.probability:.2%})"
        )


# Without the guard, importing this module to reuse a helper would load the
# model and print predictions as an unexpected side effect.
if __name__ == "__main__":
    main()

# Reflection
# 1. Fixed feature order prevents wind=40 from being interpreted as a high
#    temperature when a caller builds dictionary keys in a different order.
#    The model always receives high, low, precipitation, then wind.
# 2. A fixture builds a known, tiny model in tmp_path, so tests work without
#    the real file and don't change when the real model is retrained. Tests
#    check the component's contract rather than one model's exact scores.
# 3. Loading once and accepting batches supports repeated database batches in
#    Week 10. Before production, I would validate numeric values, reject
#    non-finite inputs, and document supported units and ranges.
