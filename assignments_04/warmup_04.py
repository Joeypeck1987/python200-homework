"""Week 4 warmups: run directly for demos, or with pytest for tests."""
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest


# Question 1
class ModelInfo:
    """Read and store model metadata once."""

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of stored features."""
        return len(self.data["features"])


# Loading in __init__ avoids disk reads and JSON parsing on every method call.


# Question 2
@dataclass
class DayForecast:
    """A small forecast with a readable summary."""
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain}"


# @dataclass generated __init__ for assigning the three fields.


# Question 3
def parse_temperature(text: str) -> float:
    """Convert text to a float, raising a helpful ValueError on bad input."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid temperature: {text!r}") from error


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, an unrelated ValueError with an unhelpful message could pass.


def main() -> None:
    """Display all three warmup results."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.json"
        path.write_text(json.dumps({"features": ["high_c", "rain_mm"]}), encoding="utf-8")
        print(f"Feature count: {ModelInfo(path).feature_count()}")
    for forecast in [DayForecast("2026-10-25", 24.1, True), DayForecast("2026-10-26", 19.0, False)]:
        print(forecast.summary())
    print(f"Parsed temperature: {parse_temperature('24.1')}")


if __name__ == "__main__":
    main()
