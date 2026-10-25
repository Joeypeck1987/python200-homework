"""Reusable weather predictions from a saved pipeline."""
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]


@dataclass
class Prediction:
    """A good/skip label and the probability (0–1) of good running weather."""
    label: str
    probability: float


class WeatherClassifier:
    """Load a weather pipeline once and reuse it across batches."""

    def __init__(self, model_path: str | Path) -> None:
        """Load a pipeline saved with joblib.

        Raises:
            FileNotFoundError: If model_path does not point to a file.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No model file at {path}. Copy your saved Week 3 model here first.")
        self.model = joblib.load(path)
        self.features = FEATURES.copy()

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Return one prediction per day, preserving input order.

        Args:
            days: Dictionaries containing all four FEATURES.
        Returns:
            Readable labels and probabilities of class 1 (good).
        Raises:
            ValueError: If any day is missing required features.
        """
        if not days:
            return []
        for index, day in enumerate(days):
            missing = [feature for feature in self.features if feature not in day]
            if missing:
                raise ValueError(f"Day {index}: missing required features: {missing}")
        frame = pd.DataFrame(days)[self.features]
        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]
        return [
            Prediction("good" if int(label) == 1 else "skip", float(probability))
            for label, probability in zip(labels, probabilities)
        ]
