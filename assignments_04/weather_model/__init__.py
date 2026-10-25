"""Public interface for the weather component."""
from .classifier import FEATURES, Prediction, WeatherClassifier

__all__ = ["WeatherClassifier", "Prediction", "FEATURES"]
