"""Load the saved weather pipeline and predict; no training or API access."""
import json
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    model_path = BASE / "models/weather_classifier.pkl"
    metadata_path = BASE / "models/weather_classifier_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise SystemExit("Missing saved model or metadata. Run python train_classifier.py first "
                         "(or python train_classifier.py --cached using the included snapshot).")
    clf = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("City:", metadata["city"])
    print("Features:", metadata["features"])
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
    names = ["Clearly good: mild and dry", "Clearly bad: wet and windy",
             "Borderline precipitation", "Borderline upper temperature"]
    new_days = pd.DataFrame([
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0, "wind_speed_10m_max": 12},
        {"temperature_2m_max": 32, "temperature_2m_min": 23, "precipitation_sum": 20, "wind_speed_10m_max": 45},
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 2.9, "wind_speed_10m_max": 20},
        {"temperature_2m_max": 26, "temperature_2m_min": 17, "precipitation_sum": 0, "wind_speed_10m_max": 15},
    ])[metadata["features"]]
    predictions = clf.predict(new_days)
    good_index = list(clf.classes_).index(1)
    probabilities = clf.predict_proba(new_days)[:, good_index]
    for name, (_, inputs), pred, probability in zip(names, new_days.iterrows(), predictions, probabilities):
        print(f"\n{name}: {inputs.to_dict()}")
        print(f"Prediction: {'good' if pred == 1 else 'skip'}; P(good): {probability:.4f}")


if __name__ == "__main__":
    main()

# Task 8: Reflection
# The day with 2.9 mm rain, a high of 18 C, a low of 10 C and 20 km/h wind
# received P(good)=0.3998. It favors skip but is fairly uncertain, not confident.
# A P(good) of 0.52 barely favors good; I would show 'uncertain' and the actual
# forecast, recommending that the runner inspect rain and wind before deciding.
# A day near a physical label threshold need not have model probability near 0.5:
# logistic regression approximates a multi-threshold rule with one linear boundary.
# Running before training leaves the model and metadata absent. Loading would
# raise FileNotFoundError; the check above explains exactly which command to run.
# For Week 10, replace the hard-coded hypothetical days with incoming daily
# weather records and write predictions to durable storage rather than only print.
