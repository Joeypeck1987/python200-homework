"""Fetch one year of weather, compare classifiers, and save a fitted pipeline."""
import argparse
import json
from pathlib import Path
import platform

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FEATURES = ["temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max"]
CITY = "Charlotte, North Carolina, USA"
PARAMS = {
    "latitude": 35.23, "longitude": -80.84,
    "start_date": "2023-01-01", "end_date": "2023-12-31",
    "daily": FEATURES, "timezone": "America/New_York",
    "temperature_unit": "celsius", "precipitation_unit": "mm",
    "wind_speed_unit": "kmh",
}


def metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true",
                        help="Use the included weather snapshot without API access.")
    args = parser.parse_args()
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "outputs").mkdir(exist_ok=True)

    # Task 1: Charlotte, North Carolina; full calendar year 2023.
    # Default execution fetches live historical data; --cached reproduces this run.
    snapshot = BASE / "weather_charlotte_2023.csv"
    if args.cached:
        df = pd.read_csv(snapshot, parse_dates=["date"])
    else:
        response = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                params=PARAMS, timeout=60)
        response.raise_for_status()
        payload = response.json()
        (BASE / "outputs/open_meteo_response.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        df = pd.DataFrame(payload["daily"])
        df["date"] = pd.to_datetime(df.pop("time"))
        df.to_csv(snapshot, index=False)
    if len(df) != 365 or df[FEATURES].isna().any().any():
        raise ValueError("Expected 365 complete daily weather records for 2023.")
    print(f"City: {CITY}; raw shape: {df.shape}")
    print(df.head().to_string(index=False))

    # Task 2: The original thresholds are unchanged; 7 and 26 C are inclusive.
    df["good_for_running"] = (
        df["temperature_2m_max"].between(7, 26)
        & df["temperature_2m_min"].ge(0)
        & df["precipitation_sum"].lt(3)
        & df["wind_speed_10m_max"].lt(30)
    ).astype(int)
    print("Class balance (0=skip, 1=good):", df.good_for_running.value_counts().to_dict())
    print(f"Fraction good: {df.good_for_running.mean():.2%}")
    # 139/365 = 38.08% of days are good; 226 are skip days. A minority of mild,
    # dry days is plausible because this rule excludes heat, cold, rain and wind.
    # The good-class precision, recall and F1 expose failures hidden by accuracy.
    X, y = df[FEATURES], df["good_for_running"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Task 3: All scaling is learned inside each training CV fold.
    cv_scores = {}
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        candidate = Pipeline([("scaler", StandardScaler()),
                              ("knn", KNeighborsClassifier(n_neighbors=k))])
        cv_scores[k] = float(cross_val_score(candidate, X_train, y_train, cv=5).mean())
        print(f"k={k:2d}: mean training CV accuracy={cv_scores[k]:.4f}")
    best_k = max(cv_scores, key=cv_scores.get)
    knn = Pipeline([("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=best_k))])
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_metrics = metrics(y_test, knn_predictions)
    print(f"Chosen k={best_k}; KNN metrics: {knn_metrics}")
    print(classification_report(y_test, knn_predictions, target_names=["skip", "good"], zero_division=0))

    # Task 4
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    logreg_metrics = metrics(y_test, predictions)
    print("Logistic regression metrics:", logreg_metrics)
    print(classification_report(y_test, predictions, target_names=["skip", "good"], zero_division=0))
    coefficients = dict(zip(FEATURES, logreg.named_steps["logreg"].coef_[0].tolist()))
    for feature, coefficient in coefficients.items():
        print(f"{feature}: {coefficient:+.4f}")
    # Rain (-2.4916) and high temperature (-0.9343) most strongly favor skip.
    # This fits wet/hot-day intuition. Wind is unexpectedly positive (+0.3198):
    # seasonal correlations and correlated temperatures can distort a linear
    # conditional coefficient. It does not imply wind makes running better.
    # Each coefficient describes log odds per training standard deviation,
    # holding the other features constant; these are associations, not causes.

    # Task 5
    # KNN won accuracy: 0.9726 versus logistic regression's 0.8767; its good
    # precision also won, 0.9333 versus 0.8276. For this dataset and an app judged
    # primarily on precision I would prefer KNN. For this instructional deployment
    # I accept that measured loss to keep the small, interpretable linear model
    # used by the later course components; it should not be sold as the score winner.
    # I weigh precision of good more heavily: a false positive sends someone
    # out on a bad day, whereas a false negative only misses a pleasant run.
    # I deploy logistic regression for compact storage, constant-size linear
    # scoring, interpretable coefficients and smooth probability estimates.
    # KNN also supplies predict_proba, but its probabilities are vote fractions;
    # it stores training examples and performs neighbor searches at prediction.
    # These probabilities are model estimates, not calibrated guarantees.
    # A linear boundary cannot exactly represent all of the threshold rule.
    chosen = logreg
    print("Deployment choice: logistic regression")

    # Task 6: Save the entire fitted pipeline, including its learned scaler.
    model_path = BASE / "models/weather_classifier.pkl"
    joblib.dump(chosen, model_path)
    loaded = joblib.load(model_path)
    assert np.array_equal(loaded.predict(X_test), chosen.predict(X_test))
    assert np.allclose(loaded.predict_proba(X_test), chosen.predict_proba(X_test))
    metadata = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES, "label": "good_for_running", "city": CITY,
        "model_type": "LogisticRegression", "classes": [0, 1],
        "label_rule": {"temperature_2m_max": "7 <= high <= 26 C",
                       "temperature_2m_min": ">= 0 C",
                       "precipitation_sum": "< 3 mm",
                       "wind_speed_10m_max": "< 30 km/h"},
        "test_accuracy": logreg_metrics["accuracy"], "test_f1": logreg_metrics["f1"],
        "test_precision": logreg_metrics["precision"], "test_recall": logreg_metrics["recall"],
        "source": "https://archive-api.open-meteo.com/v1/archive", "request_params": PARAMS,
        "train_rows": len(X_train), "test_rows": len(X_test), "random_state": 42,
        "training_feature_ranges": {c: {"min": float(X_train[c].min()),
                                           "max": float(X_train[c].max())} for c in FEATURES},
    }
    (BASE / "models/weather_classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    results = {"knn": knn_metrics, "logistic_regression": logreg_metrics,
               "best_k": best_k, "cv_scores": cv_scores, "coefficients": coefficients,
               "class_counts": df.good_for_running.value_counts().to_dict(),
               "good_fraction": float(y.mean()),
               "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()}
    (BASE / "outputs/model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1],
                                           display_labels=["skip", "good"], colorbar=False)
    plt.title("Charlotte 2023: logistic regression")
    plt.tight_layout()
    plt.savefig(BASE / "outputs/weather_confusion_matrix.png", dpi=150)
    plt.close()
    print("Model and metadata saved to models/. Reloaded labels and probabilities match.")


if __name__ == "__main__":
    main()
