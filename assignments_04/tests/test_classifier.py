"""Self-contained tests independent of the committed Week 3 model."""
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from weather_model import FEATURES, Prediction, WeatherClassifier

GOOD = dict(zip(FEATURES, [18, 9, 0.0, 10]))
BAD = dict(zip(FEATURES, [2, -6, 12.0, 40]))


@pytest.fixture
def model_path(tmp_path):
    """Train on obvious days and save in an isolated temporary directory."""
    frame = pd.DataFrame([GOOD] * 15 + [BAD] * 15)[FEATURES]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(frame, [1] * 15 + [0] * 15)
    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(model, path)
    return path


@pytest.fixture
def clf(model_path):
    return WeatherClassifier(model_path)


@pytest.mark.parametrize("day, expected", [(GOOD, "good"), (BAD, "skip")])
def test_day_label(clf, day, expected):
    # Mutation check: forcing every label to "skip" must fail the GOOD case.
    assert clf.predict([day])[0].label == expected


def test_probability_is_between_zero_and_one(clf):
    for result in clf.predict([GOOD, BAD]):
        assert isinstance(result.probability, float)
        assert 0 <= result.probability <= 1


def test_one_prediction_per_day_in_order(clf):
    results = clf.predict([GOOD, BAD, GOOD])
    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == ["good", "skip", "good"]
    assert results[0].probability == pytest.approx(results[2].probability)
    assert results[0].probability > results[1].probability


def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []


@pytest.mark.parametrize("feature", FEATURES)
def test_missing_feature_raises(clf, feature):
    incomplete = {key: value for key, value in GOOD.items() if key != feature}
    # A preceding complete day must not conceal a missing key in another row.
    with pytest.raises(ValueError, match=feature):
        clf.predict([GOOD, incomplete])


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file.*absent.pkl"):
        WeatherClassifier(tmp_path / "absent.pkl")


def test_dictionary_key_order_does_not_change_prediction(clf):
    assert clf.predict([dict(reversed(list(GOOD.items())))]) == clf.predict([GOOD])


def test_probability_is_for_good_class(clf):
    expected = clf.model.predict_proba(pd.DataFrame([GOOD, BAD])[FEATURES])[:, 1]
    assert [result.probability for result in clf.predict([GOOD, BAD])] == pytest.approx(expected)
