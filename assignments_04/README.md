# Week 4: From Model to Reusable Component

## Run

Use Python 3.12 and run these commands from assignments_04/:

```text
uv pip install -r requirements.txt
pytest -v
pytest warmup_04.py -v
python warmup_04.py
python predict_weather.py
```

The empty conftest.py enables the required pytest import layout.
The public interface is `from weather_model import WeatherClassifier, Prediction`.
Each day must supply temperature_2m_max, temperature_2m_min (C),
precipitation_sum (mm), and wind_speed_10m_max (km/h).
Prediction.probability always means the probability of good running weather,
including when the predicted label is skip.

## Model provenance

Both files in models/ are copied unchanged from Week 3 branch assignments_03,
commit 080215b55133d63846ee7ccd30b2ae227e448158. They contain the real Charlotte
2023 logistic regression pipeline and its metadata. Dependencies match the
recorded training versions. No Week 3 branch changes are needed for this PR.

## Required work

- warmup_04.py: all three exercises, printed demos, two tests and explanation comments.
- weather_model/: dataclass, load-once classifier, feature ordering, per-day missing-key validation, public exports.
- tests/test_classifier.py: eight test functions, twelve cases, temporary trained fixtures and parametrization.
- predict_weather.py: four varied days, all four inputs printed, main guard and three reflection answers.
- models/: original Week 3 model and metadata.
- conftest.py: intentionally empty.

## Validation

- pytest -v: 12 passed from assignments_04/.
- pytest warmup_04.py -v: 2 passed.
- Both demonstration scripts ran successfully.
- Deliberately changed the classifier to always return skip: test_day_label[day0-good]
  failed (one failure, one pass). Restored the correct code before the final run.
- Extra checks cover dictionary key order, class-1 probabilities, and a feature
  missing from just one record in a batch.

Optional extensions are omitted. The separate mindset assignment requires your
own job-board browsing, saved postings, and personal reflections; it is not
represented as completed by this coding submission.
