# Week 3: Classification and Model Deployment

Completed required warmups, weather mini-project, and career research. The required coding work follows the [course assignment and embedded rubric](https://github.com/Code-the-Dream-School/python-200-v1/blob/main/assignments/03_ML_classification.md). Optional extensions A–C are not included.

## Run

Use Python 3.12 and install the recorded dependencies from this directory:

```console
python -m pip install -r requirements.txt
python warmup_03.py
python train_classifier.py
python predict.py
```

Run the warmup from `assignments_03/` because its mandated setup reads the course CSV with a relative path. Training and prediction locate their files relative to the script, so they can also run from another working directory. For an offline reproduction of the included weather data, use `python train_classifier.py --cached`. Default training fetches Open-Meteo as required.

## Included files

- `warmup_03.py`: all 11 required warmup questions and explanatory comments.
- `weather_classification.csv`: the unmodified 600-row course dataset, downloaded from [the course resource](https://github.com/Code-the-Dream-School/python-200-v1/blob/main/lessons/03_ML_classification/resources/weather_classification.csv).
- `train_classifier.py`: all six training tasks using Charlotte, NC, and 2023 weather.
- `predict.py`: four example days, saved-model inference, and all three reflection answers.
- `weather_charlotte_2023.csv`: real daily weather snapshot for reproducible training.
- `models/`: both required fitted pipelines plus deployment metadata.
- `outputs/`: figures, execution logs, API response, comparison metrics, and career-search evidence.
- `mindset_03.md`: sourced career research and a proposed personal exploration plan. Chicago is an illustrative local market, not a claim about the student's residence.
- `rubric_checklist.md`: where each required rubric item is satisfied.

## Observed results

| Dataset / model | Accuracy | Precision for good | Recall for good | F1 for good |
| --- | ---: | ---: | ---: | ---: |
| Course warmup logistic regression | 0.8667 | 0.8571 | 0.8852 | 0.8710 |
| Charlotte KNN, k=3 | 0.9726 | 0.9333 | 1.0000 | 0.9655 |
| Charlotte logistic regression, deployed | 0.8767 | 0.8276 | 0.8571 | 0.8421 |

The warmup's training CV chooses k=11; the separate Charlotte data chooses k=3. Both selections use only training data. Charlotte contains 139 good and 226 skip days. Its held-out test set contains 73 days, and logistic regression makes five false positives and four false negatives. The deployment decision and its measured performance cost are explained in `train_classifier.py`.

The model learns a deterministic running-weather rule, using weather inputs from the same day. These scores measure how well it approximates that rule; they do not demonstrate forecasting future weather. Its probabilities are uncalibrated model estimates. The original multi-threshold rule is not exactly representable by a linear classifier.

## Verification and submission

The included execution logs record successful warmup, cached training, and prediction runs. Pipeline persistence checks confirm identical labels (and, for deployment, probabilities) after saving/loading. Additional validation checks the 365-day snapshot, independent inference, useful missing-model error, artifact presence, and rubric setup consistency. `requirements.txt` records the exact environment used to create the serialized models.

The course's primary submission is an open pull request into this repository's `main` branch. Submit the PR URL through the lesson/AirHub flow; the local files alone are not the course submission. Review the proposed career plan against your own interests and location before submitting that reflection.
