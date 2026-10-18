# Week 3 rubric checklist

Source: [official assignment with expanded rubric](https://github.com/Code-the-Dream-School/python-200-v1/blob/main/assignments/03_ML_classification.md), accessed October 18, 2026. No additional hidden AirHub criteria were available.

| Required item | Evidence |
| --- | --- |
| Exact warmup setup | Initial block of `warmup_03.py`, unchanged from the assignment. |
| Preprocessing Q1 | Train-only scaler; both transformations; printed means and leakage comment. |
| Preprocessing Q2 | Dense season encoding, shape `(600, 4)`, feature names, four-category explanation. |
| Preprocessing Q3 | Numeric/season ColumnTransformer in logistic pipeline; test accuracy and automation explanation. |
| KNN Q1 | Scaled k=5 model, accuracy, full report. |
| KNN Q2 | All eight required k values, five-fold training CV, measured k=11 choice in comment. |
| KNN Q3 | Raw/scaled accuracies and explanation of improvement. |
| Logistic Q1 | Accuracy, good-class precision/recall/F1. |
| Logistic Q2 | Four coefficients with measured interpretation. |
| Logistic Q3 | First five good probabilities and near-0.5 interpretation. |
| Evaluation Q1 | Confusion display rendered to `outputs/logreg_confusion_matrix.png`; 9 FP / 7 FN and error-cost comment. |
| joblib Q1 | `models/warmup_model.pkl`; reload assertion; explanation of missing scaler failure. |
| Training Task 1 | Actual Open-Meteo request; Charlotte/year comment; raw shape and head. |
| Training Task 2 | All original label thresholds; printed balance; 38.08% good comment and metric discussion. |
| Training Task 3 | Stratified 80/20 split; eight k candidates; training CV; fitted KNN and report. |
| Training Task 4 | Scaled logistic model; four metrics, report, coefficients and interpretation. |
| Training Task 5 | Measured accuracy comparison, FP/FN tradeoff, deployment justification with size/speed/interpretability/probabilities. |
| Training Task 6 | `models/weather_classifier.pkl` and `models/weather_classifier_metadata.json`; versions, features, rule, city, scores, confirmation. |
| Prediction Task 7 | Loads both artifacts, metadata printed, four hypothetical days and their inputs/labels/P(good), no training or network calls. |
| Prediction Task 8 | Bottom comment answers borderline probability, 0.52 handling, missing artifact behavior, and scheduled-pipeline change. |
| Saved model files committed | Both `.pkl` files included for repository commit; not excluded by `.gitignore`. |
| Career research: titles | Junior/associate/data/staff examples and responsibility comparison in `mindset_03.md`. |
| Career research: languages | Python/SQL and JavaScript/TypeScript co-occurrences with sources. |
| Career research: other keywords | Cloud/AI/frontend discussion plus recorded Amazon query counts. |
| Career research: location vs remote | Explicit example-location sample comparison; applicant totals marked unavailable. |
| Career reflection | Proposed next project and skills exploration; personal assumptions disclosed. |

Extensions A, B, C and the PCA lesson are explicitly optional in the assignment and are not required to pass. This checklist is a self-review, not an AirHub grade.
