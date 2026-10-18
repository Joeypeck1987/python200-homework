import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# --- Preprocessing ---
# Q1: Fit on training data only so test statistics cannot leak into training.
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)
print("Scaled training means:", scaled_train.mean(axis=0))

# Q2: One season column becomes four columns, one for each observed season.
# This standalone encoding demonstration is not used to train the later models.
encoder = OneHotEncoder(sparse_output=False)
seasons = encoder.fit_transform(df[["season"]])
print("Season shape:", seasons.shape)
print("Season names:", encoder.get_feature_names_out())

# Q3: The pipeline learns preprocessing on training data and automatically
# applies the same transformations before fitting or predicting.
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])
season_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("logreg", LogisticRegression(max_iter=1000)),
])
season_pipeline.fit(df.loc[X_train.index, numeric + ["season"]], y_train)
print("Season pipeline accuracy:", season_pipeline.score(
    df.loc[X_test.index, numeric + ["season"]], y_test))

# --- KNN ---
# Q1
knn = Pipeline([("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=5))])
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print("Scaled KNN k=5 accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, target_names=["skip", "good"]))

# Q2: Choose the highest mean training CV accuracy; ties favor the smaller k.
# I choose k=11: its mean training CV accuracy is highest (0.9417).
cv_scores = {}
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    candidate = Pipeline([("scaler", StandardScaler()),
                          ("knn", KNeighborsClassifier(n_neighbors=k))])
    cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5).mean()
    print(f"k={k:2d}, mean CV accuracy={cv_scores[k]:.4f}")
best_k = max(cv_scores, key=cv_scores.get)
print("Chosen k:", best_k)

# Q3: Scaling prevents features with larger numeric spreads dominating distance.
# Scaling helped: accuracy rose from 0.9250 to 0.9583 on this fixed test split.
raw_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("Unscaled KNN accuracy:", raw_knn.score(X_test, y_test))
print("Scaled KNN accuracy:", knn.score(X_test, y_test))

# --- Logistic Regression ---
# Q1
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print("Logistic accuracy:", accuracy_score(y_test, predictions))
print("Logistic precision (good):", precision_score(y_test, predictions, pos_label=1))
print("Logistic recall (good):", recall_score(y_test, predictions, pos_label=1))
print("Logistic F1 (good):", f1_score(y_test, predictions, pos_label=1))

# Q2: Negative coefficients lower the log odds of good, so they favor skip.
# Standardization makes coefficient magnitudes comparable per standard deviation.
# Wind (-2.4438) and rain (-1.8891) most strongly favor skip, which matches
# the expectation that windy, wet weather is less suitable for running.
for name, coefficient in zip(numeric, pipeline.named_steps["logreg"].coef_[0]):
    print(f"{name}: {coefficient:+.4f}")

# Q3: Near 0.5 means the model barely favors one class; it is uncertain.
good_index = list(pipeline.classes_).index(1)
print("First five P(good):", np.round(pipeline.predict_proba(X_test)[:5, good_index], 2))

# --- Evaluation ---
# Q1: Rows are actual [skip, good]; columns are predicted [skip, good].
# The fixed test split produced 9 false positives and 7 false negatives.
# False positives matter more to me: recommending bad weather is worse than
# missing a pleasant run, so I prioritize precision for the good class.
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"False positives: {fp}; false negatives: {fn}")
ConfusionMatrixDisplay(cm, display_labels=["skip", "good"]).plot(colorbar=False)
plt.title("Warmup logistic regression")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png", dpi=150)
plt.close()

# --- joblib ---
# Q1: Saving only logistic regression loses the learned scaling statistics;
# raw inputs would be interpreted on the wrong scale and predictions could
# silently be incorrect even though the feature count still matches.
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
