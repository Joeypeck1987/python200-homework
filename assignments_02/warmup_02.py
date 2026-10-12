from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --- scikit-learn API ---
# Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)
predictions = model.predict([[6], [18]])
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6:", predictions[0])
print("Predicted high for a low of 18:", predictions[1])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)
x = x.reshape(-1, 1)
print("Reshaped shape:", x.shape)
# X has rows for observations and columns for features, even with one feature.

# --- Linear Regression ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.figure()
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily low temperature (degrees)")
plt.ylabel("Daily high temperature (degrees)")
plt.colorbar(ticks=[0, 1], label="Rain status (0 = dry, 1 = rainy)")
plt.tight_layout()
plt.savefig(OUTPUTS / "high_vs_low.png")
plt.close()
# The colors form two overlapping upward bands. Rainy days tend to have
# lower highs at the same low temperature, suggesting rain is useful as a feature.

# Q2
X = temp_min.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, temp_max, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_single = model.score(X_test, y_test)
print("One-feature slope:", model.coef_[0])
print("One-feature intercept:", model.intercept_)
print("One-feature test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("One-feature test R²:", r2_single)
# A one-degree increase in the low corresponds to an increase in predicted
# high equal to the slope, in degrees. This describes association, not causation.

# Q4
X_full = np.column_stack([temp_min, is_rainy])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, temp_max, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, y_test_full)
print("Two-feature test R²:", r2_full)
print("One-feature test R² for comparison:", r2_single)
print("Adding the rain flag improves test R²:", r2_full > r2_single)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
# Holding the low temperature fixed, changing from dry (0) to rainy (1)
# changes the predicted high by the rain coefficient, in degrees.
# A negative coefficient means a lower predicted high on rainy days.

# Q5
plt.figure()
plt.scatter(y_pred_full, y_test_full)
lower = min(y_pred_full.min(), y_test_full.min())
upper = max(y_pred_full.max(), y_test_full.max())
plt.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted daily high (degrees)")
plt.ylabel("Actual daily high (degrees)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "predicted_vs_actual_warmup.png")
plt.close()
# Above the diagonal: actual exceeds predicted, so the model underestimates.
# Below the diagonal: actual is lower than predicted, so the model overestimates.
print("Plots saved to:", OUTPUTS)
