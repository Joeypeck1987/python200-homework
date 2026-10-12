"""Week 2 weather regression. Run with: python assignments_02/project_02.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
FEATURES = ["temperature_2m_min", "precipitation_sum",
            "wind_speed_10m_max", "is_summer"]
MEASUREMENTS = ["temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "wind_speed_10m_max"]
# Charlotte, North Carolina is the primary city; Phoenix has a drier climate.
CITIES = {
    "charlotte": (35.23, -80.84, "America/New_York"),
    "phoenix": (33.45, -112.07, "America/Phoenix"),
}


def load_weather(city, start_date, end_date):
    """Cache the original daily measurements for repeat runs and Week 3."""
    path = OUTPUTS / f"{city}_{start_date}_{end_date}_weather.csv"
    if path.exists():
        data = pd.read_csv(path, parse_dates=["date"])
    else:
        latitude, longitude, timezone = CITIES[city]
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": latitude, "longitude": longitude,
                    "start_date": start_date, "end_date": end_date,
                    "daily": MEASUREMENTS, "timezone": timezone,
                    "temperature_unit": "celsius"}, timeout=60,
        )
        response.raise_for_status()
        data = pd.DataFrame(response.json()["daily"])
        data["date"] = pd.to_datetime(data.pop("time"))
        data.to_csv(path, index=False)
    print(f"\n{city}: dataset shape:", data.shape)
    print(data.head())
    print("Missing values:\n", data.isna().sum())
    # Drop incomplete days instead of inventing target temperatures. The cache
    # retains the original data, and the number dropped is explicitly reported.
    clean = data.dropna(subset=MEASUREMENTS + ["date"]).copy()
    print("Incomplete days dropped:", len(data) - len(clean))
    clean["is_summer"] = clean["date"].dt.month.isin([6, 7, 8]).astype(int)
    return clean


def evaluate(data, columns, label):
    X = data[columns].values
    y = data["temperature_2m_max"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)
    result = {"experiment": label, "days": len(data), "test_days": len(y_test),
              "train_r2": model.score(X_train, y_train),
              "test_r2": model.score(X_test, y_test),
              "rmse_c": np.sqrt(mean_squared_error(y_test, predicted))}
    print(f"\n{label}: train R²={result['train_r2']:.4f}, "
          f"test R²={result['test_r2']:.4f}, RMSE={result['rmse_c']:.3f} °C")
    for name, coef in zip(columns, model.coef_):
        print(f"{name:22s}: {coef:+.3f}")
    return model, result, predicted, y_test


def main():
    # Task 1: one complete historical year, 2023.
    df = load_weather("charlotte", "2023-01-01", "2023-12-31")

    # Task 2: inspect the numerical distributions and target correlations.
    print("\nDataset description:\n", df.describe())
    correlations = df[MEASUREMENTS].corr()["temperature_2m_max"].sort_values()
    print("\nSorted correlations with daily high:\n", correlations)
    strongest = correlations.drop("temperature_2m_max").abs().idxmax()
    print("Strongest correlation excluding the target:", strongest)
    plt.figure()
    plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.6)
    plt.title("Charlotte Daily High vs Low (2023)")
    plt.xlabel("Daily low (°C)")
    plt.ylabel("Daily high (°C)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "charlotte_high_vs_low.png")
    plt.close()
    # The scatter shows a strong upward relationship: warmer lows usually mean
    # warmer highs, with several degrees of spread at a given low temperature.

    # Task 3: summer is a binary seasonal feature. We expect warmer highs in
    # summer, but much of that information may already be captured by the low.
    print("Summer days:", int(df["is_summer"].sum()))
    print("Non-summer days:", int((df["is_summer"] == 0).sum()))

    # Task 4: the baseline uses a 2D, single-column feature array.
    baseline_model, baseline, _, _ = evaluate(
        df, ["temperature_2m_min"], "Charlotte baseline"
    )
    print("Baseline slope:", baseline_model.coef_[0])
    # RMSE measures prediction error on the Celsius scale, with larger errors
    # weighted more heavily. It is not the mean absolute error or a guarantee.
    print(f"Baseline error scale: about {baseline['rmse_c']:.2f} °C.")
    # In the verified 2023 run, baseline RMSE is 3.16 °C: errors are on the
    # scale of a few degrees, enough to matter when choosing clothing outdoors.

    # Task 5: the split is identical to the baseline so comparison is fair.
    model, full, predicted, actual = evaluate(df, FEATURES, "Charlotte full")
    print("Test R² improvement over baseline:", full["test_r2"] - baseline["test_r2"])
    # Each coefficient describes a change with the other features fixed.
    # Temperature uses °C, precipitation mm, wind km/h, and summer a 0→1 change.
    # Raw coefficients have different units, so their magnitudes are not a
    # reliable ranking of feature importance. Compare a one-standard-deviation
    # change for a common scale, while recognizing correlated predictors.
    scaled_effects = pd.Series(model.coef_, index=FEATURES) * df[FEATURES].std()
    print("Effects for a one-standard-deviation increase (°C):\n", scaled_effects)
    print("Train/test R² gap:", full["train_r2"] - full["test_r2"])
    # Charlotte results: low +0.888 °C per °C; rain -0.201 °C per mm;
    # wind -0.045 °C per km/h; summer +1.337 °C relative to non-summer.
    # All directions seem plausible, but these are associations, not causes.
    # Train R²=0.8805 and test R²=0.9051 are close. There is no large
    # training advantage here; the test set can be easier by chance.
    # This random split evaluates held-out days from the same period. It does
    # not establish performance on future years; that needs a chronological test.
    # Same-day measurements make this a teaching task rather than a forecast.

    # Task 6: predictions on x, actual values on y.
    plt.figure()
    plt.scatter(predicted, actual, alpha=0.7)
    lower, upper = min(predicted.min(), actual.min()), max(predicted.max(), actual.max())
    plt.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
    plt.title("Predicted vs Actual (Full Model)")
    plt.xlabel("Predicted daily high (°C)")
    plt.ylabel("Actual daily high (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "predicted_vs_actual_high.png")
    plt.close()
    bands = pd.qcut(actual, 3, labels=["low", "middle", "high"])
    for band in ["low", "middle", "high"]:
        mask = bands == band
        print(f"{band} temperature range RMSE: "
              f"{np.sqrt(mean_squared_error(actual[mask], predicted[mask])):.3f} °C")
    # The plot scatters around the diagonal. Errors are somewhat larger at the
    # low end: RMSE 2.94 °C versus 2.31 in the middle and 2.49 at the high end.
    # Summary: 365 complete days, 292 training days and 73 test days.
    # Full-model RMSE=2.60 °C (about 4.67 °F), so errors of a few degrees remain.
    # Test R²=0.9051 means about 90.5% of the test target variation is explained
    # relative to predicting its mean; it does not mean 90.5% accuracy.
    # The low has the largest standardized effect, raising predicted high by
    # 6.53 °C for one standard deviation of increase, with other inputs fixed.
    # One surprise: adding three features only raises test R² by 0.0454;
    # the daily low alone already explains much of the variation.

    # Extension A: independently fit the same features in a contrasting climate.
    phoenix = load_weather("phoenix", "2023-01-01", "2023-12-31")
    _, other_city, _, _ = evaluate(phoenix, FEATURES, "Phoenix full")
    # Phoenix: RMSE=2.80 °C, R²=0.9224 versus Charlotte 2.60 °C, 0.9051.
    # Higher R² can coexist with larger absolute error because Phoenix has
    # greater temperature variation. Its dry climate may also change relations.

    # Extension B: compare random held-out days across a three-year dataset.
    # Its test period differs, so a difference cannot be attributed only to size.
    longer = load_weather("charlotte", "2021-01-01", "2023-12-31")
    _, three_years, _, _ = evaluate(longer, FEATURES, "Charlotte three years")
    # Three years: R²=0.8724 and RMSE=3.06 °C, a modest worsening from 0.9051
    # and 2.60 °C. More data does not guarantee a better score on a different test.

    # Extension C: identical rows and splits, changing only the month encoding.
    df["month"] = df["date"].dt.month
    continuous = FEATURES[:3]
    _, numeric_month, _, _ = evaluate(df, continuous + ["month"], "Numeric month")
    month_columns = pd.get_dummies(df["month"], prefix="month", dtype=float)
    encoded = pd.concat([df, month_columns], axis=1)
    _, one_hot, _, _ = evaluate(encoded, continuous + list(month_columns), "One-hot month")
    # One-hot months perform better here: R²=0.9196, RMSE=2.39 °C, compared
    # with numeric month R²=0.9094, RMSE=2.54 °C. The improvement is small,
    # consistent with allowing a seasonal pattern instead of a linear trend.
    # A numeric month forces a straight-line January-to-December effect. One-hot
    # columns allow each month a different offset. With an intercept, all twelve
    # columns are redundant, so individual month coefficients are not unique;
    # predictions remain usable with scikit-learn's least-squares solution.
    pd.DataFrame([baseline, full, other_city, three_years, numeric_month, one_hot]).to_csv(
        OUTPUTS / "model_comparison.csv", index=False
    )


if __name__ == "__main__":
    main()
