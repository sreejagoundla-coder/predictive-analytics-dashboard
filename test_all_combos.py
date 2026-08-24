"""
Headless smoke test: replicates app.py's core logic for every
(model, granularity) combination to catch runtime errors before deployment.
Not part of the deployed app — dev-only.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

raw_df = pd.read_csv("sample_sales_data.csv", parse_dates=["Date"])
target_col = "Revenue"
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

models = ["Linear Regression", "Polynomial Regression", "Random Forest", "Holt-Winters (Time Series)"]
granularities = ["Daily", "Weekly", "Monthly"]

failures = []

for granularity in granularities:
    for model_choice in models:
        try:
            clean_df = raw_df.dropna(subset=["Date", target_col]).copy()
            clean_df = clean_df.drop_duplicates()
            clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[target_col])

            series = clean_df.set_index("Date")[target_col].resample(freq_map[granularity]).sum()
            series = series.fillna(0)

            n = len(series)
            test_n = max(2, int(n * 20 / 100))
            test_n = min(test_n, n - 5)
            train_series = series.iloc[: n - test_n]
            test_series = series.iloc[n - test_n:]

            x_train = np.arange(len(train_series)).reshape(-1, 1)
            x_test = np.arange(len(train_series), len(train_series) + len(test_series)).reshape(-1, 1)
            x_future = np.arange(len(series), len(series) + 6).reshape(-1, 1)
            y_train = train_series.values
            y_test = test_series.values

            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(x_train, y_train)
                pred_test = np.clip(model.predict(x_test), 0, None)
                pred_future = np.clip(model.predict(x_future), 0, None)

            elif model_choice == "Polynomial Regression":
                degree = 2 if len(train_series) < 30 else 3
                poly = PolynomialFeatures(degree=degree)
                model = LinearRegression()
                model.fit(poly.fit_transform(x_train), y_train)
                pred_test = np.clip(model.predict(poly.transform(x_test)), 0, None)
                pred_future = np.clip(model.predict(poly.transform(x_future)), 0, None)

            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
                model.fit(x_train, y_train)
                pred_test = np.clip(model.predict(x_test), 0, None)
                pred_future = np.clip(model.predict(x_future), 0, None)

            elif model_choice == "Holt-Winters (Time Series)":
                seasonal_periods = 12 if granularity == "Monthly" else (7 if granularity == "Daily" else 4)
                use_seasonal = len(train_series) >= seasonal_periods * 2
                hw_model = ExponentialSmoothing(
                    train_series, trend="add",
                    seasonal="add" if use_seasonal else None,
                    seasonal_periods=seasonal_periods if use_seasonal else None,
                    initialization_method="estimated",
                ).fit()
                pred_test = np.clip(hw_model.forecast(len(test_series)).values, 0, None)

                use_seasonal_full = len(series) >= seasonal_periods * 2
                hw_full = ExponentialSmoothing(
                    series, trend="add",
                    seasonal="add" if use_seasonal_full else None,
                    seasonal_periods=seasonal_periods if use_seasonal_full else None,
                    initialization_method="estimated",
                ).fit()
                pred_future = np.clip(hw_full.forecast(6).values, 0, None)

            mae = mean_absolute_error(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            r2 = r2_score(y_test, pred_test) if len(y_test) > 1 else float("nan")

            print(f"OK  | {granularity:8s} | {model_choice:28s} | n={n:4d} test_n={test_n:3d} | MAE={mae:10.2f} RMSE={rmse:10.2f} R2={r2:6.3f}")

        except Exception as e:
            failures.append((granularity, model_choice, str(e)))
            print(f"FAIL| {granularity:8s} | {model_choice:28s} | {e}")

print()
if failures:
    print(f"{len(failures)} combination(s) FAILED:")
    for g, m, e in failures:
        print(f"  - {g} / {m}: {e}")
    raise SystemExit(1)
else:
    print("All model x granularity combinations passed.")
