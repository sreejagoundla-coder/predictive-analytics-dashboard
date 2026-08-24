import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Predictive Analytics Dashboard", page_icon="🔮", layout="wide")

# ---------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------
@st.cache_data
def load_default_data():
    df = pd.read_csv("sample_sales_data.csv", parse_dates=["Date"])
    return df

def load_uploaded_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.rename(columns={col: "Date"})
            break
    return df

st.sidebar.title("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    raw_df = load_uploaded_data(uploaded_file)
    st.sidebar.success(f"Loaded {len(raw_df)} rows from {uploaded_file.name}")
else:
    raw_df = load_default_data()
    st.sidebar.info("Using sample dataset (upload your own file to replace it)")

if "Date" not in raw_df.columns:
    st.error("Couldn't find a date column in this file. Please upload a file with a column containing 'date' in its name.")
    st.stop()

numeric_cols = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
if not numeric_cols:
    st.error("No numeric column found to forecast. Please upload a file with at least one numeric column (e.g. Revenue).")
    st.stop()

st.title("🔮 Predictive Analytics Dashboard")
st.caption("Forecast future trends from historical data using regression or time-series models")

# ---------------------------------------------------------------
# STEP 1: TARGET + AGGREGATION
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Forecast Setup")

default_target = "Revenue" if "Revenue" in numeric_cols else numeric_cols[0]
target_col = st.sidebar.selectbox("Column to forecast", numeric_cols, index=numeric_cols.index(default_target))

granularity = st.sidebar.selectbox("Aggregate by", ["Daily", "Weekly", "Monthly"], index=2)
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

model_choice = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Polynomial Regression", "Random Forest", "Holt-Winters (Time Series)"],
)

forecast_horizon = st.sidebar.slider("Periods to forecast into the future", min_value=3, max_value=24, value=6)
test_size_pct = st.sidebar.slider("Test set size (for accuracy evaluation)", min_value=10, max_value=40, value=20)

# ---------------------------------------------------------------
# STEP 2: CLEAN + PREPROCESS
# ---------------------------------------------------------------
clean_df = raw_df.dropna(subset=["Date", target_col]).copy()
clean_df = clean_df.drop_duplicates()
clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
clean_df = clean_df.dropna(subset=[target_col])

rows_removed = len(raw_df) - len(clean_df)

series = (
    clean_df.set_index("Date")[target_col]
    .resample(freq_map[granularity])
    .sum()
)
# Fill any gaps (periods with no data) with 0 rather than leaving NaNs
series = series.fillna(0)

with st.expander("🧹 Data Cleaning Summary"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", f"{len(raw_df):,}")
    c2.metric("Rows Removed (missing/invalid)", f"{rows_removed:,}")
    c3.metric("Time Periods After Aggregation", f"{len(series):,}")

MIN_PERIODS = 8
if len(series) < MIN_PERIODS:
    st.error(
        f"Not enough data after cleaning and aggregation ({len(series)} periods). "
        f"Need at least {MIN_PERIODS}. Try a finer granularity (e.g. Daily instead of Monthly) "
        f"or upload a larger dataset."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------
# STEP 3: TRAIN / TEST SPLIT (chronological, no shuffling)
# ---------------------------------------------------------------
n = len(series)
test_n = max(2, int(n * test_size_pct / 100))
test_n = min(test_n, n - 5)  # always leave at least 5 points to train on
train_series = series.iloc[: n - test_n]
test_series = series.iloc[n - test_n:]

x_train = np.arange(len(train_series)).reshape(-1, 1)
x_test = np.arange(len(train_series), len(train_series) + len(test_series)).reshape(-1, 1)
x_future = np.arange(len(series), len(series) + forecast_horizon).reshape(-1, 1)

y_train = train_series.values
y_test = test_series.values

# ---------------------------------------------------------------
# STEP 4: FIT MODEL + PREDICT
# ---------------------------------------------------------------
def fit_predict_regression(model, x_train, y_train, x_test, x_future, poly=None):
    if poly is not None:
        model.fit(poly.fit_transform(x_train), y_train)
        pred_test = model.predict(poly.transform(x_test))
        pred_future = model.predict(poly.transform(x_future))
    else:
        model.fit(x_train, y_train)
        pred_test = model.predict(x_test)
        pred_future = model.predict(x_future)
    return np.clip(pred_test, 0, None), np.clip(pred_future, 0, None)

model_error = None
try:
    if model_choice == "Linear Regression":
        model = LinearRegression()
        pred_test, pred_future = fit_predict_regression(model, x_train, y_train, x_test, x_future)

    elif model_choice == "Polynomial Regression":
        degree = 2 if len(train_series) < 30 else 3
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression()
        pred_test, pred_future = fit_predict_regression(model, x_train, y_train, x_test, x_future, poly=poly)

    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
        pred_test, pred_future = fit_predict_regression(model, x_train, y_train, x_test, x_future)

    elif model_choice == "Holt-Winters (Time Series)":
        seasonal_periods = 12 if granularity == "Monthly" else (7 if granularity == "Daily" else 4)
        use_seasonal = len(train_series) >= seasonal_periods * 2
        hw_model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        ).fit()
        pred_test = np.clip(hw_model.forecast(len(test_series)).values, 0, None)

        # Refit on full series for the true future forecast
        hw_full = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add" if len(series) >= seasonal_periods * 2 else None,
            seasonal_periods=seasonal_periods if len(series) >= seasonal_periods * 2 else None,
            initialization_method="estimated",
        ).fit()
        pred_future = np.clip(hw_full.forecast(forecast_horizon).values, 0, None)

except Exception as e:
    model_error = str(e)

if model_error:
    st.error(f"The model failed to fit on this data: {model_error}. Try a different model or a coarser/finer granularity.")
    st.stop()

# ---------------------------------------------------------------
# STEP 5: EVALUATE ACCURACY
# ---------------------------------------------------------------
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
nonzero_mask = y_test != 0
mape = (
    np.mean(np.abs((y_test[nonzero_mask] - pred_test[nonzero_mask]) / y_test[nonzero_mask])) * 100
    if nonzero_mask.any() else np.nan
)
r2 = r2_score(y_test, pred_test) if len(y_test) > 1 else np.nan

st.subheader("📈 Model Accuracy (on held-out test period)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{mae:,.2f}")
m2.metric("RMSE", f"{rmse:,.2f}")
m3.metric("MAPE", f"{mape:,.1f}%" if not np.isnan(mape) else "N/A")
m4.metric("R² Score", f"{r2:,.3f}" if not np.isnan(r2) else "N/A")
st.caption("Lower MAE/RMSE/MAPE = better. R² closer to 1 = better fit (can be negative for a poor model).")

st.divider()

# ---------------------------------------------------------------
# STEP 6: VISUALIZE — ACTUAL vs PREDICTED (test) + FUTURE FORECAST
# ---------------------------------------------------------------
st.subheader(f"Forecast: {target_col} ({granularity})")

future_index = pd.date_range(
    start=series.index[-1], periods=forecast_horizon + 1, freq=freq_map[granularity]
)[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_series.index, y=train_series.values, name="Training Data", line=dict(color="#4C78A8")))
fig.add_trace(go.Scatter(x=test_series.index, y=test_series.values, name="Actual (Test Period)", line=dict(color="#54A24B")))
fig.add_trace(go.Scatter(x=test_series.index, y=pred_test, name="Predicted (Test Period)", line=dict(color="#E45756", dash="dash")))
fig.add_trace(go.Scatter(x=future_index, y=pred_future, name=f"Forecast (Next {forecast_horizon})", line=dict(color="#F58518", dash="dot")))
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), xaxis_title="Date", yaxis_title=target_col)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# STEP 7: FORECAST TABLE + DOWNLOAD
# ---------------------------------------------------------------
with st.expander("📋 Forecast Table"):
    forecast_df = pd.DataFrame({"Date": future_index, f"Predicted {target_col}": pred_future.round(2)})
    st.dataframe(forecast_df, use_container_width=True)
    st.download_button(
        "Download forecast as CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        "forecast.csv",
        "text/csv",
    )

with st.expander("ℹ️ How this works"):
    st.markdown(f"""
    1. **Clean & preprocess**: rows with missing dates or invalid `{target_col}` values are dropped; data is aggregated to **{granularity.lower()}** totals.
    2. **Train/test split**: the most recent {test_size_pct}% of periods are held out as a test set (chronological split — no shuffling, since this is time-based data).
    3. **Model**: **{model_choice}** is trained on the earlier period and evaluated on the held-out test period.
    4. **Forecast**: the model is then used (refit on all available data for Holt-Winters) to project **{forecast_horizon}** periods into the future.
    """)
