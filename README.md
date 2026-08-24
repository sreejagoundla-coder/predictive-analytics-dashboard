# 🔮 Predictive Analytics Using Historical Data

A Streamlit app that cleans historical sales data and forecasts future trends using regression and time-series models — built as part of a data analytics internship task.

## Features

- **Data import**: upload your own CSV/Excel file, or use the bundled sample dataset
- **Cleaning & preprocessing**: drops missing/invalid rows, removes duplicates, aggregates to daily/weekly/monthly totals, fills time gaps
- **Models**:
  - Linear Regression
  - Polynomial Regression
  - Random Forest Regressor
  - Holt-Winters Exponential Smoothing (time-series, trend + seasonality)
- **Accuracy evaluation**: MAE, RMSE, MAPE, R² computed on a chronological (non-shuffled) held-out test period
- **Visualization**: training data, actual vs. predicted test period, and future forecast plotted together
- **Forecast export**: download the future forecast as CSV

## Tech Stack

Python, Streamlit, Pandas, NumPy, scikit-learn, statsmodels, Plotly

## Project Structure


##
predictive-analytics/
├── app.py                  # Main Streamlit app
├── test_all_combos.py      # Dev-only smoke test (all model x granularity combos)
├── sample_sales_data.csv   # Sample dataset (3000 orders, Jan 2025–Jul 2026)
├── requirements.txt        # Pinned dependency versions
└── READ.ME


##LIVE DEMO
