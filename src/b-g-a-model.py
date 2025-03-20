#!/usr/bin/env python3
"""
Unified model.py
----------------
This script demonstrates how to combine two approaches:
  - ARIMA for forecasting the log-price trajectory (mean forecast)
  - GARCH for forecasting volatility/return direction

Optionally, it attempts to use a Boostdog module (XGBoost classifier for technical indicators)
to generate an additional trade signal. If Boostdog is not available or fails, the script
prints a message and continues with ARIMA and GARCH only.

The idea is to forecast BTC-USD daily data for a one-week horizon.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

import pmdarima as pm
from arch import arch_model
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------
# Attempt to import boostdog (optional)
# ---------------------------
boostdog_available = True
try:
    from boostdog import compute_features, train_xgboost, predict_signal
except Exception as e:
    print("Boostdog module not available or failed to import. Proceeding without XGBoost signal.")
    boostdog_available = False

# ---------------------------
# ARIMA functions
# ---------------------------
def fetch_btc_data(period="730d", interval="1d"):
    symbol = "BTC-USD"
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    if df.empty:
        print("Error: No data fetched from Yahoo Finance.")
        sys.exit(1)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    return df.dropna()

def forecast_arima(log_series, forecast_days=7):
    model = pm.auto_arima(
        log_series,
        start_p=0, start_q=0,
        test='adf',
        max_p=5, max_q=5,
        d=None, seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    order = model.order  # (p, d, q)
    forecast = model.predict(n_periods=forecast_days)
    forecasted_log_price = forecast[-1]
    return forecasted_log_price, order, model

def determine_arima_signal(current_log_price, forecasted_log_price):
    """
    Compare current log price with forecasted log price.
    Ensure current_log_price is a scalar.
    """
    # If current_log_price is a Series or array, convert it to float
    if isinstance(current_log_price, (pd.Series, np.ndarray)):
        current_log_price = float(current_log_price.iloc[0] if isinstance(current_log_price, pd.Series) else current_log_price[0])
    return "Long" if forecasted_log_price > current_log_price else "Short"

# ---------------------------
# GARCH functions (from garchdog.py)
# ---------------------------
def forecast_garch_returns(returns, horizon=7):
    am = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=horizon)
    mu = res.params['mu']
    agg_mean = horizon * mu
    agg_var = forecasts.variance.iloc[-1].sum()
    return agg_mean, agg_var

def calculate_probabilities(agg_mean, agg_var):
    sigma_total = np.sqrt(agg_var)
    p_up = 1 - norm.cdf(0, loc=agg_mean, scale=sigma_total)
    return p_up

def estimate_price_range(current_price, agg_mean, agg_var):
    sigma_total = np.sqrt(agg_var)
    lower_return = agg_mean - sigma_total
    upper_return = agg_mean + sigma_total
    price_low = current_price * np.exp(lower_return / 100)
    price_high = current_price * np.exp(upper_return / 100)
    return price_low, price_high

# ---------------------------
# Ensemble Signal Function
# ---------------------------
def ensemble_signal(signal_arima, signal_garch, signal_boost=None):
    """
    If boostdog signal is available, use a simple majority vote of three.
    Otherwise, use ARIMA and GARCH:
      - If they agree, that is the overall signal.
      - If they disagree, return "Neutral".
    """
    if signal_boost is not None:
        signals = [signal_arima, signal_garch, signal_boost]
        long_count = signals.count("Long")
        short_count = signals.count("Short")
        if long_count >= 2:
            return "Long"
        elif short_count >= 2:
            return "Short"
        else:
            return "Neutral"
    else:
        if signal_arima == signal_garch:
            return signal_arima
        else:
            return "Neutral"

# ---------------------------
# Main Routine
# ---------------------------
def main():
    print("Fetching BTC historical daily data...")
    df = fetch_btc_data(period="730d", interval="1d")
    # Convert current price to a scalar
    current_price = df['Close'].iloc[-1].item()
    print(f"Current BTC Price: ${current_price:.2f}")
    
    # Prepare data for ARIMA and GARCH
    log_prices = np.log(df['Close'])
    log_returns = log_prices.diff().dropna() * 100
    
    # ARIMA Forecast
    print("\nFitting ARIMA model on log-prices...")
    forecast_days = 7  # Forecast one week ahead
    forecasted_log_price, arima_order, arima_model = forecast_arima(log_prices.dropna(), forecast_days=forecast_days)
    forecasted_price = np.exp(forecasted_log_price)
    # Convert the last log price to scalar before comparing
    current_log_price = float(log_prices.iloc[-1])
    signal_arima = determine_arima_signal(current_log_price, forecasted_log_price)
    print(f"ARIMA model order: {arima_order}")
    print(f"Forecasted BTC Price in {forecast_days} days: ${forecasted_price:.2f}")
    print(f"ARIMA Signal: {signal_arima}")
    
    # GARCH Forecast
    print("\nFitting GARCH model on log-returns...")
    agg_mean, agg_var = forecast_garch_returns(log_returns, horizon=forecast_days)
    p_up = calculate_probabilities(agg_mean, agg_var)
    LONG_THRESHOLD = 0.60
    SHORT_THRESHOLD = 0.40
    if p_up > LONG_THRESHOLD:
        signal_garch = "Long"
    elif p_up < SHORT_THRESHOLD:
        signal_garch = "Short"
    else:
        signal_garch = "Neutral"
    print(f"GARCH Aggregated Predicted Return: {agg_mean:.4f}% over {forecast_days} days")
    print(f"GARCH Predicted Volatility (std dev): {np.sqrt(agg_var):.4f}%")
    print(f"Probability of Uptrend (GARCH): {p_up*100:.1f}%")
    print(f"GARCH Signal: {signal_garch}")
    
    # Optional: Boostdog Signal (if available)
    if boostdog_available:
        print("\nAttempting to fetch multi-timeframe data for Boostdog (XGBoost)...")
        try:
            df_daily = yf.download("BTC-USD", period="180d", interval="1d", auto_adjust=True)
            df_daily.index = df_daily.index.tz_localize(None)
            df_daily.index.name = "Datetime"
            from boostdog import compute_features
            features_df = compute_features(df_daily, suffix="_daily")
            latest_features = features_df.iloc[-1:]
            import joblib
            xgb_model = joblib.load("xgboost_btc_model.pkl")
            scaler = joblib.load("scaler.pkl")
            signal_boost = predict_signal(latest_features, model=xgb_model, scaler=scaler)
            print(f"Boostdog Signal: {signal_boost}")
        except Exception as e:
            print("Boostdog signal generation failed:", e)
            signal_boost = None
    else:
        signal_boost = None
    
    # Ensemble Signal
    overall_signal = ensemble_signal(signal_arima, signal_garch, signal_boost)
    print("\n==== Combined Weekly Signal ====")
    print(f"ARIMA Signal: {signal_arima}")
    print(f"GARCH Signal: {signal_garch}")
    if signal_boost is not None:
        print(f"Boostdog Signal: {signal_boost}")
    else:
        print("Boostdog Signal: N/A")
    print(f"Overall Ensemble Signal: {overall_signal}")
    print("================================")
    
    # Save results
    results = {
        "current_price": current_price,
        "forecasted_price": forecasted_price,
        "arima_order": arima_order,
        "garch_p_up": p_up,
        "signal_arima": signal_arima,
        "signal_garch": signal_garch,
        "signal_boost": signal_boost if signal_boost is not None else "N/A",
        "overall_signal": overall_signal,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open("weekly_forecast.json", "w") as fp:
        json.dump(results, fp, indent=4)
    print("\nForecast results saved to weekly_forecast.json")

if __name__ == "__main__":
    main()

