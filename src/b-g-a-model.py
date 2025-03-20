#!/usr/bin/env python3
"""
Unified model.py
----------------
This script demonstrates how to combine two approaches:
  - ARIMA for forecasting the log-price trajectory (mean forecast)
  - GARCH for forecasting volatility/return direction

Optionally, it attempts to use a Boostdog module (XGBoost classifier for technical indicators)
to generate an additional trade signal. If Boostdog is not available or fails,
the script prints a message and continues with ARIMA and GARCH only.

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
# Multi-Timeframe Data Functions (from boostdog.py)
# ---------------------------
def fetch_hourly_data(symbol="BTC-USD", period="60d"):
    df = yf.download(symbol, period=period, interval="1h", auto_adjust=True)
    df = ensure_required_columns(df)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    return df.dropna()

def fetch_4h_data(symbol="BTC-USD", period="60d"):
    df = yf.download(symbol, period=period, interval="1h", auto_adjust=True)
    df = ensure_required_columns(df)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    df_4h = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    return df_4h.dropna()

def fetch_daily_data(symbol="BTC-USD", period="90d"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    df = ensure_required_columns(df)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    return df.dropna()

def format_columns(df):
    if isinstance(df.columns[0], tuple):
        df.columns = [" ".join(col).strip() for col in df.columns]
    else:
        df.columns = df.columns.astype(str)
    df.columns = [col.split()[0].capitalize() for col in df.columns]
    return df

def ensure_required_columns(df):
    df = format_columns(df)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if "Close" not in df.columns:
        for col in df.columns:
            if col.lower().startswith("adj"):
                df = df.rename(columns={col: "Close"})
                break
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise Exception(f"Required columns {missing} still missing.")
    return df

def merge_multitimeframe_data(df_hourly, df_4h, df_daily):
    df_hourly_raw = df_hourly[['Close']].copy().rename(columns={"Close": "Raw_Close"})
    raw_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_hourly_feat = compute_features(df_hourly, suffix="_hourly").drop(columns=raw_cols, errors="ignore")
    df_4h_feat = compute_features(df_4h, suffix="_4h").drop(columns=raw_cols, errors="ignore")
    df_daily_feat = compute_features(df_daily, suffix="_daily").drop(columns=raw_cols, errors="ignore")
    df_4h_resampled = df_4h_feat.resample("1h").ffill().drop(columns=raw_cols, errors="ignore")
    df_daily_resampled = df_daily_feat.resample("1h").ffill().drop(columns=raw_cols, errors="ignore")
    df_hourly_raw.index.name = "Datetime"
    df_4h_resampled.index.name = "Datetime"
    df_daily_resampled.index.name = "Datetime"
    df_merged = df_hourly_raw.join(df_hourly_feat, how="left")
    df_merged = df_merged.join(df_4h_resampled, how="left")
    df_merged = df_merged.join(df_daily_resampled, how="left")
    df_merged.fillna(method="ffill", inplace=True)
    return df_merged.dropna()

# ---------------------------
# Ensemble Signal Function
# ---------------------------
def ensemble_signal(signal_arima, signal_garch, signal_boost=None):
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
    current_price = df['Close'].iloc[-1].item()
    print(f"Current BTC Price: ${current_price:.2f}")
    
    log_prices = np.log(df['Close'])
    log_returns = log_prices.diff().dropna() * 100
    
    print("\nFitting ARIMA model on log-prices...")
    forecast_days = 7
    forecasted_log_price, arima_order, arima_model = forecast_arima(log_prices.dropna(), forecast_days=forecast_days)
    forecasted_price = np.exp(forecasted_log_price)
    current_log_price = float(log_prices.iloc[-1])
    signal_arima = determine_arima_signal(current_log_price, forecasted_log_price)
    print(f"ARIMA model order: {arima_order}")
    print(f"Forecasted BTC Price in {forecast_days} days: ${forecasted_price:.2f}")
    print(f"ARIMA Signal: {signal_arima}")
    
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
    
    signal_boost = None
    if boostdog_available:
        print("\nAttempting to fetch multi-timeframe data for Boostdog (XGBoost)...")
        try:
            df_hourly = fetch_hourly_data(symbol="BTC-USD", period="60d")
            df_4h = fetch_4h_data(symbol="BTC-USD", period="60d")
            df_daily = fetch_daily_data(symbol="BTC-USD", period="90d")
            features_df = merge_multitimeframe_data(df_hourly, df_4h, df_daily)
            expected_features = [
                'SMA_10_hourly', 'SMA_20_hourly', 'SMA_50_hourly',
                'EMA_10_hourly', 'EMA_20_hourly', 'EMA_50_hourly',
                'Volatility_hourly', 'Momentum_hourly', 'RSI_hourly',
                'MACD_hourly', 'BB_width_hourly',
                'SMA_10_4h', 'SMA_20_4h', 'SMA_50_4h',
                'EMA_10_4h', 'EMA_20_4h', 'EMA_50_4h',
                'Volatility_4h', 'Momentum_4h', 'RSI_4h',
                'MACD_4h', 'BB_width_4h',
                'SMA_10_daily', 'SMA_20_daily', 'SMA_50_daily',
                'EMA_10_daily', 'EMA_20_daily', 'EMA_50_daily',
                'Volatility_daily', 'Momentum_daily', 'RSI_daily',
                'MACD_daily', 'BB_width_daily'
            ]
            missing = [col for col in expected_features if col not in features_df.columns]
            if missing:
                raise Exception(f"Missing expected features: {missing}")
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
        print("Boostdog module not available. Skipping Boostdog signal.")
    
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
