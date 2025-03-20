#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress XGBoost user warnings
warnings.filterwarnings("ignore", category=UserWarning)

##############################
# Functions to format and ensure required columns exist
##############################
def format_columns(df):
    """
    If columns are a MultiIndex, flatten them and convert columns to strings.
    Then, split column names on whitespace and take the first part,
    and finally capitalize them.
    """
    if isinstance(df.columns[0], tuple):
        df.columns = [" ".join(col).strip() for col in df.columns]
    else:
        df.columns = df.columns.astype(str)
    # Split on whitespace and take the first word, then capitalize
    df.columns = [col.split()[0].capitalize() for col in df.columns]
    return df

def ensure_required_columns(df):
    """
    Ensure the DataFrame has columns: Open, High, Low, Close, Volume.
    If "Close" is missing but a column starting with "adj" exists, rename it to "Close".
    """
    df = format_columns(df)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if "Close" not in df.columns:
        # Look for a column starting with 'adj' (case-insensitive)
        for col in df.columns:
            if col.lower().startswith("adj"):
                df = df.rename(columns={col: "Close"})
                break
    missing = [col for col in required if col not in df.columns]
    if missing:
        print("DEBUG: DataFrame columns:", df.columns.tolist())
        raise Exception(f"Required columns {missing} still missing.")
    return df

##############################
# Data Fetching Functions for Multi-Timeframe Data
##############################
def fetch_hourly_data(symbol="DOGE-USD", period="60d"):
    df = yf.download(symbol, period=period, interval="1h", auto_adjust=True)
    df = ensure_required_columns(df)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    return df.dropna()

def fetch_4h_data(symbol="DOGE-USD", period="60d"):
    # Download hourly data and resample to 4h
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

def fetch_daily_data(symbol="DOGE-USD", period="90d"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    df = ensure_required_columns(df)
    df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"
    return df.dropna()

##############################
# Feature Computation Function
##############################
def compute_features(df, suffix=""):
    """
    Given a DataFrame with OHLC data (columns: Open, High, Low, Close, Volume)
    and an index named 'Datetime', compute technical indicators:
      - SMA (10, 20, 50)
      - EMA (10, 20, 50)
      - Volatility: rolling 10-period std of returns
      - Momentum: current Close minus Close 10 periods ago
      - RSI: 14-period
      - MACD: EMA_12 - EMA_26
      - Bollinger Band Width: (4 * rolling std (20)) / SMA_20
    All computed columns are suffixed with the given suffix.
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    
    df[f'SMA_10{suffix}'] = df['Close'].rolling(window=10).mean()
    df[f'SMA_20{suffix}'] = df['Close'].rolling(window=20).mean()
    df[f'SMA_50{suffix}'] = df['Close'].rolling(window=50).mean()
    
    df[f'EMA_10{suffix}'] = df['Close'].ewm(span=10, adjust=False).mean()
    df[f'EMA_20{suffix}'] = df['Close'].ewm(span=20, adjust=False).mean()
    df[f'EMA_50{suffix}'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    df[f'Volatility{suffix}'] = df['Return'].rolling(window=10).std()
    df[f'Momentum{suffix}'] = df['Close'] - df['Close'].shift(10)
    
    # RSI Calculation (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df[f'RSI{suffix}'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation: EMA_12 - EMA_26
    df[f'EMA_12{suffix}'] = df['Close'].ewm(span=12, adjust=False).mean()
    df[f'EMA_26{suffix}'] = df['Close'].ewm(span=26, adjust=False).mean()
    df[f'MACD{suffix}'] = df[f'EMA_12{suffix}'] - df[f'EMA_26{suffix}']
    
    # Bollinger Band Width: (4 * rolling std (20)) / SMA_20
    std20 = df['Close'].rolling(window=20).std()
    sma20 = df['Close'].rolling(window=20).mean()
    df[f'BB_width{suffix}'] = (4 * std20) / sma20
    
    df = df.drop(columns=['Return', f'EMA_12{suffix}', f'EMA_26{suffix}'], errors='ignore')
    return df.dropna()

##############################
# Multi-Timeframe Data Fetching & Merging
##############################
def fetch_multitimeframe_data():
    print("Fetching hourly data...")
    df_hourly = fetch_hourly_data(symbol="DOGE-USD", period="60d")
    print("Fetching 4-hour data...")
    df_4h = fetch_4h_data(symbol="DOGE-USD", period="60d")
    print("Fetching daily data...")
    df_daily = fetch_daily_data(symbol="DOGE-USD", period="90d")
    return df_hourly, df_4h, df_daily

def merge_multitimeframe_data(df_hourly, df_4h, df_daily):
    # Save raw hourly "Close" for target calculation, rename to Raw_Close
    df_hourly_raw = df_hourly[['Close']].copy().rename(columns={"Close": "Raw_Close"})
    
    # Compute features and drop raw OHLCV columns
    raw_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_hourly_feat = compute_features(df_hourly, suffix="_hourly").drop(columns=raw_cols, errors="ignore")
    df_4h_feat = compute_features(df_4h, suffix="_4h").drop(columns=raw_cols, errors="ignore")
    df_daily_feat = compute_features(df_daily, suffix="_daily").drop(columns=raw_cols, errors="ignore")
    
    # Resample 4h and daily features to hourly frequency using forward fill
    df_4h_resampled = df_4h_feat.resample("1h").ffill().drop(columns=raw_cols, errors="ignore")
    df_daily_resampled = df_daily_feat.resample("1h").ffill().drop(columns=raw_cols, errors="ignore")
    
    # Ensure index names match
    df_hourly_raw.index.name = "Datetime"
    df_4h_resampled.index.name = "Datetime"
    df_daily_resampled.index.name = "Datetime"
    
    # Merge raw hourly data with computed features from hourly, 4h, and daily
    df_merged = df_hourly_raw.join(df_hourly_feat, how="left")
    df_merged = df_merged.join(df_4h_resampled, how="left")
    df_merged = df_merged.join(df_daily_resampled, how="left")
    df_merged.fillna(method="ffill", inplace=True)
    return df_merged.dropna()

##############################
# Model Training with XGBoost & Hyperparameter Tuning
##############################
def train_xgboost(df):
    # Define features from computed multi-timeframe data
    hourly_features = ['SMA_10_hourly', 'SMA_20_hourly', 'SMA_50_hourly',
                       'EMA_10_hourly', 'EMA_20_hourly', 'EMA_50_hourly',
                       'Volatility_hourly', 'Momentum_hourly', 'RSI_hourly',
                       'MACD_hourly', 'BB_width_hourly']
    fourh_features = ['SMA_10_4h', 'SMA_20_4h', 'SMA_50_4h',
                      'EMA_10_4h', 'EMA_20_4h', 'EMA_50_4h',
                      'Volatility_4h', 'Momentum_4h', 'RSI_4h',
                      'MACD_4h', 'BB_width_4h']
    daily_features = ['SMA_10_daily', 'SMA_20_daily', 'SMA_50_daily',
                      'EMA_10_daily', 'EMA_20_daily', 'EMA_50_daily',
                      'Volatility_daily', 'Momentum_daily', 'RSI_daily',
                      'MACD_daily', 'BB_width_daily']
    features = hourly_features + fourh_features + daily_features
    
    # Define target: predict if the next hourly return (using Raw_Close) > 0
    df['Direction'] = (df['Raw_Close'].pct_change().shift(-1) > 0).astype(int)
    df = df.dropna()
    
    X = df[features]
    y = df['Direction']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_index = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(XGBClassifier(eval_metric='logloss'),
                        param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Best Parameters:", grid.best_params_)
    print("Best Cross-Validation Accuracy: {:.2%}".format(grid.best_score_))
    
    return best_model, scaler, accuracy

def predict_signal(latest_data, model, scaler):
    features = []
    hourly_features = ['SMA_10_hourly', 'SMA_20_hourly', 'SMA_50_hourly',
                       'EMA_10_hourly', 'EMA_20_hourly', 'EMA_50_hourly',
                       'Volatility_hourly', 'Momentum_hourly', 'RSI_hourly',
                       'MACD_hourly', 'BB_width_hourly']
    fourh_features = ['SMA_10_4h', 'SMA_20_4h', 'SMA_50_4h',
                      'EMA_10_4h', 'EMA_20_4h', 'EMA_50_4h',
                      'Volatility_4h', 'Momentum_4h', 'RSI_4h',
                      'MACD_4h', 'BB_width_4h']
    daily_features = ['SMA_10_daily', 'SMA_20_daily', 'SMA_50_daily',
                      'EMA_10_daily', 'EMA_20_daily', 'EMA_50_daily',
                      'Volatility_daily', 'Momentum_daily', 'RSI_daily',
                      'MACD_daily', 'BB_width_daily']
    features.extend(hourly_features)
    features.extend(fourh_features)
    features.extend(daily_features)
    
    X_latest = latest_data[features].values.reshape(1, -1)
    X_latest_scaled = scaler.transform(X_latest)
    prediction = model.predict(X_latest_scaled)[0]
    return "Long" if prediction == 1 else "Short"

def main():
    print("Fetching multi-timeframe DOGE/USDT historical data...")
    df_hourly, df_4h, df_daily = fetch_multitimeframe_data()
    
    print("Computing hourly features...")
    df_hourly_feat = compute_features(df_hourly, suffix="_hourly")
    print("Computing 4-hour features...")
    df_4h_feat = compute_features(df_4h, suffix="_4h")
    print("Computing daily features...")
    df_daily_feat = compute_features(df_daily, suffix="_daily")
    
    # Resample 4h and daily features to hourly frequency and drop any raw columns if present
    df_4h_resampled = df_4h_feat.resample("1h").ffill().drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore")
    df_daily_resampled = df_daily_feat.resample("1h").ffill().drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore")
    
    # Ensure index names match
    df_hourly_feat.index.name = "Datetime"
    df_4h_resampled.index.name = "Datetime"
    df_daily_resampled.index.name = "Datetime"
    
    # Merge raw hourly Close with computed features from hourly, 4h, and daily
    df_hourly_raw = df_hourly[['Close']].copy().rename(columns={"Close": "Raw_Close"})
    df_merged = df_hourly_raw.join(df_hourly_feat.drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore"), how="left")
    df_merged = df_merged.join(df_4h_resampled, how="left")
    df_merged = df_merged.join(df_daily_resampled, how="left")
    df_merged.ffill(inplace=True)
    df_merged = df_merged.dropna()
    
    print("Training XGBoost model with hyperparameter tuning (multi-timeframe features)...")
    best_model, scaler, accuracy = train_xgboost(df_merged)
    print(f"XGBoost Model Accuracy on Test Set: {accuracy:.2%}")
    
    # Save model and scaler for future predictions
    joblib.dump(best_model, "xgboost_doge_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Predict trade signal from the latest merged data point
    latest_data = df_merged.iloc[-1:]
    trade_signal = predict_signal(latest_data, best_model, scaler)
    print(f"Predicted Trade Signal: {trade_signal}")

if __name__ == "__main__":
    main()

