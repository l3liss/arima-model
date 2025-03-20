#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from arch import arch_model
from scipy.stats import norm

# ============================
# Phemex API Credentials (Embed your keys directly)
PHEMEX_API_KEY = "cc5c35f6-7ba1-477d-9099-fee0c8b796aa"
PHEMEX_API_SECRET = "5c0T6HsZPLAWXvjrTK2I7CYbhhS6JfVmLKof0L2S1BE2MmQ5YjdmNy04MDNjLTQ2MTYtYTdjYy00MGE3ZDExZjlkNmQ"
# ============================

# Confirmation thresholds
LONG_THRESHOLD = 0.60  # Above 60% → long confirmation
SHORT_THRESHOLD = 0.40 # Below 40% → short confirmation

def fetch_price_data(symbol="BTC-USD", interval="1h", period="60d"):
    """
    Try to fetch historical price data from Phemex first.
    If Phemex fails, fall back to Yahoo Finance.
    Returns a DataFrame with OHLCV data.
    """
    try:
        # Calculate limit from period and interval:
        days = int(period.strip('d'))
        if interval.endswith("h"):
            hours = int(interval.strip('h'))
            limit = (days * 24) // hours
        else:
            limit = 1000  # fallback value

        phemex_symbol = symbol.replace('-', '')
        url = f"https://api.phemex.com/md/kline?symbol={phemex_symbol}&resolution=3600&limit={limit}"
        headers = {"x-phemex-access-token": PHEMEX_API_KEY} if PHEMEX_API_KEY else {}
        resp = requests.get(url, headers=headers)
        try:
            json_data = resp.json() if resp and resp.content else None
        except Exception:
            json_data = None
        if json_data is None or "result" not in json_data:
            raise Exception("No valid JSON data returned from Phemex")
        data_json = json_data.get("result", {}).get("rows", [])
        if data_json and len(data_json) > 0:
            df = pd.DataFrame(data_json, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            print("Data source used for historical price data: Phemex API")
            return df
        else:
            raise Exception("No rows returned from Phemex")
    except Exception as e:
        print(f"Phemex data fetch failed: {e}", file=sys.stderr)
        try:
            print("Falling back to Yahoo Finance for historical price data...")
            data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
            if data.empty:
                raise Exception("No data returned from Yahoo Finance")
            print("Data source used for historical price data: Yahoo Finance")
            return data
        except Exception as e2:
            print(f"Yahoo Finance data fetch failed: {e2}", file=sys.stderr)
            sys.exit(1)

def forecast_garch_returns(returns, horizon=12):
    """
    Fit a GARCH(1,1) model on returns and forecast aggregated mean and variance over a given horizon.
    Assumes returns are in percentage.
    """
    am = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=horizon)
    mu = res.params['mu']  # constant mean
    agg_mean = horizon * mu
    agg_var = forecasts.variance.iloc[-1].sum()
    return agg_mean, agg_var

def calculate_probabilities(agg_mean, agg_var):
    """
    Calculate the probability that the cumulative return over the horizon is > 0.
    """
    sigma_total = np.sqrt(agg_var)
    p_up = 1 - norm.cdf(0, loc=agg_mean, scale=sigma_total)
    return p_up

def estimate_price_range(current_price, agg_mean, agg_var):
    """
    Estimate the 68% confidence range for the next period's price.
    """
    sigma_total = np.sqrt(agg_var)
    lower_return = agg_mean - sigma_total
    upper_return = agg_mean + sigma_total
    price_low = current_price * np.exp(lower_return / 100)  # convert from percentage
    price_high = current_price * np.exp(upper_return / 100)
    return price_low, price_high

def fetch_order_flow(symbol="BTCUSDT"):
    """
    Fetch order book data from Phemex first; if that fails, use Binance.
    Returns the order flow imbalance as a number between -1 and 1.
    """
    # Try Phemex
    try:
        url = f"https://api.phemex.com/md/orderbook?symbol={symbol}"
        headers = {"x-phemex-access-token": PHEMEX_API_KEY} if PHEMEX_API_KEY else {}
        resp = requests.get(url, headers=headers)
        try:
            json_data = resp.json() if resp and resp.content else None
        except Exception:
            json_data = None
        if json_data is None or "result" not in json_data:
            raise Exception("No valid JSON data from Phemex order book")
        data = json_data.get("result", {}).get("book", {})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            raise Exception("Empty bids or asks from Phemex")
        bid_vol = sum([float(level[1]) for level in bids[:10]])
        ask_vol = sum([float(level[1]) for level in asks[:10]])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        print("Order flow data source: Phemex API")
        return imbalance
    except Exception as e:
        print(f"Phemex order flow fetch failed: {e}", file=sys.stderr)
        # Fallback: Try Binance order book endpoint
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10"
            resp = requests.get(url)
            json_data = resp.json() if resp and resp.content else None
            if json_data is None or "bids" not in json_data:
                raise Exception("No valid JSON data from Binance")
            bids = json_data.get("bids", [])
            asks = json_data.get("asks", [])
            if not bids or not asks:
                raise Exception("Empty bids or asks from Binance")
            bid_vol = sum([float(bid[1]) for bid in bids])
            ask_vol = sum([float(ask[1]) for ask in asks])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            print("Order flow data source: Binance API")
            return imbalance
        except Exception as e2:
            print(f"Binance order flow fetch failed: {e2}", file=sys.stderr)
            return None

def main(args):
    # 1. Fetch price data
    df = fetch_price_data(symbol=args.symbol, interval=args.interval, period=args.period)
    if df.empty:
        print("No data available. Exiting.")
        sys.exit(1)
    
    # 2. Compute log returns (in percentage)
    df = df.dropna()
    prices = df['Close']
    log_returns = np.log(prices).diff().dropna() * 100

    # 3. Fit GARCH(1,1) and forecast for the next horizon
    agg_mean, agg_var = forecast_garch_returns(log_returns, horizon=args.horizon)
    p_up = calculate_probabilities(agg_mean, agg_var)
    
    # Convert current price to float using .item() to avoid warnings.
    current_price = prices.iloc[-1].item()
    price_low, price_high = estimate_price_range(current_price, agg_mean, agg_var)
    
    # 4. Order Flow Analysis (optional)
    if args.use_orderflow:
        imbalance = fetch_order_flow(symbol=args.orderbook_symbol)
    else:
        imbalance = None

    # 5. Print results
    print("\n==== BTC Confirmation Tool ====")
    print(f"Current Price: ${current_price:.4f}")
    print(f"Forecast Horizon: {args.horizon} hours")
    print(f"Aggregated Predicted Return (next {args.horizon}h): {agg_mean:.4f}%")
    print(f"Aggregated Predicted Volatility (std dev): {np.sqrt(agg_var):.4f}%")
    print(f"Probability of Uptrend: {p_up*100:.1f}%")
    print(f"Estimated Price Range (68% confidence): ${price_low:.4f} - ${price_high:.4f}")
    if p_up > LONG_THRESHOLD:
        print("Confirmation: Long signal valid (Probability > 60%).")
    elif p_up < SHORT_THRESHOLD:
        print("Confirmation: Short signal valid (Probability < 40%).")
    else:
        print("Confirmation: Signal ambiguous. Consider holding or closing your position early.")
    
    if imbalance is not None:
        if imbalance > 0.2:
            print(f"Order Flow: Buy-side imbalance (+{imbalance*100:.0f}%).")
        elif imbalance < -0.2:
            print(f"Order Flow: Sell-side imbalance ({imbalance*100:.0f}%).")
        else:
            print(f"Order Flow: Balanced ({imbalance*100:.0f}%).")
    else:
        print("Order Flow: Data not available.")
    print("================================\n")
    
    # 6. Optional Backtesting (placeholder)
    if args.backtest:
        threshold = 0.55  # example threshold
        signals_taken = 1 if p_up > threshold else 0
        print(f"Backtest: With threshold {threshold*100:.0f}% up probability, {signals_taken} signal(s) would have been taken (dummy value).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 12-hour Direction Confirmation Tool")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Symbol to fetch (default: BTC-USD)")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (default: 1h)")
    parser.add_argument("--period", type=str, default="60d", help="Historical period to fetch (default: 60d)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in hours (default: 12)")
    parser.add_argument("--use-orderflow", action="store_true", help="Use order flow analysis from Phemex (fallback to Binance if needed)")
    parser.add_argument("--orderbook-symbol", type=str, default="BTCUSDT", help="Symbol for order flow analysis (default: BTCUSDT)")
    parser.add_argument("--backtest", action="store_true", help="Run a simple backtest simulation")
    
    args = parser.parse_args()
    main(args)
