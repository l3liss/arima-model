# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:25:58 2022

Updated: 2025-03-20

This script fetches cryptocurrency historical data, cleans it,
applies a variance-stabilizing log transformation, tests for stationarity,
and fits an ARIMA model using auto_arima.
"""

import os
import requests
import json
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

def get_new_prices(historical_data):
    try:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%Y-%m-%d')
    except ValueError:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%Y-%m-%d')

    max_date = historical_data['Date'].max()
    today_date = pd.to_datetime("today")
    date_difference = (today_date - max_date).days
    if date_difference > 0:
        chosen_currency = historical_data['Currency'].iloc[0]
        get_historical_prices(chosen_currency, date_difference, False)
    else:
        print("No new data needed.")

def get_historical_prices(chosen_currency, num_days, first_parse):
    # Limit the number of days to 365 for free API users.
    if num_days > 365:
        print("Limiting the number of days to 365 for free API usage.")
        num_days = 365
        
    url = f'https://api.coingecko.com/api/v3/coins/{chosen_currency}/market_chart?vs_currency=usd&days={num_days}&interval=daily'
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching data:", response.status_code, response.text)
        return
    
    hist_dict = response.json()
    
    if 'prices' not in hist_dict:
        print("Key 'prices' not found in the API response. Response:", hist_dict)
        return
    
    data = pd.DataFrame.from_dict(hist_dict['prices'])
    data.rename(columns={0: 'Date', 1: 'Price(USD)'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')
    data['Date'] = data['Date'].dt.date
    data['Currency'] = chosen_currency
    
    if not first_parse:
        data.to_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv', mode='a', header=False, index=False)
    else:
        data.to_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv', index=False)

def remove_duplicate_data(df):
    df.drop_duplicates('Date', keep='last', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def clean(df):
    df = remove_duplicate_data(df)
    return df

def adf_test(target_series):
    result = adfuller(target_series)
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic: %f' % adf_statistic)
    print('p-value: %f' % p_value)
    return adf_statistic, p_value

def kpss_test(target_series):
    print("Results of KPSS Test:")
    kpss_result = kpss(target_series, regression="ct", nlags="auto")
    kpss_output = pd.Series(kpss_result[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in kpss_result[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def find_order_of_differencing(df):
    # Apply log transformation to stabilize variance
    log_series = np.log(df['Price(USD)'])
    print("Testing stationarity on log-transformed data:")
    adf_stat, p_val = adf_test(log_series)
    kpss_test(log_series)
    
    d = 0
    # Keep differencing until the ADF p-value is below 0.05
    while p_val > 0.05:
        log_series = log_series.diff().dropna()
        d += 1
        print(f"\nAfter differencing {d} time(s):")
        adf_stat, p_val = adf_test(log_series)
        kpss_test(log_series)
    print(f"\nSuccess... The log-transformed series is stationary after {d} differencing(s).")
    return d

def create_acf_pacf(series):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    plot_acf(series, lags=20, ax=ax)
    plt.title("ACF Plot")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    plot_pacf(series, lags=20, ax=ax)
    plt.title("PACF Plot")
    plt.show()

def eda(df):
    d = find_order_of_differencing(df)
    # For visualization: compute the differenced log series
    log_series = np.log(df['Price(USD)'])
    diff_series = log_series.diff(d).dropna()
    plt.figure(figsize=(12, 8))
    plt.plot(diff_series)
    plt.title(f"Log-transformed Price Series Differenced {d} time(s)")
    plt.xlabel("Time")
    plt.ylabel("Differenced Log(Price)")
    plt.show()
    create_acf_pacf(diff_series)

def auto_arima_model(df):
    # Use the log-transformed data for model fitting
    log_series = np.log(df['Price(USD)']).dropna()
    model = pm.auto_arima(
        log_series,
        start_p=0, start_q=0,
        test='adf',
        max_p=5, max_q=5,
        m=1, seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    d = model.order[1]
    differenced_series = log_series.diff(d).dropna()
    return model.order, differenced_series, model

def main():
    chosen_currency = 'bitcoin'
    # Attempt to read existing historical data
    try:
        historical_data = pd.read_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv')
    except FileNotFoundError:
        historical_data = pd.DataFrame()

    if len(historical_data) > 0:
        get_new_prices(historical_data)
    else:
        get_historical_prices(chosen_currency, 3650, True)

    # Read and clean data
    df = pd.read_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv')
    df = clean(df)
    df.to_csv(f'../dat/clean/cleaned_{chosen_currency}_daily_historical.csv', index=False)

    # Perform exploratory data analysis on log-transformed data
    eda(df)

    # Fit an ARIMA model using auto_arima on log-transformed data
    order, diff_series, model = auto_arima_model(df)
    print("\nAuto ARIMA model order (p, d, q):", order)
    
    # Save model order and differenced series to files
    with open('auto_p_d_q.json', 'w') as fp:
        json.dump(order, fp)
    diff_series.to_csv(f'../dat/clean/differenced_auto_arima_{chosen_currency}_daily_historical.csv', index=False)

    print("Data processing and analysis completed.")

if __name__ == '__main__':
    main()

