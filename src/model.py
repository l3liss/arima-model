# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:25:58 2022

@author: l3oxbit
"""
import os
import requests
import json
import datetime
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import matplotlib.pyplot as plt


def get_new_prices(historical_data):
    """
    Parameters
    ----------
    historical_data : DF
        Df read from csv containing all the historical prices by day.
    
    Returns
    -------
    None
    """
    try:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%Y/%m/%d')
    except ValueError:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%d/%m/%Y')

    max_date = historical_data['Date'].max()
    today_date = pd.to_datetime("today")
    date_difference = (today_date - max_date).days
    if date_difference > 0:
        chosen_currency = historical_data['Currency'][0]
        get_historical_prices(chosen_currency, date_difference, False)
    else:
        print("No new data needed.")


def get_historical_prices(chosen_currency, num_days, first_parse):
    """
    Parameters
    ----------
    chosen_currency : STR
        Provide a valid cryptocurrency e.g. 'bitcoin'.
    num_days : INT
        Enter the number of days of history needed.
    
    Returns
    -------
    None
    """
    response = requests.get(f'https://api.coingecko.com/api/v3/coins/{chosen_currency}/market_chart?vs_currency=usd&days={num_days}&interval=daily')
    hist_dict = response.json()
    
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
    """
    Parameters
    ----------
    df : DataFrame
        A dataframe where each row contains unique data for the respective currency.
    
    Returns
    -------
    df : DataFrame
        DataFrame with no duplicate rows.
    """
    df.drop_duplicates('Date', keep='last', inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def clean(df):
    """
    Parameters
    ----------
    df : DataFrame
        A dataframe where each row contains unique data for the respective currency.
    
    Returns
    -------
    df : DataFrame
        Cleaned dataframe with all preprocessing applied.
    """
    df = remove_duplicate_data(df)
    return df


def adf_test(target_series):
    """
    Parameters
    ----------
    target_series : Pandas Series
        Series to test for stationarity (e.g. Bitcoin Price).
    
    Returns
    -------
    adf_statistic : Float
        The ADF statistic.
    p_value : Float
        The p-value of the ADF test.
    """
    result = adfuller(target_series)
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic: %f' % adf_statistic)
    print('p-value: %f' % p_value)
    return adf_statistic, p_value


def kpss_test(target_series):
    print("Results of KPSS Test:")
    kpsstest = kpss(target_series, regression="ct", nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


def find_order_of_differencing(df):
    """
    Parameters
    ----------
    df : DataFrame
        Cleaned time series data for the currency.
    
    Returns
    -------
    d : INT
        The number of differences required to make the series stationary.
    """
    adf_statistic, p_value = adf_test(df['Price(USD)'])
    kpss_test(df['Price(USD)'])
    d = 0
    while p_value > 0.05:
        df['Price(USD)'] = df['Price(USD)'].diff()
        df.dropna(inplace=True)
        d += 1
        adf_statistic, p_value = adf_test(df['Price(USD)'])
        kpss_test(df['Price(USD)'])
    print(f"Success... TS now stationary after {d} differencing")
    return d


def create_acf_pacf(df):
    """
    Parameters
    ----------
    df : DataFrame
        The differenced dataframe.
    
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, figsize=(12,8), dpi=100)
    plot_acf(df['Price(USD)'], lags=20, ax=ax)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(12,8), dpi=100)
    plot_pacf(df['Price(USD)'], lags=20, ax=ax)
    plt.show()


def eda(df):
    """
    Parameters
    ----------
    df : DataFrame
        Cleaned time series data for the currency.
    
    Returns
    -------
    None
    """
    d = find_order_of_differencing(df)
    create_acf_pacf(df)


def auto_arima(df):
    """
    Parameters
    ----------
    df : DataFrame
        The original time series data.
    
    Returns
    -------
    model.order : Tuple
        The optimal (p, d, q) values from auto ARIMA.
    differenced_by_auto_arima : DataFrame
        The differenced dataframe from the auto ARIMA model.
    """
    orig_df = np.log(df['Price(USD)'])
    model = pm.auto_arima(orig_df, start_p=10, start_q=10, test='adf', max_p=10, max_q=10, m=1, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    differenced_by_auto_arima = orig_df.diff(model.order[1])
    return model.order, differenced_by_auto_arima


def main():
    # Select cryptocurrency
    chosen_currency = 'bitcoin'
    try:
        historical_data = pd.read_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv')
    except FileNotFoundError:
        historical_data = pd.DataFrame()

    # Fetch and clean data if necessary
    if len(historical_data) > 0:
        get_new_prices(historical_data)
    else:
        get_historical_prices(chosen_currency, 3650, True)

    # Clean the data
    df = pd.read_csv(f'../dat/raw/{chosen_currency}_daily_historical.csv')
    df = clean(df)

    # Save cleaned data
    df.to_csv(f'../dat/clean/cleaned_{chosen_currency}_daily_historical.csv', index=False)

    # Perform EDA
    eda(df)

    # Perform Auto ARIMA and save model orders
    auto_p_d_q, differenced_by_auto_arima = auto_arima(df)
    
    with open('auto_p_d_q.json', 'w') as fp:
        json.dump(auto_p_d_q, fp)

    differenced_by_auto_arima.to_csv(f'../dat/clean/differenced_auto_arima_{chosen_currency}_daily_historical.csv')

    print("Data processing and analysis completed.")

if __name__ == '__main__':
    main()
