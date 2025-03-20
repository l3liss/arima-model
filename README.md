btc-arima-model

Contributors :black_nib:
:star: Ben McKinnon
:star2: l3oxbit (Maintainer)

Table Of Contents :books:

    General Info
    Technologies
    Data Sources
    Usage
    Project Structure

General Info :page_with_curl:

This repository holds the codebase to create a predictive time series model for Bitcoin (and other desired cryptocurrencies).
The latest update integrates ARIMA for mean forecasting with a GARCH model for volatility/return direction. An optional Boostdog module (using XGBoost on multi-timeframe technical indicators) is also attempted; if Boostdog is unavailable, the model continues using ARIMA and GARCH alone.
This unified approach is designed to run at the start of a new trading week to provide a weekly price trajectory signal.
Technologies :computer:

Project is created with:

    Python 3.7.8
    Pandas 1.3.1
    Numpy 1.22.1
    JSON 2.0.9
    statsmodels 0.13.1
    pmdarima 1.8.4
    matplotlib 3.5.1
    arch (for GARCH modeling)
    yfinance

Note: The optional Boostdog (XGBoost) functionality requires additional packages (xgboost, scikit-learn, and joblib).
Data Sources :open_file_folder:

The primary data used in this repository is sourced from Coin Gecko via their free public API and from Yahoo Finance (using yfinance).

For Coin Gecko details, see: Coin Gecko API
Usage :shipit:

    Clone this repository.

    Ensure all required Python packages are installed.

    Run model.py from the src folder:

    cd src
    python model.py

    The script will:
        Fetch BTC-USD daily data.
        Fit an ARIMA model to the log-prices to forecast the weekly price trajectory.
        Fit a GARCH model to the log-returns to forecast aggregated returns and volatility.
        Optionally, attempt to generate a Boostdog (XGBoost) signal based on technical indicators.
        Combine the ARIMA and GARCH (and Boostdog, if available) signals using a simple voting mechanism.
        Print the individual and overall signals and save the forecast results to weekly_forecast.json.

    Project Structure :microscope:

    ├── dat                   
    │   ├── raw                <- Contains raw original CSV files.
    │   └── clean              <- Contains cleaned CSV files generated by clean.py.
    ├── src                    
    │   ├── crypto_price_get.py   <- Script to fetch cryptocurrency data.
    │   ├── clean.py              <- Script to clean the raw data.
    │   ├── eda.py                <- Script for exploratory data analysis.
    │   ├── model.py              <- Original foxed ARIMA model
    │   ├── b-g-a-model.py              <- Unified script that fits ARIMA & GARCH (with optional Boostdog) and outputs forecasts.
    │   ├── boostdog.py           <- (Optional) XGBoost-based signal generator.
    │   └── garchdog.py           <- GARCH-based modeling functions.
    ├── .gitignore.txt         <- Specifies files to ignore in version control.
    └── README.md              <- This file.
