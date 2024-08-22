import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timedelta
import os
import pickle

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Папка для кэширования
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_stock_data(symbol, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Data for {symbol} loaded from cache.")
        return df

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data available for {symbol} in the specified date range.")
            return None

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        logger.info(f"Data for {symbol} fetched from Yahoo Finance and cached.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def add_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.fillna(method='bfill', inplace=True)
    return df

def preprocess_data(df, look_back=60):
    df = add_technical_indicators(df)
    data = df[['Close', 'SMA', 'EMA', 'RSI']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])

    return np.array(X), np.array(y), scaler

def prepare_data_for_prediction(df, look_back=60):
    df = add_technical_indicators(df)
    data = df[['Close', 'SMA', 'EMA', 'RSI']].values[-look_back:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    return np.array([data]), scaler

def get_latest_price(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    df = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is not None and not df.empty:
        return df['Close'].iloc[-1]
    return None

def get_historical_data(symbol, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

def get_latest_data(symbol, days=1):
    return get_historical_data(symbol, days)

# Экспортируем функции
__all__ = [
    'get_stock_data',
    'add_technical_indicators',
    'preprocess_data',
    'prepare_data_for_prediction',
    'get_latest_price',
    'get_historical_data',
    'get_latest_data'
]