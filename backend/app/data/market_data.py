import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timedelta
import os
import pickle

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка для кэширования
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_stock_data(symbol, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}.pkl")

    # Проверяем кэш
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Data for {symbol} loaded from cache.")
        return df

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        # Сохраняем в кэш
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        logger.info(f"Data for {symbol} fetched from Yahoo Finance.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def add_technical_indicators(df):
    # Простая скользящая средняя
    df['SMA'] = df['Close'].rolling(window=20).mean()

    # Экспоненциальная скользящая средняя
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Относительный индекс силы (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def preprocess_data(df, look_back=60):
    # Добавляем технические индикаторы
    df = add_technical_indicators(df)

    # Выбираем нужные колонки
    data = df[['Close', 'SMA', 'EMA', 'RSI']].values

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Создаем выборки для обучения
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])

    X, y = np.array(X), np.array(y)

    return X, y, scaler


def prepare_data_for_prediction(df, look_back=60):
    df = add_technical_indicators(df)
    data = df[['Close', 'SMA', 'EMA', 'RSI']].values[-look_back:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    X = np.array([data])

    return X, scaler


def get_latest_price(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    df = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is not None and not df.empty:
        return df['Close'].iloc[-1]
    return None