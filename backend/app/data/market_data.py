import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def preprocess_data(df, look_back=60):
    # Выбираем только цены закрытия
    data = df['Close'].values
    data = data.reshape(-1, 1)

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Создаем выборки для обучения
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    
    X, y = np.array(X), np.array(y)

    # Reshape для LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def prepare_data_for_prediction(df, look_back=60):
    data = df['Close'].values[-look_back:]
    data = data.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    X = np.array([data])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, scaler
