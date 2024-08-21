import pandas as pd
import numpy as np

def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['close'].rolling(window=period).mean()
    rolling_std = data['close'].rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return pd.DataFrame({'middle': sma, 'upper': upper_band, 'lower': lower_band})

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': histogram})

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({'k': k, 'd': d})

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['close'].rolling(window=period).mean()
    rolling_std = data['close'].rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return pd.DataFrame({'middle': sma, 'upper': upper_band, 'lower': lower_band})

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': histogram})

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({'k': k, 'd': d})
