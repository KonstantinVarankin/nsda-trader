import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import os
import psycopg2
from psycopg2 import sql
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получаем параметры подключения из переменных окружения
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:2202@localhost:5432/nsda_trader')

class PredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    def train(self, data):
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)

        data = df['close'].values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)

        X, y = self.prepare_data(data_scaled)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=100, batch_size=32)

    def predict(self, last_60_days):
        df = pd.DataFrame(last_60_days)
        df['close'] = df['close'].astype(float)

        data = df['close'].values.reshape(-1, 1)
        last_60_days_scaled = self.scaler.transform(data)
        X_test = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))
        pred_price = self.model.predict(X_test)
        pred_price = self.scaler.inverse_transform(pred_price)
        return pred_price[0][0]

    def evaluate(self, test_data):
        df = pd.DataFrame(test_data)
        df['close'] = df['close'].astype(float)

        actual_prices = df['close'].values
        predicted_prices = []

        for i in range(len(df) - 60):
            last_60_days = df.iloc[i:i + 60]
            predicted_price = self.predict(last_60_days)
            predicted_prices.append(predicted_price)

        mse = np.mean((np.array(predicted_prices) - actual_prices[60:]) ** 2)
        rmse = np.sqrt(mse)
        return {'mse': mse, 'rmse': rmse}

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

class NSDATradingModel:
    def __init__(self):
        self.models = {}
        self.init_db()

    def init_db(self):
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS predictions
                           (symbol TEXT, date TIMESTAMP, prediction REAL)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS evaluations
                           (symbol TEXT, date TIMESTAMP, mse REAL, rmse REAL)''')
            conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def train(self, symbol, data):
        self.models[symbol] = PredictionModel()
        self.models[symbol].train(data)
        self.save_model(symbol, f"models/{symbol}_model")

    def predict(self, symbol, date):
        if symbol not in self.models:
            self.load_model(symbol, f"models/{symbol}_model")

        last_60_days = self.get_last_60_days(symbol, date)
        prediction = self.models[symbol].predict(last_60_days)

        conn = None
        cur = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO predictions (symbol, date, prediction) VALUES (%s, %s, %s)",
                        (symbol, date, prediction))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

        return prediction

    def evaluate(self, symbol, test_data):
        if symbol not in self.models:
            self.load_model(symbol, f"models/{symbol}_model")

        evaluation = self.models[symbol].evaluate(test_data)

        conn = None
        cur = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO evaluations (symbol, date, mse, rmse) VALUES (%s, %s, %s, %s)",
                        (symbol, datetime.now(), evaluation['mse'], evaluation['rmse']))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

        return evaluation

    def save_model(self, symbol, path):
        if symbol not in self.models:
            raise ValueError(f"Model for symbol {symbol} not trained")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.models[symbol].save(path)

    def load_model(self, symbol, path):
        if not os.path.exists(path):
            raise ValueError(f"Model for symbol {symbol} not found at {path}")

        self.models[symbol] = PredictionModel()
        self.models[symbol].load(path)

    def get_last_60_days(self, symbol, date):
        # Здесь должна быть реализация получения данных за последние 60 дней
        # Пример заглушки:
        end_date = date
        start_date = end_date - timedelta(days=60)
        # Здесь должен быть код для получения данных из вашего источника данных
        # Например, использование binance_service или запрос к базе данных
        # Возвращаем пример данных
        return pd.DataFrame({'close': np.random.rand(60)})

    def get_predictions(self, symbol, start_date, end_date):
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT * FROM predictions WHERE symbol = %s AND date BETWEEN %s AND %s",
                        (symbol, start_date, end_date))
            predictions = cur.fetchall()
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_evaluations(self, symbol, start_date, end_date):
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT * FROM evaluations WHERE symbol = %s AND date BETWEEN %s AND %s",
                        (symbol, start_date, end_date))
            evaluations = cur.fetchall()
            return evaluations
        except Exception as e:
            logger.error(f"Error getting evaluations: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

# Создаем экземпляр модели
nsda_trading_model = NSDATradingModel()

# Экспортируем необходимые компоненты
prediction_model = nsda_trading_model
predict = nsda_trading_model.predict

__all__ = ['NSDATradingModel', 'PredictionModel', 'nsda_trading_model', 'prediction_model', 'predict']