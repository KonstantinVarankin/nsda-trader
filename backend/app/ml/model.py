import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import os
import psycopg2
from psycopg2 import sql
import logging
from app.core.config import settings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получаем параметры подключения из переменных окружения
DATABASE_URL = settings.DATABASE_URL

# Настройка CUDA
device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
logger.info(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class PredictionModel:
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1):
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :])
            y.append(data[i + time_steps, 0])
        return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device)

    def train(self, data, epochs=100, batch_size=32):
        df = pd.DataFrame(data)
        df = df.astype(float)

        data_scaled = self.scaler.fit_transform(df)

        X, y = self.prepare_data(data_scaled)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    def predict(self, last_60_days):
        df = pd.DataFrame(last_60_days)
        df = df.astype(float)

        data_scaled = self.scaler.transform(df)
        X_test = torch.FloatTensor(data_scaled).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            pred_price = self.model(X_test)

        pred_price = self.scaler.inverse_transform(pred_price.cpu().numpy())
        return pred_price[0][0]

    def evaluate(self, test_data):
        df = pd.DataFrame(test_data)
        df = df.astype(float)

        actual_prices = df.iloc[:, 0].values
        predicted_prices = []

        for i in range(len(df) - 60):
            last_60_days = df.iloc[i:i + 60]
            predicted_price = self.predict(last_60_days)
            predicted_prices.append(predicted_price)

        mse = np.mean((np.array(predicted_prices) - actual_prices[60:]) ** 2)
        rmse = np.sqrt(mse)
        return {'mse': mse, 'rmse': rmse}

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        np.save(f"{path}_scaler.npy", self.scaler.scale_)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.scaler.scale_ = np.load(f"{path}_scaler.npy")

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
        self.models[symbol] = PredictionModel(input_size=data.shape[1])
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
        return pd.DataFrame({'close': np.random.rand(60), 'volume': np.random.rand(60)})

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