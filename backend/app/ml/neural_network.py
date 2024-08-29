import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from binance.client import AsyncClient
import logging
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv
import traceback

# Загрузка переменных окружения из файла .env
load_dotenv()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'BCHUSDT',
                 'LTCUSDT']
INTERVAL = AsyncClient.KLINE_INTERVAL_1MINUTE
LOOK_BACK = 60  # Количество минут для предсказания
FUTURE_PERIOD = 5  # На сколько минут вперед предсказываем
SPLIT = 0.8  # Соотношение train/test
BATCH_SIZE = 2048
EPOCHS = 30
PATIENCE = 10

# Загрузка настроек из переменных окружения
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
DATABASE_URL = os.getenv('DATABASE_URL')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL'))
USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


class TradingNeuralNetwork:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def create_model(self, symbol, input_shape):
        logging.info(f"Creating model for {symbol} with input shape: {input_shape}")
        if len(input_shape) < 2:
            raise ValueError(f"Invalid input shape: {input_shape}. Expected at least 2 dimensions.")

        input_features = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
        model = LSTMModel(input_features).to(self.device)
        self.models[symbol] = model
        return model

    async def train(self, symbol, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE):
        if symbol not in self.models:
            self.create_model(symbol, X.shape)

        model = self.models[symbol]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_samples = int(SPLIT * len(X))
        X_train, X_val = X[:train_samples], X[train_samples:]
        y_train, y_val = y[:train_samples], y[train_samples:]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), f"{symbol}_best_model.pth")
            else:
                no_improve += 1

            if no_improve == patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info(f"Model for {symbol} trained for {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")
        model.load_state_dict(torch.load(f"{symbol}_best_model.pth", map_location=self.device))

    async def predict(self, symbol, X):
        model = self.models[symbol]
        model.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X = X.to(self.device)
            else:
                X = torch.FloatTensor(X).to(self.device)
            return model(X).cpu().numpy()

    async def save_model(self, symbol, filepath):
        torch.save(self.models[symbol].state_dict(), filepath)
        logger.info(f"Model for {symbol} saved to {filepath}")

    async def load_model(self, symbol, filepath):
        if os.path.exists(filepath):
            self.models[symbol] = LSTMModel(input_size=5).to(self.device)  # Assuming 5 features
            self.models[symbol].load_state_dict(torch.load(filepath, map_location=self.device))
            self.models[symbol].eval()
            logger.info(f"Model for {symbol} loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

    async def get_binance_data(self, symbol):
        client = await AsyncClient.create(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

        # Получаем данные за последние 900 дней
        end_time = datetime.now()
        start_time = end_time - timedelta(days=900)

        klines = await client.get_historical_klines(symbol, INTERVAL, start_time.strftime("%d %b %Y %H:%M:%S"),
                                                    end_time.strftime("%d %b %Y %H:%M:%S"))

        await client.close_connection()

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        return df

    def prepare_data(self, symbol, data, look_back=LOOK_BACK, future_period=FUTURE_PERIOD, batch_size=10000):
        if len(data) < look_back + future_period:
            logger.warning(
                f"Not enough data for {symbol} to prepare. Data length: {len(data)}, Required: {look_back + future_period}")
            return None, None

        if symbol not in self.scalers:
            self.scalers[symbol] = MinMaxScaler()

        # Преобразуем 3D массив в 2D для масштабирования
        n_samples, n_steps, n_features = data.shape
        data_2d = data.reshape((n_samples * n_steps, n_features))

        # Масштабируем данные порциями
        scaled_data = []
        for i in range(0, len(data_2d), batch_size):
            batch = data_2d[i:i + batch_size]
            scaled_batch = self.scalers[symbol].fit_transform(batch)
            scaled_data.append(scaled_batch)

        scaled_data = np.concatenate(scaled_data)

        # Преобразуем обратно в 3D
        scaled_data = scaled_data.reshape((n_samples, n_steps, n_features))

        X, y = [], []
        for i in range(len(scaled_data) - look_back - future_period):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back + future_period - 1, 3])  # Предсказываем цену закрытия

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def inverse_transform(self, symbol, data):
        if symbol not in self.scalers:
            raise ValueError(f"Scaler for {symbol} not found")

        # Преобразуем в 2D массив с 5 колонками, где предсказанное значение находится в колонке для цены закрытия (индекс 3)
        full_data = np.zeros((1, 5))
        full_data[0, 3] = data

        # Выполняем обратное преобразование
        inversed = self.scalers[symbol].inverse_transform(full_data)

        # Возвращаем только предсказанное значение
        return inversed[0, 3]

    async def run_neural_network(self, symbol, data, time_steps=LOOK_BACK):
        logger.info(f"Running neural network for {symbol}")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Time steps: {time_steps}")
        logger.info(f"Future period: {FUTURE_PERIOD}")

        try:
            logger.info("Preparing data...")
            X, y = self.prepare_data(symbol, data, time_steps)
            logger.info("Data preparation completed.")

            if X is None or y is None:
                logger.warning(
                    f"Skipping model creation for {symbol} due to insufficient data. Consider adjusting parameters or collecting more data.")
                return "Insufficient data"

            logger.info(f"Prepared data shapes for {symbol} - X: {X.shape}, y: {y.shape}")
            logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
            logger.info(f"X memory usage: {X.nbytes / 1e9:.2f} GB, y memory usage: {y.nbytes / 1e9:.2f} GB")

            logger.info("Starting model training...")
            await self.train(symbol, X, y)
            logger.info("Model training completed.")

            return "Training completed"
        except Exception as e:
            logger.error(f"Error in run_neural_network for {symbol}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def evaluate_model(self, symbol, X_test, y_test):
        predictions = await self.predict(symbol, X_test)
        predictions = np.array([self.inverse_transform(symbol, pred) for pred in predictions.flatten()])
        y_test = np.array([self.inverse_transform(symbol, y) for y in y_test])
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        logger.info(f"Model evaluation for {symbol} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        return mse, mae


# Глобальный экземпляр TradingNeuralNetwork
global_nn = TradingNeuralNetwork()


async def run_neural_network(symbol, data, time_steps=LOOK_BACK):
    return await global_nn.run_neural_network(symbol, data, time_steps)


async def get_prediction(symbol, X):
    prediction = await global_nn.predict(symbol, X)
    return global_nn.inverse_transform(symbol, prediction.item())


async def main():
    # Пример использования
    symbol = 'BTCUSDT'
    data = await global_nn.get_binance_data(symbol)
    result = await run_neural_network(symbol, data.values)
    print(result)

    # Пример предсказания
    latest_data = data.values[-LOOK_BACK:].reshape(1, LOOK_BACK, -1)
    prediction = await get_prediction(symbol, latest_data)
    print(f"Predicted price for {symbol}: {prediction:.2f}")


if __name__ == "__main__":
    asyncio.run(main())