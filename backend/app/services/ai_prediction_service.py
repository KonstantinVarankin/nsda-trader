import os
import sys
import logging
import numpy as np
import pandas as pd
import asyncio
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import aiohttp
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Добавляем корневую директорию проекта в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.core.config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CURRENCY_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'BCHUSDT',
                  'LTCUSDT']
TIME_INTERVALS = ['1m', '5m', '15m', '1h', '2h', '1d', '1w', '1M']
PREDICTION_HORIZONS = ['1m', '5m', '15m', '1h', '2h', '1d', '1w', '1M']

LOOK_BACK = 60
FUTURE_PERIOD = 5

NEWS_API_KEY = 'c7b860cf1a954ce4bb0d4e531a4c7010'
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'

# Загрузка необходимых ресурсов NLTK
nltk.download('vader_lexicon', quiet=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, data):
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
        scaled_data = self.scaler.fit_transform(data[numeric_columns])
        return pd.DataFrame(scaled_data, columns=numeric_columns, index=data.index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class NewsSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    async def get_news(self, query: str, from_date: str, to_date: str) -> List[Dict]:
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'publishedAt',
            'apiKey': NEWS_API_KEY,
            'language': 'en'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(NEWS_API_ENDPOINT, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('articles', [])
                else:
                    logger.error(f"Failed to fetch news: {response.status}")
                    return []

    def analyze_sentiment(self, text: str) -> float:
        return self.sia.polarity_scores(text)['compound']

    async def get_average_sentiment(self, query: str, days: int = 7) -> float:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        news = await self.get_news(query, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if not news:
            return 0.0

        sentiments = []
        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            if title or description:
                text = f"{title} {description}".strip()
                sentiments.append(self.analyze_sentiment(text))

        return sum(sentiments) / len(sentiments) if sentiments else 0.0

class TradingNeuralNetwork:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        logger.info(f"Using device: {self.device}")

    async def load_model(self, model_path: str):
        state_dict = torch.load(model_path, map_location=self.device)

        # Проверяем размер входного слоя
        input_size = state_dict['lstm.weight_ih_l0'].size(1)

        self.model = LSTMModel(input_size=input_size, hidden_size=100, num_layers=2, output_size=1)

        # Если размеры не совпадают, инициализируем новый слой
        if input_size != 6:
            new_state_dict = self.model.state_dict()
            for name, param in state_dict.items():
                if 'lstm.weight_ih' in name or 'lstm.weight_hh' in name:
                    new_param = new_state_dict[name]
                    new_param[:, :input_size] = param
                    state_dict[name] = new_param

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path} with input size {input_size}")

    async def save_model(self, model_path: str):
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")

    async def run_neural_network(self, data: np.ndarray):
        logger.info(f"Running neural network")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Time steps: {LOOK_BACK}")
        logger.info(f"Future period: {FUTURE_PERIOD}")

        logger.info("Preparing data...")
        X, y = self.prepare_data(data)
        logger.info("Data preparation completed.")

        if X is None or y is None:
            logger.error(f"Failed to prepare data")
            return

        logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
        logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
        logger.info(f"X memory usage: {X.nbytes / 1e9:.2f} GB, y memory usage: {y.nbytes / 1e9:.2f} GB")

        logger.info("Starting model training...")
        await self.train(X, y)

    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < LOOK_BACK + FUTURE_PERIOD:
            logger.warning(f"Not enough data to prepare. Using all available data.")
            X = data[:-FUTURE_PERIOD].reshape(1, -1, data.shape[1])
            y = data[-1, 3]  # Предсказываем цену закрытия
            return X.astype(np.float32), np.array([y]).astype(np.float32)

        X, y = [], []
        for i in range(len(data) - LOOK_BACK - FUTURE_PERIOD + 1):
            X.append(data[i:(i + LOOK_BACK)])
            y.append(data[i + LOOK_BACK + FUTURE_PERIOD - 1, 3])  # Предсказываем цену закрытия

        X = np.array(X)
        y = np.array(y)

        return X.astype(np.float32), y.astype(np.float32)

    async def train(self, X: np.ndarray, y: np.ndarray):
        logger.info(f"Creating model with input shape: {X.shape}")
        self.model = LSTMModel(input_size=X.shape[-1], hidden_size=100, num_layers=2, output_size=1)
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    async def predict(self, X: torch.Tensor) -> np.ndarray:
        if self.model is None:
            logger.error("Model not initialized for prediction")
            return None
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
        return prediction.cpu().numpy()

class AIPredictionService:
    def __init__(self):
        self.models = {}
        self.data_preprocessor = DataPreprocessor()
        self.binance_client = None
        self.model_filepath = os.path.join(project_root, 'models')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.news_analyzer = NewsSentimentAnalyzer()
        logger.info(f"AI prediction service initialized. Using device: {self.device}")

    async def initialize_binance_client(self):
        if self.binance_client is None:
            self.binance_client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)

    async def initialize_models(self):
        for pair in CURRENCY_PAIRS:
            for interval in TIME_INTERVALS:
                model_key = f"{pair}_{interval}"
                model_path = os.path.join(self.model_filepath, f"{model_key}.pth")

                model = TradingNeuralNetwork()
                if os.path.exists(model_path):
                    await model.load_model(model_path)

                    # Проверяем, нужно ли переобучить модель
                    if model.model.lstm.weight_ih_l0.size(1) != 6:
                        logger.info(f"Retraining model for {model_key} due to input size change")
                        historical_data = await self.get_historical_data(pair, interval)
                        if historical_data is not None and not historical_data.empty:
                            sentiment = await self.news_analyzer.get_average_sentiment(pair.replace('USDT', ''))
                            historical_data['sentiment'] = sentiment
                            processed_data = self.data_preprocessor.preprocess(historical_data)
                            await model.run_neural_network(processed_data.values)
                            await model.save_model(model_path)

                    self.models[model_key] = model
                    logger.info(f"AI model for {model_key} loaded successfully")
                else:
                    historical_data = await self.get_historical_data(pair, interval)
                    if historical_data is not None and not historical_data.empty:
                        sentiment = await self.news_analyzer.get_average_sentiment(pair.replace('USDT', ''))
                        historical_data['sentiment'] = sentiment
                        processed_data = self.data_preprocessor.preprocess(historical_data)
                        await model.run_neural_network(processed_data.values)
                        await model.save_model(model_path)
                        self.models[model_key] = model
                        logger.info(f"AI model for {model_key} trained and saved successfully")
                    else:
                        logger.error(f"Failed to initialize model for {model_key}: No historical data available")

    async def get_historical_data(self, symbol: str, interval: str) -> pd.DataFrame:
        await self.initialize_binance_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Увеличено до 1 года

        try:
            klines = await self.binance_client.get_historical_klines(symbol, interval,
                                                                     start_date.strftime("%d %b %Y %H:%M:%S"),
                                                                     end_date.strftime("%d %b %Y %H:%M:%S"))

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.debug(f"Historical data shape for {symbol} {interval}: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {interval}: {e}")
            return None

    async def get_latest_data(self, symbol: str, interval: str):
        await self.initialize_binance_client()
        try:
            klines = await self.binance_client.get_klines(symbol=symbol, interval=interval, limit=100)

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades',
                                               'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Добавляем анализ настроений
            sentiment = await self.news_analyzer.get_average_sentiment(symbol.replace('USDT', ''))
            df['sentiment'] = sentiment

            logger.debug(f"Latest data shape for {symbol} {interval}: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol} {interval}: {e}")
            return None

    async def run_prediction_cycle(self):
        try:
            if not self.models:
                await self.initialize_models()

            results = {}

            for pair in CURRENCY_PAIRS:
                pair_results = {}
                for interval in TIME_INTERVALS:
                    model_key = f"{pair}_{interval}"
                    model = self.models.get(model_key)

                    if model is None:
                        logger.error(f"Model not found for {model_key}")
                        continue

                    latest_data = await self.get_latest_data(pair, interval)
                    if latest_data is None or latest_data.empty:
                        logger.error(f"No data available for prediction: {pair} {interval}")
                        continue

                    processed_data = self.data_preprocessor.preprocess(latest_data)

                    if len(processed_data) < LOOK_BACK:
                        logger.warning(
                            f"Not enough data for prediction: {pair} {interval}. Required: {LOOK_BACK}, Got: {len(processed_data)}")
                        continue

                    input_data = processed_data.values[-LOOK_BACK:].reshape(1, LOOK_BACK, processed_data.shape[1])
                    input_data = torch.FloatTensor(input_data).to(self.device)

                    interval_results = {}
                    for horizon in PREDICTION_HORIZONS:
                        prediction = await model.predict(input_data)
                        if prediction is not None:
                            dummy_array = np.zeros((1, 6))
                            dummy_array[0, 3] = prediction[0][0]
                            real_prediction = self.data_preprocessor.inverse_transform(dummy_array)[0, 3]

                            current_time = datetime.now()
                            prediction_time = current_time + self.convert_interval_to_timedelta(horizon)

                            interval_results[horizon] = {
                                "prediction": float(real_prediction),
                                "current_time": current_time.isoformat(),
                                "prediction_time": prediction_time.isoformat(),
                                "current_price": float(latest_data['close'].iloc[-1]),
                                "sentiment": float(latest_data['sentiment'].iloc[-1])
                            }
                        else:
                            logger.error(f"Failed to get prediction for {pair} {interval} {horizon}")

                    pair_results[interval] = interval_results

                results[pair] = pair_results

            return {"status": "success", "results": results}
        except Exception as e:
            logger.error(f"Error in prediction cycle: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def convert_interval_to_timedelta(interval: str) -> timedelta:
        value = int(interval[:-1])
        unit = interval[-1]
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        elif unit == 'M':
            return timedelta(days=value * 30)  # Приблизительно
        else:
            raise ValueError(f"Unknown interval unit: {unit}")

    async def cleanup(self):
        if self.binance_client:
            await self.binance_client.close_connection()

# Создаем экземпляр AIPredictionService после определения класса
ai_service = AIPredictionService()

# Функция для запуска сервиса
async def run_ai_service():
    await ai_service.initialize_models()
    while True:
        results = await ai_service.run_prediction_cycle()
        logger.info(f"Prediction results: {results}")
        await asyncio.sleep(300)  # Ждем 5 минут перед следующим циклом предсказаний

# Функция для запуска и остановки сервиса
def start_ai_service():
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_ai_service())
    except KeyboardInterrupt:
        logger.info("Stopping AI service...")
    finally:
        loop.run_until_complete(ai_service.cleanup())
        loop.close()

if __name__ == "__main__":
    start_ai_service()