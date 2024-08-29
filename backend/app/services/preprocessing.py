import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.column_order = ['open', 'high', 'low', 'close', 'volume']

    def normalize_ohlcv(self, df):
        cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        return df

    def calculate_technical_indicators(self, df):
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def preprocess_news(self, news_articles):
        return [article['title'] for article in news_articles]

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['timestamp'] + self.column_order
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")

        processed_data = data.copy()

        # Преобразуем временную метку в числовой формат (UNIX timestamp в секундах)
        processed_data['timestamp'] = processed_data['timestamp'].astype('int64') // 10**9

        # Нормализуем только OHLCV столбцы
        processed_data[self.column_order] = self.scaler.fit_transform(processed_data[self.column_order])

        return processed_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # Если предсказание - одно число, преобразуем его в массив нужной формы
        if data.shape == (1, 1):
            data = np.repeat(data, len(self.column_order)).reshape(1, -1)
        return self.scaler.inverse_transform(data)

# Example usage:
# preprocessor = DataPreprocessor()
# normalized_data = preprocessor.normalize_ohlcv(btc_data)
# data_with_indicators = preprocessor.calculate_technical_indicators(normalized_data)
# processed_news = preprocessor.preprocess_news(btc_news)