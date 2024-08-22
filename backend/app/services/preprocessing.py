import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

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
        # Убедимся, что у нас есть все необходимые колонки
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")

        # Преобразуем timestamp в datetime, если это еще не сделано
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Сортируем данные по времени
        data = data.sort_values('timestamp')

        # Нормализуем числовые колонки
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])

        return data

# Example usage:
# preprocessor = DataPreprocessor()
# normalized_data = preprocessor.normalize_ohlcv(btc_data)
# data_with_indicators = preprocessor.calculate_technical_indicators(normalized_data)
# processed_news = preprocessor.preprocess_news(btc_news)