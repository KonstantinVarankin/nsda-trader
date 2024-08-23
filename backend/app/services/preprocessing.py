import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def normalize_ohlcv(self, df):
        """Normalize OHLCV data."""
        cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        return df

    def calculate_technical_indicators(self, df):
        """Calculate basic technical indicators."""
        # Simple Moving Average
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()

        # Relative Strength Index
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def preprocess_news(self, news_articles):
        """Preprocess news articles."""
        # Here you would typically do things like:
        # - Extract relevant information from articles
        # - Perform sentiment analysis
        # - Convert to numerical features
        # For now, we'll just return the titles
        return [article['title'] for article in news_articles]

# Пример использования:
# preprocessor = DataPreprocessor()
# normalized_data = preprocessor.normalize_ohlcv(btc_data)
# data_with_indicators = preprocessor.calculate_technical_indicators(normalized_data)
# processed_news = preprocessor.preprocess_news(btc_news)