import ccxt
from app.core.config import settings
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        try:
            self.exchange = getattr(ccxt, settings.EXCHANGE_NAME)({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_API_SECRET,
            })
        except AttributeError as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

        try:
            logger.info(f"Initializing NewsApiClient with key: {settings.NEWS_API_KEY[:5]}...")  # Показываем только первые 5 символов ключа
            self.newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize NewsApiClient: {e}")
            raise

    def fetch_ohlcv(self, symbol, timeframe='1d', limit=100):
        """Fetch OHLCV data from the exchange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    def fetch_news(self, query, from_param, to):
        """Fetch news articles related to the given query."""
        try:
            news = self.newsapi.get_everything(q=query,
                                               from_param=from_param,
                                               to=to,
                                               language='en',
                                               sort_by='relevancy')
            return news['articles']
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return None

def get_news_data(days=7):
    """
    Fetch news data for the last specified number of days.

    :param days: Number of days to fetch news for (default is 7)
    :return: List of news articles
    """
    try:
        collector = DataCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        from_param = start_date.strftime('%Y-%m-%d')
        to = end_date.strftime('%Y-%m-%d')

        logger.info(f"Fetching news from {from_param} to {to}")
        crypto_news = collector.fetch_news('cryptocurrency OR bitcoin OR ethereum', from_param, to)

        if crypto_news is None:
            logger.warning("No news data retrieved")
        else:
            logger.info(f"Retrieved {len(crypto_news)} news articles")

        return crypto_news
    except Exception as e:
        logger.error(f"Error in get_news_data: {e}")
        return None

def get_market_data(symbol='BTC/USDT', timeframe='1d', limit=100):
    """
    Fetch market data for a given symbol.

    :param symbol: Trading pair symbol (default is 'BTC/USDT')
    :param timeframe: Timeframe for the data (default is '1d' for daily)
    :param limit: Number of candles to fetch (default is 100)
    :return: DataFrame with OHLCV data
    """
    try:
        collector = DataCollector()
        logger.info(f"Fetching market data for {symbol}, timeframe: {timeframe}, limit: {limit}")
        data = collector.fetch_ohlcv(symbol, timeframe, limit)
        if data is None:
            logger.warning("No market data retrieved")
        else:
            logger.info(f"Retrieved market data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return None

def get_historical_data(symbol='BTC/USDT', timeframe='1d', limit=1000):
    """
    Fetch historical market data for a given symbol.

    :param symbol: Trading pair symbol (default is 'BTC/USDT')
    :param timeframe: Timeframe for the data (default is '1d' for daily)
    :param limit: Number of candles to fetch (default is 1000)
    :return: DataFrame with OHLCV data
    """
    try:
        collector = DataCollector()
        logger.info(f"Fetching historical data for {symbol}, timeframe: {timeframe}, limit: {limit}")
        data = collector.fetch_ohlcv(symbol, timeframe, limit)
        if data is None:
            logger.warning("No historical data retrieved")
        else:
            logger.info(f"Retrieved historical data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error in get_historical_data: {e}")
        return None

# Список экспортируемых функций
__all__ = ['get_news_data', 'get_market_data', 'get_historical_data']

# Пример использования:
if __name__ == "__main__":
    news_data = get_news_data()
    market_data = get_market_data()
    historical_data = get_historical_data()

    if news_data:
        print(f"Retrieved {len(news_data)} news articles")
    if market_data is not None:
        print(f"Retrieved market data with shape: {market_data.shape}")
    if historical_data is not None:
        print(f"Retrieved historical data with shape: {historical_data.shape}")