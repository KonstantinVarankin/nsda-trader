import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.market_data import MarketData
from app.core.config import settings
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        try:
            self.exchange = getattr(ccxt, settings.EXCHANGE_NAME)({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        except AttributeError as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    def fetch_ohlcv(self, symbol, timeframe='1d', since=None, limit=100):
        try:
            params = {}
            if since is not None:
                params['startTime'] = since
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, params=params, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    async def get_historical_data(self, start_date: datetime, end_date: datetime = None, symbol='BTC/USDT',
                                  timeframe='1d') -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now()

        since = int(start_date.timestamp() * 1000)
        all_data = []

        while since < end_date.timestamp() * 1000:
            data = self.fetch_ohlcv(symbol, timeframe, since, 1000)
            if data is None or len(data) == 0:
                break
            all_data.append(data)
            since = int(data.iloc[-1]['timestamp'].timestamp() * 1000) + 1

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result[(result['timestamp'] >= start_date) & (result['timestamp'] <= end_date)]
        return result

    async def get_latest_data(self, symbol='BTC/USDT', timeframe='1m') -> pd.DataFrame:
        try:
            data = self.fetch_ohlcv(symbol, timeframe, limit=1)
            return data
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()

async def save_market_data(db: Session, data: pd.DataFrame):
    try:
        for _, row in data.iterrows():
            market_data = MarketData(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            db.add(market_data)
        await db.commit()
        logger.info(f"Saved {len(data)} market data points to the database")
    except Exception as e:
        logger.error(f"Error saving market data to database: {e}")
        await db.rollback()

async def get_market_data(symbol='BTC/USDT', timeframe='1d', limit=100):
    try:
        collector = DataCollector()
        logger.info(f"Fetching market data for {symbol}, timeframe: {timeframe}, limit: {limit}")
        data = collector.fetch_ohlcv(symbol, timeframe, limit=limit)
        if data is None or data.empty:
            logger.warning("No market data retrieved")
        else:
            logger.info(f"Retrieved market data with shape: {data.shape}")
            db = await anext(get_db())
            await save_market_data(db, data)
        return data
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return None

__all__ = ['DataCollector', 'get_market_data']
get_historical_data = DataCollector().get_historical_data

if __name__ == "__main__":
    async def main():
        collector = DataCollector()
        start_date = datetime.now() - timedelta(days=30)
        historical_data = await collector.get_historical_data(start_date)
        latest_data = await collector.get_latest_data()

        if historical_data is not None and not historical_data.empty:
            print(f"Retrieved historical data with shape: {historical_data.shape}")
        if latest_data is not None and not latest_data.empty:
            print(f"Retrieved latest data with shape: {latest_data.shape}")

    asyncio.run(main())