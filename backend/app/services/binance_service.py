from binance import AsyncClient
import pandas as pd
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class BinanceService:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = None

    async def initialize(self):
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        logger.info("BinanceService инициализирован")

    async def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str):
        if not self.client:
            raise ValueError("BinanceService не инициализирован. Вызовите initialize() перед использованием.")

        try:
            klines = await self.client.get_historical_klines(symbol, interval, start_date, end_date)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_exchange_info(self):
        if not self.client:
            raise ValueError("BinanceService не инициализирован. Вызовите initialize() перед использованием.")
        return await self.client.get_exchange_info()

    async def close(self):
        if self.client:
            await self.client.close_connection()
            logger.info("BinanceService соединение закрыто")

binance_service = BinanceService()