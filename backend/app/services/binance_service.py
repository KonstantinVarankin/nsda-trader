from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
from app.services.cache_service import cache_service
import os

class BinanceService:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(api_key, api_secret)

    async def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        if cache_service.is_cache_valid(symbol, interval, start_date, end_date):
            cached_data = cache_service.get_cached_data(symbol, interval, start_date, end_date)
            if cached_data is not None:
                return cached_data

        try:
            klines = self.client.get_historical_klines(symbol, interval, start_date, end_date)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)

            cache_service.cache_data(symbol, interval, start_date, end_date, df)
            return df
        try:
            klines = self.client.get_historical_klines(symbol, interval, start_date, end_date)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)
            return df
        except BinanceAPIException as e:
            print(f"An error occurred: {e}")
            return None

binance_service = BinanceService()

