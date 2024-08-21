import os
import json
import pandas as pd
from datetime import datetime, timedelta

class CacheService:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, symbol: str, interval: str, start_date: str, end_date: str) -> str:
        return f"{symbol}_{interval}_{start_date}_{end_date}.json"

    def get_cached_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        return None

    def cache_data(self, symbol: str, interval: str, start_date: str, end_date: str, df: pd.DataFrame):
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        data = df.reset_index().to_dict(orient='records')
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def is_cache_valid(self, symbol: str, interval: str, start_date: str, end_date: str) -> bool:
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        if not os.path.exists(cache_file):
            return False

        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_age = datetime.now() - file_mtime

        # Consider cache valid if it's less than 1 day old
        return cache_age < timedelta(days=1)

cache_service = CacheService()
