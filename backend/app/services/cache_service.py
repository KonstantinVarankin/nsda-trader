import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class CacheService:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_cache_key(self, symbol: str, interval: str, start_date: str, end_date: str) -> str:
        return f"{symbol}_{interval}_{start_date}_{end_date}.json"

    def get_cached_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading cache file {cache_file}: {e}")
                # If there's an error reading the cache, return None to fetch fresh data
                return None
        return None

    def cache_data(self, symbol: str, interval: str, start_date: str, end_date: str, df: pd.DataFrame):
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        data = df.reset_index().to_dict(orient='records')
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing to cache file {cache_file}: {e}")

    def is_cache_valid(self, symbol: str, interval: str, start_date: str, end_date: str) -> bool:
        cache_key = self.get_cache_key(symbol, interval, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, cache_key)

        if not os.path.exists(cache_file):
            return False

        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_age = datetime.now() - file_mtime

        # Adjust cache validity based on the interval
        if interval == '1h':
            max_age = timedelta(hours=1)
        elif interval == '4h':
            max_age = timedelta(hours=4)
        elif interval == '1d':
            max_age = timedelta(days=1)
        else:
            max_age = timedelta(days=1)  # Default to 1 day for unknown intervals

        return cache_age < max_age

    def clear_cache(self):
        """Clear all cache files."""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

cache_service = CacheService()