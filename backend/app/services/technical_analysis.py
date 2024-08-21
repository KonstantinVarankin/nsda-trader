import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from app import crud
from app.db.session import SessionLocal

class TechnicalAnalyzer:
    def __init__(self):
        self.db = SessionLocal()

    def get_market_data(self, symbol):
        data = crud.market_data.get_market_data_by_symbol(self.db, symbol)
        df = pd.DataFrame([vars(item) for item in data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df

    def calculate_indicators(self, symbol):
        df = self.get_market_data(symbol)
        df = dropna(df)
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume"
        )
        return df

    def get_latest_indicators(self, symbol):
        df = self.calculate_indicators(symbol)
        latest = df.iloc[-1].to_dict()
        return {k: v for k, v in latest.items() if not pd.isna(v)}

    def generate_signals(self, symbol):
        df = self.calculate_indicators(symbol)
        signals = {}

        # Simple Moving Average (SMA) crossover
        signals['sma_crossover'] = 'buy' if df['close'].iloc[-1] > df['trend_sma_fast'].iloc[-1] > df['trend_sma_slow'].iloc[-1] else 'sell'

        # Relative Strength Index (RSI)
        rsi = df['momentum_rsi'].iloc[-1]
        if rsi < 30:
            signals['rsi'] = 'oversold'
        elif rsi > 70:
            signals['rsi'] = 'overbought'
        else:
            signals['rsi'] = 'neutral'

        # Moving Average Convergence Divergence (MACD)
        if df['trend_macd_diff'].iloc[-1] > 0 and df['trend_macd_diff'].iloc[-2] <= 0:
            signals['macd'] = 'bullish_crossover'
        elif df['trend_macd_diff'].iloc[-1] < 0 and df['trend_macd_diff'].iloc[-2] >= 0:
            signals['macd'] = 'bearish_crossover'
        else:
            signals['macd'] = 'neutral'

        return signals

technical_analyzer = TechnicalAnalyzer()
