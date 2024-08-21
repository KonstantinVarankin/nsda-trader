import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from app.strategies.moving_average_strategy import MovingAverageStrategy
from app.strategies.macd_strategy import MACDStrategy

class BacktestingService:
    def __init__(self):
        self.data = None

    def load_data(self, symbol, interval, start_date, end_date):
        # Здесь должна быть логика загрузки данных
        pass

    def run_backtest(self, strategy, params):
        strategy.set_params(params)
        signals = strategy.generate_signals(self.data)
        returns = self.data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5)
        return sharpe_ratio, params

    def optimize_strategy(self, strategy_name: str, param_grid: dict):
        if strategy_name == "MovingAverage":
            strategy = MovingAverageStrategy()
        elif strategy_name == "MACD":
            strategy = MACDStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self.run_backtest, [(strategy, params) for params in param_combinations])

        best_sharpe, best_params = max(results, key=lambda x: x[0])
        return best_sharpe, best_params

