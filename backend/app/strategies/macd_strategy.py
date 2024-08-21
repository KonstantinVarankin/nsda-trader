from app.models.trading_models import TradingStrategy
from app.services.indicators import calculate_macd

class MACDStrategy(TradingStrategy):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.name = "MACD"
        self.params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period
        }

    def generate_signals(self, data):
        macd = calculate_macd(data, self.params['fast_period'], self.params['slow_period'], self.params['signal_period'])
        
        signals = (macd['macd'] > macd['signal']).astype(int)
        signals[macd['macd'] <= macd['signal']] = -1
        
        return signals

    def optimize_params(self, data, param_grid):
        best_sharpe = float('-inf')
        best_params = None

        for fast in param_grid['fast_period']:
            for slow in param_grid['slow_period']:
                for signal in param_grid['signal_period']:
                    if fast >= slow:
                        continue
                    
                    self.params = {
                        "fast_period": fast,
                        "slow_period": slow,
                        "signal_period": signal
                    }
                    
                    signals = self.generate_signals(data)
                    returns = data['close'].pct_change()
                    strategy_returns = signals.shift(1) * returns
                    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5)
                    
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_params = self.params.copy()

        self.params = best_params
        return best_sharpe
