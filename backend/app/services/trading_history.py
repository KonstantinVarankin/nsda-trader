import pandas as pd
import numpy as np
from typing import Dict, List
from app.db.session import get_db

class TradingHistoryAnalyzer:
    def __init__(self):
        self.db = next(get_db())

    def get_trading_history(self) -> Dict[str, List]:
        # В реальном приложении здесь будет запрос к базе данных
        # Для примера используем фиктивные данные
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        portfolio_values = np.cumsum(np.random.normal(0, 100, len(dates))) + 10000

        trades = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'profit_loss': np.diff(portfolio_values, prepend=10000)
        })

        return {
            'dates': trades['date'].dt.strftime('%Y-%m-%d').tolist(),
            'portfolioValues': trades['portfolio_value'].tolist(),
            'totalProfitLoss': trades['profit_loss'].sum(),
            'winRate': (trades['profit_loss'] > 0).mean() * 100,
            'sharpeRatio': self.calculate_sharpe_ratio(trades['profit_loss'])
        }

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()