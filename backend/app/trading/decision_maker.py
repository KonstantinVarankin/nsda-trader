import numpy as np
from app.data.market_data import get_stock_data

class TradingDecisionMaker:
    def __init__(self, risk_tolerance=0.02, position_size=1000):
        self.risk_tolerance = risk_tolerance
        self.position_size = position_size

    def make_decision(self, symbol, prediction, current_price):
        # Получаем исторические данные за последние 30 дней
        end_date = prediction['date']
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        historical_data = get_stock_data(symbol, start_date, end_date)

        # Рассчитываем волатильность
        volatility = historical_data['Close'].pct_change().std()

        # Рассчитываем ожидаемую доходность
        expected_return = (prediction['prediction'] - current_price) / current_price

        # Рассчитываем коэффициент Шарпа
        sharpe_ratio = expected_return / volatility

        # Принимаем решение на основе коэффициента Шарпа и риск-толерантности
        if sharpe_ratio > self.risk_tolerance:
            action = 'BUY'
            quantity = int(self.position_size / current_price)
        elif sharpe_ratio < -self.risk_tolerance:
            action = 'SELL'
            quantity = int(self.position_size / current_price)
        else:
            action = 'HOLD'
            quantity = 0

        return {
            'action': action,
            'quantity': quantity,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }

    def calculate_stop_loss(self, entry_price, action):
        if action == 'BUY':
            return entry_price * (1 - self.risk_tolerance)
        elif action == 'SELL':
            return entry_price * (1 + self.risk_tolerance)
        else:
            return None

    def calculate_take_profit(self, entry_price, action):
        if action == 'BUY':
            return entry_price * (1 + self.risk_tolerance * 2)
        elif action == 'SELL':
            return entry_price * (1 - self.risk_tolerance * 2)
        else:
            return None
