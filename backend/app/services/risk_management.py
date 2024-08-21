import numpy as np
from app.services.technical_analysis import technical_analyzer
from app.db.session import SessionLocal
from app import crud

class RiskManager:
    def __init__(self):
        self.db = SessionLocal()
        self.technical_analyzer = technical_analyzer

    def calculate_volatility(self, symbol, window=20):
        df = self.technical_analyzer.get_market_data(symbol)
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
        return volatility.iloc[-1]

    def calculate_position_size(self, symbol, account_balance, risk_per_trade=0.02, stop_loss_percent=0.02):
        current_price = self.technical_analyzer.get_market_data(symbol)['close'].iloc[-1]
        volatility = self.calculate_volatility(symbol)

        # Adjust stop loss based on volatility
        adjusted_stop_loss = max(stop_loss_percent, volatility)

        # Calculate position size
        risk_amount = account_balance * risk_per_trade
        shares = risk_amount / (current_price * adjusted_stop_loss)

        return round(shares, 2)

    def set_stop_loss(self, symbol, entry_price, is_long, stop_loss_percent=0.02):
        if is_long:
            return entry_price * (1 - stop_loss_percent)
        else:
            return entry_price * (1 + stop_loss_percent)

    def set_take_profit(self, symbol, entry_price, is_long, risk_reward_ratio=2):
        stop_loss = self.set_stop_loss(symbol, entry_price, is_long)
        if is_long:
            return entry_price + (entry_price - stop_loss) * risk_reward_ratio
        else:
            return entry_price - (stop_loss - entry_price) * risk_reward_ratio

    def check_max_drawdown(self, symbol, max_drawdown=0.2):
        df = self.technical_analyzer.get_market_data(symbol)
        peak = df['close'].cummax()
        drawdown = (peak - df['close']) / peak
        current_drawdown = drawdown.iloc[-1]

        if current_drawdown > max_drawdown:
            return False
        return True

    def adjust_position(self, symbol, current_position, account_balance):
        volatility = self.calculate_volatility(symbol)
        ideal_position = self.calculate_position_size(symbol, account_balance)

        if volatility > 0.03:  # High volatility
            return min(current_position, ideal_position)  # Reduce position if necessary
        elif volatility < 0.01:  # Low volatility
            return max(current_position, ideal_position)  # Increase position if possible
        else:
            return ideal_position  # Maintain current position

risk_manager = RiskManager()
