import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_capital: float, max_risk_per_trade: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_position = 0.0
        self.max_risk_per_trade = max_risk_per_trade
        self.open_positions: Dict[str, Dict] = {}

    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """
        Calculate the position size based on the risk per trade.
        """
        risk_amount = self.current_capital * self.max_risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        return risk_amount / risk_per_share

    def add_position(self, symbol: str, entry_price: float, stop_loss: float, take_profit: float) -> Dict:
        """
        Add a new position and calculate its size based on risk management rules.
        """
        position_size = self.calculate_position_size(entry_price, stop_loss, symbol)
        position = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "size": position_size
        }
        self.open_positions[symbol] = position
        logger.info(f"Added new position for {symbol}: {position}")
        return position

    def update_position(self, symbol: str, current_price: float) -> Dict:
        """
        Update the position based on the current price.
        """
        if symbol not in self.open_positions:
            raise ValueError(f"No open position for symbol {symbol}")

        position = self.open_positions[symbol]
        if current_price <= position["stop_loss"]:
            return self.close_position(symbol, current_price, "stop_loss")
        elif current_price >= position["take_profit"]:
            return self.close_position(symbol, current_price, "take_profit")
        else:
            return {"status": "open", "symbol": symbol, "current_price": current_price}

    def close_position(self, symbol: str, close_price: float, reason: str) -> Dict:
        """
        Close a position and update the current capital.
        """
        if symbol not in self.open_positions:
            raise ValueError(f"No open position for symbol {symbol}")

        position = self.open_positions.pop(symbol)
        profit_loss = (close_price - position["entry_price"]) * position["size"]
        self.current_capital += profit_loss

        result = {
            "status": "closed",
            "symbol": symbol,
            "close_price": close_price,
            "reason": reason,
            "profit_loss": profit_loss
        }
        logger.info(f"Closed position for {symbol}: {result}")
        return result

    def get_portfolio_status(self) -> Dict:
        """
        Get the current status of the portfolio.
        """
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "open_positions": self.open_positions,
            "total_profit_loss": self.current_capital - self.initial_capital
        }

    def calculate_portfolio_risk(self) -> float:
        """
        Calculate the current risk of the entire portfolio.
        """
        total_risk = sum(
            (pos["entry_price"] - pos["stop_loss"]) * pos["size"]
            for pos in self.open_positions.values()
        )
        return total_risk / self.current_capital

    def adjust_position_sizes(self):
        """
        Adjust position sizes to maintain the desired risk level.
        """
        current_risk = self.calculate_portfolio_risk()
        if current_risk > self.max_risk_per_trade:
            risk_factor = self.max_risk_per_trade / current_risk
            for symbol, position in self.open_positions.items():
                new_size = position["size"] * risk_factor
                self.open_positions[symbol]["size"] = new_size
            logger.info(f"Adjusted position sizes. New portfolio risk: {self.calculate_portfolio_risk():.4f}")

    def assess_risk(self, predicted_price: float, current_market_data: Dict[str, float]) -> Dict[str, Any]:
        current_price = current_market_data['close']
        price_change_percent = ((predicted_price - current_price) / current_price) * 100

        # Определение уровня риска на основе процентного изменения цены
        if abs(price_change_percent) > 5:
            risk_level = 'high'
        elif abs(price_change_percent) > 2:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Расчет риска портфеля (упрощенный пример)
        portfolio_risk = (abs(self.current_position) * abs(price_change_percent)) / self.current_capital

        return {
            'risk_level': risk_level,
            'price_change_percent': price_change_percent,
            'portfolio_risk': portfolio_risk,
            'current_capital': self.current_capital
        }

    def update_position(self, position_change: float):
        self.current_position += position_change

    def update_capital(self, capital_change: float):
        self.current_capital += capital_change

    def close_position(self):
        self.current_position = 0.0

risk_manager = RiskManager(initial_capital=100000)  # Здесь укажите начальный капитал

# Экспортируем risk_manager
__all__ = ['risk_manager']