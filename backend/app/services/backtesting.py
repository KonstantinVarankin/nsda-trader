import pandas as pd
import numpy as np
from app.services.technical_analysis import technical_analyzer
from app.services.decision_making import decision_maker
from app.services.risk_management import risk_manager

class Backtester:
    def __init__(self):
        self.technical_analyzer = technical_analyzer
        self.decision_maker = decision_maker
        self.risk_manager = risk_manager

    def run_backtest(self, symbol, start_date, end_date, initial_balance=10000):
        # Get historical data
        df = self.technical_analyzer.get_market_data(symbol)
        df = df.loc[start_date:end_date]

        # Initialize variables
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []

        for date, row in df.iterrows():
            # Calculate indicators
            indicators = self.technical_analyzer.calculate_indicators(symbol)
            current_price = row['close']

            # Make decision
            decision = self.decision_maker.make_decision(symbol)

            # Execute trade based on decision
            if decision in ["Strong Buy", "Buy"] and position == 0:
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(symbol, balance)
                position = position_size
                entry_price = current_price
                cost = position * entry_price
                balance -= cost
                trades.append({"date": date, "action": "buy", "price": current_price, "quantity": position})

            elif decision in ["Strong Sell", "Sell"] and position > 0:
                # Close position
                sale_value = position * current_price
                balance += sale_value
                trades.append({"date": date, "action": "sell", "price": current_price, "quantity": position})
                position = 0
                entry_price = 0

            # Check for stop loss
            if position > 0:
                stop_loss = self.risk_manager.set_stop_loss(symbol, entry_price, True)
                if current_price <= stop_loss:
                    sale_value = position * current_price
                    balance += sale_value
                    trades.append({"date": date, "action": "stop_loss", "price": current_price, "quantity": position})
                    position = 0
                    entry_price = 0

        # Close any remaining position at the end
        if position > 0:
            sale_value = position * df.iloc[-1]['close']
            balance += sale_value
            trades.append({"date": df.index[-1], "action": "close", "price": df.iloc[-1]['close'], "quantity": position})

        return {
            "initial_balance": initial_balance,
            "final_balance": balance,
            "return": (balance - initial_balance) / initial_balance * 100,
            "trades": trades
        }

    def calculate_metrics(self, backtest_result):
        trades = pd.DataFrame(backtest_result['trades'])
        if trades.empty:
            return {"error": "No trades were executed during the backtest period."}

        # Calculate returns
        trades['return'] = trades.apply(lambda row: row['price'] / trades['price'].shift(1) - 1 if row['action'] == 'sell' else 0, axis=1)

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['return'] > 0])
        losing_trades = len(trades[trades['return'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = trades[trades['return'] > 0]['return'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades['return'] < 0]['return'].mean()) if losing_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Calculate drawdown
        cumulative_returns = (1 + trades['return']).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdown.max()

        return {
            "total_return": backtest_result['return'],
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }

backtester = Backtester()
