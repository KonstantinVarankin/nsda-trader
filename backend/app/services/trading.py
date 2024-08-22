import ccxt
from app.core.config import settings

class TradingExecutor:
    def __init__(self):
        self.exchange = getattr(ccxt, settings.EXCHANGE_NAME)({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
        })

    def place_market_order(self, symbol, side, amount):
        """Place a market order."""
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            return order
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def place_limit_order(self, symbol, side, amount, price):
        """Place a limit order."""
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            return order
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_account_balance(self):
        """Get account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def execute_trading_signal(self, symbol, signal, amount):
        """Execute a trading signal."""
        if signal == 1:
            return self.place_market_order(symbol, 'buy', amount)
        elif signal == -1:
            return self.place_market_order(symbol, 'sell', amount)
        else:
            print("No trade executed (hold signal)")
            return None

# Пример использования:
# executor = TradingExecutor()
# balance = executor.get_account_balance()
# print(f"Current balance: {balance}")
#
# for signal in trading_signals:
#     result = executor.execute_trading_signal('BTC/USDT', signal, 0.01)
#     if result:
#         print(f"Order executed: {result}")