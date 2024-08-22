# app/services/binance_client.py

from binance.client import Client
from binance.exceptions import BinanceAPIException
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)

    async def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    async def execute_trade(self, symbol: str, side: str, quantity: float):
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"Order executed: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return None

    async def get_account_balance(self, asset: str) -> float:
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting balance for {asset}: {e}")
            return None

binance_client = BinanceClient()
