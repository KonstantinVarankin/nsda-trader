from binance.client import Client
from binance.exceptions import BinanceAPIException
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ExchangeAPI:
    def __init__(self):
        self.client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)

    def get_account_balance(self):
        try:
            account_info = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) for asset in account_info['balances'] if float(asset['free']) > 0}
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return None

    def place_market_order(self, symbol, side, quantity):
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(self, symbol, side, quantity, price):
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def get_open_orders(self, symbol=None):
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            return orders
        except BinanceAPIException as e:
            logger.error(f"Error getting open orders: {e}")
            return None

    def cancel_order(self, symbol, order_id):
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            return result
        except BinanceAPIException as e:
            logger.error(f"Error cancelling order: {e}")
            return None

    def get_symbol_info(self, symbol):
        try:
            info = self.client.get_symbol_info(symbol)
            return info
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

exchange_api = ExchangeAPI()
