from binance.client import Client
from app.core.config import settings

class ExchangeAPI:
    def __init__(self):
        self.client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)

    # Остальные методы класса...

exchange_api = ExchangeAPI()
