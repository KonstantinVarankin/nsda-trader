import asyncio
import logging
from app.api.api_v1.endpoints.settings import get_settings
from app.services.notification_service import send_trade_notification, send_warning_notification\nfrom app.api.api_v1.endpoints.trades import add_trade
from typing import Dict
import random  # Для симуляции цен, в реальном приложении используйте API биржи

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

import numpy as np

class TradingService:
    def __init__(self):
        self.is_running = False
        self.price_cache: Dict[str, list] = {}
        self.ma_short_cache: Dict[str, float] = {}
        self.ma_long_cache: Dict[str, float] = {}
        self.rsi_cache: Dict[str, float] = {}
    def __init__(self):
        self.is_running = False
        self.price_cache: Dict[str, list] = {}
        self.ma_short_cache: Dict[str, float] = {}
        self.ma_long_cache: Dict[str, float] = {}
        self.rsi_cache: Dict[str, float] = {}
    def __init__(self):
        self.is_running = False
        self.price_cache: Dict[str, float] = {}
        self.moving_averages: Dict[str, list] = {}

    async def start_trading(self):
        self.is_running = True
        while self.is_running:
            settings = await get_settings()
            if not settings.tradingEnabled:
                await asyncio.sleep(60)  # Проверяем каждую минуту, включена ли торговля
                continue

            for pair in settings.tradingPairs.split(','):
                await self.check_and_execute_trade(pair, settings)

            await asyncio.sleep(60)  # Ждем минуту перед следующей итерацией

    async def check_and_execute_trade(self, pair: str, settings):
        current_price = await self.get_current_price(pair)
        self.update_indicators(pair, current_price, settings)

        if self.should_buy(pair, current_price, settings):
            amount = min(settings.maxTradeAmount, self.calculate_position_size(settings.riskLevel))
            await self.execute_trade(pair, 'buy', amount, current_price)
            await send_trade_notification(pair, 'buy', amount, current_price)
            logger.info(f"Buy signal for {pair} at {current_price}")
        elif self.should_sell(pair, current_price, settings):
            amount = min(settings.maxTradeAmount, self.calculate_position_size(settings.riskLevel))
            await self.execute_trade(pair, 'sell', amount, current_price)
            await send_trade_notification(pair, 'sell', amount, current_price)
            logger.info(f"Sell signal for {pair} at {current_price}")

    async def get_current_price(self, pair: str) -> float:
        # В реальном приложении здесь будет запрос к API биржи
        # Для примера используем случайные колебания цены
        if pair not in self.price_cache:
            self.price_cache[pair] = random.uniform(1000, 50000)
        else:
            change = random.uniform(-50, 50)
            self.price_cache[pair] += change
        return self.price_cache[pair]

    def update_indicators(self, pair: str, price: float, settings):
        if pair not in self.price_cache:
            self.price_cache[pair] = []
        self.price_cache[pair].append(price)

        # Update MA
        if len(self.price_cache[pair]) >= settings.maPeriodsLong:
            self.ma_short_cache[pair] = np.mean(self.price_cache[pair][-settings.maPeriodsShort:])
            self.ma_long_cache[pair] = np.mean(self.price_cache[pair][-settings.maPeriodsLong:])

        # Update RSI
        if len(self.price_cache[pair]) >= settings.rsiPeriods + 1:
            delta = np.diff(self.price_cache[pair][-settings.rsiPeriods-1:])
            gains = delta[delta > 0]
            losses = -delta[delta < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            self.rsi_cache[pair] = 100 - (100 / (1 + rs))

        # Keep only necessary historical data
        max_periods = max(settings.maPeriodsLong, settings.rsiPeriods + 1)
        self.price_cache[pair] = self.price_cache[pair][-max_periods:]
        if pair not in self.moving_averages:
            self.moving_averages[pair] = []
        self.moving_averages[pair].append(price)
        if len(self.moving_averages[pair]) > 20:  # 20-периодная скользящая средняя
            self.moving_averages[pair].pop(0)

    def calculate_moving_average(self, pair: str) -> float:
        if pair in self.moving_averages and len(self.moving_averages[pair]) > 0:
            return sum(self.moving_averages[pair]) / len(self.moving_averages[pair])
        return 0

    def should_buy(self, pair: str, current_price: float, settings) -> bool:
        if settings.strategy == 'MA_CROSSOVER':
            if pair not in self.ma_short_cache or pair not in self.ma_long_cache:
                return False
            return self.ma_short_cache[pair] > self.ma_long_cache[pair]
        elif settings.strategy == 'RSI':
            if pair not in self.rsi_cache:
                return False
            return self.rsi_cache[pair] < settings.rsiBuyThreshold
        return False
        if settings.strategy == 'MA_CROSSOVER':
            if pair not in self.ma_short_cache or pair not in self.ma_long_cache:
                return False
            return self.ma_short_cache[pair] > self.ma_long_cache[pair]
        elif settings.strategy == 'RSI':
            if pair not in self.rsi_cache:
                return False
            return self.rsi_cache[pair] < settings.rsiBuyThreshold
        return False
        ma = self.calculate_moving_average(pair)
        if ma == 0:
            return False
        return current_price > ma * 1.02  # Покупаем, если цена на 2% выше MA

    def should_sell(self, pair: str, current_price: float, settings) -> bool:
        if settings.strategy == 'MA_CROSSOVER':
            if pair not in self.ma_short_cache or pair not in self.ma_long_cache:
                return False
            return self.ma_short_cache[pair] < self.ma_long_cache[pair]
        elif settings.strategy == 'RSI':
            if pair not in self.rsi_cache:
                return False
            return self.rsi_cache[pair] > settings.rsiSellThreshold
        return False
        if settings.strategy == 'MA_CROSSOVER':
            if pair not in self.ma_short_cache or pair not in self.ma_long_cache:
                return False
            return self.ma_short_cache[pair] < self.ma_long_cache[pair]
        elif settings.strategy == 'RSI':
            if pair not in self.rsi_cache:
                return False
            return self.rsi_cache[pair] > settings.rsiSellThreshold
        return False
        ma = self.calculate_moving_average(pair)
        if ma == 0:
            return False
        return current_price < ma * 0.98  # Продаем, если цена на 2% ниже MA

    def calculate_position_size(self, risk_level: str) -> float:
        if risk_level == 'low':
            return 100
        elif risk_level == 'medium':
            return 500
        else:
            return 1000

    async def execute_trade(self, pair: str, action: str, amount: float, price: float):
        # В реальном приложении здесь будет выполнение ордера на бирже
        logger.info(f"Executing {action} trade for {amount} of {pair} at {price}")
        # Симуляция задержки выполнения ордера
        await asyncio.sleep(1)
        logger.info(f"Trade executed: {action} {amount} {pair} at {price}")
        add_trade(pair, action, amount, price)

trading_service = TradingService()



