import asyncio
import sys
import os
from datetime import datetime, timedelta
import traceback
import pandas as pd
from tqdm import tqdm
import logging

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.ml.model import NSDATradingModel
from app.services.binance_service import binance_service
from app.services.cache_service import cache_service

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def fetch_all_symbols():
    exchange_info = await binance_service.get_exchange_info()
    return [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING']


async def fetch_historical_data(symbol, interval='1d', limit=1000):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)
        historical_data = await binance_service.get_historical_data(
            symbol,
            interval,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if historical_data is not None and not historical_data.empty:
            return historical_data
        else:
            logger.warning(f"Нет данных для {symbol}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Ошибка при получении исторических данных для {symbol}: {e}")
        return pd.DataFrame()


async def train_model_for_symbol(symbol, model):
    try:
        logger.info(f"Загрузка данных для {symbol}")
        historical_data = await fetch_historical_data(symbol)

        if historical_data.empty:
            logger.warning(f"Не удалось получить данные для {symbol}")
            return

        logger.info(f"Получено {len(historical_data)} записей для {symbol}")

        start_date = historical_data.index[0]
        end_date = historical_data.index[-1]

        logger.info(
            f"Обучение модели на данных {symbol} с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        model.train(symbol, historical_data)
        logger.info(f"Обучение модели для {symbol} завершено")

        # Оценка модели
        evaluation = model.evaluate(symbol, historical_data.tail(100))
        logger.info(f"Оценка модели для {symbol}: MSE = {evaluation['mse']:.4f}, RMSE = {evaluation['rmse']:.4f}")

        # Пример предсказания
        prediction_date = end_date + timedelta(days=1)
        prediction = model.predict(symbol, prediction_date.strftime('%Y-%m-%d'))
        logger.info(f"Предсказание для {symbol} на {prediction_date.strftime('%Y-%m-%d')}: {prediction:.2f}")

    except Exception as e:
        logger.error(f"Ошибка при обучении модели для {symbol}: {e}")


async def train_all_models():
    try:
        await binance_service.initialize()
        model = NSDATradingModel()

        symbols = await fetch_all_symbols()
        logger.info(f"Найдено {len(symbols)} торговых пар")

        tasks = [train_model_for_symbol(symbol, model) for symbol in symbols]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Обучение моделей"):
            await task

        logger.info("Обучение завершено для всех торговых пар")

    except Exception as e:
        logger.error(f"Общая ошибка: {e}")
        logger.error(traceback.format_exc())
    finally:
        await binance_service.close()


async def main():
    # Очистка кэша перед началом
    cache_service.clear_cache()

    await train_all_models()


if __name__ == "__main__":
    asyncio.run(main())