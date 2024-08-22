import logging
import os
import sys
import numpy as np
import pandas as pd

# Получаем путь к директории backend
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, backend_dir)

from app.ml.neural_network import TradingNeuralNetwork, run_neural_network, get_prediction
from app.services.data_collection import DataCollector
from app.services.preprocessing import DataPreprocessor
from app.services.trading import TradingExecutor
from app.services.risk_management import RiskManager
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIPredictionService:
    def __init__(self):
        self.model = None
        logger.info("AI prediction service initialized")

    async def initialize_model(self):
        # Здесь мы должны загрузить исторические данные для обучения модели
        # Для примера, предположим, что у нас есть функция get_historical_data()
        historical_data = await self.get_historical_data()
        self.model = run_neural_network(historical_data)
        logger.info("AI model trained successfully")

    async def get_historical_data(self):
        # Эту функцию нужно реализовать для получения исторических данных
        # Для примера, вернем случайные данные
        return np.random.rand(1000, 1)

    async def run_prediction_cycle(self):
        try:
            if self.model is None:
                await self.initialize_model()

            # Получение последних рыночных данных

            # Предобработка данных
            processed_data = preprocess_data(market_data)
            logger.info("Data preprocessed successfully")

            # Получение предсказания от модели
            prediction = get_prediction(self.model, processed_data)
            logger.info(f"Model prediction: {prediction}")

            # Проверка уровней риска
            risk_check = await check_risk_levels(prediction, market_data)
            if not risk_check['safe']:
                logger.warning(f"Risk check failed: {risk_check['reason']}")
                return

            # Выполнение торговой операции на основе предсказания
            trade_result = await execute_trade(prediction)
            logger.info(f"Trade executed: {trade_result}")

            return {
                "prediction": prediction,
                "trade_result": trade_result
            }

        except Exception as e:
            logger.error(f"Error in prediction cycle: {str(e)}")
            return {"error": str(e)}

ai_service = AIPredictionService()

if __name__ == "__main__":
    import asyncio

    async def test_run():
        result = await ai_service.run_prediction_cycle()
        print(result)

    asyncio.run(test_run())