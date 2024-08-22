import os
import sys

# Добавляем корневую директорию проекта в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import logging
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.market_data import MarketData
from app.ml.neural_network import TradingNeuralNetwork, run_neural_network, get_prediction
from app.services.data_collection import DataCollector
from app.services.preprocessing import DataPreprocessor
from app.services.trading import TradingExecutor
from app.services.risk_management import RiskManager
from app.core.config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AIPredictionService:
    def __init__(self):
        self.model = None
        self.data_collector = DataCollector()
        self.data_preprocessor = DataPreprocessor()
        self.trading_executor = TradingExecutor()
        self.risk_manager = RiskManager(initial_capital=settings.INITIAL_CAPITAL)
        logger.info("AI prediction service initialized")

    async def initialize_model(self):
        historical_data = await self.get_historical_data()
        if historical_data is not None and not historical_data.empty:
            logger.debug(f"Historical data shape: {historical_data.shape}")
            logger.debug(f"Historical data columns: {historical_data.columns}")
            logger.debug(f"Historical data head:\n{historical_data.head()}")

            processed_data = self.data_preprocessor.preprocess(historical_data)
            logger.debug(f"Processed data shape: {processed_data.shape}")
            logger.debug(f"Processed data columns: {processed_data.columns}")
            logger.debug(f"Processed data head:\n{processed_data.head()}")

            if 'timestamp' in processed_data.columns:
                processed_data = processed_data.drop('timestamp', axis=1)

            logger.debug(f"Final processed data shape: {processed_data.shape}")
            logger.debug(f"Final processed data columns: {processed_data.columns}")
            logger.debug(f"Final processed data head:\n{processed_data.head()}")

            self.model = run_neural_network(processed_data.values)
            logger.info("AI model trained successfully")
        else:
            logger.error("Failed to initialize model: No historical data available")

    async def get_historical_data(self) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        historical_data = await self.data_collector.get_historical_data(start_date, end_date)
        return historical_data

    async def get_latest_market_data(self) -> pd.DataFrame:
        latest_data = await self.data_collector.get_latest_data()
        return latest_data

    async def get_latest_data(self):
        # Здесь мы будем получать самые последние данные для предсказания
        # В реальном приложении это может быть запрос к API биржи или базе данных
        # Для примера, мы просто возьмем последние 30 записей из исторических данных
        historical_data = await self.get_historical_data()
        if historical_data is not None and not historical_data.empty:
            return historical_data.tail(30)
        else:
            return None
    async def check_risk_levels(self, prediction: float, market_data: pd.DataFrame) -> Dict[str, Any]:
        return await self.risk_manager.check_risk(prediction, market_data)

    async def execute_trade(self, prediction: float) -> Dict[str, Any]:
        return await self.trading_executor.execute_trade(prediction)

    async def run_prediction_cycle(self):
        try:
            if self.model is None:
                await self.initialize_model()

            latest_data = await self.get_latest_data()
            if latest_data is not None and not latest_data.empty:
                logger.debug(f"Latest data shape: {latest_data.shape}")
                logger.debug(f"Latest data columns: {latest_data.columns}")
                logger.debug(f"Latest data head:\n{latest_data.head()}")

                processed_data = self.data_preprocessor.preprocess(latest_data)
                logger.debug(f"Processed data shape: {processed_data.shape}")
                logger.debug(f"Processed data columns: {processed_data.columns}")
                logger.debug(f"Processed data head:\n{processed_data.head()}")

                if 'timestamp' in processed_data.columns:
                    processed_data = processed_data.drop('timestamp', axis=1)

                input_data = processed_data.values.reshape((1, processed_data.shape[0], processed_data.shape[1]))
                logger.debug(f"Input data shape: {input_data.shape}")

                prediction = get_prediction(self.model, input_data)
                logger.info(f"Prediction: {prediction}")

                return {"status": "success", "prediction": prediction.tolist()}
            else:
                return {"status": "error", "message": "No data available for prediction"}
        except Exception as e:
            logger.error(f"Error in prediction cycle: {str(e)}", exc_info=True)
            return {
                    "status": "success",
                    "prediction": denormalized_prediction,
                    "timestamp": datetime.now().isoformat(),
                    "prediction_type": "future_price",  # или "price_change" и т.д.
                    "time_horizon": "1 day",  # или любой другой период
                    "current_price": current_price  # добавьте текущую цену для сравнения
            }

ai_service = AIPredictionService()

async def main():
    result = await ai_service.run_prediction_cycle()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())