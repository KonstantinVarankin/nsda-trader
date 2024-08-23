import os
import sys
import numpy as np
import pandas as pd
from app.ml.neural_network import TradingNeuralNetwork, run_neural_network, get_prediction
from app.services.data_collection import DataCollector
from app.services.preprocessing import DataPreprocessor
from app.services.trading import TradingExecutor
from app.services.risk_management import RiskManager
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIPredictionService:
    def __init__(self):
        self.model = None
        logger.info("AI prediction service initialized")

    async def initialize_model(self):
        historical_data = await self.get_historical_data()
            logger.info("AI model trained successfully")


    async def run_prediction_cycle(self):
        try:
            if self.model is None:
                await self.initialize_model()






        except Exception as e:

ai_service = AIPredictionService()

    result = await ai_service.run_prediction_cycle()
    print(result)
