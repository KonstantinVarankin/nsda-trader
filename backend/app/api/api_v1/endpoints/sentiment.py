from fastapi import FastAPI, APIRouter
from typing import Dict
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Создание экземпляра FastAPI
app = FastAPI()

# Создание роутера
router = APIRouter()

@router.get("/api/v1/market-sentiment")
async def get_market_sentiment() -> Dict[str, str]:
    logger.debug("Received request for market sentiment")
    return {"message": "Test response"}

@router.get("/api/v1/test-sentiment")
async def test_sentiment() -> Dict[str, float]:
    logger.debug("Received request for test sentiment")
    return {
        'positive': 0.3,
        'neutral': 0.4,
        'negative': 0.3,
        'overallSentiment': 'Neutral',
        'confidence': 50
    }

# Подключение роутера к приложению
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)