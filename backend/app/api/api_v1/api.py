from fastapi import APIRouter

api_router = APIRouter()

# Импортируйте и включайте роутеры по отдельности
from .endpoints import data
api_router.include_router(data.router, prefix="/data", tags=["data"])

from .endpoints import trading
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])

from .endpoints import performance
api_router.include_router(performance.router, prefix="/performance", tags=["performance"])

from .endpoints import market_data
api_router.include_router(market_data.router, prefix="/market-data", tags=["market-data"])

from .endpoints import predictions
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])

from .endpoints import sentiment
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])