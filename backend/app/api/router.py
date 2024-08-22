from app.api.endpoints import data, predictions, trading, performance, market_data, sentiment

api_router = APIRouter()
api_router.include_router(ai_trading.router, prefix="/ai-trading", tags=["ai-trading"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(performance.router, prefix="/performance", tags=["performance"])
api_router.include_router(market_data.router, prefix="/market-data", tags=["market_data"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(sentiment.router, prefix="/market-sentiment", tags=["sentiment"])