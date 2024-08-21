from fastapi import APIRouter
from app.api.endpoints import market_data, predictions, sentiment, technical_analysis, decision, risk_management, backtesting, trading

api_router = APIRouter()
api_router.include_router(market_data.router, prefix="/market-data", tags=["market-data"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
api_router.include_router(technical_analysis.router, prefix="/technical-analysis", tags=["technical-analysis"])
api_router.include_router(decision.router, prefix="/decision", tags=["decision"])
api_router.include_router(risk_management.router, prefix="/risk-management", tags=["risk-management"])
api_router.include_router(backtesting.router, prefix="/backtesting", tags=["backtesting"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
