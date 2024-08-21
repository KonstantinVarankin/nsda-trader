from fastapi import APIRouter
from app.api.endpoints import ai_trading

api_router = APIRouter()
api_router.include_router(ai_trading.router, prefix="/ai-trading", tags=["ai-trading"])
