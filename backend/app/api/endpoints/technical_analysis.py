from fastapi import APIRouter
from app.services.technical_analysis import technical_analyzer

router = APIRouter()

@router.get("/indicators/{symbol}")
def get_technical_indicators(symbol: str):
    return technical_analyzer.get_latest_indicators(symbol)

@router.get("/signals/{symbol}")
def get_trading_signals(symbol: str):
    return technical_analyzer.generate_signals(symbol)
