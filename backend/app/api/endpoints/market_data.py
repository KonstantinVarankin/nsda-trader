from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
import ccxt

router = APIRouter()

@router.get("/")
def get_market_data(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100, db: Session = Depends(get_db)):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return {"data": ohlcv}
