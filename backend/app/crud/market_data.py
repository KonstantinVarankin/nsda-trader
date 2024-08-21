from sqlalchemy.orm import Session
from app.models.market_data import MarketData
from datetime import datetime

def create_market_data(db: Session, symbol: str, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float):
    db_market_data = MarketData(symbol=symbol, timestamp=timestamp, open=open, high=high, low=low, close=close, volume=volume)
    db.add(db_market_data)
    db.commit()
    db.refresh(db_market_data)
    return db_market_data

def get_market_data(db: Session, skip: int = 0, limit: int = 100):
    return db.query(MarketData).offset(skip).limit(limit).all()

def get_market_data_by_symbol(db: Session, symbol: str):
    return db.query(MarketData).filter(MarketData.symbol == symbol).all()
