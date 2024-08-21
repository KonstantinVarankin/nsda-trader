from pydantic import BaseModel
from datetime import datetime

class MarketDataBase(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDataCreate(MarketDataBase):
    pass

class MarketData(MarketDataBase):
    id: int

    class Config:
        orm_mode = True
