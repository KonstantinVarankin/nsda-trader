from pydantic import BaseModel
from datetime import datetime

class PredictionBase(BaseModel):
    symbol: str
    timestamp: datetime
    predicted_price: float
    confidence: float
    market_data_id: int

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: int

    class Config:
        orm_mode = True
