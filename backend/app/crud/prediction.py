from sqlalchemy.orm import Session
from app.models.prediction import Prediction
from datetime import datetime

def create_prediction(db: Session, symbol: str, timestamp: datetime, predicted_price: float, confidence: float, market_data_id: int):
    db_prediction = Prediction(symbol=symbol, timestamp=timestamp, predicted_price=predicted_price, confidence=confidence, market_data_id=market_data_id)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Prediction).offset(skip).limit(limit).all()

def get_predictions_by_symbol(db: Session, symbol: str):
    return db.query(Prediction).filter(Prediction.symbol == symbol).all()
