from sqlalchemy import Column, Integer, Float, DateTime, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    predicted_price = Column(Float)
    confidence = Column(Float)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))
    
    market_data = relationship("MarketData")
