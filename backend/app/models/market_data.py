# app/models/market_data.py
from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# app/models/sentiment.py
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from .market_data import Base

class Sentiment(Base):
    __tablename__ = "sentiments"

    id = Column(Integer, primary_key=True, index=True)
    positive = Column(Float)
    neutral = Column(Float)
    negative = Column(Float)
    overallSentiment = Column(String)
    confidence = Column(Float)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))

    market_data = relationship("MarketData", back_populates="sentiment")