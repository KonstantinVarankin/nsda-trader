from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app import crud
from app.schemas.market_data import MarketDataCreate, MarketData

router = APIRouter()

@router.post("/", response_model=MarketData)
def create_market_data(market_data: MarketDataCreate, db: Session = Depends(get_db)):
    return crud.market_data.create_market_data(db=db, **market_data.dict())

@router.get("/", response_model=List[MarketData])
def read_market_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    market_data = crud.market_data.get_market_data(db, skip=skip, limit=limit)
    return market_data

@router.get("/{symbol}", response_model=List[MarketData])
def read_market_data_by_symbol(symbol: str, db: Session = Depends(get_db)):
    market_data = crud.market_data.get_market_data_by_symbol(db, symbol=symbol)
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not found")
    return market_data
