from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app import crud
from app.schemas.prediction import PredictionCreate, Prediction

router = APIRouter()

@router.post("/", response_model=Prediction)
def create_prediction(prediction: PredictionCreate, db: Session = Depends(get_db)):
    return crud.prediction.create_prediction(db=db, **prediction.dict())

@router.get("/", response_model=List[Prediction])
def read_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    predictions = crud.prediction.get_predictions(db, skip=skip, limit=limit)
    return predictions

@router.get("/{symbol}", response_model=List[Prediction])
def read_predictions_by_symbol(symbol: str, db: Session = Depends(get_db)):
    predictions = crud.prediction.get_predictions_by_symbol(db, symbol=symbol)
    if predictions is None:
        raise HTTPException(status_code=404, detail="Predictions not found")
    return predictions
