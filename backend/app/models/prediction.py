from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.data_collection import get_market_data
from app.ml.model import predict

router = APIRouter()


@router.get("/predict")
def get_prediction(db: Session = Depends(get_db)):
    market_data = get_market_data()
    if market_data is None:
        return {"error": "Failed to retrieve market data"}

    prediction = predict(market_data)
    return {"prediction": prediction}