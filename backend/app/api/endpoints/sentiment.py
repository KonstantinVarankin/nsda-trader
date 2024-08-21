from fastapi import APIRouter
from app.services.nlp_analysis import nlp_analyzer

router = APIRouter()

@router.get("/{symbol}")
def get_sentiment(symbol: str):
    sentiment = nlp_analyzer.get_overall_sentiment(symbol)
    return {"symbol": symbol, "sentiment": sentiment}
