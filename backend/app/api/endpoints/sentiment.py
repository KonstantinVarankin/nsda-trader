from fastapi import APIRouter
from app.services.nlp_analysis import nlp_analyzer

router = APIRouter()

@router.get("/analyze")
def analyze_sentiment(query: str):
    sentiments = nlp_analyzer.analyze_news_sentiment(query)
    return {"sentiments": sentiments}
