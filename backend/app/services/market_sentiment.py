import logging
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import Dict
from app.services.data_collection import get_market_data
from app.core.cache import timed_lru_cache
from fastapi import FastAPI

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Загрузка NLTK один раз при запуске приложения
nltk.download('vader_lexicon', quiet=True)

app = FastAPI()

class MarketSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, price_change: float) -> Dict[str, float]:
        logger.debug(f"Analyzing sentiment for price change: {price_change}")
        if price_change > 0:
            return {'pos': 0.6, 'neu': 0.4, 'neg': 0, 'compound': 0.6}
        elif price_change < 0:
            return {'pos': 0, 'neu': 0.4, 'neg': 0.6, 'compound': -0.6}
        else:
            return {'pos': 0, 'neu': 1, 'neg': 0, 'compound': 0}

    @timed_lru_cache(seconds=600)  # Увеличено время кэширования до 10 минут
    def get_market_sentiment(self) -> Dict[str, float]:
        logger.debug("Starting get_market_sentiment")
        try:
            market_data = get_market_data()
            logger.debug(f"Received market data: {market_data}")
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                'positive': 0,
                'neutral': 1,
                'negative': 0,
                'overallSentiment': 'Neutral',
                'confidence': 0
            }

        if market_data is None or len(market_data) < 2:
            logger.warning("Insufficient market data")
            return {
                'positive': 0,
                'neutral': 1,
                'negative': 0,
                'overallSentiment': 'Neutral',
                'confidence': 0
            }

        last_price = market_data['close'].iloc[-1]
        prev_price = market_data['close'].iloc[-2]
        price_change = (last_price - prev_price) / prev_price

        sentiment = self.analyze_sentiment(price_change)

        overall_sentiment = sentiment['compound']

        result = {
            'positive': sentiment['pos'],
            'neutral': sentiment['neu'],
            'negative': sentiment['neg'],
            'overallSentiment': 'Positive' if overall_sentiment > 0.05 else 'Negative' if overall_sentiment < -0.05 else 'Neutral',
            'confidence': abs(overall_sentiment) * 100
        }
        logger.debug(f"Returning sentiment: {result}")
        return result

analyzer = MarketSentimentAnalyzer()

@app.get("/api/v1/market-sentiment")
async def get_market_sentiment():
    logger.debug("Received request for market sentiment")
    try:
        sentiment = analyzer.get_market_sentiment()
        logger.debug(f"Sending response for market sentiment: {sentiment}")
        return sentiment
    except Exception as e:
        logger.error(f"Error processing market sentiment request: {e}")
        return {"error": "An error occurred while processing the request"}

@app.get("/api/v1/test-sentiment")
async def test_sentiment():
    logger.debug("Received request for test sentiment")
    return {
        'positive': 0.3,
        'neutral': 0.4,
        'negative': 0.3,
        'overallSentiment': 'Neutral',
        'confidence': 50
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)