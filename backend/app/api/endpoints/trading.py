from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.market_sentiment import MarketSentimentAnalyzer
from app.services.trading_history import TradingHistoryAnalyzer
from app.services.strategy_optimizer import StrategyOptimizer
from app.trading.ai_trader import ai_trader
from typing import Dict
from app.ml.neural_network import TradingNeuralNetwork, run_neural_network, get_prediction






router = APIRouter()

sentiment_analyzer = MarketSentimentAnalyzer()
history_analyzer = TradingHistoryAnalyzer()
strategy_optimizer = StrategyOptimizer()

@router.get("/market_sentiment")
def get_market_sentiment(db: Session = Depends(get_db)):
    sentiment = sentiment_analyzer.analyze_sentiment()
    return {"market_sentiment": sentiment}

@router.get("/trading-history")
async def get_trading_history(db: Session = Depends(get_db)):
    history = await history_analyzer.get_trading_history(db)
    return {"trading_history": history}

@router.post("/optimize-strategy")
async def optimize_strategy(params: Dict, db: Session = Depends(get_db)):
    optimized_strategy = await strategy_optimizer.optimize(params, db)
    return {"optimized_strategy": optimized_strategy}

@router.post("/start-ai-trading")
async def start_ai_trading(db: Session = Depends(get_db)):
    decision = await ai_trader.make_trading_decision(db)
    return {"trading_decision": decision}


ai_trader = TradingNeuralNetwork()