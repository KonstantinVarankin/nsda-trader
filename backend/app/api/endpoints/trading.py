from app.services.market_sentiment import MarketSentimentAnalyzer
from app.services.trading_history import TradingHistoryAnalyzer
from app.services.strategy_optimizer import StrategyOptimizer
from fastapi import APIRouter
from app.trading.ai_trader import ai_trader

sentiment_analyzer = MarketSentimentAnalyzer()
history_analyzer = TradingHistoryAnalyzer()
strategy_optimizer = StrategyOptimizer()
router = APIRouter()
@router.get("/market-sentiment")
async def get_market_sentiment():
    return sentiment_analyzer.get_market_sentiment()

@router.get("/trading-history")
async def get_trading_history():
    return history_analyzer.get_trading_history()

@router.post("/optimize-strategy")
async def optimize_strategy(params: dict):
    return strategy_optimizer.optimize(params)




@router.post("/start-ai-trading")
async def start_ai_trading():
    decision = ai_trader.make_trading_decision()
    return {"trading_decision": decision}