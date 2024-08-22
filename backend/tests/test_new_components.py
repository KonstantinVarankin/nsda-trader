import pytest
from app.services.market_sentiment import MarketSentimentAnalyzer
from app.services.trading_history import TradingHistoryAnalyzer
from app.services.strategy_optimizer import StrategyOptimizer

def test_market_sentiment_analyzer():
    analyzer = MarketSentimentAnalyzer()
    sentiment = analyzer.get_market_sentiment()
    assert 'positive' in sentiment
    assert 'negative' in sentiment
    assert 'neutral' in sentiment
    assert 'overallSentiment' in sentiment
    assert 'confidence' in sentiment

def test_trading_history_analyzer():
    analyzer = TradingHistoryAnalyzer()
    history = analyzer.get_trading_history()
    assert 'dates' in history
    assert 'portfolioValues' in history
    assert 'totalProfitLoss' in history
    assert 'winRate' in history
    assert 'sharpeRatio' in history

def test_strategy_optimizer():
    optimizer = StrategyOptimizer()
    params = {'populationSize': 50, 'generations': 10}
    result = optimizer.optimize(params)
    assert 'bestFitness' in result
    assert 'bestParameters' in result