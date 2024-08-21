from pydantic import BaseModel
from typing import Dict

class TradingStrategy(BaseModel):
    name: str
    params: Dict[str, float]

class BacktestResult(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    daily_returns: List[float]
    equity_curve: List[float]
    dates: List[str]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float

