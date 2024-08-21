from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import List

router = APIRouter()

class TradingSettings(BaseModel):
    tradingEnabled: bool
    maxTradeAmount: float
    riskLevel: str
    tradingPairs: str
    stopLossPercentage: float
    takeProfitPercentage: float
    strategy: str
    maPeriodsShort: int
    maPeriodsLong: int
    rsiPeriods: int
    rsiBuyThreshold: int
    rsiSellThreshold: int
    tradingEnabled: bool
    maxTradeAmount: float
    riskLevel: str
    tradingPairs: str
    stopLossPercentage: float
    takeProfitPercentage: float
    strategy: str
    maPeriodsShort: int
    maPeriodsLong: int
    rsiPeriods: int
    rsiBuyThreshold: int
    rsiSellThreshold: int
    tradingEnabled: bool
    maxTradeAmount: float
    riskLevel: str
    tradingPairs: str
    stopLossPercentage: float
    takeProfitPercentage: float

    @validator('maxTradeAmount')
    def max_trade_amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('maxTradeAmount must be positive')
        return v

    @validator('strategy')
    def strategy_must_be_valid(cls, v):
        if v not in ['MA_CROSSOVER', 'RSI']:
            raise ValueError('strategy must be MA_CROSSOVER or RSI')
        return v

    @validator('maPeriodsShort', 'maPeriodsLong', 'rsiPeriods')
    def periods_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Periods must be positive')
        return v

    @validator('rsiBuyThreshold', 'rsiSellThreshold')
    def rsi_thresholds_must_be_valid(cls, v):
        if v < 0 or v > 100:
            raise ValueError('RSI thresholds must be between 0 and 100')
        return v

    @validator('strategy')
    def strategy_must_be_valid(cls, v):
        if v not in ['MA_CROSSOVER', 'RSI']:
            raise ValueError('strategy must be MA_CROSSOVER or RSI')
        return v

    @validator('maPeriodsShort', 'maPeriodsLong', 'rsiPeriods')
    def periods_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Periods must be positive')
        return v

    @validator('rsiBuyThreshold', 'rsiSellThreshold')
    def rsi_thresholds_must_be_valid(cls, v):
        if v < 0 or v > 100:
            raise ValueError('RSI thresholds must be between 0 and 100')
        return v

    @validator('riskLevel')
    def risk_level_must_be_valid(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('riskLevel must be low, medium, or high')
        return v

    @validator('stopLossPercentage', 'takeProfitPercentage')
    def percentages_must_be_valid(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v

# В реальном приложении эти настройки должны храниться в базе данных
current_settings = TradingSettings(
    tradingEnabled=False,
    maxTradeAmount=1000.0,
    riskLevel='medium',
    tradingPairs='BTC/USDT,ETH/USDT',
    stopLossPercentage=2.0,
    takeProfitPercentage=5.0,
    strategy='MA_CROSSOVER',
    maPeriodsShort=10,
    maPeriodsLong=20,
    rsiPeriods=14,
    rsiBuyThreshold=30,
    rsiSellThreshold=70
    tradingEnabled=False,
    maxTradeAmount=1000.0,
    riskLevel='medium',
    tradingPairs='BTC/USDT,ETH/USDT',
    stopLossPercentage=2.0,
    takeProfitPercentage=5.0,
    strategy='MA_CROSSOVER',
    maPeriodsShort=10,
    maPeriodsLong=20,
    rsiPeriods=14,
    rsiBuyThreshold=30,
    rsiSellThreshold=70
    tradingEnabled=False,
    maxTradeAmount=1000.0,
    riskLevel='medium',
    tradingPairs='BTC/USDT,ETH/USDT',
    stopLossPercentage=2.0,
    takeProfitPercentage=5.0
)

@router.get("/settings", response_model=TradingSettings)
async def get_settings():
    return current_settings

@router.post("/settings", response_model=TradingSettings)
async def update_settings(settings: TradingSettings):
    global current_settings
    current_settings = settings
    return current_settings


