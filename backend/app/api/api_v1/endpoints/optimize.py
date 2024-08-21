from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from app.services.backtesting_service import backtesting_service
from app.models.trading_models import TradingStrategy, BacktestResult

router = APIRouter()

class OptimizationParams(BaseModel):
    strategy: str
    symbol: str
    interval: str
    startDate: str
    endDate: str
    maPeriodsShortMin: int = None
    maPeriodsShortMax: int = None
    maPeriodsLongMin: int = None
    maPeriodsLongMax: int = None
    rsiPeriodsMin: int = None
    rsiPeriodsMax: int = None
    rsiBuyThresholdMin: int = None
    rsiBuyThresholdMax: int = None
    rsiSellThresholdMin: int = None
    rsiSellThresholdMax: int = None

class OptimizationResult(BaseModel):
    strategy: str
    params: dict
    performance: float
    backtest_result: BacktestResult = None

@router.post("/optimize", response_model=OptimizationResult)
async def optimize_strategy(params: OptimizationParams):
    try:
        await backtesting_service.load_data(params.symbol, params.interval, params.startDate, params.endDate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while loading data")

    if params.strategy == 'MA_CROSSOVER':
        optimized_params = {
            'maPeriodsShort': np.random.randint(params.maPeriodsShortMin, params.maPeriodsShortMax),
            'maPeriodsLong': np.random.randint(params.maPeriodsLongMin, params.maPeriodsLongMax)
        }
    elif params.strategy == 'RSI':
        optimized_params = {
            'rsiPeriods': np.random.randint(params.rsiPeriodsMin, params.rsiPeriodsMax),
            'rsiBuyThreshold': np.random.randint(params.rsiBuyThresholdMin, params.rsiBuyThresholdMax),
            'rsiSellThreshold': np.random.randint(params.rsiSellThresholdMin, params.rsiSellThresholdMax)
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    performance = np.random.uniform(0, 100)

    return OptimizationResult(
        strategy=params.strategy,
        params=optimized_params,
        performance=performance,
        backtest_result=None  # Здесь должен быть реальный результат бэктестинга
    )





