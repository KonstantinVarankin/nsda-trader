from fastapi import APIRouter, HTTPException, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from app.services.backtesting_service import backtesting_service
from app.models.trading_models import TradingStrategy, BacktestResult
import itertools

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
    fastPeriodMin: int = None
    fastPeriodMax: int = None
    slowPeriodMin: int = None
    slowPeriodMax: int = None
    signalPeriodMin: int = None
    signalPeriodMax: int = None
    strategy: str
    symbol: str
    interval: str
    startDate: str
    endDate: str
    maPeriodsShortMin: int
    maPeriodsShortMax: int
    maPeriodsLongMin: int
    maPeriodsLongMax: int
    rsiPeriodsMin: int
    rsiPeriodsMax: int
    rsiBuyThresholdMin: int
    rsiBuyThresholdMax: int
    rsiSellThresholdMin: int
    rsiSellThresholdMax: int
    strategy: str
    startDate: str
    endDate: str
    maPeriodsShortMin: int
    maPeriodsShortMax: int
    maPeriodsLongMin: int
    maPeriodsLongMax: int
    rsiPeriodsMin: int
    rsiPeriodsMax: int
    rsiBuyThresholdMin: int
    rsiBuyThresholdMax: int
    rsiSellThresholdMin: int
    rsiSellThresholdMax: int

class OptimizationResult(BaseModel):
    strategy: str
    params: dict
    performance: float
    backtest_result: BacktestResult
    strategy: str
    params: dict
    performance: float

@router.post("/optimize", response_model=OptimizationResult)
async def optimize_strategy(params: OptimizationParams):
    try:
        await backtesting_service.load_data(params.symbol, params.interval, params.startDate, params.endDate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while loading data")
    await backtesting_service.load_data(params.symbol, params.interval, params.startDate, params.endDate)
    backtesting_service.load_data(params.startDate, params.endDate)

    if params.strategy == 'MA_CROSSOVER':
        param_ranges = {
            'maPeriodsShort': range(params.maPeriodsShortMin, params.maPeriodsShortMax + 1),
            'maPeriodsLong': range(params.maPeriodsLongMin, params.maPeriodsLongMax + 1)
        }
    elif params.strategy == 'RSI':
        param_ranges = {
            'rsiPeriods': range(params.rsiPeriodsMin, params.rsiPeriodsMax + 1),
            'rsiBuyThreshold': range(params.rsiBuyThresholdMin, params.rsiBuyThresholdMax + 1),
            'rsiSellThreshold': range(params.rsiSellThresholdMin, params.rsiSellThresholdMax + 1)
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    initial_strategy = TradingStrategy(name=params.strategy, params={k: v[0] for k, v in param_ranges.items()})
    optimized_strategy = backtesting_service.optimize(initial_strategy, param_ranges)
    backtest_result = backtesting_service.backtest(optimized_strategy)

    return OptimizationResult(
        strategy=optimized_strategy.name,
        params=optimized_strategy.params,
        performance=backtest_result.sharpe_ratio,
        backtest_result=backtest_result
        strategy=optimized_strategy.name,
        params=optimized_strategy.params,
        performance=backtest_result.sharpe_ratio
    )
async def optimize_strategy(params: OptimizationParams):
    try:
        await backtesting_service.load_data(params.symbol, params.interval, params.startDate, params.endDate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while loading data")
    await backtesting_service.load_data(params.symbol, params.interval, params.startDate, params.endDate)
    # В реальном приложении здесь будет сложная логика оптимизации
    # Для примера мы просто вернем случайный результат
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

    performance = np.random.uniform(0, 100)  # Случайное значение производительности

    return OptimizationResult(
        strategy=optimized_strategy.name,
        params=optimized_strategy.params,
        performance=backtest_result.sharpe_ratio,
        backtest_result=backtest_result
        strategy=params.strategy,
        params=optimized_params,
        performance=performance
    )





