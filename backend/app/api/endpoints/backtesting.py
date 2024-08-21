from fastapi import APIRouter, HTTPException
from app.services.backtesting import backtester
from pydantic import BaseModel
from datetime import date

router = APIRouter()

class BacktestRequest(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    initial_balance: float = 10000

@router.post("/run")
def run_backtest(request: BacktestRequest):
    try:
        result = backtester.run_backtest(
            request.symbol,
            request.start_date,
            request.end_date,
            request.initial_balance
        )
        metrics = backtester.calculate_metrics(result)
        return {"backtest_result": result, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
