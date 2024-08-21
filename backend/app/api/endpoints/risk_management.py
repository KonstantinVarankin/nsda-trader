from fastapi import APIRouter, HTTPException
from app.services.risk_management import risk_manager
from pydantic import BaseModel

router = APIRouter()

class PositionSizeRequest(BaseModel):
    symbol: str
    account_balance: float
    risk_per_trade: float = 0.02
    stop_loss_percent: float = 0.02

class AdjustPositionRequest(BaseModel):
    symbol: str
    current_position: float
    account_balance: float

@router.post("/position-size")
def calculate_position_size(request: PositionSizeRequest):
    try:
        position_size = risk_manager.calculate_position_size(
            request.symbol, 
            request.account_balance, 
            request.risk_per_trade, 
            request.stop_loss_percent
        )
        return {"symbol": request.symbol, "position_size": position_size}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/adjust-position")
def adjust_position(request: AdjustPositionRequest):
    try:
        adjusted_position = risk_manager.adjust_position(
            request.symbol,
            request.current_position,
            request.account_balance
        )
        return {"symbol": request.symbol, "adjusted_position": adjusted_position}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/volatility/{symbol}")
def get_volatility(symbol: str):
    try:
        volatility = risk_manager.calculate_volatility(symbol)
        return {"symbol": symbol, "volatility": volatility}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/max-drawdown/{symbol}")
def check_max_drawdown(symbol: str):
    try:
        is_within_limit = risk_manager.check_max_drawdown(symbol)
        return {"symbol": symbol, "is_within_limit": is_within_limit}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
