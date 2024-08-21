from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import datetime

router = APIRouter()

class Trade(BaseModel):
    timestamp: datetime.datetime
    pair: str
    action: str
    amount: float
    price: float

# В реальном приложении эти данные должны храниться в базе данных
trade_history: List[Trade] = []

@router.get("/trades", response_model=List[Trade])
async def get_trades():
    return trade_history

# Функция для добавления новой сделки в историю
def add_trade(pair: str, action: str, amount: float, price: float):
    trade = Trade(
        timestamp=datetime.datetime.now(),
        pair=pair,
        action=action,
        amount=amount,
        price=price
    )
    trade_history.append(trade)
    if len(trade_history) > 100:  # Ограничиваем историю последними 100 сделками
        trade_history.pop(0)
