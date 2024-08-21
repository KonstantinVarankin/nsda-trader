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

# � �������� ���������� ��� ������ ������ ��������� � ���� ������
trade_history: List[Trade] = []

@router.get("/trades", response_model=List[Trade])
async def get_trades():
    return trade_history

# ������� ��� ���������� ����� ������ � �������
def add_trade(pair: str, action: str, amount: float, price: float):
    trade = Trade(
        timestamp=datetime.datetime.now(),
        pair=pair,
        action=action,
        amount=amount,
        price=price
    )
    trade_history.append(trade)
    if len(trade_history) > 100:  # ������������ ������� ���������� 100 ��������
        trade_history.pop(0)
