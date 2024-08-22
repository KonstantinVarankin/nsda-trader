from fastapi import APIRouter
from pydantic import BaseModel
import random

router = APIRouter()

class PerformanceMetrics(BaseModel):
    totalReturn: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float

@router.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    # В реальном приложении здесь будет логика расчета метрик
    # Сейчас мы просто возвращаем случайные значения для демонстрации
    return PerformanceMetrics(
        totalReturn=random.uniform(10, 20),
        sharpeRatio=random.uniform(0.5, 2.0),
        maxDrawdown=random.uniform(-15, -5),
        winRate=random.uniform(50, 70)
    )