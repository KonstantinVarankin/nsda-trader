# D:\nsda-trader\backend\app\api\endpoints\__init__.py

from .data import router as data_router
from .predictions import router as predictions_router
from .trading import router as trading_router
from .performance import router as performance_router
from .market_data import router as market_data_router

__all__ = ["data_router", "predictions_router", "trading_router", "performance_router", "market_data_router"]