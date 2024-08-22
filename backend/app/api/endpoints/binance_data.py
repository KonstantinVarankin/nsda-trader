from fastapi import APIRouter, HTTPException
from app.services.binance_service import binance_service
from typing import Optional
from datetime import datetime

router = APIRouter()

@router.get("/historical-data/{symbol}")
async def get_historical_data(symbol: str, interval: str, start_date: str, end_date: str):
    data = await binance_service.get_historical_data(symbol, interval, start_date, end_date)
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")
    return data.to_dict(orient='records')

@router.get("/account-info")
async def get_account_info():
    info = await binance_service.get_account_info()
    if info is None:
        raise HTTPException(status_code=500, detail="Failed to fetch account info")
    return info

@router.get("/symbol-price/{symbol}")
async def get_symbol_price(symbol: str):
    ticker = await binance_service.get_symbol_ticker(symbol)
    if ticker is None:
        raise HTTPException(status_code=500, detail="Failed to fetch symbol price")
    return ticker

@router.post("/place-order")
async def place_order(symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None):
    order = await binance_service.place_order(symbol, side, order_type, quantity, price)
    if order is None:
        raise HTTPException(status_code=500, detail="Failed to place order")
    return order

@router.get("/open-orders")
async def get_open_orders(symbol: Optional[str] = None):
    orders = await binance_service.get_open_orders(symbol)
    if orders is None:
        raise HTTPException(status_code=500, detail="Failed to fetch open orders")
    return orders