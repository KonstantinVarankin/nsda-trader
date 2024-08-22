from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.market_data import MarketData
from app.services.binance_service import binance_service
from app.ml.model import prediction_model, predict
from app.ml.model import nsda_trading_model
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

router = APIRouter()

@router.get("/historical-data")
async def get_historical_data(
    symbol: str = Query(..., description="Trading symbol"),
    interval: str = Query(..., description="Interval for the klines"),
    start_date: str = Query(..., description="Start date for historical data"),
    end_date: str = Query(..., description="End date for historical data")
):
    try:
        df = await binance_service.get_historical_data(symbol, interval, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No historical data found")
        return df.reset_index().to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

@router.get("/prediction")
async def get_prediction(
    symbol: str = Query(..., description="Trading symbol"),
    interval: str = Query(..., description="Interval for the klines")
):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Get last 60 days of data

        df = await binance_service.get_historical_data(
            symbol,
            interval,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No historical data found for prediction")

        prediction_model.train_model(df)
        prediction = prediction_model.make_prediction(df.tail(60))

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make prediction: {str(e)}")

@router.get("/account-info")
async def get_account_info():
    try:
        info = await binance_service.get_account_info()
        if info is None:
            raise HTTPException(status_code=404, detail="Account info not found")
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch account info: {str(e)}")

@router.get("/symbol-ticker")
async def get_symbol_ticker(symbol: str = Query(..., description="Trading symbol")):
    try:
        ticker = await binance_service.get_symbol_ticker(symbol)
        if ticker is None:
            raise HTTPException(status_code=404, detail="Symbol ticker not found")
        return ticker
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch symbol ticker: {str(e)}")

@router.post("/place-order")
async def place_order(
    symbol: str = Query(..., description="Trading symbol"),
    side: str = Query(..., description="Order side (BUY or SELL)"),
    order_type: str = Query(..., description="Order type"),
    quantity: float = Query(..., description="Order quantity"),
    price: Optional[float] = Query(None, description="Order price (optional for market orders)")
):
    try:
        order = await binance_service.place_order(symbol, side, order_type, quantity, price)
        if order is None:
            raise HTTPException(status_code=400, detail="Order placement failed")
        return order
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

@router.get("/open-orders")
async def get_open_orders(symbol: Optional[str] = Query(None, description="Trading symbol (optional)")):
    try:
        orders = await binance_service.get_open_orders(symbol)
        if orders is None:
            raise HTTPException(status_code=404, detail="No open orders found")
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch open orders: {str(e)}")