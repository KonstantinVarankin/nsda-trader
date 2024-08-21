from fastapi import APIRouter, HTTPException
from app.ml.model import NSDATradingModel
from app.trading.decision_maker import TradingDecisionMaker
from app.data.market_data import get_stock_data
from datetime import datetime, timedelta
import pandas as pd

router = APIRouter()
model = NSDATradingModel()
decision_maker = TradingDecisionMaker()

@router.post("/train")
async def train_model(symbol: str):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # Train on last 2 years' data
        model.train(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        return {"message": f"Model trained successfully on {symbol} data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict")
async def get_prediction(symbol: str):
    try:
        date = datetime.now().strftime('%Y-%m-%d')
        prediction = model.predict(symbol, date)
        return {"symbol": symbol, "date": date, "prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluate")
async def evaluate_model(symbol: str):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Evaluate on last month's data
        evaluation = model.evaluate(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        return {"symbol": symbol, "evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trade-decision")
async def get_trade_decision(symbol: str):
    try:
        date = datetime.now().strftime('%Y-%m-%d')
        prediction = model.predict(symbol, date)
        
        # Получаем текущую цену
        current_data = get_stock_data(symbol, date, date)
        current_price = current_data['Close'].iloc[-1]
        
        # Принимаем торговое решение
        decision = decision_maker.make_decision(symbol, {"symbol": symbol, "date": date, "prediction": prediction}, current_price)
        
        # Добавляем стоп-лосс и тейк-профит
        if decision['action'] != 'HOLD':
            decision['stop_loss'] = decision_maker.calculate_stop_loss(current_price, decision['action'])
            decision['take_profit'] = decision_maker.calculate_take_profit(current_price, decision['action'])
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "prediction": prediction,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
