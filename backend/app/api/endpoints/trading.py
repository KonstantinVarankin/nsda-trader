from fastapi import APIRouter, HTTPException
from app.services.exchange_api import exchange_api
from app.services.decision_making import decision_maker
from app.services.risk_management import risk_manager
from pydantic import BaseModel

router = APIRouter()

class TradeRequest(BaseModel):
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: float = None

@router.post("/execute")
def execute_trade(request: TradeRequest):
    try:
        # Get current account balance
        balance = exchange_api.get_account_balance()
        if not balance:
            raise HTTPException(status_code=400, detail="Failed to get account balance")

        # Get trading decision
        decision = decision_maker.make_decision(request.symbol)

        # Check if the requested action aligns with the decision
        if (request.action == 'buy' and decision not in ['Strong Buy', 'Buy']) or \
           (request.action == 'sell' and decision not in ['Strong Sell', 'Sell']):
            raise HTTPException(status_code=400, detail=f"Requested action does not align with trading decision: {decision}")

        # Calculate position size if not provided
        if not request.quantity:
            account_balance = sum(balance.values())  # Simplified; you might want to use only the relevant asset
            request.quantity = risk_manager.calculate_position_size(request.symbol, account_balance)

        # Execute the trade
        order = exchange_api.place_market_order(request.symbol, request.action.upper(), request.quantity)
        if not order:
            raise HTTPException(status_code=400, detail="Failed to place order")

        return {"status": "success", "order": order}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/open-orders")
def get_open_orders(symbol: str = None):
    orders = exchange_api.get_open_orders(symbol)
    if orders is None:
        raise HTTPException(status_code=400, detail="Failed to get open orders")
    return orders

@router.delete("/cancel-order/{symbol}/{order_id}")
def cancel_order(symbol: str, order_id: int):
    result = exchange_api.cancel_order(symbol, order_id)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to cancel order")
    return {"status": "success", "result": result}

@router.get("/account-balance")
def get_account_balance():
    balance = exchange_api.get_account_balance()
    if not balance:
        raise HTTPException(status_code=400, detail="Failed to get account balance")
    return balance
