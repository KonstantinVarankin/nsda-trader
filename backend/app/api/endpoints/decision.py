from fastapi import APIRouter
from app.services.decision_making import decision_maker

router = APIRouter()

@router.get("/{symbol}")
def get_trading_decision(symbol: str):
    decision = decision_maker.make_decision(symbol)
    confidence = decision_maker.get_confidence(symbol)
    return {
        "symbol": symbol,
        "decision": decision,
        "confidence": confidence
    }
