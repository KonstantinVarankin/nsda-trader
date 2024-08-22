# D:\nsda-trader\backend\app\api\endpoints\performance.py

from fastapi import APIRouter

router = APIRouter()


@router.get("/performance")
async def get_performance():
    return {"message": "Performance endpoint"}