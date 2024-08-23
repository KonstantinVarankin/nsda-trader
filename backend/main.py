import sys
import os
from dotenv import load_dotenv
import asyncio


# Добавляем текущую директорию и родительскую директорию в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import data, predictions, trading, performance, market_data
from app.services.ai_prediction_service import ai_service

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(data.router, prefix=f"{settings.API_V1_STR}/data", tags=["data"])
app.include_router(predictions.router, prefix=f"{settings.API_V1_STR}/predictions", tags=["predictions"])
app.include_router(trading.router, prefix=f"{settings.API_V1_STR}/trading", tags=["trading"])
app.include_router(performance.router, prefix=f"{settings.API_V1_STR}/performance", tags=["performance"])
app.include_router(market_data.router, prefix=f"{settings.API_V1_STR}/market-data", tags=["market_data"])

@app.get("/")
async def root():
    return {"message": "Welcome to NSDA-Trader API"}


async def run_ai_predictions():
    while True:
        try:
            result = await ai_service.run_prediction_cycle()
            print(result)
        except Exception as e:
            print(f"Error in AI prediction cycle: {str(e)}")
        await asyncio.sleep(settings.PREDICTION_INTERVAL)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_ai_predictions())

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print("sys.path:", sys.path)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)