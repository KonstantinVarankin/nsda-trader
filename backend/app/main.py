import sys
import os

# Добавляем текущую директорию и родительскую директорию в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

# Импортируем роутеры
from app.api.api_v1.api import api_router
from app.api.api_v1.endpoints import performance
from app.api.endpoints.market_data import router as market_data_router

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
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(performance.router, prefix=f"{settings.API_V1_STR}/performance", tags=["performance"])
app.include_router(market_data_router, prefix=f"{settings.API_V1_STR}/market-data", tags=["market_data"])


@app.get("/")
async def root():
    return {"message": "Welcome to NSDA-Trader API"}


if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print("sys.path:", sys.path)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)