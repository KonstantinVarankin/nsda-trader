import sys
import os
from dotenv import load_dotenv
import asyncio
import logging

# Добавляем текущую директорию и родительскую директорию в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

# Загружаем переменные окружения
load_dotenv()

# Теперь импортируем модули из app
from app.core.config import settings
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.services.ai_prediction_service import ai_service
from app.api.api_v1.api import api_router

# Импортируем роутеры


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем основной роутер
app.include_router(api_router, prefix=settings.API_V1_STR)



@app.get("/")
async def root():
    return {"message": "Welcome to NSDA-Trader API"}

# WebSocket для уведомлений
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

async def run_ai_predictions():
    while True:
        try:
            result = await ai_service.run_prediction_cycle()
            logger.info(f"AI prediction result: {result}")
        except Exception as e:
            logger.error(f"Error in AI prediction cycle: {str(e)}")
        await asyncio.sleep(settings.PREDICTION_INTERVAL)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_ai_predictions())

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"sys.path: {sys.path}")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)