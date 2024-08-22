from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar
from dotenv import load_dotenv
import os
from typing import Optional
from pydantic import Field

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "NSDA-Trader"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:2202@localhost:5432/nsda_trader")

    EXCHANGE_NAME: str = os.getenv("EXCHANGE_NAME", "binance")
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

    PREDICTION_INTERVAL: int = 60

settings = Settings()