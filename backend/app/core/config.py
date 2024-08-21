from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "NSDA-Trader"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    EXCHANGE_NAME: str = os.getenv("EXCHANGE_NAME", "binance")
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

settings = Settings()
