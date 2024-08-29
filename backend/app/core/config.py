from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar, Dict, Any
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
    INITIAL_CAPITAL: float = Field(1000.0, env="INITIAL_CAPITAL")

    # Настройки для использования GPU
    USE_GPU: bool = Field(True, env="USE_GPU")

    # PyTorch-специфичные настройки (если необходимо)
    PYTORCH_DEVICE: str = Field("cuda" if USE_GPU else "cpu", env="PYTORCH_DEVICE")

    PREDICTION_INTERVAL: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True
    )


settings = Settings()