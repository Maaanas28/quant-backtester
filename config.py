"""
NeuroQuant Configuration Module
Centralized configuration management using environment variables
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()


class DatabaseConfig:
    """Database configuration settings"""
    URL: str = os.getenv("DATABASE_URL", "./database/neuroquant.db")
    BACKUP_ENABLED: bool = os.getenv("DATABASE_BACKUP_ENABLED", "false").lower() == "true"
    BACKUP_INTERVAL: int = int(os.getenv("DATABASE_BACKUP_INTERVAL", "3600"))


class APIConfig:
    """API server configuration"""
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    # Render provides PORT env var, fallback to API_PORT or default 8000
    PORT: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    TITLE: str = "NeuroQuant Trading API"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "Advanced Reinforcement Learning Trading System"


class CORSConfig:
    """CORS configuration"""
    ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500"
    ).split(",")
    ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    ALLOW_METHODS: List[str] = ["*"]
    ALLOW_HEADERS: List[str] = ["*"]


class RedisConfig:
    """Redis cache configuration"""
    ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    HOST: str = os.getenv("REDIS_HOST", "localhost")
    PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    DB: int = int(os.getenv("REDIS_DB", "0"))
    PASSWORD: str = os.getenv("REDIS_PASSWORD", "")


class ModelConfig:
    """Model training and persistence configuration"""

    AUTO_SAVE: bool = os.getenv("MODEL_AUTO_SAVE", "true").lower() == "true"
    SAVE_INTERVAL: int = int(os.getenv("MODEL_SAVE_INTERVAL", "1000"))
    


class TrainingConfig:
    """RL training configuration"""
    DEFAULT_TIMESTEPS: int = int(os.getenv("DEFAULT_TIMESTEPS", "50000"))  # Increased from 20000
    DEFAULT_TRAIN_SPLIT: float = float(os.getenv("DEFAULT_TRAIN_SPLIT", "0.7"))
    MAX_EPISODE_STEPS: int = int(os.getenv("MAX_EPISODE_STEPS", "1000"))


class RLHyperparameters:
    """Reinforcement Learning hyperparameters"""
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0003"))
    GAMMA: float = float(os.getenv("GAMMA", "0.99"))
    EPSILON_START: float = float(os.getenv("EPSILON_START", "1.0"))
    EPSILON_END: float = float(os.getenv("EPSILON_END", "0.01"))
    EPSILON_DECAY: float = float(os.getenv("EPSILON_DECAY", "0.995"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
    MEMORY_SIZE: int = int(os.getenv("MEMORY_SIZE", "100000"))
    TARGET_UPDATE_FREQ: int = int(os.getenv("TARGET_UPDATE_FREQ", "10"))


class FinancialConfig:
    """Financial trading configuration"""
    INITIAL_PORTFOLIO: float = float(os.getenv("INITIAL_PORTFOLIO", "10000"))
    TRANSACTION_COST: float = float(os.getenv("TRANSACTION_COST", "0.001"))
    SLIPPAGE: float = float(os.getenv("SLIPPAGE", "0.0005"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "1.0"))


class MarketDataConfig:
    """Market data fetching configuration"""
    YFINANCE_TIMEOUT: int = int(os.getenv("YFINANCE_TIMEOUT", "30"))
    YFINANCE_MAX_RETRIES: int = int(os.getenv("YFINANCE_MAX_RETRIES", "3"))
    CACHE_ENABLED: bool = os.getenv("DATA_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("DATA_CACHE_TTL", "3600"))


class SentimentConfig:
    """Sentiment analysis configuration"""
    MODEL: str = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
    ENABLED: bool = os.getenv("SENTIMENT_ENABLED", "false").lower() == "true"  # Disabled by default
    CACHE_ENABLED: bool = os.getenv("SENTIMENT_CACHE_ENABLED", "true").lower() == "true"
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")


class LoggingConfig:
    """Logging configuration"""
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FILE: Path = BASE_DIR / os.getenv("LOG_FILE", "logs/neuroquant.log")
    MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create logs directory if it doesn't exist
    FILE.parent.mkdir(parents=True, exist_ok=True)


class SecurityConfig:
    """Security configuration"""
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    API_KEY_ENABLED: bool = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


class DevelopmentConfig:
    """Development settings"""
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")


# Create a global config object
class Config:
    """Global configuration object"""
    database = DatabaseConfig()
    api = APIConfig()
    cors = CORSConfig()
    redis = RedisConfig()
    model = ModelConfig()
    training = TrainingConfig()
    rl = RLHyperparameters()
    financial = FinancialConfig()
    market_data = MarketDataConfig()
    sentiment = SentimentConfig()
    logging = LoggingConfig()
    security = SecurityConfig()
    development = DevelopmentConfig()


config = Config()
