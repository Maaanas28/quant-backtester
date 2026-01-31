"""
NeuroQuant Logging Configuration
Centralized logging setup with rotating file handlers and structured output
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.logging.LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(config.logging.FORMAT)
    console_formatter = ColoredFormatter(config.logging.FORMAT)
    
    # File handler with rotation
    if log_to_file:
        file_handler = RotatingFileHandler(
            config.logging.FILE,
            maxBytes=config.logging.MAX_BYTES,
            backupCount=config.logging.BACKUP_COUNT,
            encoding='utf-8'  # Use UTF-8 encoding to support all characters
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        # Set encoding to UTF-8 for Windows compatibility
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass  # Fallback if reconfigure is not available
        logger.addHandler(console_handler)
    
    return logger


# Create module-level loggers
main_logger = setup_logger("neuroquant.main")
api_logger = setup_logger("neuroquant.api")
db_logger = setup_logger("neuroquant.database")
agent_logger = setup_logger("neuroquant.agent")
market_logger = setup_logger("neuroquant.market")
backtest_logger = setup_logger("neuroquant.backtest")


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator
