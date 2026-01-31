import numpy as np
import pandas as pd
import yfinance as yf
import talib
from typing import Optional, Dict
from datetime import datetime, timedelta
import hashlib
import json

from config import config
from utils.logging_config import setup_logger
from utils.exceptions import DataFetchError, DataValidationError

logger = setup_logger(__name__)


class DataCache:
    """Simple in-memory cache for market data"""
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl:
                logger.debug(f"Cache hit for {key}")
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        """Store data in cache"""
        self.cache[key] = (data, datetime.now())
        logger.debug(f"Cached data for {key}")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")


class MarketDataProvider:
    """Handles market data fetching and preprocessing"""
    
    def __init__(self):
        self.cache_enabled = config.market_data.CACHE_ENABLED
        self.cache = DataCache(ttl=config.market_data.CACHE_TTL) if self.cache_enabled else None
        logger.info("MarketDataProvider initialized")
    
    def fetch_stock_data(self, symbol: str, period: str = "1y", custom_data_df: pd.DataFrame = None) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance or use custom data"""
        if custom_data_df is not None:
            logger.info(f"Using custom data for {symbol}")
            return custom_data_df
        
        # Check cache
        cache_key = f"{symbol}_{period}"
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data.copy()
        
        try:
            logger.info(f"Fetching data for {symbol} (period: {period})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise DataFetchError(f"No data returned for {symbol}")
            
            logger.info(f"Fetched {len(data)} rows for {symbol}")
            
            # Cache the data
            if self.cache_enabled and self.cache:
                self.cache.set(cache_key, data)
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise DataFetchError(f"Error fetching data for {symbol}: {str(e)}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA-Lib"""
        data = df.copy()
        
        try:
            # Price-based indicators
            data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
            data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
            data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
            
            # Momentum indicators
            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
            data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
            data['STOCH_k'], data['STOCH_d'] = talib.STOCH(data['High'], data['Low'], data['Close'])
            
            # Volatility indicators
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
            data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = talib.SMA(data['Volume'], timeperiod=20)
            data['OBV'] = talib.OBV(data['Close'], data['Volume'])
            
            # Calculate derived features
            data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
            data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
            data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Forward fill and drop NaN values
            data = data.ffill().dropna()
            
            logger.debug(f"Calculated technical indicators, remaining rows: {len(data)}")
            return data
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise DataValidationError(f"Failed to calculate indicators: {str(e)}")
    
    def get_sentiment_score(self, symbol: str, date: Optional[str] = None) -> float:
        """
        Get sentiment score for a symbol
        
        Args:
            symbol: Stock symbol
            date: Date string (for caching/mock data)
        
        Returns:
            Sentiment compound score between -1 and 1
        """
        try:
            if config.sentiment.ENABLED:
                sentiment = self.sentiment_analyzer.get_symbol_sentiment(symbol, days_back=7)
                return sentiment['compound']
            else:
                # Use mock sentiment if disabled
                return self.sentiment_analyzer.get_mock_sentiment(symbol, date or datetime.now().isoformat())
        
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {symbol}, using neutral: {e}")
            return 0.0
    
    def get_mock_sentiment(self, symbol: str, date: str) -> float:
        """
        Backward compatibility - mock sentiment analysis
        
        Args:
            symbol: Stock symbol
            date: Date string
        
        Returns:
            Mock sentiment score
        """
        return self.sentiment_analyzer.get_mock_sentiment(symbol, date)