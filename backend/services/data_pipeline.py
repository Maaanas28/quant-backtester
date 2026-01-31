"""
Advanced Market Data Pipeline
Real-time data streaming, caching, and quality validation
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import yfinance as yf
from functools import lru_cache
import json
import logging
from collections import defaultdict
import websockets

logger = logging.getLogger(__name__)


class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.failures = 0
        self.max_failures = 3
    
    async def fetch(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        return self.failures < self.max_failures


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__("Yahoo Finance", priority=1)
    
    async def fetch(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(symbol, start=start, end=end, progress=False)
            )
            if df.empty:
                self.failures += 1
                return None
            self.failures = 0
            return df
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            self.failures += 1
            return None


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source (requires API key)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Alpha Vantage", priority=2)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    async def fetch(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if 'Time Series (Daily)' not in data:
                        self.failures += 1
                        return None
                    
                    df = pd.DataFrame.from_dict(
                        data['Time Series (Daily)'], 
                        orient='index'
                    )
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    df = df.astype(float)
                    
                    # Filter date range
                    df = df.loc[start:end]
                    self.failures = 0
                    return df
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            self.failures += 1
            return None


class DataCache:
    """In-memory cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[pd.DataFrame, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data.copy()
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        self.cache[key] = (data.copy(), datetime.now())
    
    def clear(self):
        self.cache.clear()
    
    def size(self) -> int:
        return len(self.cache)


class DataQualityValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate(df: pd.DataFrame, symbol: str) -> tuple[bool, List[str]]:
        issues = []
        
        # Check for missing data
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {null_cols}")
        
        # Check for zero/negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                # Check for negative/zero values and ensure we get a scalar boolean
                negative_check = df[col] <= 0
                count_result = negative_check.sum()
                count = int(count_result.iloc[0]) if hasattr(count_result, 'iloc') else int(count_result)
                if count > 0:
                    issues.append(f"Zero or negative values in {col}")
        
        # Check for invalid OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Get max and min as Series (axis=1 returns Series with row-wise max/min)
            max_ohlc = df[['Open', 'Low', 'Close']].max(axis=1)
            min_ohlc = df[['Open', 'High', 'Close']].min(axis=1)
            
            # Compare using values to avoid index alignment issues
            invalid_high = df['High'].values < max_ohlc.values
            invalid_low = df['Low'].values > min_ohlc.values
            
            high_count = int(invalid_high.sum())
            if high_count > 0:
                issues.append(f"High price lower than other prices: {high_count} rows")
                
            low_count = int(invalid_low.sum())
            if low_count > 0:
                issues.append(f"Low price higher than other prices: {low_count} rows")
        
        # Check for outliers (>50% daily change)
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change()
            outliers = returns.abs() > 0.5
            outlier_result = outliers.sum()
            outlier_count = int(outlier_result.iloc[0]) if hasattr(outlier_result, 'iloc') else int(outlier_result)
            if outlier_count > 0:
                issues.append(f"Extreme returns detected: {outlier_count} days")
        
        # Check for gaps in data
        if len(df) > 1:
            date_diff = df.index.to_series().diff()
            expected_gap = pd.Timedelta(days=1)
            large_gaps = date_diff > pd.Timedelta(days=7)
            gap_result = large_gaps.sum()
            gap_count = int(gap_result.iloc[0]) if hasattr(gap_result, 'iloc') else int(gap_result)
            if gap_count > 0:
                issues.append(f"Data gaps detected: {gap_count} gaps > 7 days")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and fix common data issues"""
        df = df.copy()
        
        # Forward fill missing values (max 5 days) - using Pandas 2.x syntax
        df = df.ffill(limit=5)
        
        # Remove rows with remaining nulls
        df = df.dropna()
        
        # Cap extreme returns at 50%
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change()
            extreme_mask = returns.abs() > 0.5
            extreme_result = extreme_mask.sum()
            extreme_count = int(extreme_result.iloc[0]) if hasattr(extreme_result, 'iloc') else int(extreme_result)
            if extreme_count > 0:
                # Use previous close for extreme values
                df.loc[extreme_mask, 'Close'] = df['Close'].shift(1)[extreme_mask]
        
        return df


class MarketDataPipeline:
    """
    Advanced market data pipeline with multiple sources, caching, and quality validation
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, cache_ttl: int = 3600):
        self.sources = [
            YahooFinanceSource(),
            AlphaVantageSource(alpha_vantage_key) if alpha_vantage_key else None
        ]
        self.sources = [s for s in self.sources if s is not None]
        self.cache = DataCache(ttl_seconds=cache_ttl)
        self.validator = DataQualityValidator()
        self.subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._streaming_tasks = {}
    
    async def get_data(
        self, 
        symbol: str, 
        start: str, 
        end: str,
        validate: bool = True,
        use_cache: bool = True
    ) -> tuple[Optional[pd.DataFrame], List[str]]:
        """
        Fetch market data with fallback sources
        Returns: (dataframe, list of warnings)
        """
        cache_key = f"{symbol}_{start}_{end}"
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for {symbol}")
                return cached, []
        
        # Try each source in priority order
        available_sources = [s for s in self.sources if s.is_available()]
        available_sources.sort(key=lambda x: x.priority)
        
        for source in available_sources:
            logger.info(f"Trying {source.name} for {symbol}")
            df = await source.fetch(symbol, start, end)
            
            if df is not None and not df.empty:
                warnings = []
                
                # Validate data quality
                if validate:
                    is_valid, issues = self.validator.validate(df, symbol)
                    if not is_valid:
                        warnings.extend(issues)
                        logger.warning(f"Data quality issues for {symbol}: {issues}")
                        df = self.validator.clean(df)
                
                # Cache the result
                if use_cache:
                    self.cache.set(cache_key, df)
                
                logger.info(f"Successfully fetched {symbol} from {source.name}")
                return df, warnings
        
        logger.error(f"Failed to fetch data for {symbol} from all sources")
        return None, ["Failed to fetch data from all sources"]
    
    async def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': float(info.get('currentPrice', 0)),
                'change': float(info.get('regularMarketChange', 0)),
                'change_percent': float(info.get('regularMarketChangePercent', 0)),
                'volume': int(info.get('volume', 0)),
                'bid': float(info.get('bid', 0)),
                'ask': float(info.get('ask', 0)),
                'open': float(info.get('open', 0)),
                'high': float(info.get('dayHigh', 0)),
                'low': float(info.get('dayLow', 0)),
                'prev_close': float(info.get('previousClose', 0)),
                'timestamp': datetime.now().isoformat()
            }
            return quote
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None
    
    async def stream_data(self, symbol: str, interval: int = 5):
        """
        Stream real-time data at specified interval (seconds)
        Publishes to all subscribers
        """
        while True:
            try:
                quote = await self.get_realtime_quote(symbol)
                if quote:
                    for queue in self.subscribers[symbol]:
                        try:
                            await queue.put(quote)
                        except asyncio.QueueFull:
                            logger.warning(f"Queue full for {symbol}")
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error streaming {symbol}: {e}")
                await asyncio.sleep(interval)
    
    def subscribe(self, symbol: str) -> asyncio.Queue:
        """Subscribe to real-time data stream"""
        queue = asyncio.Queue(maxsize=100)
        self.subscribers[symbol].append(queue)
        
        # Start streaming task if not already running
        if symbol not in self._streaming_tasks:
            task = asyncio.create_task(self.stream_data(symbol))
            self._streaming_tasks[symbol] = task
        
        return queue
    
    def unsubscribe(self, symbol: str, queue: asyncio.Queue):
        """Unsubscribe from data stream"""
        if symbol in self.subscribers and queue in self.subscribers[symbol]:
            self.subscribers[symbol].remove(queue)
            
            if not self.subscribers[symbol] and symbol in self._streaming_tasks:
                self._streaming_tasks[symbol].cancel()
                del self._streaming_tasks[symbol]
    
    async def get_multiple_symbols(
        self, 
        symbols: List[str], 
        start: str, 
        end: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        tasks = [self.get_data(symbol, start, end) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            symbol: df 
            for symbol, (df, _) in zip(symbols, results) 
            if df is not None
        }
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': self.cache.size(),
            'ttl_seconds': self.cache.ttl.total_seconds()
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()


# Global pipeline instance
_pipeline = None

def get_pipeline() -> MarketDataPipeline:
    """Get or create global pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = MarketDataPipeline()
    return _pipeline
