"""
Tests for market data provider
"""
import pytest
import pandas as pd
from backend.services.market_data import MarketDataProvider, DataCache
from utils.exceptions import DataFetchError


def test_data_cache():
    """Test data caching functionality"""
    cache = DataCache(ttl=60)
    
    # Test set and get
    df = pd.DataFrame({'Close': [100, 101, 102]})
    cache.set('TEST_1y', df)
    
    result = cache.get('TEST_1y')
    assert result is not None
    assert len(result) == 3
    
    # Test cache miss
    assert cache.get('NONEXISTENT') is None
    
    # Test clear
    cache.clear()
    assert cache.get('TEST_1y') is None


def test_market_data_provider_init():
    """Test MarketDataProvider initialization"""
    provider = MarketDataProvider()
    assert provider is not None
    assert hasattr(provider, 'sentiment_analyzer')


def test_calculate_technical_indicators(sample_stock_data):
    """Test technical indicator calculation"""
    provider = MarketDataProvider()
    
    result = provider.calculate_technical_indicators(sample_stock_data)
    
    # Check that indicators were added
    assert 'SMA_20' in result.columns
    assert 'SMA_50' in result.columns
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    assert 'BB_upper' in result.columns
    
    # Check no NaN values
    assert not result.isnull().any().any()


def test_get_sentiment_score():
    """Test sentiment score retrieval"""
    provider = MarketDataProvider()
    
    score = provider.get_sentiment_score('AAPL', '2023-01-01')
    
    assert isinstance(score, float)
    assert -1 <= score <= 1


@pytest.mark.parametrize("symbol,period", [
    ('INVALID_SYM_12345', '1y'),
    ('', '1y'),
])
def test_fetch_stock_data_invalid_symbol(symbol, period):
    """Test error handling for invalid symbols"""
    provider = MarketDataProvider()
    
    with pytest.raises(DataFetchError):
        provider.fetch_stock_data(symbol, period)
