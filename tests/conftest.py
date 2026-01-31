"""
Test configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    data = pd.DataFrame({
        'Open': close_prices + np.random.randn(len(dates)) * 0.5,
        'High': close_prices + abs(np.random.randn(len(dates)) * 1),
        'Low': close_prices - abs(np.random.randn(len(dates)) * 1),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data


@pytest.fixture
def sample_trades():
    """Generate sample trades for testing"""
    return [
        {
            'date': '2023-01-15',
            'action': 'buy',
            'price': 100.0,
            'shares': 10.0,
            'pnl': 0,
            'portfolio_before': 10000.0,
            'holding_before': 0.0
        },
        {
            'date': '2023-02-15',
            'action': 'sell',
            'price': 110.0,
            'shares': 10.0,
            'pnl': 100.0,
            'portfolio_before': 0.0,
            'holding_before': 10.0
        },
        {
            'date': '2023-03-15',
            'action': 'buy',
            'price': 105.0,
            'shares': 10.0,
            'pnl': 0,
            'portfolio_before': 1100.0,
            'holding_before': 0.0
        },
        {
            'date': '2023-04-15',
            'action': 'sell',
            'price': 95.0,
            'shares': 10.0,
            'pnl': -100.0,
            'portfolio_before': 0.0,
            'holding_before': 10.0
        }
    ]


@pytest.fixture
def sample_portfolio_history():
    """Generate sample portfolio history"""
    return [10000.0, 10100.0, 10200.0, 10150.0, 10300.0, 10250.0, 10400.0]


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration"""
    return {
        'id': 1,
        'name': 'Test Agent',
        'type': 'DQN',
        'parameters': {
            'learning_rate': 0.001,
            'gamma': 0.99
        }
    }
