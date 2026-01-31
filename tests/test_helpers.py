"""
Tests for utility helper functions
"""
import pytest
import pandas as pd
import numpy as np
from utils.helpers import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_sortino_ratio,
    format_currency,
    format_percentage
)


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, 0.02])
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Positive returns should give positive Sharpe
    
    # Test with zero std
    zero_returns = pd.Series([0.0, 0.0, 0.0])
    assert calculate_sharpe_ratio(zero_returns) == 0.0


def test_calculate_max_drawdown(sample_portfolio_history):
    """Test max drawdown calculation"""
    max_dd = calculate_max_drawdown(sample_portfolio_history)
    assert isinstance(max_dd, float)
    assert max_dd >= 0
    assert max_dd <= 100  # Should be a percentage
    
    # Test with empty list
    assert calculate_max_drawdown([]) == 0.0
    
    # Test with single value
    assert calculate_max_drawdown([100.0]) == 0.0


def test_calculate_win_rate(sample_trades):
    """Test win rate calculation"""
    win_rate = calculate_win_rate(sample_trades)
    assert isinstance(win_rate, float)
    assert 0 <= win_rate <= 100
    
    # With our sample trades (1 win, 1 loss), win rate should be 50%
    # But we need to ensure all trades have 'pnl'
    trades_with_pnl = [t for t in sample_trades if 'pnl' in t and t['pnl'] != 0]
    if len(trades_with_pnl) > 0:
        assert win_rate == 50.0
    
    # Test with empty trades
    assert calculate_win_rate([]) == 0.0


def test_calculate_profit_factor(sample_trades):
    """Test profit factor calculation"""
    profit_factor = calculate_profit_factor(sample_trades)
    assert isinstance(profit_factor, float)
    assert profit_factor >= 0
    
    # Test with empty trades
    assert calculate_profit_factor([]) == 0.0


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation"""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
    sortino = calculate_sortino_ratio(returns)
    assert isinstance(sortino, float)
    
    # Test with all positive returns
    positive_returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert calculate_sortino_ratio(positive_returns) == 0.0


def test_format_currency():
    """Test currency formatting"""
    assert format_currency(1000.0) == "$1,000.00"
    assert format_currency(1234567.89) == "$1,234,567.89"
    assert format_currency(0.0) == "$0.00"


def test_format_percentage():
    """Test percentage formatting"""
    assert format_percentage(50.123) == "50.12%"
    assert format_percentage(0.0) == "0.00%"
    assert format_percentage(-10.456) == "-10.46%"
