"""
Helper utility functions for NeuroQuant
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        portfolio_values: List of portfolio values over time
    
    Returns:
        Maximum drawdown as a percentage
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max
    return abs(drawdowns.min()) * 100


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate from trades
    
    Args:
        trades: List of trade dictionaries
    
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return (winning_trades / len(trades)) * 100


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profits / gross losses)
    
    Args:
        trades: List of trade dictionaries
    
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profits = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_losses = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (focuses on downside deviation)
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


def calculate_calmar_ratio(returns: pd.Series, portfolio_values: List[float]) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Args:
        returns: Series of returns
        portfolio_values: List of portfolio values
    
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annual_return = returns.mean() * 252
    max_dd = calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


def generate_trade_id(symbol: str, timestamp: datetime, action: str) -> str:
    """
    Generate unique trade ID
    
    Args:
        symbol: Trading symbol
        timestamp: Trade timestamp
        action: Trade action (buy/sell)
    
    Returns:
        Unique trade ID
    """
    trade_string = f"{symbol}_{timestamp.isoformat()}_{action}"
    return hashlib.md5(trade_string.encode()).hexdigest()[:12]


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2f}%"


def serialize_for_json(obj: Any) -> Any:
    """
    Serialize objects for JSON encoding
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    return obj


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range
    
    Args:
        start_date: Start date string
        end_date: End date string
    
    Returns:
        True if valid, False otherwise
    """
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        return start < end
    except (ValueError, TypeError):
        return False


def get_market_status() -> str:
    """
    Get current market status
    
    Returns:
        Market status string
    """
    from datetime import datetime
    import pytz
    
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    if now.weekday() >= 5:  # Weekend
        return "closed"
    
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    if market_open <= now <= market_close:
        return "open"
    elif now < market_open:
        return "pre-market"
    else:
        return "after-hours"
