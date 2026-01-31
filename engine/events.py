"""
Event System for Event-Driven Backtesting
Based on institutional trading systems architecture
"""
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class EventType(Enum):
    """Event types in the trading system"""
    MARKET = "MARKET"          # New market data
    SIGNAL = "SIGNAL"          # Trading signal generated
    ORDER = "ORDER"            # Order placed
    FILL = "FILL"              # Order filled
    RISK = "RISK"              # Risk check event
    POSITION = "POSITION"      # Position update


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class SignalType(Enum):
    """Trading signals"""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Event:
    """Base event class"""
    timestamp: datetime
    event_type: EventType


@dataclass
class MarketEvent(Event):
    """Market data update event"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    event_type: EventType = field(default=EventType.MARKET, init=False)
    
    @property
    def bid(self) -> Optional[float]:
        """Alias for backward compatibility"""
        return self.bid_price
    
    @property
    def ask(self) -> Optional[float]:
        """Alias for backward compatibility"""
        return self.ask_price
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return 0.0


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1 confidence
    strategy_name: str
    metadata: Optional[dict] = None
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent(Event):
    """Order placement event"""
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, DAY, IOC, FOK
    order_id: Optional[str] = None
    strategy_name: Optional[str] = None
    event_type: EventType = field(default=EventType.ORDER, init=False)


@dataclass
class FillEvent(Event):
    """Order fill event"""
    symbol: str
    exchange: str
    quantity: int
    side: OrderSide
    fill_price: float
    commission: float
    slippage: float  # Actual slippage incurred
    order_id: str
    fill_id: str
    event_type: EventType = field(default=EventType.FILL, init=False)
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission"""
        base_cost = self.quantity * self.fill_price
        return base_cost + self.commission


@dataclass
class RiskEvent(Event):
    """Risk management event"""
    check_type: str  # 'drawdown', 'position_limit', 'var', 'margin'
    passed: bool
    message: str
    data: Optional[dict] = None
    event_type: EventType = field(default=EventType.RISK, init=False)


@dataclass
class PositionEvent(Event):
    """Position update event"""
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    event_type: EventType = field(default=EventType.POSITION, init=False)
