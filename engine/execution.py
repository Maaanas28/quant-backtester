"""
Order Matching Engine with Realistic Market Microstructure
Simulates exchange order book and matching logic
"""
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import numpy as np

from engine.events import OrderEvent, FillEvent, OrderType, OrderSide, MarketEvent


class OrderBook:
    """Level 2 order book for a single symbol"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Price level -> list of orders
        self.bids: Dict[float, deque] = {}  # Buy orders (descending price)
        self.asks: Dict[float, deque] = {}  # Sell orders (ascending price)
        
    def add_order(self, order: OrderEvent) -> None:
        """Add order to book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = deque()
            self.bids[order.price].append(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = deque()
            self.asks[order.price].append(order)
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return min(self.asks.keys()) if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None


class ExecutionEngine:
    """
    Realistic order execution with:
    - Slippage modeling (impact + volatility)
    - Order queue simulation
    - Commission structure
    - Market impact
    """
    
    def __init__(self,
                 commission_percent: float = 0.001,  # 0.1% commission
                 slippage_impact: float = 0.0005,    # 0.05% impact
                 slippage_volatility: float = 0.0002, # 0.02% random
                 latency_ms: float = 10.0):          # 10ms latency
        self.commission_percent = commission_percent
        self.slippage_impact = slippage_impact
        self.slippage_volatility = slippage_volatility
        self.latency_ms = latency_ms
        
        # Order books per symbol
        self.order_books: Dict[str, OrderBook] = {}
        
        # Fill counter for unique IDs
        self.fill_counter = 0
    
    def _get_order_book(self, symbol: str) -> OrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]
    
    def _calculate_slippage(self,
                           market_price: float,
                           quantity: int,
                           side: OrderSide,
                           volatility: float = None) -> float:
        """
        Calculate realistic slippage based on:
        1. Market impact (square root model)
        2. Volatility-based randomness
        3. Bid-ask spread crossing
        """
        # Impact component (square root of trade size)
        # Larger trades have more impact
        impact_factor = np.sqrt(quantity / 100) * self.slippage_impact
        
        # Volatility component (random walk)
        if volatility:
            vol_factor = np.random.normal(0, volatility * self.slippage_volatility)
        else:
            vol_factor = np.random.normal(0, self.slippage_volatility)
        
        # Total slippage (positive for adverse price movement)
        if side == OrderSide.BUY:
            # Buying pushes price up
            slippage_pct = impact_factor + abs(vol_factor)
        else:
            # Selling pushes price down  
            slippage_pct = -(impact_factor + abs(vol_factor))
        
        return float(market_price * slippage_pct)
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission with minimum"""
        commission = float(quantity) * float(price) * self.commission_percent
        # Minimum $1 commission
        return float(max(commission, 1.0))
    
    def execute_market_order(self,
                            order: OrderEvent,
                            market_event: MarketEvent) -> FillEvent:
        """
        Execute market order with realistic fill simulation
        
        Market orders:
        - Fill immediately at current market price + slippage
        - Cross the spread (buy at ask, sell at bid)
        - Experience market impact
        """
        timestamp = market_event.timestamp
        
        # Market orders cross the spread
        if order.side == OrderSide.BUY:
            # Buy at ask price
            base_price = market_event.ask
        else:
            # Sell at bid price
            base_price = market_event.bid
        
        # Calculate slippage based on order size
        slippage = self._calculate_slippage(
            base_price,
            order.quantity,
            order.side
        )
        
        # Final fill price
        fill_price = base_price + slippage
        
        # Calculate commission
        commission = self._calculate_commission(order.quantity, fill_price)
        
        # Create fill event
        self.fill_counter += 1
        fill = FillEvent(
            timestamp=timestamp,
            symbol=order.symbol,
            exchange="SIMULATED",
            quantity=order.quantity,
            side=order.side,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(slippage),
            order_id=order.order_id,
            fill_id=f"FILL_{self.fill_counter:08d}"
        )
        
        return fill
    
    def execute_limit_order(self,
                           order: OrderEvent,
                           market_event: MarketEvent) -> Optional[FillEvent]:
        """
        Execute limit order if price reached
        
        Limit orders:
        - Only fill if market price crosses limit price
        - Queue position matters (FIFO)
        - May get partial fills
        """
        # Check if limit price is touched
        if order.side == OrderSide.BUY:
            # Buy limit: fill if ask <= limit price
            if market_event.ask <= order.limit_price:
                # Limit orders get better price (limit or better)
                fill_price = min(order.limit_price, market_event.ask)
            else:
                # Add to order book, not filled yet
                order_book = self._get_order_book(order.symbol)
                order_book.add_order(order)
                return None
        else:
            # Sell limit: fill if bid >= limit price
            if market_event.bid >= order.limit_price:
                fill_price = max(order.limit_price, market_event.bid)
            else:
                order_book = self._get_order_book(order.symbol)
                order_book.add_order(order)
                return None
        
        # Calculate commission (no slippage for limit orders)
        commission = self._calculate_commission(order.quantity, fill_price)
        
        # Create fill
        self.fill_counter += 1
        fill = FillEvent(
            timestamp=market_event.timestamp,
            symbol=order.symbol,
            exchange="SIMULATED",
            quantity=order.quantity,
            side=order.side,
            fill_price=fill_price,
            commission=commission,
            slippage=0.0,  # Limit orders have no slippage
            order_id=order.order_id,
            fill_id=f"FILL_{self.fill_counter:08d}"
        )
        
        return fill
    
    def execute_stop_order(self,
                          order: OrderEvent,
                          market_event: MarketEvent) -> Optional[FillEvent]:
        """
        Execute stop order (becomes market order when stop price hit)
        
        Stop orders:
        - Trigger when price crosses stop price
        - Become market orders and cross spread
        - Experience slippage
        """
        # Check if stop is triggered
        if order.side == OrderSide.BUY:
            # Buy stop: trigger if price >= stop price
            triggered = market_event.close >= order.stop_price
        else:
            # Sell stop: trigger if price <= stop price
            triggered = market_event.close <= order.stop_price
        
        if not triggered:
            # Add to book, monitoring for trigger
            order_book = self._get_order_book(order.symbol)
            order_book.add_order(order)
            return None
        
        # Triggered - execute as market order
        # Stop orders usually have worse fills due to panic
        if order.side == OrderSide.BUY:
            base_price = market_event.ask
        else:
            base_price = market_event.bid
        
        # Stops have extra slippage (adverse selection)
        slippage = self._calculate_slippage(
            base_price,
            order.quantity,
            order.side
        ) * 1.5  # 50% worse slippage
        
        fill_price = base_price + slippage
        commission = self._calculate_commission(order.quantity, fill_price)
        
        self.fill_counter += 1
        fill = FillEvent(
            timestamp=market_event.timestamp,
            symbol=order.symbol,
            exchange="SIMULATED",
            quantity=order.quantity,
            side=order.side,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(slippage),
            order_id=order.order_id,
            fill_id=f"FILL_{self.fill_counter:08d}"
        )
        
        return fill
    
    def execute_order(self,
                     order: OrderEvent,
                     market_event: MarketEvent) -> Optional[FillEvent]:
        """Main execution dispatcher"""
        if order.order_type == OrderType.MARKET:
            return self.execute_market_order(order, market_event)
        elif order.order_type == OrderType.LIMIT:
            return self.execute_limit_order(order, market_event)
        elif order.order_type == OrderType.STOP:
            return self.execute_stop_order(order, market_event)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
