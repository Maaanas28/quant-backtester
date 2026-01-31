"""
Data Handler for Market Data
Supports historical and live data
"""
from typing import Dict, List, Optional, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import yfinance as yf
from collections import deque

from engine.events import MarketEvent


@dataclass
class BarData:
    """OHLCV bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataHandler:
    """
    Market data handler:
    - Historical data loading
    - Bar streaming
    - Bid/ask simulation
    - Event generation
    """
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_idx: Dict[str, int] = {}
        self.latest_bars: Dict[str, deque] = {}
        
        # For bid/ask simulation
        self.spread_bps = 5  # 5 basis points default
        
        # Load data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical data from yfinance"""
        print(f"Loading data for {self.symbols}...")
        
        for symbol in self.symbols:
            try:
                df = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                if df.empty:
                    print(f"Warning: No data for {symbol}")
                    continue
                
                # Ensure datetime index
                df.index = pd.to_datetime(df.index)
                
                # Store
                self.data[symbol] = df
                self.current_idx[symbol] = 0
                self.latest_bars[symbol] = deque(maxlen=100)  # Keep last 100 bars
                
                print(f"Loaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    def _simulate_bid_ask(self, symbol: str, close: float) -> tuple[float, float, int, int]:
        """
        Simulate bid/ask from close price
        
        Returns: (bid_price, ask_price, bid_size, ask_size)
        """
        spread = close * (self.spread_bps / 10000)
        mid = close
        
        bid_price = mid - spread / 2
        ask_price = mid + spread / 2
        
        # Random sizes (100-1000 shares)
        import random
        bid_size = random.randint(100, 1000)
        ask_size = random.randint(100, 1000)
        
        return bid_price, ask_price, bid_size, ask_size
    
    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        """Get most recent bar for symbol"""
        if not self.latest_bars.get(symbol):
            return None
        return self.latest_bars[symbol][-1]
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[BarData]:
        """Get last N bars for symbol"""
        if not self.latest_bars.get(symbol):
            return []
        return list(self.latest_bars[symbol])[-n:]
    
    def update_bars(self) -> List[MarketEvent]:
        """
        Get next bar for all symbols
        
        Returns list of MarketEvents
        """
        events = []
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
            
            df = self.data[symbol]
            idx = self.current_idx[symbol]
            
            if idx >= len(df):
                continue  # End of data
            
            # Get bar
            row = df.iloc[idx]
            timestamp = df.index[idx]
            
            bar = BarData(
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                open=float(row['Open'].iloc[0]) if hasattr(row['Open'], 'iloc') else float(row['Open']),
                high=float(row['High'].iloc[0]) if hasattr(row['High'], 'iloc') else float(row['High']),
                low=float(row['Low'].iloc[0]) if hasattr(row['Low'], 'iloc') else float(row['Low']),
                close=float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close']),
                volume=int(row['Volume']) if 'Volume' in row and not hasattr(row['Volume'], 'iloc') else (int(row['Volume'].iloc[0]) if 'Volume' in row else 0)
            )
            
            # Store bar
            self.latest_bars[symbol].append(bar)
            
            # Simulate bid/ask
            bid, ask, bid_size, ask_size = self._simulate_bid_ask(symbol, bar.close)
            
            # Create event
            event = MarketEvent(
                timestamp=timestamp,
                symbol=symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                bid_price=bid,
                ask_price=ask,
                bid_size=bid_size,
                ask_size=ask_size
            )
            
            events.append(event)
            
            # Increment index
            self.current_idx[symbol] += 1
        
        return events
    
    def continue_backtest(self) -> bool:
        """Check if backtest can continue"""
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
            if self.current_idx[symbol] < len(self.data[symbol]):
                return True
        return False
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            bar = self.get_latest_bar(symbol)
            if bar:
                prices[symbol] = bar.close
        return prices


class LiveDataHandler(DataHandler):
    """
    Live data handler with WebSocket support
    (For future implementation)
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.latest_bars: Dict[str, deque] = {s: deque(maxlen=100) for s in symbols}
        self.live_prices: Dict[str, float] = {}
    
    def connect_websocket(self):
        """Connect to live data feed (placeholder)"""
        raise NotImplementedError("WebSocket support coming soon")
    
    def on_tick(self, symbol: str, price: float, volume: int):
        """Handle incoming tick"""
        # TODO: Aggregate ticks into bars
        pass
