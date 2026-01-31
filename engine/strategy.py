"""
Strategy Base Classes
Framework for algorithmic trading strategies
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

from engine.events import MarketEvent, SignalEvent, SignalType
from engine.data_handler import DataHandler


class Strategy(ABC):
    """
    Base strategy class
    
    All strategies must implement:
    - calculate_signals(): Generate trading signals
    """
    
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.bars_processed = 0
    
    @abstractmethod
    def calculate_signals(self, event: MarketEvent, data_handler: DataHandler) -> List[SignalEvent]:
        """
        Generate signals based on market data
        
        Returns list of SignalEvents
        """
        raise NotImplementedError("Must implement calculate_signals()")
    
    def on_fill(self, fill_event):
        """Called when order is filled (for stateful strategies)"""
        pass


class MovingAverageCrossStrategy(Strategy):
    """
    Simple moving average crossover strategy
    
    Long when fast MA crosses above slow MA
    Short when fast MA crosses below slow MA
    """
    
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50):
        super().__init__("MA_Cross", symbols)
        self.short_window = short_window
        self.long_window = long_window
        
        # Track positions
        self.positions: Dict[str, Optional[SignalType]] = {s: None for s in symbols}
    
    def calculate_signals(self, event: MarketEvent, data_handler: DataHandler) -> List[SignalEvent]:
        """Generate MA crossover signals"""
        signals = []
        
        if event.symbol not in self.symbols:
            return signals
        
        # Get historical bars (need extra bar for previous MA calculation)
        bars = data_handler.get_latest_bars(event.symbol, n=self.long_window + 1)
        
        if len(bars) < self.long_window + 1:
            return signals  # Not enough data for crossover detection
        
        # Calculate current MAs
        closes = [float(bar.close) for bar in bars]  # Ensure floats
        short_ma = sum(closes[-self.short_window:]) / self.short_window
        long_ma = sum(closes[-self.long_window:]) / self.long_window
        
        # Calculate previous MAs (one bar back)
        prev_closes = closes[:-1]
        prev_short_ma = sum(prev_closes[-self.short_window:]) / self.short_window
        prev_long_ma = sum(prev_closes[-self.long_window:]) / self.long_window
        
        symbol = event.symbol
        current_position = self.positions[symbol]
        
        # Crossover detection
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            # Bullish crossover
            if current_position != SignalType.LONG:
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=1.0,
                    strategy_name=self.name
                )
                signals.append(signal)
                self.positions[symbol] = SignalType.LONG
        
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            # Bearish crossover
            if current_position != SignalType.SHORT:
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=1.0,
                    strategy_name=self.name
                )
                signals.append(signal)
                self.positions[symbol] = SignalType.SHORT
        
        self.bars_processed += 1
        return signals


class RSIStrategy(Strategy):
    """
    RSI mean reversion strategy
    
    Long when RSI < 30 (oversold)
    Short when RSI > 70 (overbought)
    Exit when RSI returns to 50
    """
    
    def __init__(self, symbols: List[str], period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI", symbols)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
        self.positions: Dict[str, Optional[SignalType]] = {s: None for s in symbols}
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI"""
        if len(prices) < self.period + 1:
            return 50.0  # Neutral
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-self.period:]) / self.period
        avg_loss = sum(losses[-self.period:]) / self.period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_signals(self, event: MarketEvent, data_handler: DataHandler) -> List[SignalEvent]:
        """Generate RSI signals"""
        signals = []
        
        if event.symbol not in self.symbols:
            return signals
        
        # Get historical bars
        bars = data_handler.get_latest_bars(event.symbol, n=self.period + 10)
        
        if len(bars) < self.period + 1:
            return signals
        
        # Calculate RSI
        closes = [bar.close for bar in bars]
        rsi = self._calculate_rsi(closes)
        
        symbol = event.symbol
        current_position = self.positions[symbol]
        
        # Generate signals
        if rsi < self.oversold and current_position != SignalType.LONG:
            # Oversold - go long
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=(self.oversold - rsi) / self.oversold,  # Strength based on how oversold
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = SignalType.LONG
        
        elif rsi > self.overbought and current_position != SignalType.SHORT:
            # Overbought - go short
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strength=(rsi - self.overbought) / (100 - self.overbought),
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = SignalType.SHORT
        
        elif 45 < rsi < 55 and current_position is not None:
            # Return to neutral - exit
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.EXIT,
                strength=1.0,
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = None
        
        self.bars_processed += 1
        return signals


class MomentumStrategy(Strategy):
    """
    Momentum strategy
    
    Long when price momentum is positive
    Short when price momentum is negative
    """
    
    def __init__(self, symbols: List[str], lookback: int = 20):
        super().__init__("Momentum", symbols)
        self.lookback = lookback
        self.positions: Dict[str, Optional[SignalType]] = {s: None for s in symbols}
    
    def calculate_signals(self, event: MarketEvent, data_handler: DataHandler) -> List[SignalEvent]:
        """Generate momentum signals"""
        signals = []
        
        if event.symbol not in self.symbols:
            return signals
        
        bars = data_handler.get_latest_bars(event.symbol, n=self.lookback + 1)
        
        if len(bars) < self.lookback + 1:
            return signals
        
        # Calculate momentum (rate of change)
        current_price = bars[-1].close
        past_price = bars[0].close
        momentum = (current_price - past_price) / past_price
        
        symbol = event.symbol
        current_position = self.positions[symbol]
        
        # Threshold: 2% momentum
        if momentum > 0.02 and current_position != SignalType.LONG:
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=min(momentum * 10, 1.0),  # Cap at 1.0
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = SignalType.LONG
        
        elif momentum < -0.02 and current_position != SignalType.SHORT:
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strength=min(abs(momentum) * 10, 1.0),
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = SignalType.SHORT
        
        elif -0.01 < momentum < 0.01 and current_position is not None:
            # Momentum fading - exit
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.EXIT,
                strength=1.0,
                strategy_name=self.name
            )
            signals.append(signal)
            self.positions[symbol] = None
        
        self.bars_processed += 1
        return signals


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy and hold benchmark
    """
    
    def __init__(self, symbols: List[str]):
        super().__init__("BuyAndHold", symbols)
        self.invested = {s: False for s in symbols}
    
    def calculate_signals(self, event: MarketEvent, data_handler: DataHandler) -> List[SignalEvent]:
        """Buy on first bar, hold forever"""
        signals = []
        
        symbol = event.symbol
        if not self.invested.get(symbol, False):
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=1.0,
                strategy_name=self.name
            )
            signals.append(signal)
            self.invested[symbol] = True
        
        self.bars_processed += 1
        return signals
