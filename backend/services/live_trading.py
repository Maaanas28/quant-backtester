"""
Live Trading Simulation Engine
Paper trading with real-time market data
"""
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from backend.services.portfolio_manager import PortfolioManager


@dataclass
class LiveTradeSignal:
    """Real-time trading signal"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reason: str
    agent_name: str


@dataclass
class AlertRule:
    """Alert/notification rule"""
    rule_id: str
    symbol: str
    condition: str  # 'price_above', 'price_below', 'rsi_oversold', 'rsi_overbought', 'volume_spike'
    threshold: float
    enabled: bool = True
    triggered: bool = False
    last_check: Optional[datetime] = None


class LiveTradingSimulator:
    """
    Paper trading engine with real-time data
    Simulates live trading without risking real money
    """
    
    def __init__(self, 
                 symbols: List[str],
                 initial_capital: float = 10000.0,
                 update_interval: int = 60):
        """
        Args:
            symbols: List of symbols to track
            initial_capital: Starting capital for simulation
            update_interval: Update frequency in seconds
        """
        self.symbols = symbols
        self.portfolio = PortfolioManager(initial_capital)
        self.update_interval = update_interval
        
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {s: [] for s in symbols}
        self.signals: List[LiveTradeSignal] = []
        self.alerts: List[AlertRule] = []
        
        self.is_running = False
        self.agent_callbacks: Dict[str, Callable] = {}
        
    def register_strategy(self, strategy_name: str, callback: Callable):
        """
        Register a strategy for live trading
        
        Args:
            strategy_name: Name of the strategy
            callback: Function that takes (symbol, price_data) and returns action, confidence, reason
        """
        self.agent_callbacks[strategy_name] = callback
    
    def add_alert(self, alert: AlertRule):
        """Add price alert or notification rule"""
        self.alerts.append(alert)
    
    async def fetch_live_prices(self) -> Dict[str, float]:
        """Fetch current market prices"""
        prices = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='1m')
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prices[symbol] = current_price
                    
                    # Store history
                    timestamp = datetime.now()
                    self.price_history[symbol].append((timestamp, current_price))
                    
                    # Keep only last 1000 data points
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]
                        
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        return prices
    
    def check_alerts(self):
        """Check if any alert conditions are met"""
        triggered_alerts = []
        
        for alert in self.alerts:
            if not alert.enabled or alert.symbol not in self.current_prices:
                continue
            
            current_price = self.current_prices[alert.symbol]
            triggered = False
            
            if alert.condition == 'price_above' and current_price > alert.threshold:
                triggered = True
            elif alert.condition == 'price_below' and current_price < alert.threshold:
                triggered = True
            
            if triggered and not alert.triggered:
                alert.triggered = True
                alert.last_check = datetime.now()
                triggered_alerts.append(alert)
        
        return triggered_alerts
    
    async def generate_signals(self):
        """Generate trading signals from registered strategies"""
        for strategy_name, callback in self.agent_callbacks.items():
            for symbol in self.symbols:
                if symbol not in self.price_history or not self.price_history[symbol]:
                    continue
                
                # Get recent price data
                recent_data = self.price_history[symbol][-100:]  # Last 100 data points
                
                try:
                    # Call strategy callback
                    action, confidence, reason = callback(symbol, recent_data)
                    
                    signal = LiveTradeSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        reason=reason,
                        agent_name=strategy_name
                    )
                    
                    self.signals.append(signal)
                    
                    # Execute trade based on signal
                    if action == 'BUY' and confidence > 0.7:
                        self.portfolio.execute_trade(
                            symbol, 
                            'BUY', 
                            self.current_prices[symbol],
                            signal_strength=confidence
                        )
                    elif action == 'SELL' and symbol in self.portfolio.positions:
                        self.portfolio.execute_trade(
                            symbol, 
                            'SELL', 
                            self.current_prices[symbol]
                        )
                        
                except Exception as e:
                    print(f"Error generating signal for {symbol} with {strategy_name}: {e}")
    
    async def run(self, duration_hours: Optional[float] = None):
        """
        Start live trading simulation
        
        Args:
            duration_hours: How long to run (None = indefinitely)
        """
        self.is_running = True
        start_time = datetime.now()
        
        print(f"🚀 Starting live trading simulation at {start_time}")
        print(f"📊 Tracking symbols: {', '.join(self.symbols)}")
        print(f"💰 Initial capital: ${self.portfolio.initial_capital:,.2f}")
        
        try:
            while self.is_running:
                # Check if duration exceeded
                if duration_hours and (datetime.now() - start_time).total_seconds() / 3600 > duration_hours:
                    break
                
                # Fetch current prices
                self.current_prices = await self.fetch_live_prices()
                
                # Check stop losses
                stop_losses = self.portfolio.check_stop_losses(self.current_prices)
                if stop_losses:
                    print(f"⚠️  Stop loss triggered: {len(stop_losses)} positions closed")
                
                # Check alerts
                triggered_alerts = self.check_alerts()
                for alert in triggered_alerts:
                    print(f"🔔 Alert: {alert.symbol} {alert.condition} {alert.threshold}")
                
                # Generate and execute signals
                await self.generate_signals()
                
                # Update portfolio value
                current_value = self.portfolio.get_portfolio_value(self.current_prices)
                total_return = (current_value - self.portfolio.initial_capital) / self.portfolio.initial_capital
                
                # Log status
                print(f"\n{'='*60}")
                print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💼 Portfolio Value: ${current_value:,.2f} ({total_return:+.2%})")
                print(f"📈 Positions: {len(self.portfolio.positions)}")
                print(f"💵 Cash: ${self.portfolio.current_capital:,.2f}")
                
                if self.portfolio.positions:
                    print("\n📊 Open Positions:")
                    for symbol, pos in self.portfolio.positions.items():
                        pnl = (pos['current_price'] - pos['avg_price']) / pos['avg_price']
                        print(f"  {symbol}: {pos['shares']:.2f} @ ${pos['avg_price']:.2f} "
                              f"(Current: ${pos['current_price']:.2f}, P&L: {pnl:+.2%})")
                
                print(f"{'='*60}\n")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n⏸️  Simulation paused by user")
        finally:
            self.is_running = False
            self._print_final_report()
    
    def stop(self):
        """Stop the simulation"""
        self.is_running = False
    
    def _print_final_report(self):
        """Print final trading report"""
        print("\n" + "="*60)
        print("📊 LIVE TRADING SIMULATION REPORT")
        print("="*60)
        
        # Close all positions at current prices
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in self.current_prices:
                self.portfolio.execute_trade(symbol, 'SELL', self.current_prices[symbol])
        
        metrics = self.portfolio.get_performance_metrics()
        
        print(f"\n💰 Final Capital: ${self.portfolio.current_capital:,.2f}")
        print(f"📈 Total Return: {metrics.get('total_return', 0):+.2%}")
        print(f"📊 Total Trades: {metrics.get('total_trades', 0)}")
        print(f"🎯 Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"⚡ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"💵 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"📈 Largest Win: ${metrics.get('largest_win', 0):,.2f}")
        print(f"📉 Largest Loss: ${metrics.get('largest_loss', 0):,.2f}")
        
        print("\n" + "="*60)
    
    def get_status(self) -> Dict:
        """Get current simulation status"""
        current_value = self.portfolio.get_portfolio_value(self.current_prices)
        
        return {
            'is_running': self.is_running,
            'current_value': current_value,
            'total_return': (current_value - self.portfolio.initial_capital) / self.portfolio.initial_capital,
            'positions': len(self.portfolio.positions),
            'cash': self.portfolio.current_capital,
            'recent_signals': self.signals[-10:] if self.signals else [],
            'portfolio_allocation': self.portfolio.get_portfolio_allocation()
        }


# Example strategy callback
def simple_momentum_strategy(symbol: str, price_data: List[Tuple[datetime, float]]) -> tuple:
    """
    Simple momentum-based trading strategy
    Returns: (action, confidence, reason)
    """
    if len(price_data) < 20:
        return 'HOLD', 0.0, 'Insufficient data'
    
    prices = np.array([p[1] for p in price_data])
    
    # Calculate short-term and long-term moving averages
    short_ma = np.mean(prices[-5:])
    long_ma = np.mean(prices[-20:])
    
    # Calculate momentum
    momentum = (short_ma - long_ma) / long_ma
    
    # Calculate RSI
    deltas = np.diff(prices)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]
    
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signal
    if momentum > 0.02 and rsi < 70:
        return 'BUY', 0.8, f'Bullish momentum ({momentum:.2%}), RSI={rsi:.1f}'
    elif momentum < -0.02 or rsi > 80:
        return 'SELL', 0.9, f'Bearish momentum ({momentum:.2%}), RSI={rsi:.1f}'
    else:
        return 'HOLD', 0.5, f'Neutral, RSI={rsi:.1f}'
