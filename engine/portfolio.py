"""
Portfolio and Risk Management Engine
Institutional-grade risk controls
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

from engine.events import FillEvent, OrderSide, PositionEvent


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: int  # Positive = long, negative = short
    avg_price: float
    realized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.avg_price


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: Dict[str, Position]
    equity_curve: List[float]
    timestamps: List[datetime]
    

class Portfolio:
    """
    Portfolio manager with:
    - Position tracking
    - PnL calculation
    - Margin accounting
    - Portfolio metrics
    """
    
    def __init__(self, initial_capital: float = 100000.0, enable_shadow: bool = True):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Track history
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        self.fill_history: List[FillEvent] = []
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        # Shadow Portfolio - executes opposite trades
        self.enable_shadow = enable_shadow
        self.shadow_cash = initial_capital
        self.shadow_positions: Dict[str, Position] = {}
        self.shadow_equity_curve = [initial_capital]
        self.shadow_fill_history: List[FillEvent] = []
        self.shadow_total_commission = 0.0
    
    def update_fill(self, fill: FillEvent, current_prices: Dict[str, float]):
        """
        Update portfolio based on fill
        
        Handles:
        - Opening positions
        - Adding to positions
        - Closing positions
        - Partial closes
        - Shadow portfolio (opposite trades)
        """
        # Process main portfolio
        self._process_fill(fill, is_shadow=False)
        
        # Process shadow portfolio with inverse trade
        if self.enable_shadow:
            inverse_fill = self._create_inverse_fill(fill)
            self._process_fill(inverse_fill, is_shadow=True)
        
        # Update equity curves
        self._update_equity(current_prices)
    
    def _create_inverse_fill(self, fill: FillEvent) -> FillEvent:
        """Create inverse fill for shadow portfolio"""
        inverse_side = OrderSide.SELL if fill.side == OrderSide.BUY else OrderSide.BUY
        
        # Handle potentially None order_id and fill_id
        order_id = f"{fill.order_id}_shadow" if fill.order_id else "shadow_order"
        fill_id = f"{fill.fill_id}_shadow" if fill.fill_id else "shadow_fill"
        
        return FillEvent(
            symbol=fill.symbol,
            exchange=fill.exchange,
            quantity=fill.quantity,
            side=inverse_side,
            fill_price=fill.fill_price,
            commission=fill.commission,
            slippage=fill.slippage,
            order_id=order_id,
            fill_id=fill_id,
            timestamp=fill.timestamp
        )
    
    def _process_fill(self, fill: FillEvent, is_shadow: bool = False):
        """Process fill for main or shadow portfolio"""
        # Select portfolio state
        if is_shadow:
            cash = self.shadow_cash
            positions = self.shadow_positions
            fill_history = self.shadow_fill_history
            total_commission = self.shadow_total_commission
        else:
            cash = self.cash
            positions = self.positions
            fill_history = self.fill_history
            total_commission = self.total_commission_paid
        
        symbol = fill.symbol
        
        # Track fill
        fill_history.append(fill)
        total_commission += fill.commission
        
        if is_shadow:
            self.shadow_total_commission = total_commission
        else:
            self.total_commission_paid = total_commission
            self.total_slippage_cost += fill.slippage * fill.quantity
        
        # Get or create position
        if symbol not in positions:
            positions[symbol] = Position(symbol, 0, 0.0)
        
        position = positions[symbol]
        transaction_cost = fill.commission
        
        if fill.side == OrderSide.BUY:
            # Buying
            cost = fill.quantity * fill.fill_price + transaction_cost
            
            if position.quantity < 0:
                # Covering short - realize PnL
                cover_qty = min(fill.quantity, abs(position.quantity))
                realized_pnl = cover_qty * (position.avg_price - fill.fill_price) - transaction_cost
                position.realized_pnl += realized_pnl
                position.quantity += cover_qty
                
                if fill.quantity > cover_qty:
                    # Going long after covering
                    remaining = fill.quantity - cover_qty
                    position.quantity = remaining
                    position.avg_price = fill.fill_price
            else:
                # Adding to long or opening long
                total_qty = position.quantity + fill.quantity
                new_avg = ((position.quantity * position.avg_price) + 
                          (fill.quantity * fill.fill_price)) / total_qty if total_qty > 0 else fill.fill_price
                position.quantity = total_qty
                position.avg_price = new_avg
            
            cash -= cost
            
        else:  # SELL
            # Selling
            proceeds = fill.quantity * fill.fill_price - transaction_cost
            
            if position.quantity > 0:
                # Closing long - realize PnL
                close_qty = min(fill.quantity, position.quantity)
                realized_pnl = close_qty * (fill.fill_price - position.avg_price) - transaction_cost
                position.realized_pnl += realized_pnl
                position.quantity -= close_qty
                
                if fill.quantity > close_qty:
                    # Going short after closing
                    remaining = fill.quantity - close_qty
                    position.quantity = -remaining
                    position.avg_price = fill.fill_price
            else:
                # Adding to short or opening short
                total_qty = position.quantity - fill.quantity
                new_avg = ((abs(position.quantity) * position.avg_price) + 
                          (fill.quantity * fill.fill_price)) / abs(total_qty) if total_qty != 0 else fill.fill_price
                position.quantity = total_qty
                position.avg_price = new_avg
            
            cash += proceeds
        
        # Remove position if flat
        if position.quantity == 0:
            del positions[symbol]
        
        # Update cash
        if is_shadow:
            self.shadow_cash = cash
        else:
            self.cash = cash
    
    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for position"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        if position.quantity > 0:
            # Long position
            return position.quantity * (current_price - position.avg_price)
        else:
            # Short position
            return abs(position.quantity) * (position.avg_price - current_price)
    
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Total PnL (realized + unrealized)"""
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(
            self.get_unrealized_pnl(symbol, current_prices[symbol])
            for symbol in self.positions.keys()
            if symbol in current_prices
        )
        return realized + unrealized
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Current portfolio equity"""
        positions_value = sum(
            self.get_unrealized_pnl(symbol, current_prices[symbol])
            for symbol in self.positions.keys()
            if symbol in current_prices
        )
        return self.cash + positions_value
    
    def get_shadow_equity(self, current_prices: Dict[str, float]) -> float:
        """Current shadow portfolio equity"""
        if not self.enable_shadow:
            return 0.0
        
        positions_value = sum(
            self._get_shadow_unrealized_pnl(symbol, current_prices[symbol])
            for symbol in self.shadow_positions.keys()
            if symbol in current_prices
        )
        return self.shadow_cash + positions_value
    
    def _get_shadow_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for shadow position"""
        if symbol not in self.shadow_positions:
            return 0.0
        
        position = self.shadow_positions[symbol]
        if position.quantity > 0:
            return position.quantity * (current_price - position.avg_price)
        else:
            return abs(position.quantity) * (position.avg_price - current_price)
    
    def _update_equity(self, current_prices: Dict[str, float]):
        """Update equity curve and drawdown for both main and shadow portfolios"""
        # Update main portfolio
        equity = self.get_equity(current_prices)
        self.equity_curve.append(equity)
        self.timestamps.append(datetime.now())
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Update shadow portfolio if enabled
        if self.enable_shadow:
            shadow_equity = self.get_shadow_equity(current_prices)
            self.shadow_equity_curve.append(shadow_equity)
    
    def get_position_exposure(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get exposure per position as % of equity"""
        equity = self.get_equity(current_prices)
        exposures = {}
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_value = abs(position.quantity) * current_prices[symbol]
                exposures[symbol] = market_value / equity
        
        return exposures
    
    def calculate_returns(self, is_shadow: bool = False) -> np.ndarray:
        """Calculate period returns for main or shadow portfolio"""
        equity_curve = self.shadow_equity_curve if is_shadow else self.equity_curve
        
        if len(equity_curve) < 2:
            return np.array([])
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        return returns
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02, is_shadow: bool = False) -> float:
        """Sharpe ratio (annualized) for main or shadow portfolio"""
        returns = self.calculate_returns(is_shadow=is_shadow)
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        std_returns = np.std(returns)
        
        if std_returns == 0 or np.isnan(std_returns):
            return 0.0
        
        sharpe = np.mean(excess_returns) / std_returns * np.sqrt(252)
        
        # Handle NaN or inf
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        
        return float(sharpe)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02, is_shadow: bool = False) -> float:
        """Sortino ratio (downside deviation) for main or shadow portfolio"""
        returns = self.calculate_returns(is_shadow=is_shadow)
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        # Handle NaN or inf
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0
        
        return float(sortino)
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Value at Risk (parametric)"""
        returns = self.calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        var = -(mean_return + z_score * std_return)
        return var * self.get_equity({})  # Dollar VaR
    
    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """Expected Shortfall / CVaR"""
        returns = self.calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        var_percentile = np.percentile(returns, (1 - confidence) * 100)
        es = returns[returns <= var_percentile].mean()
        return -es * self.get_equity({})


class RiskManager:
    """
    Risk management with circuit breakers:
    - Max drawdown limits
    - Position size limits
    - Concentration limits
    - VaR limits
    - Margin requirements
    """
    
    def __init__(self,
                 max_drawdown_pct: float = 0.20,      # 20% max drawdown
                 max_position_pct: float = 0.25,       # 25% max per position
                 max_total_exposure_pct: float = 1.0,  # 100% max exposure
                 var_limit_pct: float = 0.05,          # 5% daily VaR limit
                 margin_requirement: float = 0.25):    # 25% margin
        
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.var_limit_pct = var_limit_pct
        self.margin_requirement = margin_requirement
        
        self.circuit_breaker_triggered = False
        self.violations: List[str] = []
    
    def check_drawdown(self, portfolio: Portfolio) -> bool:
        """Check if drawdown limit breached"""
        if portfolio.max_drawdown > self.max_drawdown_pct:
            self.violations.append(
                f"Max drawdown breached: {portfolio.max_drawdown:.2%} > {self.max_drawdown_pct:.2%}"
            )
            self.circuit_breaker_triggered = True
            return False
        return True
    
    def check_position_size(self,
                           portfolio: Portfolio,
                           symbol: str,
                           quantity: int,
                           price: float,
                           current_prices: Dict[str, float]) -> bool:
        """Check if position size limit breached"""
        equity = portfolio.get_equity(current_prices)
        position_value = quantity * price
        position_pct = position_value / equity
        
        if position_pct > self.max_position_pct:
            self.violations.append(
                f"Position size limit breached for {symbol}: {position_pct:.2%} > {self.max_position_pct:.2%}"
            )
            return False
        return True
    
    def check_total_exposure(self,
                            portfolio: Portfolio,
                            current_prices: Dict[str, float]) -> bool:
        """Check total exposure limit"""
        exposures = portfolio.get_position_exposure(current_prices)
        total_exposure = sum(exposures.values())
        
        if total_exposure > self.max_total_exposure_pct:
            self.violations.append(
                f"Total exposure limit breached: {total_exposure:.2%} > {self.max_total_exposure_pct:.2%}"
            )
            return False
        return True
    
    def check_var(self, portfolio: Portfolio) -> bool:
        """Check VaR limit"""
        equity = portfolio.get_equity({})
        var = portfolio.calculate_var(confidence=0.95)
        var_pct = var / equity
        
        if var_pct > self.var_limit_pct:
            self.violations.append(
                f"VaR limit breached: {var_pct:.2%} > {self.var_limit_pct:.2%}"
            )
            return False
        return True
    
    def check_margin(self,
                    portfolio: Portfolio,
                    current_prices: Dict[str, float]) -> bool:
        """Check margin requirements"""
        equity = portfolio.get_equity(current_prices)
        
        # Calculate required margin
        required_margin = 0
        for symbol, position in portfolio.positions.items():
            if symbol in current_prices:
                position_value = abs(position.quantity) * current_prices[symbol]
                required_margin += position_value * self.margin_requirement
        
        if portfolio.cash < required_margin:
            self.violations.append(
                f"Margin call: Cash ${portfolio.cash:.2f} < Required ${required_margin:.2f}"
            )
            return False
        return True
    
    def check_all(self,
                  portfolio: Portfolio,
                  current_prices: Dict[str, float]) -> bool:
        """Run all risk checks"""
        self.violations = []
        
        checks = [
            self.check_drawdown(portfolio),
            self.check_total_exposure(portfolio, current_prices),
            self.check_var(portfolio),
            self.check_margin(portfolio, current_prices)
        ]
        
        return all(checks)
