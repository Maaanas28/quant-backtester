"""
Position Sizer - Kelly Criterion and Fixed Risk
"""
from typing import Dict
from engine.events import SignalEvent
from engine.portfolio import Portfolio


class PositionSizer:
    """
    Position sizing algorithms
    """
    
    def __init__(self, method: str = "fixed", **kwargs):
        """
        method: "fixed", "kelly", "percent_equity"
        kwargs: method-specific parameters
        """
        self.method = method
        self.params = kwargs
    
    def calculate_quantity(self,
                          signal: SignalEvent,
                          portfolio: Portfolio,
                          current_prices: Dict[str, float]) -> int:
        """Calculate position size"""
        
        if signal.symbol not in current_prices:
            return 0
        
        price = current_prices[signal.symbol]
        equity = portfolio.get_equity(current_prices)
        
        if self.method == "fixed":
            # Fixed dollar amount
            fixed_amount = self.params.get("amount", 10000)
            quantity = int(fixed_amount / price)
        
        elif self.method == "kelly":
            # Kelly Criterion
            win_rate = self.params.get("win_rate", 0.55)
            win_loss_ratio = self.params.get("win_loss_ratio", 1.5)
            kelly_fraction = self.params.get("fraction", 0.25)  # Quarter Kelly
            
            # Kelly formula: (p * b - q) / b
            # where p = win rate, q = loss rate, b = win/loss ratio
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly = max(0, kelly) * kelly_fraction  # Apply fractional Kelly
            
            position_value = equity * kelly * signal.strength
            quantity = int(position_value / price)
        
        elif self.method == "percent_equity":
            # Percentage of equity
            pct = self.params.get("percent", 0.10)  # 10% default
            position_value = equity * pct * signal.strength
            quantity = int(position_value / price)
        
        elif self.method == "volatility_adjusted":
            target_risk = self.params.get("target_risk", 0.02)  # 2% risk
            quantity = int((equity * target_risk) / price)
        
        else:
            quantity = 100  # Default
        
        return max(quantity, 1)  # At least 1 share
