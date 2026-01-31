"""
Advanced analytics and metrics module for NeuroQuant
Provides institutional-grade performance metrics and risk analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annual_return: float
    daily_volatility: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    recovery_factor: float
    avg_win: float
    avg_loss: float
    avg_hold_time: float


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    conditional_sharpe: float
    downside_deviation: float
    beta: float
    correlation_to_benchmark: float


class AdvancedAnalytics:
    """Enterprise-grade analytics engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        trades: List[Dict],
        trading_days: int = 252
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        
        # Volatility
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(trading_days)
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / trading_days
        sharpe_ratio = excess_returns.mean() / daily_volatility * np.sqrt(trading_days) if daily_volatility > 0 else 0
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std()
        sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(trading_days) if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        max_drawdown_duration = drawdown_periods.sum()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information Ratio (vs. market average)
        benchmark_return = 0.10 / trading_days  # Assume 10% annual benchmark
        info_ratio = (returns.mean() - benchmark_return) / returns.std() * np.sqrt(trading_days) if returns.std() > 0 else 0
        
        # Trade metrics
        if trades:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            total_wins = sum(t.get('pnl', 0) for t in wins)
            total_losses = abs(sum(t.get('pnl', 0) for t in losses))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            avg_win = total_wins / len(wins) if wins else 0
            avg_loss = total_losses / len(losses) if losses else 0
            
            # Average hold time (in days)
            hold_times = [t.get('hold_days', 0) for t in trades]
            avg_hold_time = np.mean(hold_times) if hold_times else 0
        else:
            win_rate = 0
            profit_factor = 0
            recovery_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_hold_time = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            daily_volatility=daily_volatility,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=info_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(max_drawdown_duration),
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_time=avg_hold_time
        )
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> RiskMetrics:
        """Calculate advanced risk metrics"""
        
        # Value at Risk (historical)
        var_95 = returns.quantile(1 - confidence_level)
        var_99 = returns.quantile(0.01)
        
        # Expected Shortfall (CVaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Conditional Sharpe Ratio
        downside_returns = returns[returns < 0]
        conditional_sharpe = downside_returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Downside deviation
        downside_variance = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
        
        # Beta
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Correlation to benchmark
        correlation = returns.corr(benchmark_returns)
        
        return RiskMetrics(
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            conditional_sharpe=conditional_sharpe,
            downside_deviation=downside_variance,
            beta=beta,
            correlation_to_benchmark=correlation
        )
    
    def calculate_attribution_analysis(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Perform factor attribution analysis"""
        
        attribution = {}
        for factor_name, factor_ret in factor_returns.items():
            # Simple regression coefficient as contribution
            covariance = portfolio_returns.cov(factor_ret)
            factor_variance = factor_ret.var()
            attribution[factor_name] = covariance / factor_variance if factor_variance > 0 else 0
        
        return attribution
    
    def stress_test(
        self,
        returns: pd.Series,
        shock_magnitude: float = 0.2
    ) -> Dict[str, float]:
        """Perform stress test analysis"""
        
        scenarios = {
            "market_crash": -shock_magnitude,
            "market_rally": shock_magnitude,
            "volatility_spike": returns.std() * shock_magnitude,
            "tail_risk": returns.quantile(0.05)
        }
        
        stressed_returns = {}
        for scenario_name, shock in scenarios.items():
            stressed_ret = returns + shock
            stressed_returns[f"{scenario_name}_return"] = (1 + stressed_ret).prod() - 1
        
        return stressed_returns
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        periods: int = 252,
        simulations: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Monte Carlo simulation for future returns"""
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        simulated_paths = np.zeros((simulations, periods))
        
        for i in range(simulations):
            path = np.zeros(periods)
            for j in range(periods):
                random_return = np.random.normal(mean_return, volatility)
                path[j] = random_return
            simulated_paths[i] = path
        
        cumulative_paths = (1 + simulated_paths).cumprod(axis=1)
        
        return {
            "paths": simulated_paths,
            "cumulative_paths": cumulative_paths,
            "mean_path": cumulative_paths.mean(axis=0),
            "percentile_5": np.percentile(cumulative_paths[:, -1], 5),
            "percentile_95": np.percentile(cumulative_paths[:, -1], 95),
            "median_path": np.median(cumulative_paths, axis=0)
        }
    
    def portfolio_optimization(
        self,
        asset_returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize portfolio weights using Modern Portfolio Theory"""
        
        n_assets = len(asset_returns.columns)
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # Simple equal-weight optimization (can be enhanced with quadratic programming)
        weights = np.array([1.0 / n_assets] * n_assets)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        return {
            "weights": {col: w for col, w in zip(asset_returns.columns, weights)},
            "expected_return": portfolio_return,
            "expected_volatility": portfolio_volatility,
            "sharpe_ratio": portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        }


class ReportGenerator:
    """Generate comprehensive trading reports"""
    
    @staticmethod
    def generate_performance_report(
        metrics: PerformanceMetrics,
        risk_metrics: RiskMetrics,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate comprehensive performance report"""
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "returns": {
                "total_return": f"{metrics.total_return * 100:.2f}%",
                "annual_return": f"{metrics.annual_return * 100:.2f}%",
                "monthly_return": f"{(metrics.total_return / ((end_date - start_date).days / 30)) * 100:.2f}%"
            },
            "risk": {
                "annual_volatility": f"{metrics.annual_volatility * 100:.2f}%",
                "max_drawdown": f"{metrics.max_drawdown * 100:.2f}%",
                "value_at_risk_95": f"{risk_metrics.value_at_risk_95 * 100:.2f}%",
                "downside_deviation": f"{risk_metrics.downside_deviation * 100:.2f}%"
            },
            "ratios": {
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                "calmar_ratio": f"{metrics.calmar_ratio:.2f}",
                "information_ratio": f"{metrics.information_ratio:.2f}"
            },
            "trading": {
                "win_rate": f"{metrics.win_rate * 100:.2f}%",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "avg_win": f"${metrics.avg_win:.2f}",
                "avg_loss": f"${metrics.avg_loss:.2f}",
                "avg_hold_time": f"{metrics.avg_hold_time:.1f} days"
            }
        }


if __name__ == "__main__":
    # Example usage
    analytics = AdvancedAnalytics()
    
    # Generate sample returns
    returns = pd.Series(np.random.normal(0.001, 0.015, 252))
    trades = [
        {"pnl": 1000, "hold_days": 5},
        {"pnl": -500, "hold_days": 3},
        {"pnl": 1500, "hold_days": 7},
    ]
    
    metrics = analytics.calculate_performance_metrics(returns, trades)
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
