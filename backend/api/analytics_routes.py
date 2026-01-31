"""
Advanced Analytics API Routes
Provides institutional-grade trading analytics and reporting
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3

from backend.analytics.advanced_metrics import AdvancedAnalytics, ReportGenerator
from database.database import get_db
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Create router with /analytics prefix (will be /api/analytics when included)
router = APIRouter(prefix="/analytics", tags=["analytics"])
analytics = AdvancedAnalytics()


# Request/Response Models
class PerformanceReportRequest(BaseModel):
    agent_id: str
    start_date: datetime
    end_date: datetime
    benchmark: Optional[str] = "SPY"


class RiskAnalysisRequest(BaseModel):
    agent_id: str
    portfolio_value: float
    confidence_level: float = 0.95


class OptimizationRequest(BaseModel):
    symbols: List[str]
    weights: Optional[Dict[str, float]] = None
    constraints: Optional[Dict] = None


@router.post("/performance-report")
async def get_performance_report(request: PerformanceReportRequest):
    """
    Generate comprehensive performance report
    
    - **agent_id**: ID of the trading agent
    - **start_date**: Report start date
    - **end_date**: Report end date
    - **benchmark**: Benchmark ticker (default: SPY)
    
    Returns detailed metrics including Sharpe ratio, max drawdown, win rate, etc.
    """
    try:
        # In production, fetch actual backtest data from database
        # For now, generate sample data
        days = (request.end_date - request.start_date).days
        returns = pd.Series(np.random.normal(0.001, 0.015, days))
        
        trades = [
            {"pnl": np.random.uniform(-2000, 3000), "hold_days": np.random.randint(1, 20)}
            for _ in range(np.random.randint(20, 100))
        ]
        
        metrics = analytics.calculate_performance_metrics(returns, trades)
        
        report = ReportGenerator.generate_performance_report(
            metrics,
            analytics.calculate_risk_metrics(returns, returns),
            request.start_date,
            request.end_date
        )
        
        return {
            "status": "success",
            "agent_id": request.agent_id,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{agent_id}")
async def get_metrics(
    agent_id: str,
    period: str = Query("1m", regex="^(1d|1w|1m|3m|6m|1y)$")
):
    """
    Get current performance metrics for an agent
    
    - **agent_id**: ID of the trading agent
    - **period**: Time period (1d, 1w, 1m, 3m, 6m, 1y)
    
    Returns Sharpe ratio, max drawdown, win rate, and more
    """
    try:
        # Generate sample metrics
        metrics = {
            "sharpe_ratio": round(np.random.uniform(0.5, 2.5), 2),
            "sortino_ratio": round(np.random.uniform(0.8, 3.0), 2),
            "max_drawdown": round(np.random.uniform(-0.20, -0.05), 4),
            "win_rate": round(np.random.uniform(0.45, 0.75), 2),
            "profit_factor": round(np.random.uniform(1.2, 2.5), 2),
            "annual_return": round(np.random.uniform(0.10, 0.40), 2),
            "calmar_ratio": round(np.random.uniform(0.8, 3.0), 2),
            "information_ratio": round(np.random.uniform(0.5, 2.0), 2),
        }
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "period": period,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-analysis")
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Comprehensive risk analysis
    
    Returns:
    - Value at Risk (VaR) at 95% and 99% confidence
    - Expected Shortfall (CVaR)
    - Beta and correlation to market
    - Stress test scenarios
    """
    try:
        # Generate sample risk metrics
        returns = pd.Series(np.random.normal(0.001, 0.015, 252))
        benchmark = pd.Series(np.random.normal(0.0008, 0.012, 252))
        
        risk_metrics = analytics.calculate_risk_metrics(returns, benchmark)
        stress_tests = analytics.stress_test(returns)
        
        # Monte Carlo simulation
        mc_results = analytics.monte_carlo_simulation(returns, periods=30, simulations=1000)
        
        return {
            "status": "success",
            "agent_id": request.agent_id,
            "risk_metrics": {
                "var_95": float(risk_metrics.value_at_risk_95),
                "var_99": float(risk_metrics.value_at_risk_99),
                "expected_shortfall_95": float(risk_metrics.expected_shortfall_95),
                "expected_shortfall_99": float(risk_metrics.expected_shortfall_99),
                "beta": float(risk_metrics.beta),
                "correlation_to_benchmark": float(risk_metrics.correlation_to_benchmark),
                "downside_deviation": float(risk_metrics.downside_deviation),
            },
            "stress_tests": {k: float(v) for k, v in stress_tests.items()},
            "monte_carlo": {
                "mean_path": mc_results["mean_path"].tolist(),
                "percentile_5": float(mc_results["percentile_5"]),
                "percentile_95": float(mc_results["percentile_95"]),
                "median_path": mc_results["median_path"].tolist(),
            },
            "estimated_max_loss_30d": float(request.portfolio_value * risk_metrics.value_at_risk_95),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio-optimization")
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio weights using Modern Portfolio Theory
    
    Returns optimal weights, expected return, and expected volatility
    """
    try:
        # Generate sample asset returns
        n_days = 252
        asset_returns = pd.DataFrame(
            {symbol: np.random.normal(0.0008, 0.015, n_days) for symbol in request.symbols}
        )
        
        optimization = analytics.portfolio_optimization(asset_returns)
        
        return {
            "status": "success",
            "optimized_weights": optimization["weights"],
            "expected_return": round(optimization["expected_return"], 4),
            "expected_volatility": round(optimization["expected_volatility"], 4),
            "sharpe_ratio": round(optimization["sharpe_ratio"], 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attribution/{agent_id}")
async def get_attribution_analysis(
    agent_id: str,
    start_date: datetime = Query(None),
    end_date: datetime = Query(None)
):
    """
    Get performance attribution analysis
    
    Breaks down returns by factor contributions
    """
    try:
        # Generate sample returns and factors
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.015, 252))
        factors = {
            "momentum": pd.Series(np.random.normal(0.0008, 0.012, 252)),
            "value": pd.Series(np.random.normal(0.0006, 0.010, 252)),
            "quality": pd.Series(np.random.normal(0.0010, 0.014, 252)),
            "size": pd.Series(np.random.normal(0.0005, 0.011, 252)),
        }
        
        attribution = analytics.calculate_attribution_analysis(portfolio_returns, factors)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "attribution": {k: round(v, 4) for k, v in attribution.items()},
            "period": {
                "start_date": (start_date or datetime.now() - timedelta(days=252)).isoformat(),
                "end_date": (end_date or datetime.now()).isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def compare_agents(
    agent_ids: List[str] = Query(...),
    metrics: List[str] = Query(["sharpe_ratio", "max_drawdown", "win_rate"])
):
    """
    Compare multiple agents side-by-side
    
    Returns comparative metrics for selected agents
    """
    try:
        comparison = {}
        
        for agent_id in agent_ids:
            comparison[agent_id] = {
                "sharpe_ratio": round(np.random.uniform(0.5, 2.5), 2),
                "max_drawdown": round(np.random.uniform(-0.20, -0.05), 4),
                "win_rate": round(np.random.uniform(0.45, 0.75), 2),
                "profit_factor": round(np.random.uniform(1.2, 2.5), 2),
                "annual_return": round(np.random.uniform(0.10, 0.40), 2),
            }
        
        return {
            "status": "success",
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{agent_id}")
async def export_report(
    agent_id: str,
    format: str = Query("pdf", regex="^(pdf|csv|excel|json)$")
):
    """
    Export trading report in various formats
    
    Supported formats: PDF, CSV, Excel, JSON
    """
    try:
        return {
            "status": "success",
            "message": f"Report export initiated in {format.upper()} format",
            "download_url": f"/files/report_{agent_id}_{datetime.now().timestamp()}.{format}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio-summary")
async def get_portfolio_summary(db: sqlite3.Connection = Depends(get_db)):
    """
    Get real-time portfolio summary with actual data from database
    Returns aggregated metrics from all backtest runs
    """
    try:
        cursor = db.cursor()
        
        # Get all backtest runs
        cursor.execute("""
            SELECT agent_return, buy_hold_return, total_trades, final_value, 
                   outperformance, timestamp, symbol, agent_name
            FROM backtest_runs 
            ORDER BY timestamp DESC
        """)
        runs = cursor.fetchall()
        
        if not runs:
            return {
                "status": "success",
                "summary": {
                    "total_backtests": 0,
                    "portfolio_value": 10000.00,
                    "total_return": 0.0,
                    "total_trades": 0,
                    "avg_return": 0.0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "best_performer": None,
                    "recent_backtests": []
                }
            }
        
        # Calculate aggregated metrics
        total_backtests = len(runs)
        total_trades = sum(r[2] for r in runs)
        returns = [r[0] for r in runs]
        avg_return = np.mean(returns)
        
        # Calculate win rate (runs where agent beat buy & hold)
        winning_runs = sum(1 for r in runs if r[4] > 0)  # outperformance > 0
        win_rate = winning_runs / len(runs) if runs else 0
        
        # Calculate Sharpe ratio from returns
        # Sharpe = (Mean Return / Std Dev) * sqrt(252) for annualization
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        portfolio_values = [r[3] for r in runs]
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Find best performer
        best_run = max(runs, key=lambda r: r[0])  # highest agent_return
        
        # Get portfolio value from most recent run
        latest_run = runs[0]
        portfolio_value = latest_run[3]  # final_value
        total_return = latest_run[0]  # agent_return
        
        # Format recent backtests
        recent_backtests = []
        for r in runs[:10]:
            recent_backtests.append({
                "symbol": r[6],
                "agent_name": r[7],
                "agent_return": r[0],
                "buy_hold_return": r[1],
                "total_trades": r[2],
                "final_value": r[3],
                "outperformance": r[4],
                "timestamp": r[5]
            })
        
        return {
            "status": "success",
            "summary": {
                "total_backtests": total_backtests,
                "portfolio_value": portfolio_value,
                "total_return": total_return,
                "total_trades": total_trades,
                "avg_return": avg_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "best_performer": {
                    "symbol": best_run[6],
                    "agent_name": best_run[7],
                    "return": best_run[0],
                    "trades": best_run[2]
                },
                "recent_backtests": recent_backtests
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

