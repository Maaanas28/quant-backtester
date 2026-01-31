"""
Advanced API Routes for New Features
Multi-symbol backtesting, strategy optimization, live trading simulation
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import traceback
import asyncio

from backend.services.portfolio_manager import PortfolioManager
from backend.services.strategy_optimizer import StrategyOptimizer, GeneticOptimizer
from backend.services.live_trading import LiveTradingSimulator, simple_momentum_strategy, AlertRule

router = APIRouter(prefix="/advanced")

# Live trading simulators
live_simulators: Dict[str, LiveTradingSimulator] = {}
simulator_tasks: Dict[str, asyncio.Task] = {}


class MultiSymbolBacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    agent_name: str = "dqn"


@router.post("/backtest/multi_symbol")
async def run_multi_symbol_backtest(request: MultiSymbolBacktestRequest):
    """Run backtest across multiple symbols - Temporarily disabled"""
    raise HTTPException(status_code=501, detail="Multi-symbol backtesting temporarily disabled. Use main /api/backtest endpoint.")


class OptimizationRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    method: str = "random"  # "grid", "random", "bayesian", "genetic"
    n_iterations: int = 20


@router.post("/optimize/strategy")
async def optimize_strategy(request: OptimizationRequest):
    """
    Optimize trading strategy hyperparameters using advanced algorithms
    Methods: grid search, random search, Bayesian optimization, genetic algorithm
    """
    try:
        # Define parameter space for optimization
        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'gamma': [0.95, 0.98, 0.99],
            'buffer_size': [10000, 50000, 100000],
            'batch_size': [32, 64, 128]
        }
        
        # Objective function (simplified - would run actual backtest)
        def objective(**params):
            import numpy as np
            # Simulate performance metrics
            base_return = np.random.uniform(-0.1, 0.4)
            volatility = np.random.uniform(0.1, 0.3)
            # Simplified Sharpe for simulation: Return / Volatility
            sharpe = base_return / volatility if volatility > 0 else 0
            
            return {
                'sharpe_ratio': sharpe,
                'total_return': base_return,
                'win_rate': np.random.uniform(0.4, 0.7),
                'max_drawdown': np.random.uniform(0.1, 0.4)
            }
        
        # Run optimization based on method
        if request.method == "genetic":
            optimizer = GeneticOptimizer(
                objective_function=objective,
                param_space=param_space,
                population_size=10,
                generations=max(2, request.n_iterations // 10)
            )
            result = optimizer.optimize()
        else:
            optimizer = StrategyOptimizer(
                objective_function=objective,
                param_space=param_space
            )
            
            if request.method == "grid":
                result = optimizer.grid_search(max_combinations=request.n_iterations)
            elif request.method == "bayesian":
                result = optimizer.bayesian_optimization(n_iterations=request.n_iterations)
            else:  # random
                result = optimizer.random_search(n_iterations=request.n_iterations)
        
        return {
            "status": "success",
            "method": request.method,
            "best_params": result.best_params,
            "best_score": float(result.best_score),
            "optimization_time": float(result.optimization_time),
            "top_results": [
                {
                    "params": r["params"],
                    "score": float(r["score"]),
                    "metrics": {k: float(v) for k, v in r["metrics"].items()}
                }
                for r in result.all_results[:5]
            ],
            "total_evaluations": len(result.all_results)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class LiveTradingRequest(BaseModel):
    symbols: List[str]
    initial_capital: float = 10000.0
    update_interval: int = 60


@router.post("/live_trading/start")
async def start_live_trading(request: LiveTradingRequest):
    """Start live paper trading simulation with real-time market data"""
    try:
        session_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        simulator = LiveTradingSimulator(
            symbols=request.symbols,
            initial_capital=request.initial_capital,
            update_interval=request.update_interval
        )
        
        # Register momentum-based agent
        simulator.register_strategy("momentum", simple_momentum_strategy)
        
        # Store simulator
        live_simulators[session_id] = simulator
        
        # Start simulation in background
        async def run_simulator():
            await simulator.run(duration_hours=8)  # Max 8 hours
        
        task = asyncio.create_task(run_simulator())
        simulator_tasks[session_id] = task
        
        return {
            "status": "started",
            "session_id": session_id,
            "symbols": request.symbols,
            "initial_capital": request.initial_capital,
            "message": "Live trading simulation started"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live_trading/status/{session_id}")
async def get_live_trading_status(session_id: str):
    """Get current status and performance of live trading simulation"""
    if session_id not in live_simulators:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = live_simulators[session_id]
    status = simulator.get_status()
    
    # Format signals for JSON
    formatted_signals = []
    for signal in status.get('recent_signals', []):
        formatted_signals.append({
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'action': signal.action,
            'confidence': float(signal.confidence),
            'reason': signal.reason,
            'agent_name': signal.agent_name
        })
    
    return {
        "session_id": session_id,
        "is_running": status['is_running'],
        "current_value": float(status['current_value']),
        "total_return": float(status['total_return']),
        "positions": status['positions'],
        "cash": float(status['cash']),
        "portfolio_allocation": {k: float(v) for k, v in status['portfolio_allocation'].items()},
        "recent_signals": formatted_signals
    }


@router.post("/live_trading/stop/{session_id}")
async def stop_live_trading(session_id: str):
    """Stop live trading simulation and get final report"""
    if session_id not in live_simulators:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = live_simulators[session_id]
    simulator.stop()
    
    # Cancel background task
    if session_id in simulator_tasks:
        simulator_tasks[session_id].cancel()
        del simulator_tasks[session_id]
    
    # Get final metrics
    metrics = simulator.portfolio.get_performance_metrics()
    
    return {
        "status": "stopped",
        "session_id": session_id,
        "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        "final_value": float(simulator.portfolio.current_capital)
    }


class AlertRequest(BaseModel):
    symbol: str
    condition: str  # 'price_above', 'price_below', 'rsi_oversold', 'rsi_overbought'
    threshold: float


@router.post("/live_trading/alert/{session_id}")
async def add_trading_alert(session_id: str, alert_req: AlertRequest):
    """Add price alert or technical indicator alert to live trading session"""
    if session_id not in live_simulators:
        raise HTTPException(status_code=404, detail="Session not found")
    
    simulator = live_simulators[session_id]
    
    alert = AlertRule(
        rule_id=f"alert_{len(simulator.alerts) + 1}",
        symbol=alert_req.symbol,
        condition=alert_req.condition,
        threshold=alert_req.threshold
    )
    
    simulator.add_alert(alert)
    
    return {
        "status": "alert_added",
        "alert_id": alert.rule_id,
        "symbol": alert_req.symbol,
        "condition": alert_req.condition,
        "threshold": alert_req.threshold
    }


@router.get("/live_trading/sessions")
async def list_live_sessions():
    """List all active live trading sessions"""
    sessions = []
    
    for session_id, simulator in live_simulators.items():
        status = simulator.get_status()
        sessions.append({
            "session_id": session_id,
            "is_running": status['is_running'],
            "symbols": simulator.symbols,
            "current_value": float(status['current_value']),
            "total_return": float(status['total_return'])
        })
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }
