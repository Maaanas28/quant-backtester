"""
Backtesting API Routes
Institutional-grade backtesting endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from engine.backtester import Backtester, WalkForwardAnalysis
from engine.strategy import (
    MovingAverageCrossStrategy,
    RSIStrategy,
    MomentumStrategy,
    BuyAndHoldStrategy
)

router = APIRouter()


class BacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    initial_capital: float = 100000.0
    strategy: str    # "ma_cross", "rsi", "momentum", "buy_hold"
    strategy_params: Optional[Dict] = None


class BacktestResponse(BaseModel):
    success: bool
    results: Optional[Dict] = None
    error: Optional[str] = None


class WalkForwardRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    train_period_days: int = 180
    test_period_days: int = 60
    strategy: str
    strategy_params: Optional[Dict] = None


@router.post("/backtest/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run institutional-grade backtest
    
    Supports:
    - MA Crossover
    - RSI Mean Reversion
    - Momentum
    - Buy & Hold (benchmark)
    """
    try:
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # Create strategy
        strategy = None
        params = request.strategy_params or {}
        
        if request.strategy == "ma_cross":
            strategy = MovingAverageCrossStrategy(
                symbols=request.symbols,
                short_window=params.get("short_window", 20),
                long_window=params.get("long_window", 50)
            )
        
        elif request.strategy == "rsi":
            strategy = RSIStrategy(
                symbols=request.symbols,
                period=params.get("period", 14),
                oversold=params.get("oversold", 30),
                overbought=params.get("overbought", 70)
            )
        
        elif request.strategy == "momentum":
            strategy = MomentumStrategy(
                symbols=request.symbols,
                lookback=params.get("lookback", 20)
            )
        
        elif request.strategy == "buy_hold":
            strategy = BuyAndHoldStrategy(symbols=request.symbols)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        # Run backtest
        backtester = Backtester(
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=request.initial_capital,
            strategy=strategy
        )
        
        results = backtester.run()
        
        # Convert timestamps to strings for JSON
        results['timestamps'] = [ts.isoformat() for ts in results['timestamps']]
        
        return BacktestResponse(success=True, results=results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return BacktestResponse(success=False, error=str(e))


@router.post("/backtest/walk-forward")
async def run_walk_forward(request: WalkForwardRequest):
    """
    Run walk-forward analysis
    
    Optimizes on training data, validates on test data
    Rolling window approach
    """
    try:
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # Get strategy class
        strategy_class = None
        params = request.strategy_params or {}
        
        if request.strategy == "ma_cross":
            strategy_class = MovingAverageCrossStrategy
            if 'symbols' in params:
                del params['symbols']
        elif request.strategy == "rsi":
            strategy_class = RSIStrategy
            if 'symbols' in params:
                del params['symbols']
        elif request.strategy == "momentum":
            strategy_class = MomentumStrategy
            if 'symbols' in params:
                del params['symbols']
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        # Run walk-forward
        wfa = WalkForwardAnalysis(
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            train_period_days=request.train_period_days,
            test_period_days=request.test_period_days
        )
        
        results = wfa.run(strategy_class, params)
        
        # Convert timestamps
        for r in results:
            r['timestamps'] = [ts.isoformat() for ts in r['timestamps']]
            r['train_start'] = r['train_start'].isoformat()
            r['train_end'] = r['train_end'].isoformat()
            r['test_start'] = r['test_start'].isoformat()
            r['test_end'] = r['test_end'].isoformat()
        
        return {"success": True, "results": results}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/backtest/strategies")
async def get_available_strategies():
    """List available strategies"""
    return {
        "strategies": [
            {
                "id": "ma_cross",
                "name": "Moving Average Crossover",
                "description": "Long when fast MA crosses above slow MA",
                "parameters": {
                    "short_window": {"type": "int", "default": 20, "description": "Fast MA period"},
                    "long_window": {"type": "int", "default": 50, "description": "Slow MA period"}
                }
            },
            {
                "id": "rsi",
                "name": "RSI Mean Reversion",
                "description": "Long when oversold, short when overbought",
                "parameters": {
                    "period": {"type": "int", "default": 14, "description": "RSI period"},
                    "oversold": {"type": "int", "default": 30, "description": "Oversold threshold"},
                    "overbought": {"type": "int", "default": 70, "description": "Overbought threshold"}
                }
            },
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Follow price momentum",
                "parameters": {
                    "lookback": {"type": "int", "default": 20, "description": "Lookback period"}
                }
            },
            {
                "id": "buy_hold",
                "name": "Buy & Hold",
                "description": "Buy and hold benchmark",
                "parameters": {}
            }
        ]
    }


@router.get("/backtest/example")
async def run_example_backtest():
    """Run a quick example backtest"""
    try:
        # 1 year backtest on SPY
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        strategy = MovingAverageCrossStrategy(
            symbols=['SPY'],
            short_window=20,
            long_window=50
        )
        
        backtester = Backtester(
            symbols=['SPY'],
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            strategy=strategy
        )
        
        results = backtester.run()
        
        # Convert timestamps
        results['timestamps'] = [ts.isoformat() for ts in results['timestamps']]
        
        return {
            "success": True,
            "message": "Example backtest: MA Crossover on SPY (1 year)",
            "results": results
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
