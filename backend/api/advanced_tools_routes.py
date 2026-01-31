"""
Advanced Financial Tools API Routes
Portfolio Optimization, Financial Modeling, Market Data Pipeline
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging

from backend.services.data_pipeline import get_pipeline
from backend.services.portfolio_optimizer import PortfolioOptimizer
from backend.services.financial_models import (
    DCFModel, OptionsModels, ScenarioAnalysis, SensitivityAnalysis
)

router = APIRouter(prefix="/api/advanced", tags=["advanced"])
logger = logging.getLogger(__name__)


# ============================================================================
# MARKET DATA PIPELINE
# ============================================================================

class MarketDataRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    validate: bool = Field(True, description="Validate data quality")


@router.post("/market-data")
async def get_market_data(request: MarketDataRequest):
    """Fetch market data with quality validation"""
    pipeline = get_pipeline()
    
    results = {}
    warnings_dict = {}
    
    for symbol in request.symbols:
        df, warnings = await pipeline.get_data(
            symbol,
            request.start_date,
            request.end_date,
            validate=request.validate
        )
        
        if df is not None:
            results[symbol] = {
                'data': df.reset_index().to_dict(orient='records'),
                'start': df.index[0].isoformat() if len(df) > 0 else None,
                'end': df.index[-1].isoformat() if len(df) > 0 else None,
                'rows': len(df)
            }
            
            if warnings:
                warnings_dict[symbol] = warnings
    
    return {
        'success': True,
        'data': results,
        'warnings': warnings_dict,
        'cache_stats': pipeline.get_cache_stats()
    }


@router.get("/market-data/quote/{symbol}")
async def get_realtime_quote(symbol: str):
    """Get real-time quote for a symbol"""
    pipeline = get_pipeline()
    quote = await pipeline.get_realtime_quote(symbol)
    
    if quote is None:
        raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
    
    return quote


@router.websocket("/ws/market-data/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streaming"""
    await websocket.accept()
    
    pipeline = get_pipeline()
    queue = pipeline.subscribe(symbol.upper())
    
    try:
        while True:
            # Get data from queue
            quote = await queue.get()
            
            # Send to client
            await websocket.send_json(quote)
            
    except WebSocketDisconnect:
        pipeline.unsubscribe(symbol.upper(), queue)
    except Exception as e:
        print(f"WebSocket error: {e}")
        pipeline.unsubscribe(symbol.upper(), queue)


@router.delete("/market-data/cache")
async def clear_cache():
    """Clear market data cache"""
    pipeline = get_pipeline()
    pipeline.clear_cache()
    return {'success': True, 'message': 'Cache cleared'}


# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., description="Asset symbols")
    start_date: str = Field(..., description="Historical data start")
    end_date: str = Field(..., description="Historical data end")
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate")
    optimization_type: str = Field(
        "max_sharpe",
        description="Type: max_sharpe, min_volatility, risk_parity, max_diversification"
    )
    constraints: Optional[Dict] = Field(None, description="Min/max weight constraints")


@router.post("/optimize-portfolio")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio allocation"""
    try:
        # Fetch data
        pipeline = get_pipeline()
        data_dict = await pipeline.get_multiple_symbols(
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        if not data_dict:
            raise HTTPException(status_code=400, detail="Failed to fetch data for symbols")
        
        # Calculate returns - create DataFrame directly from dict of Series
        returns_list = []
        for symbol, df in data_dict.items():
            returns = df['Close'].pct_change().dropna()
            returns.name = symbol
            returns_list.append(returns)
        
        # Concatenate all returns into a DataFrame
        returns_df = pd.concat(returns_list, axis=1)
        
        # Drop any rows with NaN values
        returns_df = returns_df.dropna()
        
        # Remove duplicate indices (timestamps) by keeping first occurrence
        if returns_df.index.duplicated().any():
            logger.warning(f"Removing {returns_df.index.duplicated().sum()} duplicate timestamps")
            returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
        
        if returns_df.empty or len(returns_df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for optimization after cleaning")
        
        # Create optimizer
        optimizer = PortfolioOptimizer(returns_df, request.risk_free_rate)
        
        # Run optimization
        constraints = request.constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        if request.optimization_type == 'max_sharpe':
            result = optimizer.max_sharpe_ratio(min_weight, max_weight)
        elif request.optimization_type == 'min_volatility':
            target_return = constraints.get('target_return')
            result = optimizer.min_volatility(target_return, min_weight, max_weight)
        elif request.optimization_type == 'risk_parity':
            result = optimizer.risk_parity()
        elif request.optimization_type == 'max_diversification':
            result = optimizer.max_diversification()
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization type")
        
        # Add statistics
        result['statistics'] = optimizer.get_statistics_summary()
        
        # Convert correlation matrix to nested dict format
        corr_matrix = optimizer.get_correlation_matrix()
        # Ensure index and columns are unique
        if corr_matrix.index.duplicated().any():
            corr_matrix = corr_matrix[~corr_matrix.index.duplicated(keep='first')]
        if corr_matrix.columns.duplicated().any():
            corr_matrix = corr_matrix.loc[:, ~corr_matrix.columns.duplicated(keep='first')]
        
        # Convert using index and column names (which are symbol names)
        symbols = corr_matrix.index.tolist()
        result['correlation_matrix'] = {
            symbol: {col: float(corr_matrix.loc[symbol, col]) for col in symbols}
            for symbol in symbols
        }
        
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"Portfolio optimization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/efficient-frontier")
async def calculate_efficient_frontier(request: PortfolioOptimizationRequest):
    """Calculate efficient frontier"""
    try:
        # Fetch data
        pipeline = get_pipeline()
        data_dict = await pipeline.get_multiple_symbols(
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        # Calculate returns - create DataFrame directly from dict of Series
        returns_list = []
        for symbol, df in data_dict.items():
            returns = df['Close'].pct_change().dropna()
            returns.name = symbol
            returns_list.append(returns)
        
        # Concatenate all returns into a DataFrame
        returns_df = pd.concat(returns_list, axis=1)
        returns_df = returns_df.dropna()
        
        # Remove duplicate indices (timestamps) by keeping first occurrence
        if returns_df.index.duplicated().any():
            logger.warning(f"Removing {returns_df.index.duplicated().sum()} duplicate timestamps from efficient frontier data")
            returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
        
        if returns_df.empty or len(returns_df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for optimization after cleaning")
        
        # Create optimizer
        optimizer = PortfolioOptimizer(returns_df, request.risk_free_rate)
        
        # Calculate frontier
        n_portfolios = request.constraints.get('n_portfolios', 100) if request.constraints else 100
        frontier_df = optimizer.efficient_frontier(n_portfolios)
        
        # Monte Carlo for comparison
        mc_df = optimizer.monte_carlo_simulation(n_simulations=5000)
        
        return {
            'efficient_frontier': frontier_df.to_dict(orient='records'),
            'monte_carlo': mc_df.to_dict(orient='records'),
            'statistics': optimizer.get_statistics_summary()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BlackLittermanRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    views: Dict[str, float] = Field(..., description="Asset -> expected return")
    view_confidences: Dict[str, float] = Field(..., description="Asset -> confidence (0-1)")
    market_cap_weights: Optional[Dict[str, float]] = None
    risk_free_rate: float = 0.02


@router.post("/black-litterman")
async def black_litterman_optimization(request: BlackLittermanRequest):
    """Black-Litterman optimization with investor views"""
    try:
        # Fetch data
        pipeline = get_pipeline()
        data_dict = await pipeline.get_multiple_symbols(
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        # Calculate returns
        returns_data = {symbol: df['Close'].pct_change().dropna() for symbol, df in data_dict.items()}
        returns_df = pd.DataFrame(returns_data)
        
        # Create optimizer
        optimizer = PortfolioOptimizer(returns_df, request.risk_free_rate)
        
        # Run Black-Litterman
        result = optimizer.black_litterman(
            request.views,
            request.view_confidences,
            request.market_cap_weights
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FINANCIAL MODELING
# ============================================================================

class DCFRequest(BaseModel):
    free_cash_flows: List[float] = Field(..., description="Projected FCF for each year")
    discount_rate: float = Field(..., description="WACC/discount rate")
    terminal_growth_rate: float = Field(0.025, description="Terminal growth rate")
    shares_outstanding: Optional[float] = None


@router.post("/dcf-valuation")
async def dcf_valuation(request: DCFRequest):
    """DCF valuation model"""
    try:
        model = DCFModel(
            request.free_cash_flows,
            request.discount_rate,
            request.terminal_growth_rate,
            request.shares_outstanding
        )
        
        valuation = model.calculate_enterprise_value()
        
        # Sensitivity analysis
        discount_rates = [request.discount_rate - 0.02, request.discount_rate, request.discount_rate + 0.02]
        terminal_rates = [request.terminal_growth_rate - 0.01, request.terminal_growth_rate, 
                         request.terminal_growth_rate + 0.01]
        
        sensitivity = model.sensitivity_analysis(discount_rates, terminal_rates)
        
        valuation['sensitivity_analysis'] = sensitivity.to_dict(orient='records')
        
        return valuation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class OptionPricingRequest(BaseModel):
    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiration (years)")
    risk_free_rate: float = Field(..., description="Risk-free rate")
    volatility: float = Field(..., description="Implied volatility")
    option_type: str = Field("call", description="'call' or 'put'")


@router.post("/option-pricing")
async def option_pricing(request: OptionPricingRequest):
    """Black-Scholes option pricing with Greeks"""
    try:
        result = OptionsModels.black_scholes(
            request.spot,
            request.strike,
            request.time_to_expiry,
            request.risk_free_rate,
            request.volatility,
            request.option_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImpliedVolatilityRequest(BaseModel):
    option_price: float
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    option_type: str = "call"


@router.post("/implied-volatility")
async def calculate_implied_volatility(request: ImpliedVolatilityRequest):
    """Calculate implied volatility from option price"""
    try:
        iv = OptionsModels.implied_volatility(
            request.option_price,
            request.spot,
            request.strike,
            request.time_to_expiry,
            request.risk_free_rate,
            request.option_type
        )
        
        if iv is None:
            raise HTTPException(status_code=400, detail="Could not calculate implied volatility")
        
        return {'implied_volatility': iv}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class OptionStrategyRequest(BaseModel):
    strategies: List[Dict] = Field(..., description="List of option legs")
    spot_range: tuple[float, float] = Field(..., description="(min, max) spot prices")
    n_points: int = 100


@router.post("/option-strategy")
async def analyze_option_strategy(request: OptionStrategyRequest):
    """Analyze option strategy P&L"""
    try:
        df = OptionsModels.option_strategy(
            request.strategies,
            request.spot_range,
            request.n_points
        )
        
        # Find breakeven points
        breakeven_points = df[df['breakeven']]['spot_price'].tolist()
        
        # Calculate max profit/loss
        max_profit = float(df['pnl'].max())
        max_loss = float(df['pnl'].min())
        
        return {
            'pnl_data': df.to_dict(orient='records'),
            'breakeven_points': breakeven_points,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward_ratio': abs(max_profit / max_loss) if max_loss != 0 else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MonteCarloRequest(BaseModel):
    initial_price: float
    expected_return: float
    volatility: float
    time_horizon: int
    n_simulations: int = 1000


@router.post("/monte-carlo")
async def monte_carlo_simulation(request: MonteCarloRequest):
    """Monte Carlo price simulation"""
    try:
        result = ScenarioAnalysis.monte_carlo_price_simulation(
            request.initial_price,
            request.expected_return,
            request.volatility,
            request.time_horizon,
            request.n_simulations
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScenarioAnalysisRequest(BaseModel):
    base_case: Dict[str, float]
    variables: List[str]
    scenarios: Dict[str, Dict[str, float]]
    model_type: str = Field("dcf", description="Type of model for valuation")


@router.post("/scenario-analysis")
async def scenario_analysis(request: ScenarioAnalysisRequest):
    """Multi-variable scenario analysis"""
    try:
        # Define valuation function based on model type
        if request.model_type == "dcf":
            def valuation_func(**params):
                model = DCFModel(
                    params['free_cash_flows'],
                    params['discount_rate'],
                    params.get('terminal_growth_rate', 0.025),
                    params.get('shares_outstanding')
                )
                result = model.calculate_enterprise_value()
                return result['enterprise_value']
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        df = ScenarioAnalysis.scenario_analysis(
            request.base_case,
            request.variables,
            request.scenarios,
            valuation_func
        )
        
        return {
            'scenarios': df.to_dict(orient='records'),
            'base_case_value': float(df[df['scenario'] == 'Base Case']['value'].iloc[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SensitivityRequest(BaseModel):
    base_params: Dict[str, Any]
    variable: str
    values: List[float]
    model_type: str = "dcf"


@router.post("/sensitivity-analysis")
async def sensitivity_analysis(request: SensitivityRequest):
    """One-way sensitivity analysis"""
    try:
        # Define valuation function
        if request.model_type == "dcf":
            def valuation_func(**params):
                model = DCFModel(
                    params['free_cash_flows'],
                    params['discount_rate'],
                    params.get('terminal_growth_rate', 0.025),
                    params.get('shares_outstanding')
                )
                result = model.calculate_enterprise_value()
                return result['enterprise_value']
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        df = SensitivityAnalysis.one_way_sensitivity(
            request.base_params,
            request.variable,
            request.values,
            valuation_func
        )
        
        return {'results': df.to_dict(orient='records')}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    pipeline = get_pipeline()
    return {
        'status': 'healthy',
        'cache_size': pipeline.cache.size(),
        'timestamp': datetime.now().isoformat()
    }
