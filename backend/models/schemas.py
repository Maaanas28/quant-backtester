"""
Pydantic models for data validation and serialization
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, field_validator, model_validator
from enum import Enum


class AgentType(str, Enum):
    """Supported agent types"""
    DQN = "DQN"
    PPO = "PPO"
    A3C = "A3C"
    INDICATOR_BASED = "IndicatorBased"
    RANDOM = "Random"


class DataSource(str, Enum):
    """Supported data sources"""
    YFINANCE = "yfinance"
    CUSTOM = "custom"


class TradeAction(str, Enum):
    """Trade actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# Request Models
class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Trading symbol")
    train_split: float = Field(0.7, ge=0.1, le=0.9, description="Train/test split ratio")
    agent_id: Optional[int] = Field(None, ge=1, description="Agent ID to use")
    data_source: DataSource = Field(DataSource.YFINANCE, description="Data source")
    custom_dataset_id: Optional[int] = Field(None, ge=1, description="Custom dataset ID")
    start_date: Optional[str] = Field(None, description="Start date for data")
    end_date: Optional[str] = Field(None, description="End date for data")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()
    
    @model_validator(mode='after')
    def validate_data_source(self):
        if self.data_source == DataSource.CUSTOM and not self.custom_dataset_id:
            raise ValueError("custom_dataset_id required when data_source is 'custom'")
        return self


class AgentCreateRequest(BaseModel):
    """Request model for creating an agent"""
    name: str = Field(..., min_length=3, max_length=100, description="Agent name")
    type: AgentType = Field(..., description="Agent type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent parameters")
    description: Optional[str] = Field(None, max_length=500, description="Agent description")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class DatasetUploadRequest(BaseModel):
    """Request model for dataset upload"""
    name: str = Field(..., min_length=3, max_length=100, description="Dataset name")
    description: Optional[str] = Field(None, max_length=500, description="Dataset description")


# Response Models
class TradeInfo(BaseModel):
    """Trade information"""
    date: str
    action: TradeAction
    price: float = Field(..., ge=0)
    shares: float = Field(..., ge=0)
    pnl: Optional[float] = None
    portfolio_before: float = Field(..., ge=0)
    holding_before: float = Field(..., ge=0)
    commission: Optional[float] = Field(None, ge=0)
    slippage: Optional[float] = None
    fill_price: Optional[float] = Field(None, ge=0)


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    sharpe_ratio: float
    max_drawdown: float = Field(..., ge=0, le=100)
    win_rate: float = Field(..., ge=0, le=100)
    profit_factor: float = Field(..., ge=0)
    sortino_ratio: float
    calmar_ratio: float
    total_return: float
    annual_return: float
    volatility: float


class ShadowMetrics(BaseModel):
    """Shadow portfolio metrics"""
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    equity_curve: List[float]
    num_trades: int
    total_commission: float


class ComparisonMetrics(BaseModel):
    """Comparison between main and shadow portfolios"""
    equity_difference: float
    return_difference_pct: float
    main_outperformed: bool


class BacktestResponse(BaseModel):
    """Response model for backtesting"""
    symbol: str
    test_period: str
    agent_return: float
    buy_hold_return: float
    outperformance: float
    total_trades: int = Field(..., ge=0)
    final_value: float = Field(..., ge=0)
    trades: List[TradeInfo] = []
    portfolio_history: List[float] = []
    portfolio_dates: List[str] = []
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    value_at_risk_95: Optional[float] = None
    value_at_risk_99: Optional[float] = None
    conditional_var_95: Optional[float] = None
    conditional_var_99: Optional[float] = None
    metrics: Optional[PerformanceMetrics] = None
    agent_name: Optional[str] = None
    # Execution details
    total_commission: Optional[float] = Field(None, ge=0)
    total_slippage: Optional[float] = None
    signals_generated: Optional[int] = Field(None, ge=0)
    orders_placed: Optional[int] = Field(None, ge=0)
    fills_received: Optional[int] = Field(None, ge=0)
    bars_processed: Optional[int] = Field(None, ge=0)
    shadow: Optional[ShadowMetrics] = None
    comparison: Optional[ComparisonMetrics] = None


class AgentResponse(BaseModel):
    """Response model for agent"""
    id: int
    name: str
    type: AgentType
    parameters: Dict[str, Any]
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BacktestRunResponse(BaseModel):
    """Response model for backtest run"""
    id: int
    timestamp: datetime
    symbol: str
    agent_id: Optional[int]
    agent_name: Optional[str]
    test_period: str
    agent_return: float
    buy_hold_return: float
    outperformance: float
    total_trades: int
    final_value: float
    trades: List[Dict[str, Any]]
    portfolio_history: List[float]
    portfolio_dates: List[str]
    metrics: Optional[Dict[str, float]] = None


class CustomDatasetResponse(BaseModel):
    """Response model for custom dataset"""
    id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    row_count: Optional[int] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    database_status: str
    cache_status: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SymbolListResponse(BaseModel):
    """Symbol list response"""
    yfinance_symbols: List[str]
    custom_datasets: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    path: str
    created_at: datetime
    size_bytes: int
    hyperparameters: Dict[str, Any]


class TrainingProgress(BaseModel):
    """Training progress information"""
    current_step: int
    total_steps: int
    episode: int
    reward: float
    loss: Optional[float] = None
    epsilon: Optional[float] = None
    progress_percentage: float
