"""
Custom exceptions for NeuroQuant
"""


class NeuroQuantException(Exception):
    """Base exception for all NeuroQuant errors"""
    pass


class ConfigurationError(NeuroQuantException):
    """Raised when there's a configuration issue"""
    pass


class DataFetchError(NeuroQuantException):
    """Raised when market data fetching fails"""
    pass


class DataValidationError(NeuroQuantException):
    """Raised when data validation fails"""
    pass


class ModelNotFoundError(NeuroQuantException):
    """Raised when a model cannot be found"""
    pass


class ModelLoadError(NeuroQuantException):
    """Raised when model loading fails"""
    pass


class ModelSaveError(NeuroQuantException):
    """Raised when model saving fails"""
    pass


class TrainingError(NeuroQuantException):
    """Raised when training fails"""
    pass


class BacktestError(NeuroQuantException):
    """Raised when backtesting fails"""
    pass


class AgentNotFoundError(NeuroQuantException):
    """Raised when an agent cannot be found"""
    pass


class InvalidAgentTypeError(NeuroQuantException):
    """Raised when an invalid agent type is specified"""
    pass


class DatabaseError(NeuroQuantException):
    """Raised when database operations fail"""
    pass


class CacheError(NeuroQuantException):
    """Raised when cache operations fail"""
    pass


class SentimentAnalysisError(NeuroQuantException):
    """Raised when sentiment analysis fails"""
    pass
