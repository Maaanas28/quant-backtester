"""
Portfolio Optimization Engine
Modern Portfolio Theory, Efficient Frontier, Risk Parity
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple strategies
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize optimizer
        
        Args:
            returns: DataFrame with asset returns (columns = assets, rows = periods)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        
        # Calculate statistics
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.corr_matrix = returns.corr()
    
    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio statistics"""
        portfolio_return = float(np.sum(self.mean_returns.values * weights))
        portfolio_std = float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights))))
        sharpe_ratio = float((portfolio_return - self.risk_free_rate) / portfolio_std) if portfolio_std > 0 else 0
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective function: negative Sharpe ratio"""
        _, _, sharpe = self._portfolio_stats(weights)
        return -sharpe
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Objective function: portfolio volatility"""
        _, std, _ = self._portfolio_stats(weights)
        return std
    
    def max_sharpe_ratio(
        self, 
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio
        
        Args:
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        
        Returns:
            Dict with weights and portfolio stats
        """
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds for each weight
        bounds = Bounds(
            lb=[min_weight] * self.n_assets,
            ub=[max_weight] * self.n_assets
        )
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self._negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)
        
        return {
            'type': 'max_sharpe',
            'weights': {asset: float(w) for asset, w in zip(self.assets, weights)},
            'expected_return': float(ret),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'success': bool(result.success)
        }
    
    def min_volatility(
        self,
        target_return: Optional[float] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict:
        """
        Find minimum volatility portfolio
        
        Args:
            target_return: Target annual return (optional)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        
        Returns:
            Dict with weights and portfolio stats
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns.values * x) - target_return
            })
        
        bounds = Bounds(
            lb=[min_weight] * self.n_assets,
            ub=[max_weight] * self.n_assets
        )
        
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            self._portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)
        
        return {
            'type': 'min_volatility',
            'weights': {asset: float(w) for asset, w in zip(self.assets, weights)},
            'expected_return': float(ret),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'target_return': target_return,
            'success': bool(result.success)
        }
    
    def efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            n_portfolios: Number of portfolios to calculate
        
        Returns:
            DataFrame with portfolio weights and stats
        """
        # Get min and max possible returns
        min_ret = float(self.mean_returns.min())
        max_ret = float(self.mean_returns.max())
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        results = []
        for target_ret in target_returns:
            try:
                portfolio = self.min_volatility(target_return=float(target_ret))
                if portfolio['success']:
                    results.append({
                        'return': portfolio['expected_return'],
                        'volatility': portfolio['volatility'],
                        'sharpe_ratio': portfolio['sharpe_ratio'],
                        **portfolio['weights']
                    })
            except Exception as e:
                logger.debug(f"Failed to optimize for return {target_ret}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def risk_parity(self) -> Dict:
        """
        Risk parity portfolio (equal risk contribution)
        
        Returns:
            Dict with weights and portfolio stats
        """
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
            marginal_contrib = np.dot(self.cov_matrix.values, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            risk_contrib = risk_contribution(weights)
            target_risk = 1.0 / self.n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = Bounds(lb=[0.0] * self.n_assets, ub=[1.0] * self.n_assets)
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)
        
        return {
            'type': 'risk_parity',
            'weights': {asset: float(w) for asset, w in zip(self.assets, weights)},
            'expected_return': float(ret),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'risk_contributions': {
                asset: float(rc) 
                for asset, rc in zip(self.assets, risk_contribution(weights))
            },
            'success': bool(result.success)
        }
    
    def max_diversification(self) -> Dict:
        """
        Maximum diversification portfolio
        
        Returns:
            Dict with weights and portfolio stats
        """
        def diversification_ratio(weights):
            weighted_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix.values)))
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
            return -weighted_vol / portfolio_vol  # Negative for minimization
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = Bounds(lb=[0.0] * self.n_assets, ub=[1.0] * self.n_assets)
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)
        
        return {
            'type': 'max_diversification',
            'weights': {asset: float(w) for asset, w in zip(self.assets, weights)},
            'expected_return': float(ret),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'diversification_ratio': float(-diversification_ratio(weights)),
            'success': bool(result.success)
        }
    
    def monte_carlo_simulation(
        self, 
        n_simulations: int = 10000,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """
        Monte Carlo simulation of random portfolios
        
        Args:
            n_simulations: Number of random portfolios to generate
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        
        Returns:
            DataFrame with simulated portfolios
        """
        results = []
        
        for _ in range(n_simulations):
            # Generate random weights
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)  # Normalize
            
            # Apply bounds
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / np.sum(weights)  # Renormalize
            
            ret, vol, sharpe = self._portfolio_stats(weights)
            
            results.append({
                'return': float(ret),
                'volatility': float(vol),
                'sharpe_ratio': float(sharpe),
                **{asset: float(w) for asset, w in zip(self.assets, weights)}
            })
        
        return pd.DataFrame(results)
    
    def black_litterman(
        self,
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        market_cap_weights: Optional[Dict[str, float]] = None,
        tau: float = 0.025
    ) -> Dict:
        """
        Black-Litterman model with investor views
        
        Args:
            views: Dict of asset -> expected return view
            view_confidences: Dict of asset -> confidence (0-1)
            market_cap_weights: Prior market cap weights
            tau: Scaling factor for uncertainty
        
        Returns:
            Dict with adjusted weights and stats
        """
        # Use equal weights if no market cap provided
        if market_cap_weights is None:
            market_cap_weights = {asset: 1.0 / self.n_assets for asset in self.assets}
        
        pi = np.array([market_cap_weights.get(asset, 0) for asset in self.assets])
        
        # Prepare views
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in self.assets:
                idx = self.assets.index(asset)
                P[i, idx] = 1.0
                Q[i] = view_return
        
        # Omega: diagonal matrix of view uncertainties
        omega_diag = []
        for asset in views.keys():
            if asset in self.assets:
                idx = self.assets.index(asset)
                confidence = view_confidences.get(asset, 0.5)
                omega_diag.append(float(tau * self.cov_matrix.values[idx, idx] / confidence))
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau_sigma = tau * self.cov_matrix.values
        
        # Posterior expected returns
        M1 = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
        M2 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
        posterior_returns = M1 @ M2
        
        # Optimize with posterior returns
        self.mean_returns = pd.Series(posterior_returns, index=self.assets)
        portfolio = self.max_sharpe_ratio()
        
        portfolio['type'] = 'black_litterman'
        portfolio['views'] = views
        portfolio['view_confidences'] = view_confidences
        
        return portfolio
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix"""
        return self.corr_matrix
    
    def get_statistics_summary(self) -> Dict:
        """Get summary statistics for all assets"""
        stats = {
            'annualized_returns': self.mean_returns.to_dict(),
            'annualized_volatility': {
                asset: float(np.sqrt(self.cov_matrix.values[i, i]))
                for i, asset in enumerate(self.assets)
            },
            'sharpe_ratios': {
                asset: float((float(self.mean_returns.iloc[i]) - self.risk_free_rate) / 
                           float(np.sqrt(self.cov_matrix.values[i, i])))
                for i, asset in enumerate(self.assets)
            }
        }
        return stats
