"""
Financial Modeling Laboratory
DCF, Options Pricing, Scenario Analysis, Sensitivity Testing
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DCFModel:
    """
    Discounted Cash Flow valuation model
    """
    
    def __init__(
        self,
        free_cash_flows: List[float],
        discount_rate: float,
        terminal_growth_rate: float = 0.025,
        shares_outstanding: Optional[float] = None
    ):
        """
        Initialize DCF model
        
        Args:
            free_cash_flows: List of projected FCF for each year
            discount_rate: WACC or discount rate
            terminal_growth_rate: Perpetual growth rate
            shares_outstanding: Number of shares (for per-share value)
        """
        self.fcf = free_cash_flows
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth_rate
        self.shares = shares_outstanding
    
    def calculate_enterprise_value(self) -> Dict:
        """Calculate enterprise value"""
        n_years = len(self.fcf)
        
        # Present value of projected cash flows
        pv_fcf = []
        for i, fcf in enumerate(self.fcf, start=1):
            pv = fcf / ((1 + self.discount_rate) ** i)
            pv_fcf.append(float(pv))
        
        pv_projected = sum(pv_fcf)
        
        # Terminal value
        terminal_fcf = self.fcf[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.discount_rate - self.terminal_growth)
        pv_terminal = terminal_value / ((1 + self.discount_rate) ** n_years)
        
        enterprise_value = pv_projected + pv_terminal
        
        result = {
            'enterprise_value': float(enterprise_value),
            'pv_projected_fcf': float(pv_projected),
            'pv_terminal_value': float(pv_terminal),
            'terminal_value': float(terminal_value),
            'fcf_breakdown': [
                {
                    'year': i,
                    'fcf': float(fcf),
                    'pv': float(pv),
                    'discount_factor': float(1 / ((1 + self.discount_rate) ** i))
                }
                for i, (fcf, pv) in enumerate(zip(self.fcf, pv_fcf), start=1)
            ]
        }
        
        if self.shares:
            result['equity_value_per_share'] = float(enterprise_value / self.shares)
        
        return result
    
    def sensitivity_analysis(
        self,
        discount_rates: List[float],
        terminal_growth_rates: List[float]
    ) -> pd.DataFrame:
        """
        Two-way sensitivity analysis
        
        Args:
            discount_rates: List of discount rates to test
            terminal_growth_rates: List of terminal growth rates to test
        
        Returns:
            DataFrame with sensitivity matrix
        """
        results = []
        
        for dr in discount_rates:
            row = {'discount_rate': dr}
            for tgr in terminal_growth_rates:
                # Temporarily change rates
                orig_dr = self.discount_rate
                orig_tgr = self.terminal_growth
                
                self.discount_rate = dr
                self.terminal_growth = tgr
                
                valuation = self.calculate_enterprise_value()
                value = valuation['enterprise_value']
                if self.shares:
                    value = valuation['equity_value_per_share']
                
                row[f'tgr_{tgr:.1%}'] = float(value)
                
                # Restore original
                self.discount_rate = orig_dr
                self.terminal_growth = orig_tgr
            
            results.append(row)
        
        return pd.DataFrame(results)


class OptionsModels:
    """
    Options pricing models (Black-Scholes, Greeks)
    """
    
    @staticmethod
    def black_scholes(
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call'
    ) -> Dict:
        """
        Black-Scholes option pricing
        
        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration (years)
            risk_free_rate: Risk-free rate
            volatility: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dict with price and Greeks
        """
        # Handle edge case
        if time_to_expiry <= 0:
            if option_type == 'call':
                return {'price': max(0, spot - strike), 'delta': 1 if spot > strike else 0}
            else:
                return {'price': max(0, strike - spot), 'delta': -1 if spot < strike else 0}
        
        # Calculate d1 and d2
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # put
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        # Calculate Greeks
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  # Per 1% change
        
        if option_type == 'call':
            theta = (
                -spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry))
                - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            ) / 365  # Per day
            rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:
            theta = (
                -spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry))
                + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            ) / 365
            rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'rho': float(rho),
            'd1': float(d1),
            'd2': float(d2)
        }
    
    @staticmethod
    def implied_volatility(
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        # Initial guess
        volatility = 0.3
        
        for i in range(max_iterations):
            bs_result = OptionsModels.black_scholes(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            price_diff = bs_result['price'] - option_price
            vega = bs_result['vega'] * 100  # Convert back to per 100% change
            
            if abs(price_diff) < tolerance:
                return float(volatility)
            
            if vega == 0:
                return None
            
            # Newton-Raphson update
            volatility = volatility - price_diff / vega
            
            # Keep volatility positive
            if volatility <= 0:
                volatility = 0.01
        
        logger.warning("Implied volatility calculation did not converge")
        return None
    
    @staticmethod
    def option_strategy(
        strategies: List[Dict],
        spot_range: Tuple[float, float],
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Calculate P&L for option strategies
        
        Args:
            strategies: List of option legs with:
                - type: 'call' or 'put'
                - strike: strike price
                - premium: option premium paid/received
                - quantity: number of contracts (negative for short)
                - action: 'buy' or 'sell'
            spot_range: (min, max) spot prices to evaluate
            n_points: Number of points to calculate
        
        Returns:
            DataFrame with spot prices and P&L
        """
        spot_prices = np.linspace(spot_range[0], spot_range[1], n_points)
        
        results = []
        for spot in spot_prices:
            total_pnl = 0
            
            for leg in strategies:
                option_type = leg['type']
                strike = leg['strike']
                premium = leg['premium']
                quantity = leg['quantity']
                
                # Calculate intrinsic value at expiration
                if option_type == 'call':
                    intrinsic = max(0, spot - strike)
                else:  # put
                    intrinsic = max(0, strike - spot)
                
                # P&L = (intrinsic value - premium paid) * quantity
                leg_pnl = (intrinsic - premium) * quantity
                total_pnl += leg_pnl
            
            results.append({
                'spot_price': float(spot),
                'pnl': float(total_pnl),
                'breakeven': abs(total_pnl) < 0.01
            })
        
        return pd.DataFrame(results)


class ScenarioAnalysis:
    """
    Monte Carlo simulation and scenario analysis
    """
    
    @staticmethod
    def monte_carlo_price_simulation(
        initial_price: float,
        expected_return: float,
        volatility: float,
        time_horizon: int,
        n_simulations: int = 1000,
        time_step: float = 1/252  # Daily steps
    ) -> Dict:
        """
        Monte Carlo simulation for price paths
        
        Args:
            initial_price: Starting price
            expected_return: Annual expected return (drift)
            volatility: Annual volatility
            time_horizon: Number of time periods
            n_simulations: Number of simulation paths
            time_step: Time step size (1/252 for daily)
        
        Returns:
            Dict with simulation results and statistics
        """
        dt = time_step
        prices = np.zeros((n_simulations, time_horizon + 1))
        prices[:, 0] = initial_price
        
        # Generate random returns
        for t in range(1, time_horizon + 1):
            random_returns = np.random.normal(
                (expected_return - 0.5 * volatility ** 2) * dt,
                volatility * np.sqrt(dt),
                n_simulations
            )
            prices[:, t] = prices[:, t - 1] * np.exp(random_returns)
        
        final_prices = prices[:, -1]
        
        # Calculate statistics
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(final_prices, percentiles)
        
        result = {
            'simulations': n_simulations,
            'time_horizon': time_horizon,
            'initial_price': float(initial_price),
            'final_price_mean': float(np.mean(final_prices)),
            'final_price_median': float(np.median(final_prices)),
            'final_price_std': float(np.std(final_prices)),
            'percentiles': {
                f'p{p}': float(v) for p, v in zip(percentiles, percentile_values)
            },
            'probability_profit': float(np.mean(final_prices > initial_price)),
            'expected_return': float(np.mean(final_prices) / initial_price - 1),
            'max_price': float(np.max(final_prices)),
            'min_price': float(np.min(final_prices)),
            'sample_paths': [
                [float(p) for p in prices[i]]
                for i in range(min(10, n_simulations))  # Return first 10 paths
            ]
        }
        
        return result
    
    @staticmethod
    def scenario_analysis(
        base_case: Dict[str, float],
        variables: List[str],
        scenarios: Dict[str, Dict[str, float]],
        valuation_function
    ) -> pd.DataFrame:
        """
        Multi-variable scenario analysis
        
        Args:
            base_case: Base case assumptions
            variables: Variables to adjust
            scenarios: Dict of scenario_name -> variable adjustments
            valuation_function: Function that takes params and returns value
        
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        # Base case
        base_value = valuation_function(**base_case)
        results.append({
            'scenario': 'Base Case',
            'value': float(base_value),
            'change': 0.0,
            **base_case
        })
        
        # Other scenarios
        for scenario_name, adjustments in scenarios.items():
            params = base_case.copy()
            params.update(adjustments)
            
            value = valuation_function(**params)
            change = (value / base_value - 1) * 100
            
            results.append({
                'scenario': scenario_name,
                'value': float(value),
                'change': float(change),
                **params
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def stress_test(
        base_value: float,
        risk_factors: Dict[str, float],
        shocks: Dict[str, float],
        correlations: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Stress testing with correlated shocks
        
        Args:
            base_value: Base portfolio value
            risk_factors: Dict of factor -> sensitivity
            shocks: Dict of factor -> shock size
            correlations: Correlation matrix between factors
        
        Returns:
            Dict with stressed values
        """
        factors = list(risk_factors.keys())
        sensitivities = np.array([risk_factors[f] for f in factors])
        shock_values = np.array([shocks[f] for f in factors])
        
        # Apply uncorrelated shocks
        uncorrelated_change = np.sum(sensitivities * shock_values)
        uncorrelated_value = base_value * (1 + uncorrelated_change)
        
        result = {
            'base_value': float(base_value),
            'uncorrelated_stressed_value': float(uncorrelated_value),
            'uncorrelated_change': float(uncorrelated_change * 100),
            'factor_impacts': {
                factor: float(risk_factors[factor] * shocks[factor] * 100)
                for factor in factors
            }
        }
        
        # If correlations provided, adjust for correlation
        if correlations is not None:
            # Get correlation matrix for these factors
            corr_matrix = correlations.loc[factors, factors].values
            
            # Calculate correlated impact
            variance = sensitivities @ corr_matrix @ (sensitivities * shock_values ** 2)
            std_dev = np.sqrt(variance)
            
            result['correlated_stressed_value'] = float(base_value * (1 - std_dev))
            result['correlated_change'] = float(-std_dev * 100)
        
        return result


class SensitivityAnalysis:
    """
    One-way and multi-way sensitivity analysis
    """
    
    @staticmethod
    def one_way_sensitivity(
        base_params: Dict,
        variable: str,
        values: List[float],
        valuation_function
    ) -> pd.DataFrame:
        """
        One-way sensitivity analysis
        
        Args:
            base_params: Base case parameters
            variable: Variable to adjust
            values: List of values to test
            valuation_function: Function that returns valuation
        
        Returns:
            DataFrame with results
        """
        results = []
        
        for value in values:
            params = base_params.copy()
            params[variable] = value
            
            result_value = valuation_function(**params)
            
            results.append({
                variable: float(value),
                'output': float(result_value)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def tornado_chart_data(
        base_params: Dict,
        variables: List[str],
        percentage_changes: List[float],
        valuation_function
    ) -> pd.DataFrame:
        """
        Generate data for tornado chart (sensitivity ranking)
        
        Args:
            base_params: Base case parameters
            variables: Variables to test
            percentage_changes: List of % changes to apply (e.g., [-0.1, 0.1])
            valuation_function: Function that returns valuation
        
        Returns:
            DataFrame sorted by sensitivity magnitude
        """
        base_value = valuation_function(**base_params)
        
        results = []
        for var in variables:
            impacts = []
            
            for pct_change in percentage_changes:
                params = base_params.copy()
                params[var] = params[var] * (1 + pct_change)
                
                value = valuation_function(**params)
                impact = value - base_value
                
                impacts.append({
                    'variable': var,
                    'change_pct': pct_change * 100,
                    'output_value': float(value),
                    'impact': float(impact),
                    'impact_pct': float((value / base_value - 1) * 100)
                })
            
            # Calculate range of impact
            impact_range = max(i['impact'] for i in impacts) - min(i['impact'] for i in impacts)
            
            results.append({
                'variable': var,
                'base_value': float(base_value),
                'impact_range': float(impact_range),
                'sensitivities': impacts
            })
        
        # Sort by impact range (most sensitive first)
        df = pd.DataFrame(results)
        df = df.sort_values('impact_range', ascending=False)
        
        return df
