"""
Strategy Optimization Engine
Hyperparameter tuning and strategy optimization using genetic algorithms and grid search
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm


@dataclass
class OptimizationResult:
    """Results from strategy optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_time: float


class StrategyOptimizer:
    """Optimize trading strategy hyperparameters"""
    
    def __init__(self, 
                 objective_function: Callable,
                 param_space: Dict[str, List],
                 scoring_metric: str = 'sharpe_ratio'):
        """
        Args:
            objective_function: Function that takes params and returns performance metrics
            param_space: Dictionary of parameter names and their possible values
            scoring_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate', etc.)
        """
        self.objective_function = objective_function
        self.param_space = param_space
        self.scoring_metric = scoring_metric
        
    def grid_search(self, max_combinations: int = 100) -> OptimizationResult:
        """
        Exhaustive grid search over parameter space
        
        Args:
            max_combinations: Maximum number of combinations to test
        """
        import time
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        results = []
        best_score = float('-inf')
        best_params = None
        
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            
            try:
                metrics = self.objective_function(**params)
                score = metrics.get(self.scoring_metric, 0)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=sorted(results, key=lambda x: x['score'], reverse=True),
            optimization_time=optimization_time
        )
    
    def random_search(self, n_iterations: int = 50) -> OptimizationResult:
        """
        Random search over parameter space
        
        Args:
            n_iterations: Number of random configurations to test
        """
        import time
        start_time = time.time()
        
        results = []
        best_score = float('-inf')
        best_params = None
        
        for _ in range(n_iterations):
            # Sample random parameters
            params = {
                name: np.random.choice(values) 
                for name, values in self.param_space.items()
            }
            
            try:
                metrics = self.objective_function(**params)
                score = metrics.get(self.scoring_metric, 0)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=sorted(results, key=lambda x: x['score'], reverse=True),
            optimization_time=optimization_time
        )
    
    def bayesian_optimization(self, n_iterations: int = 30) -> OptimizationResult:
        """
        Bayesian optimization using Gaussian Process
        More efficient than grid/random search for expensive objective functions
        """
        import time
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        start_time = time.time()
        
        # Convert param space to numerical bounds
        param_names = list(self.param_space.keys())
        bounds = []
        param_mappings = {}
        
        for name, values in self.param_space.items():
            if isinstance(values[0], (int, float)):
                bounds.append((min(values), max(values)))
                param_mappings[name] = None
            else:
                # Categorical - map to indices
                param_mappings[name] = {v: i for i, v in enumerate(values)}
                bounds.append((0, len(values) - 1))
        
        bounds = np.array(bounds)
        
        # Initialize with random samples
        n_initial = min(5, n_iterations // 3)
        X_sample = []
        y_sample = []
        
        for _ in range(n_initial):
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            params = self._decode_params(x, param_names, param_mappings)
            
            try:
                metrics = self.objective_function(**params)
                score = metrics.get(self.scoring_metric, 0)
                X_sample.append(x)
                y_sample.append(score)
            except Exception as e:
                continue
        
        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)
        
        # Bayesian optimization loop
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            normalize_y=True
        )
        
        results = []
        
        for iteration in range(n_initial, n_iterations):
            # Fit GP
            gp.fit(X_sample, y_sample)
            
            # Find next point using Expected Improvement
            next_x = self._propose_location(gp, X_sample, y_sample, bounds)
            params = self._decode_params(next_x, param_names, param_mappings)
            
            try:
                metrics = self.objective_function(**params)
                score = metrics.get(self.scoring_metric, 0)
                
                X_sample = np.vstack([X_sample, next_x])
                y_sample = np.append(y_sample, score)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
            except Exception as e:
                continue
        
        # Get best result
        best_idx = np.argmax(y_sample)
        best_params = self._decode_params(X_sample[best_idx], param_names, param_mappings)
        best_score = y_sample[best_idx]
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=sorted(results, key=lambda x: x['score'], reverse=True),
            optimization_time=optimization_time
        )
    
    def _decode_params(self, x: np.ndarray, param_names: List[str], mappings: Dict) -> Dict:
        """Convert numerical array to parameter dictionary"""
        params = {}
        for i, name in enumerate(param_names):
            if mappings[name] is None:
                # Numerical parameter
                if name in self.param_space and isinstance(self.param_space[name][0], int):
                    params[name] = int(round(x[i]))
                else:
                    params[name] = x[i]
            else:
                # Categorical parameter
                idx = int(round(x[i]))
                idx = np.clip(idx, 0, len(mappings[name]) - 1)
                reverse_map = {v: k for k, v in mappings[name].items()}
                params[name] = reverse_map[idx]
        return params
    
    def _propose_location(self, gp, X_sample, y_sample, bounds, n_restarts=25):
        """Propose next sampling point using Expected Improvement"""
        from scipy.optimize import minimize
        
        best_y = np.max(y_sample)
        
        def expected_improvement(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            
            with np.errstate(divide='ignore'):
                Z = (mu - best_y) / sigma
                ei = (mu - best_y) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            return -ei[0]  # Negative for minimization
        
        # Multi-start optimization
        best_x = None
        best_ei = float('inf')
        
        for _ in range(n_restarts):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            result = minimize(expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.fun < best_ei:
                best_ei = result.fun
                best_x = result.x
        
        return best_x


class GeneticOptimizer:
    """Genetic algorithm for strategy optimization"""
    
    def __init__(self,
                 objective_function: Callable,
                 param_space: Dict[str, List],
                 scoring_metric: str = 'sharpe_ratio',
                 population_size: int = 20,
                 generations: int = 10):
        self.objective_function = objective_function
        self.param_space = param_space
        self.scoring_metric = scoring_metric
        self.population_size = population_size
        self.generations = generations
        
    def optimize(self) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        import time
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        
        results = []
        best_score = float('-inf')
        best_params = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            
            for individual in population:
                try:
                    metrics = self.objective_function(**individual)
                    score = metrics.get(self.scoring_metric, 0)
                    fitness_scores.append(score)
                    
                    results.append({
                        'params': individual.copy(),
                        'score': score,
                        'metrics': metrics,
                        'generation': generation
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()
                        
                except Exception as e:
                    fitness_scores.append(float('-inf'))
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            population = self._crossover_and_mutate(selected)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=sorted(results, key=lambda x: x['score'], reverse=True),
            optimization_time=optimization_time
        )
    
    def _initialize_population(self) -> List[Dict]:
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            individual = {
                name: np.random.choice(values)
                for name, values in self.param_space.items()
            }
            population.append(individual)
        return population
    
    def _selection(self, population: List[Dict], fitness: List[float]) -> List[Dict]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament_idx = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover_and_mutate(self, population: List[Dict]) -> List[Dict]:
        """Apply crossover and mutation"""
        new_population = []
        
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            # Crossover
            if np.random.random() < 0.7:  # Crossover probability
                child1, child2 = {}, {}
                for param in parent1.keys():
                    if np.random.random() < 0.5:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            for child in [child1, child2]:
                if np.random.random() < 0.2:  # Mutation probability
                    param_to_mutate = np.random.choice(list(child.keys()))
                    child[param_to_mutate] = np.random.choice(
                        self.param_space[param_to_mutate]
                    )
            
            new_population.extend([child1, child2])
        
        return new_population
