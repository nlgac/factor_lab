"""
returns_simulator.py - Returns Simulation

Simulates returns from factor models using arbitrary distributions.

Design Philosophy
-----------------
Single Responsibility: Returns simulation only (Steps 2-4)
- Sample factor returns from k potentially different distributions
- Sample idiosyncratic returns from 1 distribution
- Combine them via matrix multiplication

Does NOT create models. See FactorModelBuilder for that.

Key Feature: Same model, different return distributions!
- Build model once with FactorModelBuilder
- Simulate many times with different distributions
- Compare normal vs t-distributed vs mixture returns

Usage
-----
>>> from returns_simulator import ReturnsSimulator
>>> from distributions import create_sampler
>>> 
>>> rng = np.random.default_rng(42)
>>> factory = lambda name, **p: create_sampler(name, rng, **p)
>>> 
>>> # Simulate from existing model
>>> simulator = ReturnsSimulator(rng=rng)
>>> results = simulator.simulate(
...     model=model,  # From FactorModelBuilder
...     n_periods=1000,
...     factor_return_samplers=factory("normal", loc=0, scale=1),
...     idio_return_sampler=factory("normal", loc=0, scale=1)
... )
>>> 
>>> returns = results['security_returns']  # (1000, 100)
"""

from typing import Union, List, Callable, Optional, Dict
import numpy as np

from .factor_types import FactorModelData

# Universal sampler interface
Sampler = Callable[[int], np.ndarray]


class ReturnsSimulator:
    """
    Simulate returns from factor models.
    
    This class implements Steps 2-4 of the factor model simulation pipeline:
    2. Sample factor returns (f) for each factor
    3. Sample idiosyncratic returns (ε) for each asset
    4. Combine: r = B'f + ε
    
    The simulator is stateless - you can reuse it with different models
    and different return distributions.
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        If None, creates a new default generator.
    
    Attributes
    ----------
    rng : np.random.Generator
        The random number generator used for sampling
        
    Examples
    --------
    Basic usage:
        >>> from model_builder import FactorModelBuilder
        >>> from returns_simulator import ReturnsSimulator
        >>> from distributions import create_sampler
        >>> 
        >>> rng = np.random.default_rng(42)
        >>> factory = lambda name, **p: create_sampler(name, rng, **p)
        >>> 
        >>> # Build model once
        >>> builder = FactorModelBuilder(rng=rng)
        >>> model = builder.build(
        ...     p=100, k=2,
        ...     beta_samplers=[
        ...         factory("normal", loc=1.0, scale=0.2),
        ...         factory("student_t", df=5)
        ...     ],
        ...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
        ...     factor_variances=[0.04, 0.01]
        ... )
        >>> 
        >>> # Simulate with normal returns
        >>> simulator = ReturnsSimulator(rng=rng)
        >>> results = simulator.simulate(
        ...     model=model,
        ...     n_periods=1000,
        ...     factor_return_samplers=factory("normal", loc=0, scale=1),
        ...     idio_return_sampler=factory("normal", loc=0, scale=1)
        ... )
    
    Reusing same model with different return distributions:
        >>> # Experiment 1: Normal returns
        >>> results_normal = simulator.simulate(
        ...     model, 1000,
        ...     factory("normal", loc=0, scale=1),
        ...     factory("normal", loc=0, scale=1)
        ... )
        >>> 
        >>> # Experiment 2: Heavy-tailed returns (same model!)
        >>> results_t = simulator.simulate(
        ...     model, 1000,
        ...     factory("student_t", df=5),
        ...     factory("student_t", df=7)
        ... )
        >>> 
        >>> # Experiment 3: Mixed (same model!)
        >>> results_mixed = simulator.simulate(
        ...     model, 1000,
        ...     [factory("normal", loc=0, scale=1), factory("student_t", df=4)],
        ...     factory("normal", loc=0, scale=1)
        ... )
    
    Notes
    -----
    Factor Returns (Step 2):
        - Each factor sampled independently
        - Raw samples scaled by sqrt(variance from F)
        - Ensures correct factor variance
    
    Idiosyncratic Returns (Step 3):
        - All assets use same distribution
        - Each asset's samples scaled by its volatility (from D)
        - Ensures correct idio variance per asset
    
    Combination (Step 4):
        - Matrix multiplication: r = f @ B + ε
        - Shape: (n_periods, k) @ (k, p) + (n_periods, p) = (n_periods, p)
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator. If None, creates default_rng().
        """
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def simulate(
        self,
        model: FactorModelData,
        n_periods: int,
        factor_return_samplers: Union[Sampler, List[Sampler]],
        idio_return_sampler: Sampler
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns from a factor model.
        
        Samples factor and idiosyncratic returns, then combines them
        according to the factor model structure.
        
        Parameters
        ----------
        model : FactorModelData
            Factor model created by FactorModelBuilder.
            Contains B (loadings), F (factor cov), D (idio cov).
            
        n_periods : int
            Number of time periods to simulate. Must be positive.
            
        factor_return_samplers : Sampler or List[Sampler]
            Distribution(s) for factor returns.
            
            If Sampler (single callable):
                Broadcast to all k factors (all use same distribution)
            
            If List[Sampler] (list of k callables):
                Each factor gets its own distribution
                factor_return_samplers[i] generates returns for factor i
                
            Each sampler called with n_periods to generate n_periods samples
            
            IMPORTANT: Samplers should return standardized values (mean=0, std=1)
            They will be scaled by sqrt(factor variance) automatically
            
        idio_return_sampler : Sampler
            Distribution for idiosyncratic returns.
            Called n_periods × p times total (once per asset per period).
            
            IMPORTANT: Should return standardized values (mean=0, std=1)
            They will be scaled by asset volatilities automatically
        
        Returns
        -------
        results : dict
            Dictionary with keys:
            
            'security_returns' : np.ndarray, shape (n_periods, p)
                Simulated returns for p securities over n_periods
                
            'factor_returns' : np.ndarray, shape (n_periods, k)
                Simulated factor returns (properly scaled by F)
                
            'idio_returns' : np.ndarray, shape (n_periods, p)
                Simulated idiosyncratic returns (properly scaled by D)
        
        Raises
        ------
        ValueError
            If n_periods is not positive
            If factor_return_samplers list has wrong length
        TypeError
            If factor_return_samplers or idio_return_sampler not callable
            
        Examples
        --------
        Normal returns:
            >>> results = simulator.simulate(
            ...     model, 1000,
            ...     factory("normal", loc=0, scale=1),
            ...     factory("normal", loc=0, scale=1)
            ... )
        
        Heavy-tailed returns:
            >>> results = simulator.simulate(
            ...     model, 1000,
            ...     factory("student_t", df=5),
            ...     factory("student_t", df=7)
            ... )
        
        Different distribution per factor:
            >>> results = simulator.simulate(
            ...     model, 1000,
            ...     [factory("normal", loc=0, scale=1), factory("student_t", df=4)],
            ...     factory("normal", loc=0, scale=1)
            ... )
        
        Custom scipy distributions:
            >>> from scipy.stats import skewnorm
            >>> skew = lambda n: skewnorm.rvs(a=5, size=n, random_state=42)
            >>> results = simulator.simulate(model, 1000, skew, skew)
        
        Notes
        -----
        The scaling works as follows:
        
        Factor returns:
            raw_f[i] ~ sampler(n_periods)  # Raw samples
            f[i] = raw_f[i] * sqrt(F[i,i])  # Scaled to have variance F[i,i]
        
        Idiosyncratic returns:
            raw_ε[j] ~ sampler(n_periods)  # Raw samples
            ε[j] = raw_ε[j] * sqrt(D[j,j])  # Scaled to have variance D[j,j]
        
        Security returns:
            r[j] = Σ(B[i,j] * f[i]) + ε[j]  # Linear combination
        """
        # Validate inputs
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        
        k, p = model.k, model.p
        
        # Resolve factor_return_samplers to list of k samplers
        factor_samplers_list = self._resolve_to_list(
            factor_return_samplers, k, "factor_return_samplers"
        )
        
        # Validate idio_return_sampler is callable
        if not callable(idio_return_sampler):
            raise TypeError(
                f"idio_return_sampler must be callable, got {type(idio_return_sampler)}"
            )
        
        # Step 2: Sample factor returns
        factor_returns = self._sample_factor_returns(
            factor_samplers_list, n_periods, model.F
        )  # Shape: (n_periods, k)
        
        # Step 3: Sample idiosyncratic returns
        idio_returns = self._sample_idio_returns(
            idio_return_sampler, n_periods, model.D
        )  # Shape: (n_periods, p)
        
        # Step 4: Combine via matrix multiplication
        # r = f @ B + ε
        # (n_periods, k) @ (k, p) + (n_periods, p) = (n_periods, p)
        security_returns = factor_returns @ model.B + idio_returns
        
        return {
            'security_returns': security_returns,
            'factor_returns': factor_returns,
            'idio_returns': idio_returns
        }
    
    def _sample_factor_returns(
        self,
        samplers: List[Sampler],
        n_periods: int,
        F: np.ndarray
    ) -> np.ndarray:
        """
        Sample factor returns and scale by factor variances.
        
        For each factor:
        1. Sample n_periods raw values from its distribution
        2. Scale by sqrt(variance) to achieve correct variance
        
        Parameters
        ----------
        samplers : List[Sampler]
            One sampler per factor (length k)
        n_periods : int
            Number of periods to simulate
        F : np.ndarray, shape (k, k)
            Factor covariance matrix (diagonal)
            
        Returns
        -------
        factor_returns : np.ndarray, shape (n_periods, k)
            Factor returns with correct variances
        """
        k = len(samplers)
        
        # Sample raw returns for each factor
        # Each column is one factor's returns
        raw_returns = np.column_stack([
            sampler(n_periods) for sampler in samplers
        ])  # Shape: (n_periods, k)
        
        # Extract factor standard deviations (sqrt of diagonal of F)
        factor_stds = np.sqrt(np.diag(F))  # Shape: (k,)
        
        # Scale each factor's returns by its standard deviation
        # Broadcasting: (n_periods, k) * (k,) -> (n_periods, k)
        factor_returns = raw_returns * factor_stds[np.newaxis, :]
        
        return factor_returns
    
    def _sample_idio_returns(
        self,
        sampler: Sampler,
        n_periods: int,
        D: np.ndarray
    ) -> np.ndarray:
        """
        Sample idiosyncratic returns and scale by asset volatilities.
        
        For each asset:
        1. Sample n_periods raw values from the distribution
        2. Scale by asset's volatility (sqrt of D[i,i])
        
        All assets use the same distribution, but scaled differently.
        
        Parameters
        ----------
        sampler : Sampler
            Distribution for idiosyncratic returns
        n_periods : int
            Number of periods to simulate
        D : np.ndarray, shape (p, p)
            Idiosyncratic covariance matrix (diagonal)
            
        Returns
        -------
        idio_returns : np.ndarray, shape (n_periods, p)
            Idiosyncratic returns with correct variances per asset
        """
        p = D.shape[0]
        
        # Sample raw returns for each asset
        # Each column is one asset's idio returns
        raw_returns = np.column_stack([
            sampler(n_periods) for _ in range(p)
        ])  # Shape: (n_periods, p)
        
        # Extract idio standard deviations (sqrt of diagonal of D)
        idio_stds = np.sqrt(np.diag(D))  # Shape: (p,)
        
        # Scale each asset's returns by its idio volatility
        # Broadcasting: (n_periods, p) * (p,) -> (n_periods, p)
        idio_returns = raw_returns * idio_stds[np.newaxis, :]
        
        return idio_returns
    
    def _resolve_to_list(
        self,
        sampler_spec: Union[Sampler, List[Sampler]],
        expected_length: int,
        param_name: str
    ) -> List[Sampler]:
        """
        Resolve sampler specification to list of samplers.
        
        Same as in FactorModelBuilder - handles broadcasting.
        
        Parameters
        ----------
        sampler_spec : Sampler or List[Sampler]
            Either a single sampler or list of samplers
        expected_length : int
            Required list length (k for factor returns)
        param_name : str
            Parameter name for error messages
            
        Returns
        -------
        samplers : List[Sampler]
            List of exactly expected_length samplers
        """
        if isinstance(sampler_spec, list):
            if len(sampler_spec) != expected_length:
                raise ValueError(
                    f"{param_name}: expected list of length {expected_length}, "
                    f"got {len(sampler_spec)}"
                )
            
            for i, sampler in enumerate(sampler_spec):
                if not callable(sampler):
                    raise TypeError(
                        f"{param_name}[{i}] is not callable: {type(sampler)}"
                    )
            
            return sampler_spec
        
        elif callable(sampler_spec):
            return [sampler_spec] * expected_length
        
        else:
            raise TypeError(
                f"{param_name} must be callable or list of callables, "
                f"got {type(sampler_spec)}"
            )
