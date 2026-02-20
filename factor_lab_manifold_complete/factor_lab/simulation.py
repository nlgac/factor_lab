"""
simulation.py - Returns Simulation

Simulate returns from factor models with arbitrary distributions.

Design Philosophy
-----------------
- Backward Compatible: Default behavior unchanged (normal distributions)
- Flexible: Accept any callable (n: int) -> np.ndarray
- Decoupled: Simulator doesn't know distribution source
- Composable: Method chaining for configuration

Examples
--------
Default normal distribution (backward compatible):
    >>> from factor_lab.types import FactorModelData
    >>> from factor_lab.simulation import ReturnsSimulator
    >>> 
    >>> model = FactorModelData(B, F, D)
    >>> sim = ReturnsSimulator(model, rng=rng)
    >>> results = sim.simulate(1000)

Custom t-distribution for factors:
    >>> from factor_lab.distributions import create_sampler
    >>> 
    >>> t_sampler = create_sampler("student_t", rng, df=5)
    >>> sim = ReturnsSimulator(model, rng=rng)
    >>> sim.set_factor_distribution(t_sampler)
    >>> results = sim.simulate(1000)

Custom scipy distribution:
    >>> from scipy.stats import levy_stable
    >>> 
    >>> levy = lambda n: levy_stable.rvs(alpha=1.5, beta=0, size=n)
    >>> sim = ReturnsSimulator(model, rng=rng)
    >>> sim.set_factor_distribution(levy)
    >>> results = sim.simulate(1000)

Mixture model for regime-switching:
    >>> def mixture(n):
    ...     regime = rng.binomial(1, 0.95, n)
    ...     normal = rng.normal(0, 1, n)
    ...     crisis = rng.normal(0, 3, n)
    ...     return np.where(regime, normal, crisis)
    >>> 
    >>> sim = ReturnsSimulator(model, rng=rng)
    >>> sim.set_factor_distribution(mixture)
    >>> results = sim.simulate(1000)

Method chaining:
    >>> results = (ReturnsSimulator(model, rng=rng)
    ...           .set_factor_distribution(t_sampler)
    ...           .set_idio_distribution(normal_sampler)
    ...           .simulate(1000))
"""

from typing import Optional, Callable, Dict
import numpy as np
try:
    from .factor_types import FactorModelData
except ImportError:
    from factor_types import FactorModelData

# Universal sampler interface
Sampler = Callable[[int], np.ndarray]


class ReturnsSimulator:
    """
    Simulate returns from factor model with configurable distributions.
    
    Model: r_t = B' f_t + ε_t
    
    By default, both factor and idiosyncratic returns use normal distributions
    (maintains backward compatibility). Custom distributions can be set via
    set_factor_distribution() and set_idio_distribution().
    
    Any callable with signature (n: int) -> np.ndarray can be used as a
    distribution sampler.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model containing B (loadings), F (factor cov), D (idio cov)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Attributes
    ----------
    model : FactorModelData
        The factor model
    rng : np.random.Generator
        Random number generator
        
    Examples
    --------
    Basic usage (normal distributions):
        >>> model = FactorModelData(B, F, D)
        >>> sim = ReturnsSimulator(model, rng=rng)
        >>> results = sim.simulate(1000)
        >>> returns = results['security_returns']  # (1000, p)
    
    Custom factor distribution:
        >>> from factor_lab.distributions import t_sampler
        >>> sim = ReturnsSimulator(model, rng=rng)
        >>> sim.set_factor_distribution(t_sampler(rng, df=5))
        >>> results = sim.simulate(1000)
    
    Both custom distributions:
        >>> t5 = t_sampler(rng, df=5)
        >>> t7 = t_sampler(rng, df=7)
        >>> sim = ReturnsSimulator(model, rng=rng)
        >>> sim.set_factor_distribution(t5)
        >>> sim.set_idio_distribution(t7)
        >>> results = sim.simulate(1000)
    
    Method chaining:
        >>> results = (ReturnsSimulator(model, rng=rng)
        ...           .set_factor_distribution(t5)
        ...           .simulate(1000))
    
    Notes
    -----
    The covariance structure (F and D) is applied to the sampled distributions
    via Cholesky decomposition. For non-normal distributions, samples are
    standardized before applying covariance to ensure correct variance.
    """
    
    def __init__(
        self, 
        model: FactorModelData, 
        rng: Optional[np.random.Generator] = None
    ):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Optional custom distributions (None = use normal)
        self._factor_sampler: Optional[Sampler] = None
        self._idio_sampler: Optional[Sampler] = None
    
    def set_factor_distribution(self, sampler: Sampler) -> 'ReturnsSimulator':
        """
        Set custom distribution for factor returns.
        
        Parameters
        ----------
        sampler : Callable[[int], np.ndarray]
            Distribution sampler with signature (n) -> array of n samples
            
        Returns
        -------
        self : ReturnsSimulator
            Returns self for method chaining
            
        Examples
        --------
        Using built-in distributions:
            >>> from factor_lab.distributions import create_sampler
            >>> t_dist = create_sampler("student_t", rng, df=5)
            >>> sim.set_factor_distribution(t_dist)
        
        Using scipy:
            >>> from scipy.stats import skewnorm
            >>> skew = lambda n: skewnorm.rvs(a=2, size=n)
            >>> sim.set_factor_distribution(skew)
        
        Custom mixture:
            >>> def mixture(n):
            ...     regime = rng.binomial(1, 0.95, n)
            ...     return np.where(regime, 
            ...                    rng.normal(0, 1, n),
            ...                    rng.normal(0, 3, n))
            >>> sim.set_factor_distribution(mixture)
        """
        self._factor_sampler = sampler
        return self
    
    def set_idio_distribution(self, sampler: Sampler) -> 'ReturnsSimulator':
        """
        Set custom distribution for idiosyncratic returns.
        
        Parameters
        ----------
        sampler : Callable[[int], np.ndarray]
            Distribution sampler with signature (n) -> array of n samples
            
        Returns
        -------
        self : ReturnsSimulator
            Returns self for method chaining
            
        Examples
        --------
        >>> from factor_lab.distributions import create_sampler
        >>> t_dist = create_sampler("student_t", rng, df=7)
        >>> sim.set_idio_distribution(t_dist)
        """
        self._idio_sampler = sampler
        return self
    
    def simulate(self, n_periods: int) -> Dict[str, np.ndarray]:
        """
        Simulate returns from the factor model.
        
        Generates returns using: r_t = B' f_t + ε_t
        
        Parameters
        ----------
        n_periods : int
            Number of time periods to simulate
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'security_returns': (n_periods, p) array of asset returns
            - 'factor_returns': (n_periods, k) array of factor returns
            - 'idio_returns': (n_periods, p) array of idiosyncratic returns
            
        Examples
        --------
        >>> results = sim.simulate(1000)
        >>> returns = results['security_returns']  # (1000, p)
        >>> factors = results['factor_returns']    # (1000, k)
        >>> idio = results['idio_returns']         # (1000, p)
        
        Notes
        -----
        If custom distributions are set via set_factor_distribution() or
        set_idio_distribution(), they will be used instead of the default
        normal distributions.
        
        The covariance structure from F and D is applied to all distributions,
        so the resulting returns will have the correct covariance even when
        using non-normal distributions.
        """
        k, p = self.model.k, self.model.p
        
        # Simulate factor returns
        if self._factor_sampler is not None:
            # Custom distribution
            factor_returns = self._apply_covariance(
                self._factor_sampler,
                n_periods,
                k,
                self.model.F
            )
        else:
            # Default: normal distribution (backward compatible)
            F_chol = np.linalg.cholesky(self.model.F)
            factor_returns = self.rng.normal(0, 1, (n_periods, k)) @ F_chol.T
        
        # Simulate idiosyncratic returns
        if self._idio_sampler is not None:
            # Custom distribution
            idio_returns = self._apply_covariance(
                self._idio_sampler,
                n_periods,
                p,
                self.model.D
            )
        else:
            # Default: normal distribution (backward compatible)
            D_sqrt = np.sqrt(np.diag(self.model.D))
            idio_returns = self.rng.normal(0, 1, (n_periods, p)) * D_sqrt
        
        # Combine: r = B'f + ε
        security_returns = factor_returns @ self.model.B + idio_returns
        
        return {
            'security_returns': security_returns,
            'factor_returns': factor_returns,
            'idio_returns': idio_returns
        }
    
    def _apply_covariance(
        self,
        sampler: Sampler,
        n_periods: int,
        dim: int,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Sample from distribution and apply covariance structure.
        
        Strategy:
        1. Sample each dimension independently using the sampler
        2. Standardize to unit variance (important for t-distributions)
        3. Apply covariance structure via Cholesky: X = Z @ L'
        
        Parameters
        ----------
        sampler : Callable[[int], np.ndarray]
            Distribution sampler
        n_periods : int
            Number of time periods
        dim : int
            Dimension (k for factors, p for idiosyncratic)
        covariance : np.ndarray
            Covariance matrix (F or D)
            
        Returns
        -------
        samples : np.ndarray, shape (n_periods, dim)
            Samples with specified covariance structure
            
        Notes
        -----
        For t-distributions with df > 2, the variance is df/(df-2), not 1.
        We standardize to unit variance before applying the covariance to
        ensure the resulting samples have the correct covariance matrix.
        
        For normal distributions, standardization is a no-op (already unit var).
        
        This approach works correctly for any distribution with finite variance.
        """
        # Sample each dimension independently
        # Each column represents one dimension
        Z = np.column_stack([sampler(n_periods) for _ in range(dim)])
        
        # Standardize to unit variance
        # For t(df), this converts var=df/(df-2) to var=1
        # For normal(0,1), this is a no-op
        Z = Z / Z.std(axis=0, ddof=1)
        
        # Apply covariance structure via Cholesky
        chol = np.linalg.cholesky(covariance)
        return Z @ chol.T
