"""
model_builder.py - Factor Model Construction

Builds factor models by sampling from distributions.

Design Philosophy
-----------------
Single Responsibility: Model creation only (Step 1)
- Sample factor loadings from k potentially different distributions
- Sample idiosyncratic volatilities from 1 distribution  
- Construct FactorModelData with user-specified factor variances

Does NOT simulate returns. See ReturnsSimulator for that.

Usage
-----
>>> from model_builder import FactorModelBuilder
>>> from distributions import create_sampler
>>> 
>>> rng = np.random.default_rng(42)
>>> factory = lambda name, **p: create_sampler(name, rng, **p)
>>> 
>>> # Build a 2-factor model
>>> builder = FactorModelBuilder(rng=rng)
>>> model = builder.build(
...     p=100,  # 100 assets
...     k=2,    # 2 factors
...     beta_samplers=[
...         factory("normal", loc=1.0, scale=0.2),  # Market factor
...         factory("student_t", df=5)               # Size factor
...     ],
...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
...     factor_variances=[0.04, 0.01]  # Market 4x more volatile
... )
>>> 
>>> # Now simulate returns with this model (see ReturnsSimulator)
>>> # Can reuse this model with different return distributions!
"""

from typing import Union, List, Callable, Optional
import numpy as np

from .factor_types import FactorModelData

# Universal sampler interface: (n: int) -> np.ndarray
Sampler = Callable[[int], np.ndarray]


class FactorModelBuilder:
    """
    Build factor models by sampling from distributions.
    
    This class implements Step 1 of the factor model simulation pipeline:
    1. Sample factor loadings (β) for each factor
    2. Sample idiosyncratic volatilities (σ) for each asset
    3. Construct diagonal covariance matrices F and D
    
    The resulting FactorModelData can be used repeatedly with different
    return distributions via ReturnsSimulator.
    
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
    Basic usage with different distributions per factor:
        >>> from distributions import create_sampler
        >>> rng = np.random.default_rng(42)
        >>> 
        >>> builder = FactorModelBuilder(rng=rng)
        >>> 
        >>> # Helper for creating samplers
        >>> factory = lambda name, **params: create_sampler(name, rng, **params)
        >>> 
        >>> # Build 2-factor model: normal market, heavy-tailed size
        >>> model = builder.build(
        ...     p=100,
        ...     k=2,
        ...     beta_samplers=[
        ...         factory("normal", loc=1.0, scale=0.2),
        ...         factory("student_t", df=5)
        ...     ],
        ...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
        ...     factor_variances=[0.04, 0.01]
        ... )
        >>> 
        >>> print(f"Built model: {model.k} factors, {model.p} assets")
        Built model: 2 factors, 100 assets
    
    Broadcasting - single distribution for all factors:
        >>> # All factors use same distribution (broadcast)
        >>> model = builder.build(
        ...     p=100,
        ...     k=3,
        ...     beta_samplers=factory("normal", loc=0, scale=1),  # Broadcast!
        ...     idio_vol_sampler=factory("constant", value=0.03),
        ...     factor_variances=[0.04, 0.02, 0.01]
        ... )
    
    Notes
    -----
    Factor Covariance (F):
        - Diagonal matrix (factors are uncorrelated)
        - User specifies variances explicitly
        - F = diag(factor_variances)
    
    Idiosyncratic Covariance (D):
        - Diagonal matrix (assets have independent idio risk)
        - Sampled volatilities, then D = diag(σ²)
        
    The model represents returns as:
        r_i = β_i' * f + ε_i
    where:
        - r_i: Return for asset i
        - β_i: (k,) vector of factor loadings for asset i
        - f: (k,) vector of factor returns, f ~ N(0, F)
        - ε_i: Idiosyncratic return, ε_i ~ N(0, d_i)
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize the builder.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator. If None, creates default_rng().
        """
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def build(
        self,
        p: int,
        k: int,
        beta_samplers: Union[Sampler, List[Sampler]],
        idio_vol_sampler: Sampler,
        factor_variances: List[float]
    ) -> FactorModelData:
        """
        Build a factor model by sampling from distributions.
        
        Samples factor loadings for each factor, idiosyncratic volatilities
        for each asset, and constructs the complete FactorModelData.
        
        Parameters
        ----------
        p : int
            Number of assets/securities. Must be positive.
        k : int
            Number of factors. Must be positive.
        beta_samplers : Sampler or List[Sampler]
            Distribution(s) for factor loadings.
            
            If Sampler (single callable):
                Broadcast to all k factors (each factor uses same distribution)
            
            If List[Sampler] (list of k callables):
                Each factor gets its own distribution
                beta_samplers[i] generates loadings for factor i
                
            Each sampler called with p to generate p loadings (one per asset)
            
        idio_vol_sampler : Sampler
            Distribution for idiosyncratic volatilities.
            Called once with p to generate p volatilities (one per asset).
            
        factor_variances : List[float]
            Variance for each factor (length k).
            These form the diagonal of the factor covariance matrix F.
            Must all be positive.
            
            Example: [0.04, 0.02, 0.01] means:
                - Factor 1 has variance 0.04 (20% vol)
                - Factor 2 has variance 0.02 (14% vol)
                - Factor 3 has variance 0.01 (10% vol)
        
        Returns
        -------
        model : FactorModelData
            Complete factor model with:
            - B: (k, p) factor loadings matrix
            - F: (k, k) diagonal factor covariance
            - D: (p, p) diagonal idiosyncratic covariance
            
        Raises
        ------
        ValueError
            If p or k are not positive
            If len(factor_variances) != k
            If any factor variance is not positive
            If beta_samplers list has wrong length
        TypeError
            If beta_samplers or idio_vol_sampler are not callable
            
        Examples
        --------
        Different distribution per factor:
            >>> model = builder.build(
            ...     p=100,
            ...     k=2,
            ...     beta_samplers=[
            ...         factory("normal", loc=1.0, scale=0.2),
            ...         factory("student_t", df=5)
            ...     ],
            ...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
            ...     factor_variances=[0.04, 0.01]
            ... )
        
        Single distribution broadcast to all factors:
            >>> model = builder.build(
            ...     p=100,
            ...     k=3,
            ...     beta_samplers=factory("normal", loc=0, scale=1),
            ...     idio_vol_sampler=factory("constant", value=0.03),
            ...     factor_variances=[0.04, 0.02, 0.01]
            ... )
        
        Using scipy distributions:
            >>> from scipy.stats import skewnorm
            >>> skew = lambda n: skewnorm.rvs(a=5, size=n, random_state=42)
            >>> model = builder.build(
            ...     p=100,
            ...     k=1,
            ...     beta_samplers=skew,  # Custom scipy distribution
            ...     idio_vol_sampler=factory("constant", value=0.03),
            ...     factor_variances=[0.04]
            ... )
        """
        # Validate inputs
        if p <= 0:
            raise ValueError(f"p (number of assets) must be positive, got {p}")
        if k <= 0:
            raise ValueError(f"k (number of factors) must be positive, got {k}")
        
        if len(factor_variances) != k:
            raise ValueError(
                f"factor_variances must have length k={k}, got {len(factor_variances)}"
            )
        
        factor_variances_array = np.array(factor_variances)
        if np.any(factor_variances_array <= 0):
            raise ValueError("All factor variances must be positive")
        
        # Resolve beta_samplers to list of k samplers
        beta_samplers_list = self._resolve_to_list(
            beta_samplers, k, "beta_samplers"
        )
        
        # Validate idio_vol_sampler is callable
        if not callable(idio_vol_sampler):
            raise TypeError(
                f"idio_vol_sampler must be callable, got {type(idio_vol_sampler)}"
            )
        
        # Step 1.1: Sample factor loadings
        # Each factor (row of B) is sampled independently
        # B[i, :] = loadings for factor i across all p assets
        B = np.vstack([
            sampler(p) for sampler in beta_samplers_list
        ])  # Shape: (k, p)
        
        # Step 1.2: Sample idiosyncratic volatilities
        # One volatility per asset
        idio_vols = idio_vol_sampler(p)  # Shape: (p,)
        
        # Construct D: diagonal matrix of idio variances
        # D[i, i] = idio_vols[i]²
        D = np.diag(idio_vols ** 2)  # Shape: (p, p)
        
        # Construct F: diagonal matrix of factor variances
        # F[i, i] = factor_variances[i]
        F = np.diag(factor_variances_array)  # Shape: (k, k)
        
        return FactorModelData(B=B, F=F, D=D)
    
    def _resolve_to_list(
        self,
        sampler_spec: Union[Sampler, List[Sampler]],
        expected_length: int,
        param_name: str
    ) -> List[Sampler]:
        """
        Resolve sampler specification to list of samplers.
        
        Handles two cases:
        1. Single sampler → broadcast to list of length expected_length
        2. List of samplers → validate length and callability
        
        Parameters
        ----------
        sampler_spec : Sampler or List[Sampler]
            Either a single sampler or list of samplers
        expected_length : int
            Required list length (k for beta_samplers)
        param_name : str
            Parameter name for error messages
            
        Returns
        -------
        samplers : List[Sampler]
            List of exactly expected_length samplers
            
        Raises
        ------
        ValueError
            If list has wrong length
        TypeError
            If any element is not callable
        """
        if isinstance(sampler_spec, list):
            # Explicit list provided - validate
            if len(sampler_spec) != expected_length:
                raise ValueError(
                    f"{param_name}: expected list of length {expected_length}, "
                    f"got {len(sampler_spec)}"
                )
            
            # Validate all elements are callable
            for i, sampler in enumerate(sampler_spec):
                if not callable(sampler):
                    raise TypeError(
                        f"{param_name}[{i}] is not callable: {type(sampler)}"
                    )
            
            return sampler_spec
        
        elif callable(sampler_spec):
            # Single sampler - broadcast to all factors
            return [sampler_spec] * expected_length
        
        else:
            raise TypeError(
                f"{param_name} must be callable or list of callables, "
                f"got {type(sampler_spec)}"
            )
