"""
samplers.py - Statistical Distribution Management and Data Sampling

This module provides tools for sampling from statistical distributions:
- DistributionRegistry: Repository of available distributions
- DistributionFactory: Creates sampler functions from distribution specs
- DataSampler: Generates synthetic factor model data

Design Principles:
-----------------
1. Dependency Injection: All samplers accept an explicit RNG for reproducibility
2. Registry Pattern: Distributions are registered and accessed by name
3. Validation: Parameter requirements are checked before sampler creation
4. Broadcasting: Single samplers can be broadcast across multiple dimensions

Example Usage:
-------------
    >>> import numpy as np
    >>> from factor_lab.samplers import DistributionFactory, DataSampler
    >>> 
    >>> # Create samplers with explicit RNG
    >>> rng = np.random.default_rng(42)
    >>> factory = DistributionFactory(rng=rng)
    >>> 
    >>> # Create a normal sampler
    >>> normal_sampler = factory.create("normal", mean=0.0, std=1.0)
    >>> samples = normal_sampler(1000)  # 1000 samples
    >>> 
    >>> # Generate a complete factor model
    >>> data_sampler = DataSampler(p=100, k=3, rng=rng)
    >>> model = data_sampler.configure(
    ...     beta=factory.create("normal", mean=0.0, std=1.0),
    ...     factor_vol=factory.create("uniform", low=0.1, high=0.3),
    ...     idio_vol=factory.create("constant", value=0.05)
    ... ).generate()
"""

from __future__ import annotations

import inspect
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Set, Any
from dataclasses import dataclass, field

from .types import SamplerCallable, FactorModelData


# =============================================================================
# DISTRIBUTION REGISTRY
# =============================================================================

@dataclass
class DistributionInfo:
    """
    Metadata about a registered distribution.
    
    Attributes
    ----------
    name : str
        Canonical name of the distribution (lowercase).
    func : Callable
        The sampling function. Signature: f(rng, n, **params) -> array.
    required_params : Set[str]
        Parameter names that must be provided.
    optional_params : Dict[str, Any]
        Parameter names with their default values.
    description : str
        Human-readable description of the distribution.
    """
    name: str
    func: Callable
    required_params: Set[str]
    optional_params: Dict[str, Any]
    description: str = ""


class DistributionRegistry:
    """
    Repository of available probability distributions.
    
    This class maintains a registry of distribution functions that can be
    used to create samplers. Built-in distributions are pre-registered,
    and users can add custom distributions.
    
    Examples
    --------
    >>> registry = DistributionRegistry()
    >>> print(registry.list_distributions())
    ['beta', 'constant', 'exponential', 'normal', 'student_t', 'uniform']
    >>> 
    >>> # Register a custom distribution
    >>> registry.register(
    ...     name="laplace",
    ...     func=lambda rng, n, loc, scale: rng.laplace(loc, scale, n),
    ...     required_params={"loc", "scale"},
    ...     description="Laplace (double exponential) distribution"
    ... )
    
    Notes
    -----
    All distribution functions must have the signature:
        func(rng: np.random.Generator, n: int, **params) -> np.ndarray
    
    The `rng` is passed by the factory, `n` is the sample size, and
    `**params` are distribution-specific parameters.
    """
    
    def __init__(self):
        """Initialize with built-in distributions."""
        self._distributions: Dict[str, DistributionInfo] = {}
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register the standard set of distributions."""
        
        # Normal distribution
        self.register(
            name="normal",
            func=lambda rng, n, mean, std: rng.normal(mean, std, n),
            required_params={"mean", "std"},
            description="Gaussian/Normal distribution with given mean and std"
        )
        
        # Uniform distribution
        self.register(
            name="uniform",
            func=lambda rng, n, low, high: rng.uniform(low, high, n),
            required_params={"low", "high"},
            description="Uniform distribution on [low, high)"
        )
        
        # Constant (degenerate) distribution
        self.register(
            name="constant",
            func=lambda rng, n, value: np.full(n, value, dtype=float),
            required_params={"value"},
            description="Degenerate distribution returning a constant value"
        )
        
        # Student's t distribution
        self.register(
            name="student_t",
            func=lambda rng, n, df: rng.standard_t(df, size=n),
            required_params={"df"},
            description="Student's t distribution with df degrees of freedom"
        )
        
        # Beta distribution
        self.register(
            name="beta",
            func=lambda rng, n, a, b: rng.beta(a, b, size=n),
            required_params={"a", "b"},
            description="Beta distribution with shape parameters a and b"
        )
        
        # Exponential distribution
        self.register(
            name="exponential",
            func=lambda rng, n, scale: rng.exponential(scale, size=n),
            required_params={"scale"},
            description="Exponential distribution with given scale (1/rate)"
        )
        
        # Log-normal distribution
        self.register(
            name="lognormal",
            func=lambda rng, n, mean, sigma: rng.lognormal(mean, sigma, n),
            required_params={"mean", "sigma"},
            description="Log-normal distribution (mean and sigma of log)"
        )
        
        # Chi-squared distribution
        self.register(
            name="chi2",
            func=lambda rng, n, df: rng.chisquare(df, size=n),
            required_params={"df"},
            description="Chi-squared distribution with df degrees of freedom"
        )
    
    def register(
        self,
        name: str,
        func: Callable,
        required_params: Set[str],
        optional_params: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> None:
        """
        Register a new distribution.
        
        Parameters
        ----------
        name : str
            Name for the distribution (will be lowercased).
        func : Callable
            Sampling function with signature f(rng, n, **params) -> array.
        required_params : Set[str]
            Set of parameter names that must be provided.
        optional_params : Dict[str, Any], optional
            Parameter names with default values.
        description : str, optional
            Human-readable description.
        
        Examples
        --------
        >>> registry.register(
        ...     name="truncated_normal",
        ...     func=lambda rng, n, mean, std, low, high: ...,
        ...     required_params={"mean", "std", "low", "high"},
        ...     description="Normal distribution truncated to [low, high]"
        ... )
        """
        name_lower = name.lower()
        
        self._distributions[name_lower] = DistributionInfo(
            name=name_lower,
            func=func,
            required_params=required_params,
            optional_params=optional_params or {},
            description=description
        )
    
    def get(self, name: str) -> DistributionInfo:
        """
        Retrieve a registered distribution.
        
        Parameters
        ----------
        name : str
            Distribution name (case-insensitive).
        
        Returns
        -------
        DistributionInfo
            The distribution's metadata and function.
        
        Raises
        ------
        KeyError
            If the distribution is not registered.
        """
        name_lower = name.lower()
        
        if name_lower not in self._distributions:
            available = ", ".join(sorted(self._distributions.keys()))
            raise KeyError(
                f"Unknown distribution '{name}'. Available: {available}"
            )
        
        return self._distributions[name_lower]
    
    def list_distributions(self) -> List[str]:
        """
        Get a list of all registered distribution names.
        
        Returns
        -------
        List[str]
            Sorted list of distribution names.
        """
        return sorted(self._distributions.keys())
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a distribution.
        
        Parameters
        ----------
        name : str
            Distribution name.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: name, required_params, optional_params, description.
        """
        info = self.get(name)
        return {
            "name": info.name,
            "required_params": info.required_params,
            "optional_params": info.optional_params,
            "description": info.description
        }


# =============================================================================
# DISTRIBUTION FACTORY
# =============================================================================

class DistributionFactory:
    """
    Factory for creating sampler functions from distribution specifications.
    
    This class creates callable samplers that draw from specified distributions.
    All samplers use a shared (or provided) random number generator for
    reproducibility.
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator to use. If None, creates a new default RNG.
    registry : DistributionRegistry, optional
        Distribution registry to use. If None, creates a new registry with
        built-in distributions.
    
    Examples
    --------
    >>> # Reproducible sampling
    >>> rng = np.random.default_rng(seed=42)
    >>> factory = DistributionFactory(rng=rng)
    >>> 
    >>> sampler = factory.create("normal", mean=0.0, std=1.0)
    >>> samples = sampler(1000)
    >>> print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
    
    Notes
    -----
    The factory validates parameters at creation time, so errors are
    caught early rather than during sampling.
    """
    
    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        registry: Optional[DistributionRegistry] = None
    ):
        """
        Initialize the factory.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator. If None, creates a new one.
        registry : DistributionRegistry, optional
            Distribution registry. If None, uses default with builtins.
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self._registry = registry if registry is not None else DistributionRegistry()
    
    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used by this factory."""
        return self._rng
    
    @property
    def registry(self) -> DistributionRegistry:
        """The distribution registry used by this factory."""
        return self._registry
    
    def list_distributions(self) -> List[str]:
        """
        Get a list of available distributions.
        
        Returns
        -------
        List[str]
            Sorted list of distribution names.
        """
        return self._registry.list_distributions()
    
    def create(self, dist_name: str, **params) -> SamplerCallable:
        """
        Create a sampler function for the specified distribution.
        
        Parameters
        ----------
        dist_name : str
            Name of the distribution (case-insensitive).
        **params
            Distribution-specific parameters (e.g., mean, std for normal).
        
        Returns
        -------
        SamplerCallable
            A callable that takes sample size n and returns an array of samples.
        
        Raises
        ------
        KeyError
            If the distribution is not registered.
        ValueError
            If required parameters are missing.
        
        Examples
        --------
        >>> factory = DistributionFactory()
        >>> 
        >>> # Normal distribution
        >>> normal = factory.create("normal", mean=0.0, std=1.0)
        >>> samples = normal(100)
        >>> 
        >>> # Student's t with 4 degrees of freedom (fat tails)
        >>> t_dist = factory.create("student_t", df=4)
        >>> samples = t_dist(100)
        """
        # Get distribution info
        info = self._registry.get(dist_name)
        
        # Validate parameters
        self._validate_params(info, params)
        
        # Merge with optional defaults
        full_params = {**info.optional_params, **params}
        
        # Capture references for closure
        rng = self._rng
        func = info.func
        
        # Create and return the sampler
        def sampler(n: Union[int, tuple]) -> np.ndarray:
            """
            Sample from the distribution.
            
            Parameters
            ----------
            n : int or tuple
                Sample size. If tuple, draws np.prod(n) samples and reshapes.
            
            Returns
            -------
            np.ndarray
                Array of samples with shape determined by n.
            """
            if isinstance(n, tuple):
                size = int(np.prod(n))
                return func(rng, size, **full_params).reshape(n)
            else:
                return func(rng, n, **full_params)
        
        return sampler
    
    def _validate_params(
        self, 
        info: DistributionInfo, 
        params: Dict[str, Any]
    ) -> None:
        """Validate that required parameters are provided."""
        missing = info.required_params - set(params.keys())
        
        if missing:
            raise ValueError(
                f"Distribution '{info.name}' requires parameters: {missing}. "
                f"Got: {set(params.keys())}"
            )


# =============================================================================
# DATA SAMPLER
# =============================================================================

class DataSampler:
    """
    Generator for synthetic factor model data.
    
    This class creates random FactorModelData instances by sampling
    factor loadings, factor volatilities, and idiosyncratic volatilities
    from specified distributions.
    
    Parameters
    ----------
    p : int
        Number of assets.
    k : int
        Number of factors.
    rng : np.random.Generator, optional
        Random number generator. If None, creates a new one.
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> factory = DistributionFactory(rng=rng)
    >>> 
    >>> # Create a sampler for 100 assets, 3 factors
    >>> sampler = DataSampler(p=100, k=3, rng=rng)
    >>> 
    >>> # Configure with distributions
    >>> model = (sampler
    ...     .configure(
    ...         beta=factory.create("normal", mean=0.0, std=1.0),
    ...         factor_vol=factory.create("uniform", low=0.15, high=0.25),
    ...         idio_vol=factory.create("constant", value=0.05)
    ...     )
    ...     .generate())
    >>> 
    >>> print(f"Model: {model.k} factors, {model.p} assets")
    
    Notes
    -----
    The configure() method allows specifying a single sampler (broadcast
    to all factors/assets) or a list of samplers (one per factor/asset).
    
    Factor covariance F is diagonal (factors are orthogonal).
    Idiosyncratic covariance D is diagonal (assets have independent idio risk).
    """
    
    def __init__(
        self,
        p: int,
        k: int,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the data sampler.
        
        Parameters
        ----------
        p : int
            Number of assets. Must be positive.
        k : int
            Number of factors. Must be positive.
        rng : np.random.Generator, optional
            Random number generator.
        """
        if p <= 0:
            raise ValueError(f"Number of assets must be positive, got {p}")
        if k <= 0:
            raise ValueError(f"Number of factors must be positive, got {k}")
        
        self.p = p
        self.k = k
        self._rng = rng if rng is not None else np.random.default_rng()
        
        # Sampler storage
        self._beta_samplers: List[SamplerCallable] = []
        self._factor_vol_samplers: List[SamplerCallable] = []
        self._idio_vol_samplers: List[SamplerCallable] = []
        
        self._configured = False
    
    def configure(
        self,
        beta: Union[SamplerCallable, List[SamplerCallable]],
        factor_vol: Union[SamplerCallable, List[SamplerCallable]],
        idio_vol: Union[SamplerCallable, List[SamplerCallable]]
    ) -> "DataSampler":
        """
        Configure the sampling distributions.
        
        Parameters
        ----------
        beta : SamplerCallable or List[SamplerCallable]
            Sampler(s) for factor loadings. Each factor's loadings are drawn
            independently. If a single sampler, it's broadcast to all k factors.
        factor_vol : SamplerCallable or List[SamplerCallable]
            Sampler(s) for factor volatilities (sqrt of diagonal of F).
            If a single sampler, broadcast to all k factors.
        idio_vol : SamplerCallable or List[SamplerCallable]
            Sampler(s) for idiosyncratic volatilities (sqrt of diagonal of D).
            If a single sampler, broadcast to all p assets.
        
        Returns
        -------
        DataSampler
            Self, for method chaining.
        
        Raises
        ------
        ValueError
            If list lengths don't match expected dimensions.
        TypeError
            If arguments are neither callable nor lists.
        
        Examples
        --------
        >>> # Single samplers (broadcast)
        >>> sampler.configure(
        ...     beta=factory.create("normal", mean=0, std=1),
        ...     factor_vol=factory.create("constant", value=0.2),
        ...     idio_vol=factory.create("uniform", low=0.03, high=0.08)
        ... )
        >>> 
        >>> # Explicit per-factor samplers
        >>> sampler.configure(
        ...     beta=[
        ...         factory.create("normal", mean=1.0, std=0.2),  # Market factor
        ...         factory.create("normal", mean=0.0, std=0.5),  # Size factor
        ...     ],
        ...     factor_vol=[
        ...         factory.create("constant", value=0.20),  # Market vol
        ...         factory.create("constant", value=0.10),  # Size vol
        ...     ],
        ...     idio_vol=factory.create("uniform", low=0.02, high=0.05)
        ... )
        """
        self._beta_samplers = self._resolve_samplers(beta, self.k, "beta")
        self._factor_vol_samplers = self._resolve_samplers(
            factor_vol, self.k, "factor_vol"
        )
        self._idio_vol_samplers = self._resolve_samplers(
            idio_vol, self.p, "idio_vol"
        )
        
        self._configured = True
        return self
    
    def generate(self) -> FactorModelData:
        """
        Generate a factor model by sampling from configured distributions.
        
        Returns
        -------
        FactorModelData
            A complete factor model with randomly sampled parameters.
        
        Raises
        ------
        RuntimeError
            If configure() has not been called.
        
        Examples
        --------
        >>> sampler.configure(...)
        >>> model = sampler.generate()
        >>> print(f"Generated model with k={model.k}, p={model.p}")
        """
        if not self._configured:
            raise RuntimeError(
                "DataSampler not configured. Call configure() first."
            )
        
        # Sample factor loadings: each row is drawn from its sampler
        # B has shape (k, p)
        B = np.vstack([
            sampler(self.p) for sampler in self._beta_samplers
        ])
        
        # Sample factor volatilities and construct F (diagonal)
        # Each sampler returns a single value (sample size 1)
        factor_vols = np.array([
            sampler(1)[0] for sampler in self._factor_vol_samplers
        ])
        F = np.diag(factor_vols ** 2)  # Variance = vol^2
        
        # Sample idiosyncratic volatilities and construct D (diagonal)
        idio_vols = np.array([
            sampler(1)[0] for sampler in self._idio_vol_samplers
        ])
        D = np.diag(idio_vols ** 2)
        
        return FactorModelData(B=B, F=F, D=D)
    
    def _resolve_samplers(
        self,
        sampler_spec: Union[SamplerCallable, List[SamplerCallable]],
        target_len: int,
        name: str
    ) -> List[SamplerCallable]:
        """
        Resolve a sampler specification into a list of samplers.
        
        Handles broadcasting (single sampler -> list) and validation.
        """
        if isinstance(sampler_spec, list):
            if len(sampler_spec) != target_len:
                raise ValueError(
                    f"{name}: expected list of length {target_len}, "
                    f"got {len(sampler_spec)}"
                )
            # Validate all are callable
            for i, s in enumerate(sampler_spec):
                if not callable(s):
                    raise TypeError(
                        f"{name}[{i}] is not callable: {type(s)}"
                    )
            return sampler_spec
        
        elif callable(sampler_spec):
            # Broadcast single sampler to all elements
            return [sampler_spec] * target_len
        
        else:
            raise TypeError(
                f"{name} must be callable or list of callables, "
                f"got {type(sampler_spec)}"
            )
