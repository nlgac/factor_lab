"""
distributions.py - Distribution Samplers

Simple factory for creating distribution samplers.
Any callable with signature (n: int) -> np.ndarray works.

Design Philosophy
-----------------
- Universal Interface: Sampler = Callable[[int], np.ndarray]
- Maximum Decoupling: Simulator doesn't know distribution source
- Extensibility: Just write any callable, no registration needed
- Convenience: create_sampler() helper for numpy distributions

Examples
--------
Built-in numpy distributions:
    >>> rng = np.random.default_rng(42)
    >>> normal = create_sampler("normal", rng, loc=0, scale=1)
    >>> samples = normal(1000)
    
Custom scipy distribution:
    >>> from scipy.stats import levy_stable
    >>> levy = lambda n: levy_stable.rvs(alpha=1.5, beta=0, size=n)
    >>> samples = levy(1000)  # Just works!
    
Mixture model:
    >>> def mixture(n):
    ...     regime = rng.binomial(1, 0.95, n)
    ...     normal = rng.normal(0, 1, n)
    ...     crisis = rng.normal(0, 3, n)
    ...     return np.where(regime, normal, crisis)
    >>> samples = mixture(1000)
"""

from typing import Callable
import numpy as np

# Universal sampler interface
Sampler = Callable[[int], np.ndarray]

# Dispatch table for built-in distributions
# Maps distribution name to builder function
_SAMPLERS = {
    'normal': lambda rng, p: lambda n: rng.normal(
        p.get('loc', 0), 
        p.get('scale', 1), 
        n
    ),
    
    'student_t': lambda rng, p: lambda n: rng.standard_t(
        p['df'],  # Required
        n
    ),
    
    'uniform': lambda rng, p: lambda n: rng.uniform(
        p['low'],   # Required
        p['high'],  # Required
        n
    ),
    
    'beta': lambda rng, p: lambda n: rng.beta(
        p['a'],  # Required
        p['b'],  # Required
        n
    ),
    
    'exponential': lambda rng, p: lambda n: rng.exponential(
        p['scale'],  # Required
        n
    ),
    
    'gamma': lambda rng, p: lambda n: rng.gamma(
        p['shape'],  # Required
        p['scale'],  # Required
        n
    ),
    
    'constant': lambda rng, p: lambda n: np.full(
        n, 
        p['value']  # Required
    ),
}


def create_sampler(
    name: str,
    rng: np.random.Generator,
    **params
) -> Sampler:
    """
    Create a sampler for built-in numpy distributions.
    
    This is a CONVENIENCE function for common distributions.
    For scipy or custom distributions, just write your own callable.
    
    Parameters
    ----------
    name : str
        Distribution name. Available:
        - 'normal': Normal distribution (loc, scale)
        - 'student_t': Student's t distribution (df)
        - 'uniform': Uniform distribution (low, high)
        - 'beta': Beta distribution (a, b)
        - 'exponential': Exponential distribution (scale)
        - 'gamma': Gamma distribution (shape, scale)
        - 'constant': Constant value (value)
    rng : np.random.Generator
        Random number generator for reproducibility
    **params
        Distribution-specific parameters (see name descriptions)
        
    Returns
    -------
    sampler : Callable[[int], np.ndarray]
        Function that takes n (sample size) and returns n samples
        
    Raises
    ------
    ValueError
        If distribution name is unknown or required parameters are missing
        
    Examples
    --------
    Normal distribution with defaults:
        >>> rng = np.random.default_rng(42)
        >>> normal = create_sampler("normal", rng)
        >>> samples = normal(1000)  # N(0, 1)
    
    Normal distribution with custom parameters:
        >>> normal = create_sampler("normal", rng, loc=100, scale=15)
        >>> samples = normal(1000)  # N(100, 15²)
    
    Student's t distribution:
        >>> t_dist = create_sampler("student_t", rng, df=5)
        >>> samples = t_dist(1000)  # t(5)
    
    Uniform distribution:
        >>> uniform = create_sampler("uniform", rng, low=0, high=1)
        >>> samples = uniform(1000)  # U(0, 1)
    
    Notes
    -----
    For distributions not in the built-in list, or for custom logic,
    just write your own callable directly:
    
        >>> from scipy.stats import skewnorm
        >>> skew_sampler = lambda n: skewnorm.rvs(a=5, size=n)
        >>> samples = skew_sampler(1000)
    
    The simulator accepts ANY callable with signature (n) -> array.
    """
    if name not in _SAMPLERS:
        available = ', '.join(sorted(_SAMPLERS.keys()))
        raise ValueError(
            f"Unknown distribution '{name}'. "
            f"Available: {available}\n"
            f"For custom distributions, write your own callable."
        )
    
    try:
        return _SAMPLERS[name](rng, params)
    except KeyError as e:
        raise ValueError(
            f"Missing required parameter for '{name}': {e.args[0]}"
        ) from e


def list_available_distributions() -> list[str]:
    """
    List all built-in distributions.
    
    Returns
    -------
    distributions : list of str
        Sorted list of available distribution names
        
    Examples
    --------
    >>> list_available_distributions()
    ['beta', 'constant', 'exponential', 'gamma', 'normal', 'student_t', 'uniform']
    """
    return sorted(_SAMPLERS.keys())


# Convenience aliases for common cases
def normal_sampler(
    rng: np.random.Generator, 
    loc: float = 0, 
    scale: float = 1
) -> Sampler:
    """Convenience function for normal distribution."""
    return create_sampler("normal", rng, loc=loc, scale=scale)


def t_sampler(rng: np.random.Generator, df: float) -> Sampler:
    """Convenience function for Student's t distribution."""
    return create_sampler("student_t", rng, df=df)


def uniform_sampler(
    rng: np.random.Generator, 
    low: float, 
    high: float
) -> Sampler:
    """Convenience function for uniform distribution."""
    return create_sampler("uniform", rng, low=low, high=high)
