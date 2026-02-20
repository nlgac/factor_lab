"""
types.py - Core Data Structures

Pure data containers with no business logic.

This module contains only dataclasses and type definitions.
For algorithms and behavior, see:
- estimation.py: Model estimation (SVD, etc.)
- simulation.py: Returns simulation
- distributions.py: Distribution samplers
- io.py: File I/O
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FactorModelData:
    """
    Factor model representation: r = B'f + e
    
    This is a pure data structure representing a factor model.
    All parameters are numpy arrays.
    
    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Factor loadings matrix
        - k: number of factors
        - p: number of assets
    F : np.ndarray, shape (k, k)
        Factor covariance matrix (must be positive definite)
    D : np.ndarray, shape (p, p)
        Idiosyncratic covariance matrix (typically diagonal)
    factor_transform : np.ndarray, optional
        Optional transformation matrix for factors
    idio_transform : np.ndarray, optional
        Optional transformation matrix for idiosyncratic component
        
    Attributes
    ----------
    k : int
        Number of factors (derived from B.shape[0])
    p : int
        Number of assets (derived from B.shape[1])
        
    Examples
    --------
    Create a simple 3-factor model for 100 assets:
        >>> import numpy as np
        >>> k, p = 3, 100
        >>> B = np.random.randn(k, p)
        >>> F = np.diag([0.04, 0.02, 0.01])
        >>> D = np.eye(p) * 0.01
        >>> model = FactorModelData(B=B, F=F, D=D)
    
    Access dimensions:
        >>> model.k  # 3 factors
        >>> model.p  # 100 assets
    
    Compute implied covariance:
        >>> Sigma = model.implied_covariance()
        >>> Sigma.shape  # (100, 100)
    
    Notes
    -----
    The factor model represents asset returns as:
        r_t = B' f_t + ε_t
    
    Where:
    - r_t: (p×1) vector of asset returns at time t
    - f_t: (k×1) vector of factor returns at time t, f_t ~ (0, F)
    - ε_t: (p×1) vector of idiosyncratic returns, ε_t ~ (0, D)
    - B: (k×p) matrix of factor loadings
    
    The implied covariance of returns is:
        Σ = B' F B + D
    """
    
    B: np.ndarray  # (k, p) factor loadings
    F: np.ndarray  # (k, k) factor covariance
    D: np.ndarray  # (p, p) idiosyncratic covariance
    factor_transform: Optional[np.ndarray] = None
    idio_transform: Optional[np.ndarray] = None
    
    @property
    def k(self) -> int:
        """Number of factors."""
        return self.B.shape[0]
    
    @property
    def p(self) -> int:
        """Number of assets."""
        return self.B.shape[1]
    
    def implied_covariance(self) -> np.ndarray:
        """
        Compute implied covariance matrix: Σ = B' F B + D
        
        Returns
        -------
        Sigma : np.ndarray, shape (p, p)
            Implied covariance matrix of returns
            
        Examples
        --------
        >>> Sigma = model.implied_covariance()
        >>> # Verify it's positive definite
        >>> eigvals = np.linalg.eigvalsh(Sigma)
        >>> assert np.all(eigvals > 0)
        
        Notes
        -----
        This is the theoretical covariance of returns r_t = B'f_t + ε_t
        when f_t ~ N(0, F) and ε_t ~ N(0, D) are independent.
        """
        return self.B.T @ self.F @ self.B + self.D


# Note: This types.py is now CLEAN - just the FactorModelData dataclass
# All other functionality has been moved to appropriate modules:
#
# - estimation.py: svd_decomposition
# - simulation.py: ReturnsSimulator  
# - distributions.py: create_sampler (replaces DistributionFactory)
# - io.py: save_model, load_model
#
# For backward compatibility during transition period, you can add re-exports:
#
# from .estimation import svd_decomposition
# from .simulation import ReturnsSimulator
# from .distributions import create_sampler as DistributionFactory
# from .io import save_model
#
# But imports should be updated to use the new modules directly.
