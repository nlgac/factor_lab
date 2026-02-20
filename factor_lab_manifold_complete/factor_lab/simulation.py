"""
simulation.py - Returns Simulation (Old API - Backward Compatible)

This module provides the OLD API for ReturnsSimulator to maintain
backward compatibility with existing tests.
"""

from typing import Optional, Dict
import numpy as np

from .factor_types import FactorModelData

__all__ = ['ReturnsSimulator']


class ReturnsSimulator:
    """
    Simulate returns from factor model (OLD API).
    
    This class maintains backward compatibility:
        simulator = ReturnsSimulator(model, rng=rng)
        results = simulator.simulate(n_periods)
    
    Parameters
    ----------
    model : FactorModelData
        The factor model
    rng : np.random.Generator, optional
        Random number generator
    """
    
    def __init__(
        self,
        model: FactorModelData,
        rng: Optional[np.random.Generator] = None
    ):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def simulate(
        self,
        n_periods: int,
        factor_samplers=None,
        idio_samplers=None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns using normal distributions.
        
        Parameters
        ----------
        n_periods : int
            Number of time periods to simulate
        factor_samplers : optional
            Ignored (for compatibility)
        idio_samplers : optional
            Ignored (for compatibility)
        
        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'security_returns': (n_periods, p)
            - 'factor_returns': (n_periods, k)
            - 'idio_returns': (n_periods, p)
        """
        k, p = self.model.k, self.model.p
        
        # Use Cholesky for proper covariance
        F_chol = np.linalg.cholesky(self.model.F)
        D_sqrt = np.sqrt(np.diag(self.model.D))
        
        # Sample factor returns: f ~ N(0, F)
        factor_returns = self.rng.normal(0, 1, (n_periods, k)) @ F_chol.T
        
        # Sample idiosyncratic returns: ε ~ N(0, D)
        idio_returns = self.rng.normal(0, 1, (n_periods, p)) * D_sqrt
        
        # Combine: r = B'f + ε
        security_returns = factor_returns @ self.model.B + idio_returns
        
        return {
            'security_returns': security_returns,
            'factor_returns': factor_returns,
            'idio_returns': idio_returns
        }
