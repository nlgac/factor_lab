"""
simulation.py - Legacy Returns Simulator (deprecated)

Use factor_lab.flexible_simulator.ReturnsSimulator instead.
"""

import warnings
from typing import Optional, Dict
import numpy as np

from .factor_types import FactorModelData

__all__ = ['ReturnsSimulator']


class ReturnsSimulator:
    """
    Simulate returns from a factor model (legacy API — deprecated).

    Use FlexibleReturnsSimulator from factor_lab.flexible_simulator instead:

        from factor_lab.flexible_simulator import ReturnsSimulator
        simulator = ReturnsSimulator(rng=rng)
        results = simulator.simulate(model, n_periods, factor_samplers, idio_sampler)
    """

    def __init__(
        self,
        model: FactorModelData,
        rng: Optional[np.random.Generator] = None
    ):
        warnings.warn(
            "factor_lab.simulation.ReturnsSimulator is deprecated. "
            "Use factor_lab.flexible_simulator.ReturnsSimulator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
