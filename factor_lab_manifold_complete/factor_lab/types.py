"""
Backward compatibility wrapper for types.py
"""

from .factor_types import FactorModelData
from .estimation import svd_decomposition
from .simulation import ReturnsSimulator
from .model_io import save_model
from .distributions import create_sampler
import numpy as np


class DistributionFactory:
    """
    Backward compatibility wrapper for DistributionFactory.
    
    Old code: factory = DistributionFactory(rng)
    New code: Uses create_sampler function directly
    """
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
    
    def create(self, name: str, **params):
        """Create a sampler (delegates to create_sampler)"""
        return create_sampler(name, self.rng, **params)


__all__ = [
    'FactorModelData',
    'svd_decomposition',
    'ReturnsSimulator',
    'DistributionFactory',
    'save_model',
]