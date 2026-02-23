"""
types.py - Backward Compatibility Layer
"""

from .factor_types import FactorModelData
from .decomposition import svd_decomposition
from .simulation import ReturnsSimulator

__all__ = [
    'FactorModelData',
    'svd_decomposition', 
    'ReturnsSimulator',
]
