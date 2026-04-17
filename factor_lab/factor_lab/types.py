"""
types.py - Deprecated backward-compatibility shim.

Import directly from the canonical modules instead:
    from factor_lab.factor_types import FactorModelData
    from factor_lab.decomposition import svd_decomposition
    from factor_lab.simulation import ReturnsSimulator
"""

import warnings

warnings.warn(
    "factor_lab.types is deprecated and will be removed in a future release. "
    "Import FactorModelData from factor_lab.factor_types, "
    "svd_decomposition from factor_lab.decomposition, "
    "and ReturnsSimulator from factor_lab.simulation.",
    DeprecationWarning,
    stacklevel=2,
)

from .factor_types import FactorModelData
from .decomposition import svd_decomposition
from .simulation import ReturnsSimulator

__all__ = [
    'FactorModelData',
    'svd_decomposition',
    'ReturnsSimulator',
]
