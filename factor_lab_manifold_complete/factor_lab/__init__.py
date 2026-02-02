"""Factor Lab Manifold Analysis - Standalone Package"""
__version__ = "2.2.0"

from .types import (
    FactorModelData, 
    svd_decomposition,
    ReturnsSimulator,
    DistributionFactory,
    save_model
)
from .analysis import SimulationAnalysis, SimulationContext
from .analyses import (
    Analyses,
    ManifoldDistanceAnalysis,
    ImplicitEigenAnalysis,
    EigenvectorAlignment,
)

__all__ = [
    'FactorModelData', 'svd_decomposition', 'ReturnsSimulator',
    'DistributionFactory', 'save_model', 'SimulationAnalysis',
    'SimulationContext', 'Analyses', 'ManifoldDistanceAnalysis',
    'ImplicitEigenAnalysis', 'EigenvectorAlignment',
]
