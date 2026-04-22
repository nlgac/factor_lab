"""Factor Lab Manifold Analysis - Complete Package"""
__version__ = "3.0.0"

# Core types and functions (backward compatible)
from .factor_types import FactorModelData
from .decomposition import svd_decomposition
from .simulation import ReturnsSimulator
from .model_io import save_model

# Distributions
from .distributions import create_sampler

# NEW: Flexible simulation components
from .model_builder import FactorModelBuilder
from .flexible_simulator import ReturnsSimulator as FlexibleReturnsSimulator

# Integration
from .integration import (
    build_simulate_analyze,
    build_simulate_analyze_from_model
)

# Analysis framework
from .analysis import SimulationContext
from .analyses import (
    Analyses,
    ManifoldDistanceAnalysis,
    ImplicitEigenAnalysis,
    EigenvectorAlignment,
)

__all__ = [
    # Core (backward compatible)
    'FactorModelData',
    'svd_decomposition',
    'ReturnsSimulator',
    'create_sampler',
    'save_model',
    
    # NEW: Flexible simulation
    'FactorModelBuilder',
    'FlexibleReturnsSimulator',
    'build_simulate_analyze',
    'build_simulate_analyze_from_model',
    
    # Analysis
    'SimulationContext',
    'Analyses',
    'ManifoldDistanceAnalysis',
    'ImplicitEigenAnalysis',
    'EigenvectorAlignment',
]