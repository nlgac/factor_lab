# factor_lab/__init__.py

# Core data structures
from .factor_types import FactorModelData

# Estimation
from .estimation import svd_decomposition

# Simulation
from .simulation import ReturnsSimulator

# Distributions
from .distributions import create_sampler, list_available_distributions

# I/O
from .model_io import save_model, load_model

# For backward compatibility, you can alias:
from .types import DistributionFactory  # Import the CLASS from types.py

__all__ = [
    'FactorModelData',
    'svd_decomposition',
    'ReturnsSimulator',
    'create_sampler',
    'DistributionFactory',  # Alias for compatibility
    'save_model',
    'load_model',
    'list_available_distributions',
]