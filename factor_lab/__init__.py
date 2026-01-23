"""
factor_lab - A Python Library for Factor Model Construction and Simulation
"""

__version__ = "2.1.0"

# =============================================================================
# CORE TYPES
# =============================================================================
from .types import (
    FactorModelData,
    OptimizationResult,
    Scenario,
    CovarianceValidationResult,
    CovarianceTransform,
    TransformType,
    SamplerCallable,
)

# =============================================================================
# DECOMPOSITION
# =============================================================================
from .decomposition import (
    svd_decomposition,
    pca_decomposition,
    compute_explained_variance,
    select_k_by_variance,
    PCAMethod,
)

# =============================================================================
# SIMULATION
# =============================================================================
from .simulation import (
    ReturnsSimulator,
    CovarianceValidator,
    simulate_returns,
)

# =============================================================================
# OPTIMIZATION
# =============================================================================
from .optimization import (
    FactorOptimizer,
    ScenarioBuilder,
    minimum_variance_portfolio,
)

# =============================================================================
# SAMPLERS
# =============================================================================
from .samplers import (
    DistributionFactory,
    DistributionRegistry,
    DataSampler,
    DistributionInfo,
)

# =============================================================================
# =============================================================================
# I/O
# =============================================================================
from .io import (
    save_model,
    load_model,
    ModelFormat,
)

# PUBLIC API
# =============================================================================
__all__ = [
    "__version__",
    "FactorModelData",
    "OptimizationResult",
    "Scenario",
    "CovarianceValidationResult",
    "CovarianceTransform",
    "TransformType",
    "SamplerCallable",
    "svd_decomposition",
    "pca_decomposition",
    "compute_explained_variance",
    "select_k_by_variance",
    "PCAMethod",
    "ReturnsSimulator",
    "CovarianceValidator",
    "simulate_returns",
    "FactorOptimizer",
    "ScenarioBuilder",
    "minimum_variance_portfolio",
    "DistributionFactory",
    "DistributionRegistry",
    "DataSampler",
    "DistributionInfo",
    "save_model",
    "load_model",
    "ModelFormat",
]