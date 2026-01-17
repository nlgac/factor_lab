"""
conftest.py - Pytest Configuration and Shared Fixtures

This file contains fixtures used across all test modules. Fixtures are
organized by category:
- Random number generators (for reproducibility)
- Factor models (various configurations)
- Samplers and factories
"""

import pytest
import numpy as np

from factor_lab import (
    FactorModelData,
    DistributionFactory,
    CovarianceTransform,
    TransformType,
)


# =============================================================================
# RANDOM NUMBER GENERATORS
# =============================================================================

@pytest.fixture
def rng():
    """
    Provide a seeded random number generator for reproducible tests.
    
    All tests should use this fixture (or derive from it) to ensure
    reproducibility across runs.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture
def rng_alternate():
    """Alternate RNG with different seed for comparison tests."""
    return np.random.default_rng(seed=12345)


# =============================================================================
# FACTOR MODELS - SIMPLE
# =============================================================================

@pytest.fixture
def simple_diagonal_model():
    """
    A minimal 10-asset, 2-factor model with diagonal covariances.
    
    This is the simplest case where both F and D are diagonal,
    allowing the O(n) simulation path.
    
    Structure:
    - Factor 1 affects assets 0-4 (loadings = 1.0)
    - Factor 2 affects assets 5-9 (loadings = 1.0)
    - Factor vols: 20% and 30%
    - Idio vol: 10% for all assets
    """
    p, k = 10, 2
    
    # Block-diagonal loadings
    B = np.zeros((k, p))
    B[0, :5] = 1.0   # Factor 1 -> first 5 assets
    B[1, 5:] = 1.0   # Factor 2 -> last 5 assets
    
    # Factor covariance (diagonal): variances = vol^2
    F = np.diag([0.04, 0.09])  # Vols: 0.20, 0.30
    
    # Idiosyncratic covariance (diagonal)
    D = np.diag(np.full(p, 0.01))  # Vol: 0.10
    
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def simple_model_with_transforms(simple_diagonal_model):
    """
    Same as simple_diagonal_model but with pre-computed transforms.
    
    This simulates the output of svd_decomposition which provides
    transforms for efficient simulation.
    """
    model = simple_diagonal_model
    
    # Create diagonal transforms (as SVD would)
    factor_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(model.F)),
        transform_type=TransformType.DIAGONAL
    )
    
    idio_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(model.D)),
        transform_type=TransformType.DIAGONAL
    )
    
    return FactorModelData(
        B=model.B,
        F=model.F,
        D=model.D,
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )


# =============================================================================
# FACTOR MODELS - MEDIUM COMPLEXITY
# =============================================================================

@pytest.fixture
def medium_model(rng):
    """
    A 50-asset, 3-factor model with random loadings.
    
    Loadings are drawn from N(0, 1), providing a more realistic
    test case than the block-diagonal simple model.
    """
    p, k = 50, 3
    
    B = rng.standard_normal((k, p))
    F = np.diag([0.04, 0.02, 0.01])  # Decreasing factor variances
    D = np.diag(rng.uniform(0.005, 0.02, p))  # Heterogeneous idio
    
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def correlated_factors_model(rng):
    """
    A model with non-diagonal factor covariance (correlated factors).
    
    This tests the dense (Cholesky) simulation path.
    """
    p, k = 20, 2
    
    B = rng.standard_normal((k, p))
    
    # Non-diagonal F: factors are correlated
    F = np.array([
        [0.04, 0.01],
        [0.01, 0.02]
    ])
    
    D = np.diag(np.full(p, 0.01))
    
    return FactorModelData(B=B, F=F, D=D)


# =============================================================================
# FACTOR MODELS - SPECIAL CASES
# =============================================================================

@pytest.fixture
def single_factor_model():
    """A minimal single-factor model (like CAPM)."""
    p, k = 30, 1
    
    B = np.ones((k, p))  # All assets have beta = 1
    F = np.array([[0.04]])  # Market vol = 20%
    D = np.diag(np.full(p, 0.01))
    
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def large_model(rng):
    """
    A large 500-asset, 10-factor model for performance testing.
    
    Use sparingly as tests with this fixture may be slow.
    """
    p, k = 500, 10
    
    B = rng.standard_normal((k, p)) * 0.5
    F = np.diag(np.exp(np.linspace(np.log(0.05), np.log(0.005), k)))
    D = np.diag(rng.uniform(0.002, 0.01, p))
    
    return FactorModelData(B=B, F=F, D=D)


# =============================================================================
# SAMPLERS AND FACTORIES
# =============================================================================

@pytest.fixture
def factory(rng):
    """A DistributionFactory using the seeded RNG."""
    return DistributionFactory(rng=rng)


@pytest.fixture
def normal_sampler(factory):
    """A standard normal sampler."""
    return factory.create("normal", mean=0.0, std=1.0)


@pytest.fixture
def student_t_sampler(factory):
    """A Student's t sampler with df=5 (moderate fat tails)."""
    return factory.create("student_t", df=5)


# =============================================================================
# RETURNS DATA
# =============================================================================

@pytest.fixture
def sample_returns(rng):
    """
    Synthetic returns data for SVD decomposition tests.
    
    Shape: (1000, 50) - 1000 time periods, 50 assets.
    """
    T, p = 1000, 50
    k = 3
    
    # Generate with known factor structure
    true_factors = rng.standard_normal((T, k)) * np.array([0.2, 0.1, 0.05])
    true_loadings = rng.standard_normal((k, p))
    true_idio = rng.standard_normal((T, p)) * 0.02
    
    returns = true_factors @ true_loadings + true_idio
    
    return returns


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def tolerance():
    """Standard numerical tolerance for float comparisons."""
    return {"rtol": 1e-5, "atol": 1e-8}


@pytest.fixture
def large_sample_tolerance():
    """Looser tolerance for statistical convergence tests."""
    return {"rtol": 0.05, "atol": 0.01}
