import pytest
import numpy as np
from factor_lab import svd_decomposition, ReturnsSimulator, FactorModelData

def test_svd_recovery():
    """
    Test that SVD correctly recovers factors from a known structure.
    
    CRITICAL MATH NOTE:
    For PCA/SVD eigenvalues to match the generating factor variances,
    the generating Loadings matrix B must have orthonormal rows.
    Otherwise, the eigenvalues absorb the magnitude of B.
    """
    # 1. Setup
    T, p = 5000, 50 # Increase size for statistical stability
    k = 3
    rng = np.random.default_rng(42)

    # 2. Create True Factors (Variances: 100, 25, 1)
    vars_true = np.array([100.0, 25.0, 1.0])
    # Generate orthogonal factors to prevent random correlation noise
    # Start with random, then scale
    F_raw = rng.standard_normal((T, k))
    F_Q, _ = np.linalg.qr(F_raw) # Orthogonal columns
    # Scale columns to have exact desired variance
    factors = F_Q * np.sqrt(vars_true * (T-1))

    # 3. Create Orthonormal Loadings B
    # If B rows are orthonormal (B @ B.T = I), then Cov(R) eigenvalues = Vars_true
    B_raw = rng.standard_normal((p, k)) 
    Q_B, _ = np.linalg.qr(B_raw)
    B_true = Q_B.T # Shape (k, p), rows are orthonormal

    # 4. Generate Returns
    # R = F @ B
    returns = factors @ B_true
    
    # Add tiny noise to avoid singular matrices, but small enough not to distort eigenvalues
    returns += rng.normal(0, 0.001, (T, p))

    # 5. Run SVD Decomposition
    model = svd_decomposition(returns, k=k)

    # 6. Assertions
    vars_est = np.diag(model.F)
    
    # Comparison
    print(f"\nTrue Vars: {vars_true}")
    print(f"Est Vars:  {vars_est}")
    
    # We expect very close match because we controlled orthogonality
    assert np.allclose(vars_est, vars_true, rtol=0.1), \
        f"Variance Mismatch. Expected {vars_true}, Got {vars_est}"

def test_cached_roots_usage(caplog):
    """
    Verify that the Simulator uses the pre-computed transforms from SVD.
    """
    import logging
    from factor_lab import CovarianceTransform, TransformType
    
    # 1. Create dummy model with pre-computed transforms
    B = np.zeros((1, 5))
    F = np.eye(1)
    D = np.eye(5)
    
    # Create transforms (as SVD would)
    factor_transform = CovarianceTransform(
        matrix=np.ones(1),  # sqrt of diagonal
        transform_type=TransformType.DIAGONAL
    )
    idio_transform = CovarianceTransform(
        matrix=np.ones(5),  # sqrt of diagonal
        transform_type=TransformType.DIAGONAL
    )
    
    model = FactorModelData(
        B=B, F=F, D=D, 
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )
    
    # 2. Initialize Simulator
    sim = ReturnsSimulator(model)
    
    # The transforms should be used
    assert sim._factor_transform is not None
    assert sim._idio_transform is not None
    assert sim._factor_transform.is_diagonal
    assert sim._idio_transform.is_diagonal
    
    # Check that the transform matches our input
    assert np.array_equal(sim._factor_transform.matrix, np.ones(1))
    assert np.array_equal(sim._idio_transform.matrix, np.ones(5))
