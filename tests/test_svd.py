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
    Verify that the Simulator uses the pre-computed roots from SVD.
    """
    import logging
    
    # 1. Create dummy SVD model
    B = np.zeros((1, 5))
    F = np.eye(1)
    D = np.eye(5)
    # Manually attach roots
    F_sqrt = np.ones(1) 
    D_sqrt = np.ones(5)
    
    model = FactorModelData(B, F, D, F_sqrt=F_sqrt, D_sqrt=D_sqrt)
    
    # 2. Initialize Simulator
    sim = ReturnsSimulator(model)
    
    # The 'True' flag indicates diagonal/scalar transform was selected
    assert sim.F_diag is True
    assert sim.D_diag is True
    
    # Check that the transform matches our cached input
    assert np.array_equal(sim.F_tx, F_sqrt)
    assert np.array_equal(sim.D_tx, D_sqrt)