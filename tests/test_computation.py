import pytest
import numpy as np
from factor_lab import pca_decomposition, FactorOptimizer

def test_pca_method_equivalence(simple_model):
    """
    Verify that Scipy (sparse) and Numpy (dense) PCA methods 
    return mathematically equivalent results (Loadings and Variances).
    """
    # Construct true Covariance Matrix: M = B'FB + D
    M = (simple_model.B.T @ simple_model.F @ simple_model.B) + simple_model.D
    
    k = simple_model.k
    
    # 1. Run Numpy
    B_np, F_np = pca_decomposition(M, k=k, method='numpy')
    
    # 2. Run Scipy
    B_sp, F_sp = pca_decomposition(M, k=k, method='scipy')
    
    # 3. Assert Equivalence
    # Note: Eigenvectors can flip sign, so we check absolute values of B
    # or reconstructed covariance. Simplest is to check eigenvalues (F).
    
    # Check Eigenvalues (Factor Variances)
    assert np.allclose(np.diag(F_np), np.diag(F_sp)), "Eigenvalues mismatch between Numpy/Scipy"
    
    # Check Reconstruction Error consistency
    rec_np = B_np.T @ F_np @ B_np
    rec_sp = B_sp.T @ F_sp @ B_sp
    assert np.allclose(rec_np, rec_sp), "Reconstructed matrices mismatch"

def test_pca_sorting(simple_model):
    """Ensure factors are returned in descending order of variance."""
    M = np.diag([1.0, 5.0, 2.0]) # Variances: 1, 5, 2
    # Top 2 should be 5.0 and 2.0
    
    B, F = pca_decomposition(M, k=2, method='scipy')
    vars = np.diag(F)
    
    assert vars[0] > vars[1], "Factors not sorted descending"
    assert np.isclose(vars[0], 5.0)
    assert np.isclose(vars[1], 2.0)

def test_optimizer_basic(simple_model):
    """Test Minimum Variance with Fully Invested Constraint."""
    opt = FactorOptimizer(simple_model)
    
    # Constraint: Sum(w) == 1
    p = simple_model.p
    opt.add_eq(np.ones((1, p)), np.array([1.0]))
    
    # Solve
    res = opt.solve()
    
    assert res.solved is True
    assert np.isclose(res.weights.sum(), 1.0)
    assert res.risk > 0