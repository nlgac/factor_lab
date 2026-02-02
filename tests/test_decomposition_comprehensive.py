"""
test_decomposition_comprehensive.py - Comprehensive Test Suite
==============================================================

Additional tests to ensure complete coverage of decomposition module.
Tests edge cases, numerical stability, and all parameter combinations.
"""

import pytest
import numpy as np
from scipy.sparse.linalg import LinearOperator

# Note: These imports assume factor_lab package structure
# Adjust if needed for your environment
try:
    from factor_lab.decomposition import (
        svd_decomposition,
        pca_decomposition,
        compute_explained_variance,
        select_k_by_variance,
        _compute_svd
    )
    from factor_lab.types import FactorModelData, CovarianceTransform, TransformType
except ImportError:
    # Fallback for testing outside package
    import sys
    sys.path.insert(0, '.')


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def small_returns():
    """Small dataset for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def medium_returns():
    """Medium dataset for testing."""
    np.random.seed(42)
    return np.random.randn(500, 50)


@pytest.fixture
def large_returns():
    """Large dataset for edge case testing."""
    np.random.seed(42)
    return np.random.randn(1000, 200)


# =============================================================================
# SVD Decomposition Tests
# =============================================================================

class TestSVDDecompositionComprehensive:
    """Comprehensive tests for SVD decomposition."""
    
    def test_demean_parameter(self, medium_returns):
        """Test demean parameter works correctly."""
        # With demeaning
        model_demean = svd_decomposition(medium_returns, k=3, demean=True)
        
        # Without demeaning
        model_no_demean = svd_decomposition(medium_returns, k=3, demean=False)
        
        # They should be different
        assert not np.allclose(model_demean.B, model_no_demean.B)
    
    def test_center_parameter(self, medium_returns):
        """Test center parameter works correctly."""
        # With centering
        model_center = svd_decomposition(medium_returns, k=3, center=True)
        
        # Without centering
        model_no_center = svd_decomposition(medium_returns, k=3, center=False)
        
        # They should be different
        assert not np.allclose(model_center.B, model_no_center.B)
    
    def test_transforms_populated(self, medium_returns):
        """Test that transforms are properly populated."""
        model = svd_decomposition(medium_returns, k=3)
        
        assert model.factor_transform is not None
        assert model.idio_transform is not None
        assert isinstance(model.factor_transform, CovarianceTransform)
        assert isinstance(model.idio_transform, CovarianceTransform)
        # Check transform_type is TransformType.DIAGONAL enum
        assert model.factor_transform.transform_type == TransformType.DIAGONAL
        assert model.idio_transform.transform_type == TransformType.DIAGONAL
    
    def test_transform_shapes(self, medium_returns):
        """Test transform shapes are correct."""
        k = 3
        model = svd_decomposition(medium_returns, k=k)
        
        T, p = medium_returns.shape
        
        # Diagonal transforms store just the diagonal as 1D vectors
        assert model.factor_transform.matrix.shape == (k,)  # 1D vector
        assert model.idio_transform.matrix.shape == (p,)    # 1D vector
    
    def test_transform_sqrt_property(self, medium_returns):
        """Test that transforms are sqrt of covariances."""
        model = svd_decomposition(medium_returns, k=3)
        
        # For diagonal transforms, matrix is 1D vector of diagonal elements
        # Squaring the vector gives back the diagonal of the covariance
        F_diag_reconstruct = model.factor_transform.matrix ** 2
        assert np.allclose(F_diag_reconstruct, np.diag(model.F), atol=1e-10)
        
        D_diag_reconstruct = model.idio_transform.matrix ** 2
        assert np.allclose(D_diag_reconstruct, np.diag(model.D), atol=1e-10)
    
    def test_k_bounds_lower(self, medium_returns):
        """Test k < 1 raises error."""
        with pytest.raises(ValueError, match="k must be in range"):
            svd_decomposition(medium_returns, k=0)
    
    def test_k_bounds_upper(self, medium_returns):
        """Test k > max raises error."""
        T, p = medium_returns.shape
        max_k = min(T, p)
        with pytest.raises(ValueError, match="k must be in range"):
            svd_decomposition(medium_returns, k=max_k + 1)
    
    def test_1d_input_raises(self):
        """Test 1D input raises proper error."""
        returns_1d = np.random.randn(100)
        with pytest.raises(ValueError, match="2D"):
            svd_decomposition(returns_1d, k=2)
    
    def test_3d_input_raises(self):
        """Test 3D input raises proper error."""
        returns_3d = np.random.randn(100, 10, 5)
        with pytest.raises(ValueError, match="2D"):
            svd_decomposition(returns_3d, k=2)
    
    def test_single_factor(self, medium_returns):
        """Test extraction of single factor."""
        model = svd_decomposition(medium_returns, k=1)
        
        assert model.B.shape[0] == 1
        assert model.F.shape == (1, 1)
    
    def test_max_factors(self, small_returns):
        """Test extraction of maximum possible factors."""
        T, p = small_returns.shape
        max_k = min(T, p)
        
        model = svd_decomposition(small_returns, k=max_k)
        
        assert model.B.shape[0] == max_k
        assert model.F.shape == (max_k, max_k)
    
    def test_variance_decomposition(self, medium_returns):
        """Test that variance is properly decomposed."""
        model = svd_decomposition(medium_returns, k=5)
        
        # Total variance from returns
        total_var = np.var(medium_returns, axis=0, ddof=1).sum()
        
        # Systematic + idiosyncratic
        sys_var = np.trace(model.F)
        idio_var = np.trace(model.D)
        model_total = sys_var + idio_var
        
        # Should match (approximately due to centering)
        assert np.abs(total_var - model_total) / total_var < 0.01
    
    def test_explained_variance_integration(self, medium_returns):
        """Test that explained variance function works with svd_decomposition."""
        model = svd_decomposition(medium_returns, k=3)
        
        explained = compute_explained_variance(model)
        
        assert 0 < explained < 1
        assert isinstance(explained, float)
    
    def test_reproducibility(self, medium_returns):
        """Test that decomposition is deterministic."""
        model1 = svd_decomposition(medium_returns, k=3)
        model2 = svd_decomposition(medium_returns, k=3)
        
        assert np.allclose(model1.B, model2.B)
        assert np.allclose(model1.F, model2.F)
        assert np.allclose(model1.D, model2.D)


# =============================================================================
# PCA Decomposition Tests
# =============================================================================

class TestPCADecompositionComprehensive:
    """Comprehensive tests for PCA decomposition."""
    
    def test_numpy_method(self):
        """Test numpy method explicitly."""
        np.random.seed(42)
        cov = np.random.randn(20, 20)
        cov = cov @ cov.T  # Make symmetric PSD
        
        B, F = pca_decomposition(cov, k=5, method='numpy')
        
        assert B.shape == (5, 20)
        assert F.shape == (5, 5)
    
    def test_scipy_method(self):
        """Test scipy method explicitly."""
        np.random.seed(42)
        cov = np.random.randn(20, 20)
        cov = cov @ cov.T
        
        B, F = pca_decomposition(cov, k=5, method='scipy')
        
        assert B.shape == (5, 20)
        assert F.shape == (5, 5)
    
    def test_numpy_scipy_equivalence(self):
        """Test that numpy and scipy methods give similar results."""
        np.random.seed(42)
        cov = np.random.randn(20, 20)
        cov = cov @ cov.T
        
        B_np, F_np = pca_decomposition(cov, k=5, method='numpy')
        B_sp, F_sp = pca_decomposition(cov, k=5, method='scipy')
        
        # Eigenvalues should match
        assert np.allclose(np.diag(F_np), np.diag(F_sp), rtol=0.01)
        
        # Eigenvectors may differ by sign, check absolute values
        assert np.allclose(np.abs(B_np), np.abs(B_sp), atol=0.1)
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        cov = np.eye(10)
        
        with pytest.raises(ValueError, match="Unknown method"):
            pca_decomposition(cov, k=3, method='invalid')
    
    def test_k_bounds_lower_pca(self):
        """Test k < 1 raises error."""
        cov = np.eye(10)
        
        with pytest.raises(ValueError, match="k must be in range"):
            pca_decomposition(cov, k=0)
    
    def test_k_bounds_upper_pca(self):
        """Test k > p raises error."""
        cov = np.eye(10)
        
        with pytest.raises(ValueError, match="k must be in range"):
            pca_decomposition(cov, k=11)
    
    def test_linear_operator_input(self):
        """Test that LinearOperator input works."""
        p = 50
        
        # Create LinearOperator
        def matvec(v):
            return v  # Identity operator
        
        A = LinearOperator(shape=(p, p), matvec=matvec)
        
        B, F = pca_decomposition(A, k=5, method='numpy')
        
        assert B.shape == (5, p)
        assert F.shape == (5, 5)
    
    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are in descending order."""
        np.random.seed(42)
        cov = np.diag([5, 4, 3, 2, 1, 0.5, 0.1])
        
        B, F = pca_decomposition(cov, k=5)
        
        eigenvalues = np.diag(F)
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


# =============================================================================
# Explained Variance Tests
# =============================================================================

class TestExplainedVarianceComprehensive:
    """Comprehensive tests for explained variance."""
    
    def test_zero_factor_variance(self):
        """Test when factor variance is zero."""
        B = np.random.randn(3, 10)
        F = np.zeros((3, 3))
        D = np.eye(10) * 0.1
        
        model = FactorModelData(B=B, F=F, D=D)
        
        explained = compute_explained_variance(model)
        
        assert explained == 0.0
    
    def test_zero_idio_variance(self):
        """Test when idiosyncratic variance is zero."""
        B = np.random.randn(3, 10)
        F = np.eye(3) * 1.0
        D = np.zeros((10, 10))
        
        model = FactorModelData(B=B, F=F, D=D)
        
        explained = compute_explained_variance(model)
        
        assert explained == 1.0
    
    def test_both_zero_variance(self):
        """Test when both variances are zero."""
        B = np.random.randn(3, 10)
        F = np.zeros((3, 3))
        D = np.zeros((10, 10))
        
        model = FactorModelData(B=B, F=F, D=D)
        
        explained = compute_explained_variance(model)
        
        assert explained == 0.0
    
    def test_explained_variance_range(self, medium_returns):
        """Test explained variance is in [0, 1]."""
        for k in [1, 3, 5, 10]:
            model = svd_decomposition(medium_returns, k=k)
            explained = compute_explained_variance(model)
            
            assert 0 <= explained <= 1
    
    def test_explained_variance_increases_with_k(self, medium_returns):
        """Test that explained variance increases with k."""
        explained_vars = []
        
        for k in [1, 3, 5, 7, 10]:
            model = svd_decomposition(medium_returns, k=k)
            explained = compute_explained_variance(model)
            explained_vars.append(explained)
        
        # Should be monotonically increasing
        for i in range(len(explained_vars) - 1):
            assert explained_vars[i] <= explained_vars[i+1]


# =============================================================================
# Select K by Variance Tests
# =============================================================================

class TestSelectKComprehensive:
    """Comprehensive tests for select_k_by_variance."""
    
    def test_1d_input_raises(self):
        """Test that 1D input raises proper error."""
        returns = np.random.randn(100)
        
        with pytest.raises(ValueError, match="2D"):
            select_k_by_variance(returns)
    
    def test_3d_input_raises(self):
        """Test that 3D input raises proper error."""
        returns = np.random.randn(100, 10, 5)
        
        with pytest.raises(ValueError, match="2D"):
            select_k_by_variance(returns)
    
    def test_target_50_percent(self, medium_returns):
        """Test selection with 50% target."""
        k = select_k_by_variance(medium_returns, target_explained=0.5)
        
        assert k >= 1
        assert isinstance(k, int)
    
    def test_target_90_percent(self, medium_returns):
        """Test selection with 90% target."""
        k = select_k_by_variance(medium_returns, target_explained=0.9)
        
        assert k >= 1
        assert isinstance(k, int)
    
    def test_higher_target_more_factors(self, medium_returns):
        """Test that higher target requires more factors."""
        k_50 = select_k_by_variance(medium_returns, target_explained=0.5)
        k_90 = select_k_by_variance(medium_returns, target_explained=0.9)
        k_99 = select_k_by_variance(medium_returns, target_explained=0.99)
        
        assert k_50 <= k_90 <= k_99
    
    def test_max_k_constraint(self, medium_returns):
        """Test that max_k constrains result."""
        k = select_k_by_variance(medium_returns, target_explained=0.99, max_k=5)
        
        assert k <= 5
    
    def test_achieves_target(self, medium_returns):
        """Test that selected k achieves target variance."""
        target = 0.8
        k = select_k_by_variance(medium_returns, target_explained=target)
        
        # Extract model and check
        model = svd_decomposition(medium_returns, k=k)
        achieved = compute_explained_variance(model)
        
        # Should achieve at least target (or close)
        assert achieved >= target - 0.05


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_svd_to_pca_consistency(self, medium_returns):
        """Test that SVD and PCA give consistent results."""
        # SVD decomposition
        model_svd = svd_decomposition(medium_returns, k=5)
        
        # PCA on sample covariance
        cov = np.cov(medium_returns, rowvar=False)
        B_pca, F_pca = pca_decomposition(cov, k=5)
        
        # Eigenvalues should be similar (not exact due to different algorithms)
        evs_svd = np.sort(np.diag(model_svd.F))[::-1]
        evs_pca = np.sort(np.diag(F_pca))[::-1]
        
        assert np.allclose(evs_svd, evs_pca, rtol=0.1)
    
    def test_select_then_decompose(self, medium_returns):
        """Test selecting k then decomposing."""
        # Select k
        k = select_k_by_variance(medium_returns, target_explained=0.8)
        
        # Decompose with selected k
        model = svd_decomposition(medium_returns, k=k)
        
        # Check variance achieved
        explained = compute_explained_variance(model)
        assert explained >= 0.75  # Should be close to target
    
    def test_full_pipeline(self, large_returns):
        """Test complete pipeline: select k → decompose → validate."""
        # Step 1: Select k
        k = select_k_by_variance(large_returns, target_explained=0.85, max_k=20)
        
        # Step 2: Decompose
        model = svd_decomposition(large_returns, k=k)
        
        # Step 3: Validate
        explained = compute_explained_variance(model)
        
        # Should have reasonable properties
        assert 1 <= k <= 20
        # Random data won't achieve 85% with just 20 factors - adjust expectation
        assert 0.10 <= explained <= 1.0  # At least 10% explained
        assert model.B.shape == (k, large_returns.shape[1])
        assert model.factor_transform is not None
        assert model.idio_transform is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
