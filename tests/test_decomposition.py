"""
test_decomposition.py - Tests for PCA and SVD Factor Extraction

Tests cover:
- PCA decomposition from covariance matrices
- SVD decomposition from returns matrices
- Variance recovery accuracy
- Pre-computed transforms from SVD
- Helper functions (select_k_by_variance, etc.)
"""

import pytest
import numpy as np

from factor_lab import (
    pca_decomposition,
    svd_decomposition,
    compute_explained_variance,
    select_k_by_variance,
    FactorModelData,
)


class TestPCADecomposition:
    """Tests for pca_decomposition function."""
    
    def test_basic_extraction(self, rng):
        """Test basic PCA extraction from a covariance matrix."""
        # Create a covariance with known eigenstructure
        p, k = 50, 3
        
        # Create factors with known variances
        true_vecs = np.linalg.qr(rng.standard_normal((p, k)))[0]
        true_vals = np.array([10.0, 5.0, 1.0])
        
        cov = true_vecs @ np.diag(true_vals) @ true_vecs.T
        
        B, F = pca_decomposition(cov, k=k)
        
        # Check shapes
        assert B.shape == (k, p)
        assert F.shape == (k, k)
        
        # F should be diagonal
        assert np.allclose(F, np.diag(np.diag(F)))
        
        # Eigenvalues should match (up to ordering)
        recovered_vals = np.sort(np.diag(F))[::-1]
        expected_vals = np.sort(true_vals)[::-1]
        assert np.allclose(recovered_vals, expected_vals, rtol=1e-10)
    
    def test_scipy_method(self, rng):
        """Test PCA with scipy method gives same results as numpy."""
        p, k = 30, 5
        cov = rng.standard_normal((p, p))
        cov = cov @ cov.T / p  # Make positive definite
        
        B_numpy, F_numpy = pca_decomposition(cov, k=k, method="numpy")
        B_scipy, F_scipy = pca_decomposition(cov, k=k, method="scipy")
        
        # Eigenvalues should match
        vals_numpy = np.sort(np.diag(F_numpy))[::-1]
        vals_scipy = np.sort(np.diag(F_scipy))[::-1]
        assert np.allclose(vals_numpy, vals_scipy)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        cov = np.eye(10)
        
        with pytest.raises(ValueError, match="Unknown method"):
            pca_decomposition(cov, k=3, method="invalid")
    
    def test_non_square_matrix(self):
        """Test that non-square input raises error."""
        cov = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="square"):
            pca_decomposition(cov, k=3)
    
    def test_k_bounds(self):
        """Test k validation."""
        cov = np.eye(10)
        
        # k = 0 should fail
        with pytest.raises(ValueError, match="k must be in"):
            pca_decomposition(cov, k=0)
        
        # k > n should fail
        with pytest.raises(ValueError, match="k must be in"):
            pca_decomposition(cov, k=15)
    
    def test_eigenvalue_ordering(self, rng):
        """Test that eigenvalues are returned in descending order."""
        p, k = 20, 5
        cov = rng.standard_normal((p, p))
        cov = cov @ cov.T
        
        B, F = pca_decomposition(cov, k=k)
        
        eigenvalues = np.diag(F)
        # Check descending order
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


class TestSVDDecomposition:
    """Tests for svd_decomposition function."""
    
    def test_basic_extraction(self, sample_returns):
        """Test basic SVD extraction from returns."""
        k = 5
        model = svd_decomposition(sample_returns, k=k)
        
        T, p = sample_returns.shape
        
        # Check shapes
        assert model.k == k
        assert model.p == p
        assert model.B.shape == (k, p)
        assert model.F.shape == (k, k)
        assert model.D.shape == (p, p)
    
    def test_transforms_provided(self, sample_returns):
        """Test that SVD provides pre-computed transforms."""
        model = svd_decomposition(sample_returns, k=3)
        
        assert model.factor_transform is not None
        assert model.idio_transform is not None
        
        # Both should be diagonal (SVD produces orthogonal factors)
        assert model.factor_transform.is_diagonal
        assert model.idio_transform.is_diagonal
    
    def test_variance_recovery(self, rng):
        """
        Test that SVD correctly recovers factor variances.
        
        This is a critical mathematical test. For accurate recovery:
        - Loadings should be orthonormal
        - Factors should be orthogonal (uncorrelated)
        """
        T, p, k = 5000, 50, 3
        
        # Create orthogonal factors with known variances
        true_vars = np.array([100.0, 25.0, 1.0])
        
        # Generate orthogonal factor time series
        F_raw = rng.standard_normal((T, k))
        F_Q, _ = np.linalg.qr(F_raw)
        factors = F_Q * np.sqrt(true_vars * (T - 1))  # Scale to target variance
        
        # Create orthonormal loadings
        B_raw = rng.standard_normal((p, k))
        B_Q, _ = np.linalg.qr(B_raw)
        B_true = B_Q.T  # Shape (k, p)
        
        # Generate returns
        returns = factors @ B_true
        returns += rng.normal(0, 0.001, (T, p))  # Tiny noise
        
        # Extract via SVD
        model = svd_decomposition(returns, k=k)
        
        # Check variance recovery
        recovered_vars = np.diag(model.F)
        assert np.allclose(recovered_vars, true_vars, rtol=0.1)
    
    def test_demean_option(self, rng):
        """Test the demean parameter."""
        T, p = 100, 10
        
        # Returns with non-zero mean
        returns = rng.standard_normal((T, p)) + 5.0
        
        # With demean=True (default), mean is removed
        model_demeaned = svd_decomposition(returns, k=2, demean=True)
        
        # With demean=False, mean is not removed
        model_not_demeaned = svd_decomposition(returns, k=2, demean=False)
        
        # The models should differ
        assert not np.allclose(model_demeaned.F, model_not_demeaned.F)
    
    def test_input_validation(self):
        """Test input validation."""
        # 1D input
        with pytest.raises(ValueError, match="2D"):
            svd_decomposition(np.random.randn(100), k=2)
        
        # Too few time periods
        with pytest.raises(ValueError, match="at least 2"):
            svd_decomposition(np.random.randn(1, 10), k=2)
        
        # k out of bounds
        with pytest.raises(ValueError, match="k must be in"):
            svd_decomposition(np.random.randn(100, 10), k=0)
        
        with pytest.raises(ValueError, match="k must be in"):
            svd_decomposition(np.random.randn(100, 10), k=20)
    
    def test_covariance_reconstruction(self, sample_returns, large_sample_tolerance):
        """Test that SVD model approximates sample covariance."""
        model = svd_decomposition(sample_returns, k=3)
        
        # Compute sample covariance
        sample_cov = np.cov(sample_returns, rowvar=False)
        
        # Compute model-implied covariance
        model_cov = model.implied_covariance()
        
        # They should be close (not exact due to truncation to k factors)
        # Use relative error normalized by sample variance
        relative_error = np.abs(model_cov - sample_cov) / np.abs(sample_cov + 1e-10)
        mean_relative_error = np.mean(relative_error)
        
        # With enough factors, should be reasonably close
        assert mean_relative_error < 0.5


class TestExplainedVariance:
    """Tests for explained variance utilities."""
    
    def test_compute_explained_variance(self, simple_diagonal_model):
        """Test explained variance computation."""
        ratio = compute_explained_variance(simple_diagonal_model)
        
        # Should be between 0 and 1
        assert 0 <= ratio <= 1
    
    def test_explained_variance_extremes(self):
        """Test explained variance at extremes."""
        p, k = 10, 2
        
        # Case 1: No idiosyncratic variance (factors explain everything)
        B = np.random.randn(k, p)
        F = np.eye(k)
        D = np.zeros((p, p))  # No idio variance
        
        model = FactorModelData(B=B, F=F, D=D)
        ratio = compute_explained_variance(model)
        assert ratio == pytest.approx(1.0)
        
        # Case 2: High idiosyncratic variance (factors explain little)
        D = np.eye(p) * 1000  # Huge idio variance
        model = FactorModelData(B=B, F=F, D=D)
        ratio = compute_explained_variance(model)
        assert ratio < 0.1
    
    def test_select_k_by_variance(self, sample_returns):
        """Test automatic k selection."""
        # Request 80% explained variance
        k = select_k_by_variance(sample_returns, target_explained=0.80)
        
        assert k >= 1
        assert k <= min(sample_returns.shape)
        
        # Higher target should require more factors
        k_high = select_k_by_variance(sample_returns, target_explained=0.95)
        assert k_high >= k
    
    def test_select_k_with_max(self, sample_returns):
        """Test k selection with max_k constraint."""
        k = select_k_by_variance(sample_returns, target_explained=0.99, max_k=2)
        
        assert k <= 2
    
    def test_select_k_input_validation(self):
        """Test select_k input validation."""
        with pytest.raises(ValueError, match="2D"):
            select_k_by_variance(np.random.randn(100), target_explained=0.8)


class TestSVDAndPCAConsistency:
    """Tests comparing SVD and PCA results."""
    
    def test_same_eigenvalues(self, sample_returns):
        """Test that SVD and PCA give same eigenvalues."""
        k = 3
        
        # SVD on returns
        model_svd = svd_decomposition(sample_returns, k=k)
        svd_eigenvalues = np.diag(model_svd.F)
        
        # PCA on covariance
        cov = np.cov(sample_returns, rowvar=False)
        _, F_pca = pca_decomposition(cov, k=k)
        pca_eigenvalues = np.diag(F_pca)
        
        # Should be very close
        assert np.allclose(
            np.sort(svd_eigenvalues)[::-1],
            np.sort(pca_eigenvalues)[::-1],
            rtol=1e-10
        )
    
    def test_eigenvector_span(self, sample_returns):
        """Test that SVD and PCA span the same subspace."""
        k = 3
        
        # Get loadings from both methods
        model_svd = svd_decomposition(sample_returns, k=k)
        B_svd = model_svd.B  # (k, p)
        
        cov = np.cov(sample_returns, rowvar=False)
        B_pca, _ = pca_decomposition(cov, k=k)
        
        # The row spaces should be the same (up to rotation)
        # Check by seeing if B_pca @ B_svd.T is approximately orthogonal
        # Actually, check that B_svd rows lie in span of B_pca rows
        
        # Project B_svd onto B_pca
        proj = B_svd @ B_pca.T @ np.linalg.pinv(B_pca @ B_pca.T) @ B_pca
        
        # Projection should recover B_svd if they span same space
        # This is a bit loose due to sign/rotation ambiguity
        # Just check the norms are similar
        assert np.linalg.norm(proj) / np.linalg.norm(B_svd) > 0.9
