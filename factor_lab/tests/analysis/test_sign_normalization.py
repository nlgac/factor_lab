"""
Test sign normalization for factor loadings and eigenvectors.

This test verifies that sign normalization is applied correctly to:
1. Factor loadings from SVD (in types.py)
2. True eigenvectors from eigendecomposition (in spectral.py)
3. Sample eigenvectors from PCA (uses svd_decomposition)
"""

import numpy as np
import pytest
from factor_lab import FactorModelData, ReturnsSimulator, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses


class TestSignNormalization:
    """Test suite for sign normalization of loadings and eigenvectors."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic returns data for testing."""
        np.random.seed(42)
        k, p, T = 3, 50, 200
        
        # Generate synthetic data
        true_factors = np.random.randn(T, k)
        true_loadings = np.random.randn(k, p)
        noise = np.random.randn(T, p) * 0.1
        returns = true_factors @ true_loadings + noise
        
        return returns, k
    
    def test_svd_factor_loadings_positive_mean(self, synthetic_data):
        """Test that SVD factor loadings have positive mean."""
        returns, k = synthetic_data
        
        # Extract model using SVD
        model = svd_decomposition(returns, k=k, center=True)
        
        # Check that all factor loadings have positive (or near-zero) mean
        B_means = model.B.mean(axis=1)
        
        assert model.B.shape == (k, returns.shape[1]), "Wrong shape for B"
        assert np.all(B_means >= -1e-10), \
            f"Factor loadings have negative means: {B_means}"
        
    def test_true_eigenvectors_positive_mean(self, synthetic_data):
        """Test that true eigenvectors have positive mean."""
        returns, k = synthetic_data
        
        # Extract model and simulate
        model = svd_decomposition(returns, k=k, center=True)
        simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
        results = simulator.simulate(n_periods=200)
        
        # Create context
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns']
        )
        
        # Run eigenvector comparison
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
        
        # Check true eigenvectors
        true_evecs = eigvec['true_eigenvectors']
        true_means = true_evecs.mean(axis=1)
        
        assert true_evecs.shape[0] == k, "Wrong number of eigenvectors"
        assert np.all(true_means >= -1e-10), \
            f"True eigenvectors have negative means: {true_means}"
    
    def test_sample_eigenvectors_positive_mean(self, synthetic_data):
        """Test that sample eigenvectors have positive mean."""
        returns, k = synthetic_data
        
        # Extract model and simulate
        model = svd_decomposition(returns, k=k, center=True)
        simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
        results = simulator.simulate(n_periods=200)
        
        # Create context
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns']
        )
        
        # Run eigenvector comparison
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
        
        # Check sample eigenvectors
        sample_evecs = eigvec['sample_eigenvectors']
        sample_means = sample_evecs.mean(axis=1)
        
        assert sample_evecs.shape[0] == k, "Wrong number of eigenvectors"
        assert np.all(sample_means >= -1e-10), \
            f"Sample eigenvectors have negative means: {sample_means}"
    
    def test_all_components_consistent_signs(self, synthetic_data):
        """Test that all components (loadings, true evecs, sample evecs) have consistent positive signs."""
        returns, k = synthetic_data
        
        # Extract model and simulate
        model = svd_decomposition(returns, k=k, center=True)
        simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
        results = simulator.simulate(n_periods=200)
        
        # Create context
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns']
        )
        
        # Get all components
        B_means = model.B.mean(axis=1)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
        true_means = eigvec['true_eigenvectors'].mean(axis=1)
        sample_means = eigvec['sample_eigenvectors'].mean(axis=1)
        
        # All should be non-negative
        assert np.all(B_means >= -1e-10), \
            f"Factor loadings have negative means: {B_means}"
        assert np.all(true_means >= -1e-10), \
            f"True eigenvectors have negative means: {true_means}"
        assert np.all(sample_means >= -1e-10), \
            f"Sample eigenvectors have negative means: {sample_means}"
    
    def test_sign_normalization_preserves_subspace(self, synthetic_data):
        """Test that sign normalization doesn't change the subspace."""
        returns, k = synthetic_data
        
        # Extract model
        model = svd_decomposition(returns, k=k, center=True)
        
        # Manually flip all signs (should not change subspace)
        B_flipped = -model.B
        
        # Both should span the same subspace
        # (We can't directly test this without implementing subspace comparison,
        #  but we can verify the model still works)
        model_flipped = FactorModelData(B=B_flipped, F=model.F, D=model.D)
        
        # Both models should have same implied covariance
        Sigma1 = model.implied_covariance()
        Sigma2 = model_flipped.implied_covariance()
        
        assert np.allclose(Sigma1, Sigma2), \
            "Sign flips changed the implied covariance!"
    
    def test_sign_normalization_idempotent(self, synthetic_data):
        """Test that applying sign normalization twice gives same result."""
        returns, k = synthetic_data
        
        # Extract model (sign normalized once)
        model1 = svd_decomposition(returns, k=k, center=True)
        
        # Manually apply normalization again
        B = model1.B.copy()
        row_means = B.mean(axis=1)
        sign_flips = np.where(row_means < 0, -1, 1)
        B_normalized = B * sign_flips[:, np.newaxis]
        
        # Should be identical (all means already positive)
        assert np.allclose(B, B_normalized), \
            "Sign normalization is not idempotent!"
        
        # All sign flips should be +1
        assert np.all(sign_flips == 1), \
            f"Some factors would be flipped again: {sign_flips}"


def test_sign_normalization_comprehensive():
    """
    Comprehensive test for sign normalization across all components.
    
    This is the main test that verifies the complete workflow.
    """
    # Setup
    np.random.seed(42)
    k, p, T = 3, 50, 200
    
    # Generate synthetic data
    true_factors = np.random.randn(T, k)
    true_loadings = np.random.randn(k, p)
    noise = np.random.randn(T, p) * 0.1
    returns = true_factors @ true_loadings + noise
    
    # Extract model using SVD (sign normalization happens here)
    model = svd_decomposition(returns, k=k, center=True)
    
    # Check factor loadings
    B_means = model.B.mean(axis=1)
    assert np.all(B_means >= -1e-10), \
        f"Factor loadings have negative means: {B_means}"
    
    # Simulate
    simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
    results = simulator.simulate(n_periods=T)
    
    # Create context
    context = SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns']
    )
    
    # Run eigenvector comparison
    eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
    
    # Check true eigenvectors
    true_means = eigvec['true_eigenvectors'].mean(axis=1)
    assert np.all(true_means >= -1e-10), \
        f"True eigenvectors have negative means: {true_means}"
    
    # Check sample eigenvectors
    sample_means = eigvec['sample_eigenvectors'].mean(axis=1)
    assert np.all(sample_means >= -1e-10), \
        f"Sample eigenvectors have negative means: {sample_means}"
