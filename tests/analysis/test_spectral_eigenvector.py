"""
test_spectral_eigenvector.py - Tests for Spectral and Eigenvector Analyses
===========================================================================
"""

import numpy as np
import pytest
from factor_lab.analyses.spectral import (
    compute_true_eigenvalues,
    ImplicitEigenAnalysis,
)
from factor_lab.analyses.eigenvector import EigenvectorAlignment
from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext


class TestComputeTrueEigenvalues:
    """Test implicit eigenvalue computation."""
    
    def test_matches_dense_method(self):
        """Should match dense eigenvalue computation for small p."""
        np.random.seed(42)
        k, p = 3, 50
        
        B = np.random.randn(k, p)
        F = np.diag([0.09, 0.04, 0.01])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # Compute via LinearOperator
        evals_sparse, _ = compute_true_eigenvalues(model, k_top=k)
        
        # Compute via dense method
        Sigma = B.T @ F @ B + D
        evals_dense = np.linalg.eigvalsh(Sigma)[::-1][:k]
        
        # Should match within tolerance
        assert np.allclose(evals_sparse, evals_dense, rtol=1e-6)
    
    def test_returns_descending_order(self):
        """Eigenvalues should be in descending order."""
        k, p = 5, 100
        B = np.random.randn(k, p)
        F = np.diag([0.25, 0.16, 0.09, 0.04, 0.01])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        evals, _ = compute_true_eigenvalues(model, k_top=k)
        
        # Check descending order
        assert np.all(evals[:-1] >= evals[1:])
    
    def test_eigenvectors_orthonormal(self):
        """Eigenvectors should be orthonormal."""
        k, p = 3, 50
        B = np.random.randn(k, p)
        F = np.diag([0.09, 0.04, 0.01])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        _, evecs = compute_true_eigenvalues(model, k_top=k)
        
        # evecs is (k, p) - row vectors
        # Check orthonormality: evecs @ evecs.T = I
        assert np.allclose(evecs @ evecs.T, np.eye(k), atol=1e-8)
    
    def test_works_for_large_p(self):
        """Should work efficiently for large p."""
        k, p = 5, 1000
        B = np.random.randn(k, p)
        F = np.diag(np.random.uniform(0.01, 0.1, k))
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # Should complete without error
        evals, evecs = compute_true_eigenvalues(model, k_top=k)
        
        assert len(evals) == k
        assert evecs.shape == (k, p)


class TestImplicitEigenAnalysis:
    """Test ImplicitEigenAnalysis class."""
    
    def create_context(self, k=3, p=50, T=100):
        """Create mock context."""
        B = np.random.randn(k, p)
        F = np.diag(np.random.uniform(0.01, 0.1, k))
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # Simulate from true model
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p) * 0.1
        returns = factors @ B + idio
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        return context
    
    def test_analyze_returns_all_metrics(self):
        """Should return all expected metrics."""
        context = self.create_context()
        analysis = ImplicitEigenAnalysis(k_top=3)
        
        results = analysis.analyze(context)
        
        # Check expected keys
        assert 'true_eigenvalues' in results
        assert 'sample_eigenvalues' in results
        assert 'eigenvalue_errors' in results
        assert 'eigenvalue_relative_errors' in results
        assert 'eigenvalue_rmse' in results
        assert 'eigenvalue_max_error' in results
        assert 'eigenvalue_mean_relative_error' in results
    
    def test_eigenvalue_errors_decrease_with_T(self):
        """Errors should decrease as T increases."""
        np.random.seed(42)
        k, p = 3, 30
        
        B = np.random.randn(k, p)
        F = np.diag([0.09, 0.04, 0.01])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        rmse_values = []
        for T in [50, 100, 200, 500]:
            factors = np.random.randn(T, k)
            idio = np.random.randn(T, p) * 0.1
            returns = factors @ B + idio
            
            context = SimulationContext(
                model=model,
                security_returns=returns,
                factor_returns=factors,
                idio_returns=idio,
            )
            
            analysis = ImplicitEigenAnalysis(k_top=k)
            results = analysis.analyze(context)
            rmse_values.append(results['eigenvalue_rmse'])
        
        # RMSE should generally decrease
        # (Not strictly monotonic due to randomness, but trend should be down)
        assert rmse_values[-1] < rmse_values[0]
    
    def test_with_eigenvector_comparison(self):
        """Should include eigenvector metrics when requested."""
        context = self.create_context()
        analysis = ImplicitEigenAnalysis(k_top=3, compare_eigenvectors=True)
        
        results = analysis.analyze(context)
        
        # Should have eigenvector metrics
        assert 'eigenvector_subspace_distance' in results
        assert 'eigenvector_principal_angles' in results
        assert 'eigenvector_canonical_correlations' in results
        assert 'eigenvector_mean_correlation' in results


class TestEigenvectorAlignment:
    """Test EigenvectorAlignment class."""
    
    def create_context(self, k=3, p=50, T=100):
        """Create mock context with good factor recovery."""
        np.random.seed(42)
        B = np.random.randn(k, p)
        F = np.diag([0.16, 0.09, 0.04])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # Generate from true model with large T for good recovery
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p) * 0.1
        returns = factors @ B + idio
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        return context
    
    def test_analyze_returns_all_metrics(self):
        """Should return all expected metrics."""
        context = self.create_context(T=500)  # Large T for good recovery
        analysis = EigenvectorAlignment(k_components=3)
        
        results = analysis.analyze(context)
        
        # Core metrics
        assert 'subspace_distance' in results
        assert 'principal_angles' in results
        assert 'max_principal_angle' in results
        assert 'grassmann_distance' in results
        
        # Correlation metrics
        assert 'vector_correlations' in results
        assert 'mean_correlation' in results
        assert 'min_correlation' in results
        assert 'max_correlation' in results
        
        # Procrustes metrics
        assert 'procrustes_distance' in results
        assert 'optimal_rotation' in results
        assert 'aligned_eigenvectors' in results
        
        # Reference data
        assert 'true_eigenvectors' in results
        assert 'sample_eigenvectors' in results
        assert 'k_components' in results
    
    def test_correlations_improve_with_T(self):
        """Mean correlation should improve with more data."""
        k, p = 3, 30
        B = np.random.randn(k, p)
        F = np.diag([0.16, 0.09, 0.04])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        correlations = []
        for T in [50, 200, 500]:
            np.random.seed(42 + T)  # Different seed for each T
            factors = np.random.randn(T, k)
            idio = np.random.randn(T, p) * 0.1
            returns = factors @ B + idio
            
            context = SimulationContext(
                model=model,
                security_returns=returns,
                factor_returns=factors,
                idio_returns=idio,
            )
            
            analysis = EigenvectorAlignment(k_components=k)
            results = analysis.analyze(context)
            correlations.append(results['mean_correlation'])
        
        # Should generally improve
        assert correlations[-1] > correlations[0]
    
    def test_sign_alignment_works(self):
        """Sign alignment should improve correlations."""
        context = self.create_context(T=500)
        
        # Without sign alignment
        analysis_no_align = EigenvectorAlignment(
            k_components=3,
            align_signs=False
        )
        results_no_align = analysis_no_align.analyze(context)
        
        # With sign alignment
        analysis_align = EigenvectorAlignment(
            k_components=3,
            align_signs=True
        )
        results_align = analysis_align.analyze(context)
        
        # Aligned version should have equal or better correlation
        assert results_align['mean_correlation'] >= results_no_align['mean_correlation'] - 0.05
    
    def test_optimal_rotation_is_orthogonal(self):
        """Optimal rotation matrix should be orthogonal."""
        context = self.create_context()
        analysis = EigenvectorAlignment(k_components=3)
        
        results = analysis.analyze(context)
        
        R = results['optimal_rotation']
        # R.T @ R should be identity
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)


class TestIntegration:
    """Integration tests combining multiple analyses."""
    
    def test_all_analyses_together(self):
        """Should be able to run all analyses on same context."""
        from factor_lab.analyses import Analyses
        
        # Create context
        k, p, T = 3, 50, 200
        B = np.random.randn(k, p)
        F = np.diag([0.16, 0.09, 0.04])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p) * 0.1
        returns = factors @ B + idio
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # Run all analyses
        analyses = [
            Analyses.manifold_distances(),
            Analyses.eigenvalue_analysis(compare_eigenvectors=True),
            Analyses.eigenvector_comparison(),
        ]
        
        all_results = {}
        for analysis in analyses:
            result = analysis.analyze(context)
            all_results.update(result)
        
        # Should have metrics from all three analyses
        assert 'dist_grassmannian' in all_results  # Manifold
        assert 'eigenvalue_rmse' in all_results  # Spectral
        assert 'mean_correlation' in all_results  # Eigenvector
        
        # All metrics should be reasonable
        assert 0 <= all_results['mean_correlation'] <= 1
        assert all_results['eigenvalue_rmse'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
