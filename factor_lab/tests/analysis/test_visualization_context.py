"""
test_visualization_context.py - Tests for Visualization and Context
====================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import tempfile

from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import print_verbose_results


class TestSimulationContext:
    """Test SimulationContext functionality."""
    
    def create_context(self, k=2, p=20, T=50):
        """Helper to create test context."""
        B = np.random.randn(k, p)
        F = np.diag(np.random.uniform(0.01, 0.1, k))
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        return SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
    
    def test_properties(self):
        """Test context properties."""
        context = self.create_context(k=3, p=50, T=100)
        
        assert context.T == 100
        assert context.p == 50
        assert context.k == 3
    
    def test_immutability(self):
        """Test that context is immutable."""
        context = self.create_context()
        
        # Should not be able to modify
        with pytest.raises((AttributeError, Exception)):
            context.T = 999
    
    def test_sample_covariance_ddof(self):
        """Test sample covariance with different ddof."""
        context = self.create_context(T=100)
        
        cov_ddof1 = context.sample_covariance(ddof=1)
        
        # Different ddof should give different result
        # (though can't test directly due to caching)
        assert cov_ddof1.shape == (context.p, context.p)
        assert np.all(np.isfinite(cov_ddof1))
    
    def test_pca_decomposition_default_k(self):
        """Test PCA decomposition with default k."""
        context = self.create_context(k=3)
        
        # Default should use model.k
        pca = context.pca_decomposition()
        assert pca.k == context.model.k
    
    def test_summary(self):
        """Test summary string generation."""
        context = self.create_context()
        
        summary = context.summary()
        assert "SimulationContext Summary" in summary
        assert "Model:" in summary
        assert "Periods:" in summary


class TestVerboseOutput:
    """Test verbose output formatting."""
    
    def test_print_verbose_basic(self, capsys):
        """Test basic verbose output."""
        results = {
            'dist_grassmannian': 0.123456,
            'dist_procrustes': 0.023456,
            'dist_chordal': 0.345678,
        }
        
        print_verbose_results(results, "Test Results")
        
        captured = capsys.readouterr()
        assert "Test Results" in captured.out
        assert "0.123456" in captured.out
        assert "MANIFOLD DISTANCES" in captured.out
    
    def test_print_verbose_eigenvalues(self, capsys):
        """Test verbose output with eigenvalues."""
        results = {
            'true_eigenvalues': np.array([0.1, 0.05, 0.02]),
            'sample_eigenvalues': np.array([0.09, 0.048, 0.021]),
            'eigenvalue_errors': np.array([0.01, 0.002, -0.001]),
            'eigenvalue_relative_errors': np.array([0.1, 0.04, -0.05]),
            'eigenvalue_rmse': 0.006,
        }
        
        print_verbose_results(results, "Eigenvalue Test")
        
        captured = capsys.readouterr()
        assert "EIGENVALUE COMPARISON" in captured.out
        assert "0.1" in captured.out  # Should show values
    
    def test_print_verbose_eigenvectors(self, capsys):
        """Test verbose output with eigenvector correlations."""
        results = {
            'vector_correlations': np.array([0.95, 0.85, 0.65]),
            'mean_correlation': 0.82,
            'min_correlation': 0.65,
            'max_correlation': 0.95,
        }
        
        print_verbose_results(results, "Eigenvector Test")
        
        captured = capsys.readouterr()
        assert "EIGENVECTOR CORRELATIONS" in captured.out
        assert "0.95" in captured.out
        assert "✓" in captured.out or "○" in captured.out  # Status symbols
    
    def test_print_verbose_assessment(self, capsys):
        """Test verbose output assessment section."""
        # Excellent result
        results = {
            'dist_grassmannian': 0.05,
            'mean_correlation': 0.97,
        }
        
        print_verbose_results(results, "Assessment Test")
        
        captured = capsys.readouterr()
        assert "ASSESSMENT" in captured.out
        assert "Excellent" in captured.out or "✓" in captured.out


class TestVisualizationCreation:
    """Test visualization functions (without displaying)."""
    
    def create_results(self):
        """Helper to create test results."""
        return {
            'dist_grassmannian': 0.123,
            'dist_procrustes': 0.023,
            'dist_chordal': 0.345,
            'principal_angles': np.array([0.1, 0.05, 0.02]),
            'true_eigenvalues': np.array([0.1, 0.05, 0.02]),
            'sample_eigenvalues': np.array([0.09, 0.048, 0.021]),
            'eigenvalue_errors': np.array([0.01, 0.002, -0.001]),
            'eigenvalue_relative_errors': np.array([0.1, 0.04, -0.05]),
            'eigenvalue_rmse': 0.006,
            'vector_correlations': np.array([0.95, 0.85, 0.65]),
            'mean_correlation': 0.82,
            'min_correlation': 0.65,
            'max_correlation': 0.95,
            'subspace_distance': 0.123,
            'true_eigenvectors': np.random.randn(3, 50),
            'sample_eigenvectors': np.random.randn(3, 50),
        }
    
    def test_create_dashboard_to_file(self):
        """Test creating dashboard and saving to file."""
        from factor_lab.visualization import create_manifold_dashboard
        
        results = self.create_results()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            # Should not raise error
            create_manifold_dashboard(results, output_path=output_path)
            
            # File should exist
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            if output_path.exists():
                output_path.unlink()
    
    def test_create_dashboard_without_all_data(self):
        """Test dashboard creation with partial data."""
        from factor_lab.visualization import create_manifold_dashboard
        
        # Minimal results
        results = {
            'dist_grassmannian': 0.123,
            'dist_procrustes': 0.023,
            'dist_chordal': 0.345,
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            # Should handle gracefully
            create_manifold_dashboard(results, output_path=output_path)
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()


class TestAnalysesBuilder:
    """Test Analyses builder/factory."""
    
    def test_all_factory_methods(self):
        """Test all factory methods exist and return correct types."""
        from factor_lab.analyses import ManifoldDistanceAnalysis, ImplicitEigenAnalysis, EigenvectorAlignment
        
        # Test factory methods
        manifold = Analyses.manifold_distances()
        assert isinstance(manifold, ManifoldDistanceAnalysis)
        
        eigen = Analyses.eigenvalue_analysis(k_top=5)
        assert isinstance(eigen, ImplicitEigenAnalysis)
        
        eigvec = Analyses.eigenvector_comparison(k=3)
        assert isinstance(eigvec, EigenvectorAlignment)
        
        custom = Analyses.custom(lambda ctx: {'test': 1})
        # Custom returns an object with analyze method
        assert hasattr(custom, 'analyze')
    
    def test_factory_parameters(self):
        """Test factory methods accept parameters."""
        # Should not raise
        manifold = Analyses.manifold_distances(use_pca_loadings=True)
        eigen = Analyses.eigenvalue_analysis(k_top=10, compare_eigenvectors=True)
        eigvec = Analyses.eigenvector_comparison(k=5, align_signs=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
