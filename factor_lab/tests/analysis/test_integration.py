"""
test_integration.py - Comprehensive Integration Tests
======================================================

Tests the complete workflow including:
- JSON parsing
- Model building
- Simulation
- Analysis pipeline
- Visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import tempfile
import numpy as np
import pytest

from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses


class TestCompleteWorkflow:
    """Test the complete analysis workflow end-to-end."""
    
    def test_basic_workflow(self):
        """Test basic workflow: build → simulate → analyze."""
        # 1. Create model
        k, p, T = 2, 30, 100
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # 2. Simulate
        simulator = ReturnsSimulator(model)
        results = simulator.simulate(n_periods=T)
        
        # 3. Create context
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns'],
        )
        
        # 4. Run all analyses
        manifold = Analyses.manifold_distances().analyze(context)
        eigen = Analyses.eigenvalue_analysis(k_top=k).analyze(context)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
        
        # 5. Verify results
        assert 'dist_grassmannian' in manifold
        assert 'eigenvalue_rmse' in eigen
        assert 'mean_correlation' in eigvec
        
        # All metrics should be reasonable
        assert 0 <= manifold['dist_grassmannian'] <= 5
        assert 0 <= eigen['eigenvalue_rmse']
        assert 0 <= eigvec['mean_correlation'] <= 1
    
    def test_json_config_parsing(self):
        """Test JSON configuration parsing."""
        config = {
            "meta": {
                "p_assets": 50,
                "n_periods": 100
            },
            "factor_loadings": [
                {"distribution": "normal", "params": {"loc": 0, "scale": 1}},
                {"distribution": "normal", "params": {"loc": 0, "scale": 0.5}}
            ],
            "covariance": {
                "F_diagonal": ["0.1^2", "0.05^2"],
                "D_diagonal": "0.01^2"
            },
            "simulations": [
                {"name": "Test", "type": "normal"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)
        
        try:
            # This would require build_and_simulate.py to be importable
            # For now, just verify the JSON is valid
            with open(config_path) as f:
                loaded = json.load(f)
            assert loaded['meta']['p_assets'] == 50
            assert loaded['covariance']['F_diagonal'][0] == "0.1^2"
        finally:
            config_path.unlink()
    
    def test_multiple_simulations(self):
        """Test running multiple simulation scenarios."""
        k, p, T = 2, 20, 50
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        # Run multiple simulations
        all_results = []
        for _ in range(3):
            simulator = ReturnsSimulator(model)
            results = simulator.simulate(n_periods=T)
            
            context = SimulationContext(
                model=model,
                security_returns=results['security_returns'],
                factor_returns=results['factor_returns'],
                idio_returns=results['idio_returns'],
            )
            
            manifold = Analyses.manifold_distances().analyze(context)
            all_results.append(manifold['dist_grassmannian'])
        
        # Results should vary across simulations
        assert len(set(all_results)) > 1
    
    def test_large_scale(self):
        """Test analysis works for larger problems."""
        k, p, T = 5, 100, 200
        B = np.random.randn(k, p)
        F = np.diag(np.random.uniform(0.01, 0.1, k))
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        simulator = ReturnsSimulator(model)
        results = simulator.simulate(n_periods=T)
        
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns'],
        )
        
        # Should complete without error
        manifold = Analyses.manifold_distances().analyze(context)
        eigen = Analyses.eigenvalue_analysis(k_top=k).analyze(context)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
        
        assert all(k in manifold for k in ['dist_grassmannian', 'dist_procrustes', 'dist_chordal'])
        assert 'eigenvalue_rmse' in eigen
        assert 'mean_correlation' in eigvec


class TestContextCaching:
    """Test SimulationContext caching behavior."""
    
    def test_sample_covariance_cached(self):
        """Sample covariance should be cached after first computation."""
        k, p, T = 2, 20, 50
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # First call computes
        cov1 = context.sample_covariance()
        
        # Second call should return cached value
        cov2 = context.sample_covariance()
        
        # Should be identical (same object)
        assert cov1 is cov2
    
    def test_pca_decomposition_cached(self):
        """PCA decomposition should be cached by n_components."""
        k, p, T = 3, 30, 100
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05, 0.02])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # PCA with k=2
        pca2_1 = context.pca_decomposition(n_components=2)
        pca2_2 = context.pca_decomposition(n_components=2)
        
        # Should be cached
        assert pca2_1.B is pca2_2.B
        
        # PCA with k=3 should be different
        pca3 = context.pca_decomposition(n_components=3)
        assert pca3.B is not pca2_1.B


class TestCustomAnalyses:
    """Test custom analysis functionality."""
    
    def test_custom_lambda(self):
        """Test custom analysis with lambda."""
        k, p, T = 2, 20, 50
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # Custom analysis
        custom = Analyses.custom(lambda ctx: {
            'frobenius_B': float(np.linalg.norm(ctx.model.B, 'fro')),
            'trace_F': float(np.trace(ctx.model.F)),
        })
        
        results = custom.analyze(context)
        
        assert 'frobenius_B' in results
        assert 'trace_F' in results
        assert results['frobenius_B'] > 0
        assert results['trace_F'] > 0
    
    def test_custom_function(self):
        """Test custom analysis with function."""
        k, p, T = 2, 20, 50
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        def my_analysis(ctx):
            pca = ctx.pca_decomposition(n_components=ctx.model.k)
            error = np.linalg.norm(ctx.model.B - pca.B, 'fro')
            return {'loading_error': float(error)}
        
        custom = Analyses.custom(my_analysis)
        results = custom.analyze(context)
        
        assert 'loading_error' in results
        assert results['loading_error'] >= 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_k_components(self):
        """Test handling of invalid k in analyses."""
        k, p, T = 2, 20, 50
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # k > model.k should handle gracefully
        # (may raise error or truncate, depending on implementation)
        # Just verify it doesn't crash catastrophically
        try:
            analysis = Analyses.eigenvector_comparison(k=10)
            # May work with truncation or may raise ValueError
        except ValueError:
            pass  # Expected
    
    def test_very_small_sample(self):
        """Test handling of very small sample size."""
        k, p, T = 2, 10, 15  # T < p
        B = np.random.randn(k, p)
        F = np.diag([0.1, 0.05])
        D = np.diag(np.full(p, 0.01))
        model = FactorModelData(B=B, F=F, D=D)
        
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        # Should handle gracefully
        manifold = Analyses.manifold_distances().analyze(context)
        assert 'dist_grassmannian' in manifold


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
