"""
test_manifold.py - Tests for Manifold Distance Computations
============================================================
"""

import numpy as np
import pytest
import scipy.stats
from factor_lab.analyses.manifold import (
    orthonormalize,
    compute_grassmannian_distance,
    compute_procrustes_distance,
    compute_chordal_distance,
    ManifoldDistanceAnalysis,
)
from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext


class TestOrthonormalize:
    """Test orthonormalization function."""
    
    def test_orthonormal_output(self):
        """Output should have orthonormal columns."""
        B = np.random.randn(3, 100)
        Q = orthonormalize(B)
        
        # Check Q.T @ Q = I
        assert Q.shape == (100, 3)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)
    
    def test_preserves_subspace(self):
        """Should preserve the subspace spanned by B."""
        B = np.random.randn(3, 100)
        Q = orthonormalize(B)
        
        # Q and B.T should span the same subspace
        # Check that B.T can be expressed as linear combination of Q
        proj = Q @ (Q.T @ B.T)  # Project B.T onto span(Q)
        assert np.allclose(proj, B.T, atol=1e-10)


class TestGrassmannianDistance:
    """Test Grassmannian distance computation."""
    
    def test_identical_subspaces(self):
        """Distance should be zero for identical subspaces."""
        B = np.random.randn(3, 100)
        
        dist, angles = compute_grassmannian_distance(B, B)
        
        assert dist < 1e-10
        assert np.allclose(angles, 0, atol=1e-10)
    
    def test_rotation_invariance(self):
        """Distance should be zero after rotation."""
        np.random.seed(42)
        k, p = 3, 100
        B_true = np.random.randn(k, p)
        
        # Apply random orthogonal rotation
        Q = scipy.stats.ortho_group.rvs(k)
        B_rotated = Q @ B_true
        
        dist, _ = compute_grassmannian_distance(B_true, B_rotated)
        
        # Should be invariant to rotation
        assert dist < 1e-8, f"Distance {dist} > 1e-8"
    
    def test_sign_flip_invariance(self):
        """Distance should be zero after sign flip."""
        B_true = np.random.randn(3, 100)
        B_flipped = B_true.copy()
        B_flipped[0, :] *= -1  # Flip first factor
        B_flipped[1, :] *= -1  # Flip second factor
        
        dist, _ = compute_grassmannian_distance(B_true, B_flipped)
        
        # Should be invariant to sign flip
        assert dist < 1e-8
    
    def test_permutation_invariance(self):
        """Distance should be zero after permutation."""
        B_true = np.random.randn(3, 100)
        # Permute factors
        B_permuted = B_true[[1, 2, 0], :]
        
        dist, _ = compute_grassmannian_distance(B_true, B_permuted)
        
        # Should be invariant to permutation
        assert dist < 1e-8
    
    def test_orthogonal_subspaces(self):
        """Distance should be large for orthogonal subspaces."""
        k, p = 3, 100
        # Create orthogonal subspaces
        B1 = np.zeros((k, p))
        B1[:, :k] = np.eye(k)
        
        B2 = np.zeros((k, p))
        B2[:, k:2*k] = np.eye(k)
        
        dist, angles = compute_grassmannian_distance(B1, B2)
        
        # All principal angles should be π/2
        assert np.allclose(angles, np.pi/2, atol=1e-6)


class TestProcrustesDistance:
    """Test Procrustes distance computation."""
    
    def test_identical_frames(self):
        """Distance should be zero for identical frames."""
        B = np.random.randn(3, 100)
        
        result = compute_procrustes_distance(B, B)
        
        assert result['distance'] < 1e-10
        assert np.allclose(result['optimal_rotation'], np.eye(3), atol=1e-8)
    
    def test_handles_sign_flip(self):
        """Should find alignment despite sign flips."""
        B_true = np.random.randn(3, 100)
        B_flipped = B_true.copy()
        B_flipped[0, :] *= -1
        B_flipped[1, :] *= -1
        
        result = compute_procrustes_distance(B_true, B_flipped)
        
        # Distance should be small after optimal alignment
        assert result['distance'] < 1e-8
        
        # Rotation should be diagonal with ±1 entries
        R = result['optimal_rotation']
        # R should be close to a diagonal matrix with ±1
        assert np.allclose(np.abs(np.diag(R)), 1, atol=1e-8)
    
    def test_handles_rotation(self):
        """Should find alignment despite rotation."""
        k, p = 3, 100
        B_true = np.random.randn(k, p)
        
        # Apply random rotation
        Q = scipy.stats.ortho_group.rvs(k)
        B_rotated = Q @ B_true
        
        result = compute_procrustes_distance(B_true, B_rotated)
        
        # Distance should be near zero after finding optimal rotation
        assert result['distance'] < 1e-6


class TestChordalDistance:
    """Test chordal distance computation."""
    
    def test_identical_frames(self):
        """Distance should be zero for identical frames."""
        B = np.random.randn(3, 100)
        
        dist = compute_chordal_distance(B, B)
        
        assert dist < 1e-10
    
    def test_sensitive_to_rotation(self):
        """Should be sensitive to rotation (unlike Grassmannian)."""
        k, p = 3, 100
        B_true = np.random.randn(k, p)
        
        # Small rotation
        angle = 0.1  # radians
        Q = np.eye(k)
        Q[0, 0] = Q[1, 1] = np.cos(angle)
        Q[0, 1] = -np.sin(angle)
        Q[1, 0] = np.sin(angle)
        
        B_rotated = Q @ B_true
        
        chordal = compute_chordal_distance(B_true, B_rotated)
        grassmann, _ = compute_grassmannian_distance(B_true, B_rotated)
        
        # Chordal should be larger than Grassmannian
        assert chordal > grassmann


class TestManifoldDistanceAnalysis:
    """Test ManifoldDistanceAnalysis class."""
    
    def create_mock_context(self, k=3, p=50, T=100):
        """Create a mock SimulationContext for testing."""
        B = np.random.randn(k, p)
        F = np.eye(k) * 0.1
        D = np.eye(p) * 0.01
        model = FactorModelData(B=B, F=F, D=D)
        
        # Simulate returns
        returns = np.random.randn(T, p)
        factors = np.random.randn(T, k)
        idio = np.random.randn(T, p)
        
        context = SimulationContext(
            model=model,
            security_returns=returns,
            factor_returns=factors,
            idio_returns=idio,
        )
        
        return context
    
    def test_analyze_returns_all_metrics(self):
        """analyze() should return all expected metrics."""
        context = self.create_mock_context()
        analysis = ManifoldDistanceAnalysis(use_pca_loadings=True)
        
        results = analysis.analyze(context)
        
        # Check all expected keys present
        assert 'dist_grassmannian' in results
        assert 'dist_procrustes' in results
        assert 'dist_chordal' in results
        assert 'principal_angles' in results
        assert 'optimal_rotation' in results
        assert 'aligned_frame' in results
        
        # Check types
        assert isinstance(results['dist_grassmannian'], float)
        assert isinstance(results['dist_procrustes'], float)
        assert isinstance(results['dist_chordal'], float)
        assert isinstance(results['principal_angles'], np.ndarray)
        assert isinstance(results['optimal_rotation'], np.ndarray)
        assert isinstance(results['aligned_frame'], np.ndarray)
    
    def test_analyze_self_comparison(self):
        """When comparing model to itself, distances should be zero."""
        context = self.create_mock_context()
        analysis = ManifoldDistanceAnalysis(use_pca_loadings=False)
        
        results = analysis.analyze(context)
        
        # All distances should be near zero
        assert results['dist_grassmannian'] < 1e-8
        assert results['dist_procrustes'] < 1e-8
        assert results['dist_chordal'] < 1e-8
    
    def test_analyze_works_with_different_k(self):
        """Should work for different numbers of factors."""
        for k in [2, 5, 10]:
            context = self.create_mock_context(k=k, p=100, T=200)
            analysis = ManifoldDistanceAnalysis()
            
            results = analysis.analyze(context)
            
            assert len(results['principal_angles']) == k
            assert results['optimal_rotation'].shape == (k, k)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
