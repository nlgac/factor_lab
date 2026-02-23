#!/usr/bin/env python3
"""
Test script to verify eigenvector sign normalization is working.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude/factor_lab_manifold_complete')

from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses

def test_eigenvector_sign_normalization():
    """Test that all eigenvectors have positive mean after normalization."""
    
    print("=" * 70)
    print("Testing Eigenvector Sign Normalization")
    print("=" * 70)
    
    # Create model from SVD (this is where sign normalization happens)
    np.random.seed(42)
    k, p, T = 3, 50, 200
    
    # Generate synthetic data
    true_factors = np.random.randn(T, k)
    true_loadings = np.random.randn(k, p)
    noise = np.random.randn(T, p) * 0.1
    returns = true_factors @ true_loadings + noise
    
    print(f"\nExtracting {k}-factor model from {T}x{p} returns...")
    
    # Extract model using SVD (sign normalization happens here)
    from factor_lab import svd_decomposition
    model = svd_decomposition(returns, k=k, center=True)
    
    print(f"Factor loadings B shape: {model.B.shape}")
    
    # Check that model B has positive means (from SVD normalization)
    B_means = model.B.mean(axis=1)
    print(f"\nFactor loading means (should all be ≥ 0):")
    for i, mean in enumerate(B_means):
        status = "✓" if mean >= 0 else "✗"
        print(f"  Factor {i}: {mean:+.6f} {status}")
    
    assert np.all(B_means >= -1e-10), f"Some factor loadings have negative mean! Means: {B_means}"
    print("✓ All factor loadings have positive mean")
    
    # Now create a proper simulation from this model
    simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
    results = simulator.simulate(n_periods=T)
    
    # Create context
    context = SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns']
    )
    
    # Run eigenvector comparison analysis
    print("\n" + "=" * 70)
    print("Running Eigenvector Comparison Analysis")
    print("=" * 70)
    
    eigvec = Analyses.eigenvector_comparison(k=k).analyze(context)
    
    # Check true eigenvectors
    true_evecs = eigvec['true_eigenvectors']  # (k, p)
    true_means = true_evecs.mean(axis=1)
    
    print(f"\nTrue eigenvector means (should all be ≥ 0):")
    for i, mean in enumerate(true_means):
        status = "✓" if mean >= 0 else "✗"
        print(f"  Eigenvector {i}: {mean:+.6f} {status}")
    
    assert np.all(true_means >= 0), "Some true eigenvectors have negative mean!"
    print("✓ All true eigenvectors have positive mean")
    
    # Check sample eigenvectors
    sample_evecs = eigvec['sample_eigenvectors']  # (k, p)
    sample_means = sample_evecs.mean(axis=1)
    
    print(f"\nSample eigenvector means (should all be ≥ 0):")
    for i, mean in enumerate(sample_means):
        status = "✓" if mean >= 0 else "✗"
        print(f"  Eigenvector {i}: {mean:+.6f} {status}")
    
    assert np.all(sample_means >= 0), "Some sample eigenvectors have negative mean!"
    print("✓ All sample eigenvectors have positive mean")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Factor loadings (SVD): All positive means")
    print("✓ True eigenvectors (eigendecomposition): All positive means")
    print("✓ Sample eigenvectors (PCA): All positive means")
    print("\n✅ Sign normalization is working correctly!")
    print("=" * 70)
    
    # Pytest convention: test functions should return None

if __name__ == "__main__":
    try:
        test_eigenvector_sign_normalization()
        print("\n✅ Test passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
