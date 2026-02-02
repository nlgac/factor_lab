#!/usr/bin/env python3
"""
demo.py - Comprehensive Demonstration of Factor Lab Manifold Analysis
======================================================================

This script demonstrates all features with clear explanations and
verbose output. Perfect for learning and testing.

Demonstrates:
1. Model building from specifications
2. Monte Carlo simulation
3. Factor extraction via SVD
4. Manifold distance analysis
5. Eigenvalue analysis (O(kp) memory!)
6. Eigenvector comparison
7. Custom analyses
8. Rich visualization
9. Integration with JSON configs

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add current directory to path for standalone operation
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import scipy.stats
from datetime import datetime

from factor_lab import (
    FactorModelData,
    svd_decomposition,
    ReturnsSimulator,
    DistributionFactory,
    save_model,
)
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import (
    create_manifold_dashboard,
    create_interactive_plotly_dashboard,
    print_verbose_results,
)


def print_section(title):
    """Print styled section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print styled subsection header."""
    print(f"\n{title}")
    print("-" * 50)


def demo_model_building():
    """Demonstrate model building."""
    print_section("1. MODEL BUILDING")
    
    print("""
This demonstrates creating a factor model with:
- 3 factors
- 100 assets
- Specific variance structure
    """)
    
    # Setup
    k, p = 3, 100
    rng = np.random.default_rng(42)
    
    print(f"Building model: k={k} factors, p={p} assets")
    
    # Create factor loadings
    print("\n📊 Factor Loadings:")
    B = np.random.randn(k, p) * 0.5
    print(f"   Shape: {B.shape}")
    print(f"   Mean: {B.mean():.4f}, Std: {B.std():.4f}")
    
    # Factor covariance
    print("\n📊 Factor Covariance (F):")
    factor_vols = [0.18, 0.10, 0.05]
    F = np.diag([v**2 for v in factor_vols])
    print(f"   Diagonal: {np.diag(F)}")
    print(f"   Factor volatilities: {factor_vols}")
    
    # Idiosyncratic variance
    print("\n📊 Idiosyncratic Variance (D):")
    idio_vol = 0.16
    D = np.diag(np.full(p, idio_vol**2))
    print(f"   All assets: {idio_vol**2:.4f}")
    
    # Create model
    model = FactorModelData(B=B, F=F, D=D)
    
    print(f"\n✓ Model created successfully")
    print(f"   Implied covariance Σ = B'FB + D: {model.implied_covariance().shape}")
    print(f"   Total variance: {np.trace(model.implied_covariance()):.4f}")
    
    return model, rng


def demo_simulation(model, rng):
    """Demonstrate Monte Carlo simulation."""
    print_section("2. MONTE CARLO SIMULATION")
    
    print("""
This demonstrates simulating returns from the factor model:
- Returns = Factors @ Loadings + Idiosyncratic
- Using Gaussian innovations
    """)
    
    T = 500
    print(f"\nSimulating {T} periods...")
    
    # Simulate
    simulator = ReturnsSimulator(model, rng=rng)
    results = simulator.simulate(n_periods=T)
    
    returns = results['security_returns']
    factors = results['factor_returns']
    idio = results['idio_returns']
    
    print(f"\n✓ Simulation complete")
    print(f"\n📊 Security Returns:")
    print(f"   Shape: {returns.shape}")
    print(f"   Mean: {returns.mean():.6f}")
    print(f"   Std: {returns.std():.4f}")
    print(f"   Min: {returns.min():.4f}, Max: {returns.max():.4f}")
    
    print(f"\n📊 Factor Returns:")
    print(f"   Shape: {factors.shape}")
    for i in range(min(3, factors.shape[1])):
        print(f"   Factor {i+1}: Mean={factors[:,i].mean():.4f}, Std={factors[:,i].std():.4f}")
    
    # Check kurtosis
    kurt = scipy.stats.kurtosis(returns.flatten())
    print(f"\n📊 Distribution Properties:")
    print(f"   Excess kurtosis: {kurt:.2f}")
    print(f"   {'(Fat tails)' if kurt > 1 else '(Normal-ish)'}")
    
    return results


def demo_factor_extraction(returns):
    """Demonstrate factor extraction via SVD."""
    print_section("3. FACTOR EXTRACTION (SVD)")
    
    print("""
This demonstrates extracting factors from returns using SVD.
The extracted factors should approximate the true factors.
    """)
    
    k = 3
    print(f"\nExtracting {k} factors via SVD...")
    
    # Extract
    extracted = svd_decomposition(returns, k=k, center=True)
    
    print(f"\n✓ Extraction complete")
    print(f"\n📊 Extracted Model:")
    print(f"   Factor loadings B: {extracted.B.shape}")
    print(f"   Factor covariance F: {extracted.F.shape}")
    print(f"   Idiosyncratic variance D: {extracted.D.shape}")
    
    print(f"\n📊 Explained Variance:")
    explained_var = np.trace(extracted.F) / np.var(returns)
    print(f"   {explained_var:.1%} of total variance")
    
    print(f"\n📊 Factor Variances:")
    for i, fvar in enumerate(np.diag(extracted.F)):
        print(f"   Factor {i+1}: {fvar:.6f} (vol={np.sqrt(fvar):.4f})")
    
    return extracted


def demo_manifold_analysis(model, results):
    """Demonstrate manifold distance analysis."""
    print_section("4. MANIFOLD DISTANCE ANALYSIS")
    
    print("""
This demonstrates geometric comparison of factor loadings using:
- Grassmannian distance (subspace comparison, rotation-invariant)
- Procrustes distance (optimal frame alignment)
- Chordal distance (raw frame difference)

These metrics handle the fundamental ambiguities in factor models:
rotation, sign flips, and permutations.
    """)
    
    # Create context
    context = SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns'],
    )
    
    print("\n📐 Running manifold analysis...")
    
    # Run analysis
    manifold = Analyses.manifold_distances()
    manifold_results = manifold.analyze(context)
    
    # Print results
    print_verbose_results(manifold_results, "Manifold Analysis Results")
    
    return context, manifold_results


def demo_eigenvalue_analysis(context):
    """Demonstrate eigenvalue analysis."""
    print_section("5. EIGENVALUE ANALYSIS (O(kp) Memory!)")
    
    print("""
This demonstrates efficient eigenvalue computation using LinearOperator.

Instead of forming the full p×p covariance matrix Σ = B'FB + D,
we define an implicit operator that computes Σv efficiently:
    Σv = B'(F(Bv)) + Dv

This reduces:
- Memory: O(p²) → O(kp)
- For p=10,000, k=10: 100× reduction (800 MB → 8 MB)
    """)
    
    print("\n🔢 Running eigenvalue analysis...")
    
    # Run analysis
    eigen = Analyses.eigenvalue_analysis(
        k_top=context.model.k,
        compare_eigenvectors=True
    )
    eigen_results = eigen.analyze(context)
    
    # Print results
    print_verbose_results(eigen_results, "Eigenvalue Analysis Results")
    
    return eigen_results


def demo_eigenvector_comparison(context):
    """Demonstrate eigenvector comparison."""
    print_section("6. EIGENVECTOR COMPARISON")
    
    print("""
This demonstrates comparing eigenvectors of the true covariance
matrix with eigenvectors obtained from PCA.

Eigenvectors are the directions of maximum variance. Even if
eigenvalues match, the eigenvectors might not!

Metrics computed:
- Subspace distance (principal angles)
- Procrustes distance (after optimal rotation)
- Canonical correlations (per-vector similarity)
- Sign alignment (handles ambiguity)
    """)
    
    print("\n🎯 Running eigenvector comparison...")
    
    # Run analysis
    eigvec = Analyses.eigenvector_comparison(
        k=context.model.k,
        align_signs=True,
        compute_rotation=True
    )
    eigvec_results = eigvec.analyze(context)
    
    # Print results
    print_verbose_results(eigvec_results, "Eigenvector Comparison Results")
    
    return eigvec_results


def demo_custom_analysis(context):
    """Demonstrate custom analyses."""
    print_section("7. CUSTOM ANALYSIS")
    
    print("""
This demonstrates the extensibility framework.
You can easily add custom analyses without modifying core code.
    """)
    
    print_subsection("Example 1: Simple Lambda")
    
    custom1 = Analyses.custom(lambda ctx: {
        'frobenius_B': float(np.linalg.norm(ctx.model.B, 'fro')),
        'trace_F': float(np.trace(ctx.model.F)),
        'mean_D': float(np.mean(np.diag(ctx.model.D))),
        'total_variance': float(np.trace(ctx.model.implied_covariance())),
    })
    
    results1 = custom1.analyze(context)
    
    print("\nResults:")
    for key, value in results1.items():
        print(f"   {key}: {value:.6f}")
    
    print_subsection("Example 2: Loading Reconstruction Error")
    
    def loading_error_analysis(ctx):
        """Compute Frobenius error in loading reconstruction."""
        pca = ctx.pca_decomposition(n_components=ctx.model.k)
        
        # Direct Frobenius error (not accounting for rotation)
        direct_error = np.linalg.norm(ctx.model.B - pca.B, 'fro')
        
        # Procrustes error (accounting for rotation)
        Q_true, _ = scipy.linalg.qr(ctx.model.B.T, mode='economic')
        Q_est, _ = scipy.linalg.qr(pca.B.T, mode='economic')
        M = Q_est.T @ Q_true
        U, _, Vt = scipy.linalg.svd(M)
        R = U @ Vt
        B_aligned = R @ pca.B
        procrustes_error = np.linalg.norm(ctx.model.B - B_aligned, 'fro')
        
        return {
            'loading_direct_error': float(direct_error),
            'loading_procrustes_error': float(procrustes_error),
            'improvement_ratio': float(direct_error / procrustes_error) if procrustes_error > 0 else float('inf'),
        }
    
    custom2 = Analyses.custom(loading_error_analysis)
    results2 = custom2.analyze(context)
    
    print("\nResults:")
    for key, value in results2.items():
        print(f"   {key}: {value:.6f}")
    
    print("\n✓ Custom analyses demonstrate extensibility!")
    
    return {**results1, **results2}


def demo_visualization(all_results):
    """Demonstrate visualization."""
    print_section("8. VISUALIZATION")
    
    print("""
This demonstrates rich visualization with:
- Seaborn: Clean, publication-quality static plots
- Plotly: Interactive dashboards (zoom, hover, pan)
    """)
    
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}/")
    
    # Static dashboard
    print("\n📊 Creating static dashboard (seaborn)...")
    static_path = output_dir / "demo_dashboard.png"
    create_manifold_dashboard(all_results, output_path=static_path)
    print(f"   ✓ Saved to: {static_path}")
    
    # Interactive dashboard
    print("\n📊 Creating interactive dashboard (plotly)...")
    interactive_path = output_dir / "demo_interactive.html"
    try:
        create_interactive_plotly_dashboard(all_results, output_path=interactive_path)
        print(f"   ✓ Saved to: {interactive_path}")
    except Exception as e:
        print(f"   ⚠ Skipped (plotly not available): {e}")
    
    print(f"\n✓ Visualizations created!")
    print(f"\nTo view:")
    print(f"   Static:      open {static_path}")
    print(f"   Interactive: open {interactive_path}")


def demo_comprehensive_pipeline(model, results):
    """Demonstrate running all analyses in one pipeline."""
    print_section("9. COMPREHENSIVE ANALYSIS PIPELINE")
    
    print("""
This demonstrates running all analyses together in a single pipeline.
This is the typical workflow for production use.
    """)
    
    # Create context
    context = SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns'],
    )
    
    print("\n🔬 Running all analyses...")
    
    # Define analyses
    analyses = [
        ("Manifold Distances", Analyses.manifold_distances()),
        ("Eigenvalue Analysis", Analyses.eigenvalue_analysis(k_top=model.k)),
        ("Eigenvector Comparison", Analyses.eigenvector_comparison(k=model.k)),
    ]
    
    # Run all
    all_results = {}
    for name, analysis in analyses:
        print(f"   ↳ {name}...")
        result = analysis.analyze(context)
        all_results.update(result)
        print(f"      ✓ Complete ({len(result)} metrics)")
    
    print(f"\n✓ All analyses complete!")
    print(f"\n📊 Summary:")
    print(f"   Total metrics: {len(all_results)}")
    
    if 'dist_grassmannian' in all_results:
        print(f"   Grassmannian Distance:  {all_results['dist_grassmannian']:.6f}")
    if 'eigenvalue_rmse' in all_results:
        print(f"   Eigenvalue RMSE:        {all_results['eigenvalue_rmse']:.6f}")
    if 'mean_correlation' in all_results:
        print(f"   Mean Eigenvector Corr:  {all_results['mean_correlation']:.4f}")
    
    return all_results


def main():
    """Run comprehensive demonstration."""
    print("\n" + "=" * 70)
    print("  FACTOR LAB MANIFOLD ANALYSIS - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Random seed: 42 (for reproducibility)")
    
    # 1. Model Building
    model, rng = demo_model_building()
    
    # 2. Simulation
    results = demo_simulation(model, rng)
    
    # 3. Factor Extraction
    extracted = demo_factor_extraction(results['security_returns'])
    
    # 4. Manifold Analysis
    context, manifold_results = demo_manifold_analysis(model, results)
    
    # 5. Eigenvalue Analysis
    eigen_results = demo_eigenvalue_analysis(context)
    
    # 6. Eigenvector Comparison
    eigvec_results = demo_eigenvector_comparison(context)
    
    # 7. Custom Analysis
    custom_results = demo_custom_analysis(context)
    
    # 8. Combine all results
    all_results = {
        **manifold_results,
        **eigen_results,
        **eigvec_results,
        **custom_results,
    }
    
    # 9. Visualization
    demo_visualization(all_results)
    
    # 10. Comprehensive Pipeline
    pipeline_results = demo_comprehensive_pipeline(model, results)
    
    # Summary
    print_section("DEMONSTRATION COMPLETE")
    
    print("""
✓ All demonstrations completed successfully!

What we covered:
1. Model building with specific variance structure
2. Monte Carlo simulation
3. Factor extraction via SVD
4. Manifold distance analysis (Grassmannian, Procrustes, Chordal)
5. Eigenvalue analysis via LinearOperator (O(kp) memory!)
6. Eigenvector comparison and alignment
7. Custom analysis framework
8. Rich visualization (seaborn + plotly)
9. Comprehensive analysis pipeline

Files created:
• demo_output/demo_dashboard.png          (static visualization)
• demo_output/demo_interactive.html       (interactive dashboard)

Next steps:
1. Open the visualizations
2. Try modifying parameters (k, p, T)
3. Add your own custom analyses
4. Use build_and_simulate.py with JSON configs
5. Integrate into your research workflow

For more information:
• README.md - Complete documentation
• API.md - API reference
• CHEATSHEET.md - Quick reference
• examples/ - More examples
    """)
    
    print("\n" + "=" * 70)
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
