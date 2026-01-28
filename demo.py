#!/usr/bin/env python3
"""
demo.py - Comprehensive Demonstration of Factor Lab
===================================================

This script demonstrates the complete factor_lab workflow:
1. Generative Model Creation - Build synthetic factor models
2. Returns Simulation - Generate Monte Carlo samples
3. SVD Decomposition - Extract factors from returns data
4. Covariance Validation - Verify model accuracy
5. Portfolio Optimization - Find minimum variance portfolios
6. PCA vs SVD Comparison - Check numerical consistency

Usage:
    python demo.py
"""

import numpy as np
import scipy.stats
from typing import Dict, Any

# =============================================================================
# IMPORTS
# =============================================================================

from factor_lab import (
    # Decomposition
    svd_decomposition,
    pca_decomposition,
    compute_explained_variance,
    select_k_by_variance,
    
    # Simulation
    ReturnsSimulator,
    CovarianceValidator,
    simulate_returns,
    
    # Optimization
    FactorOptimizer,
    ScenarioBuilder,
    minimum_variance_portfolio,
    
    # Samplers
    DistributionFactory,
    DataSampler,
    
    # Types
    FactorModelData,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_model_summary(model: FactorModelData, name: str = "Model") -> None:
    """Print summary statistics for a factor model."""
    print(f"\n{name} Summary:")
    print(f"  Factors (k): {model.k}")
    print(f"  Assets (p):  {model.p}")
    
    # Print factor variances (diagonal of F)
    f_vars = np.diag(model.F)
    print(f"  Factor variances: {f_vars[:3]}{'...' if len(f_vars)>3 else ''}")
    
    # Calculate explained variance ratio
    explained = compute_explained_variance(model)
    print(f"  Explained variance: {explained:.1%}")
    
    # Print sample loadings
    print(f"\n  Sample loadings (first 5 assets):")
    factor_names = ["Market", "Value", "Momentum", "Quality", "Vol"]
    for i in range(min(model.k, 3)):
        fname = factor_names[i] if i < len(factor_names) else f"Factor {i+1}"
        print(f"    {fname:<8}: {np.round(model.B[i, :5], 3)}")


# =============================================================================
# 1. GENERATIVE MODEL
# =============================================================================

def demo_generative_model(rng: np.random.Generator, factory: DistributionFactory) -> FactorModelData:
    """Demonstrate creating a synthetic factor model."""
    print_section("1. GENERATIVE MODEL CREATION")
    
    p_assets = 100
    k_factors = 3
    
    print(f"Creating a {k_factors}-factor model for {p_assets} assets...")
    
    # Configure samplers
    # 1. Betas: Normal distribution
    # 2. Factor Vols: Uniform [0.1, 0.2] (squared for covariance)
    # 3. Idio Vols: Constant 0.1
    
    sampler = DataSampler(p=p_assets, k=k_factors, rng=rng)
    model = sampler.configure(
        beta=factory.create("normal", mean=0.2, std=0.5), # Market-like betas
        factor_vol=factory.create("uniform", low=0.1, high=0.2),
        idio_vol=factory.create("constant", value=0.10)
    ).generate()
    
    print_model_summary(model, "Generative Model")
    return model


# =============================================================================
# 2. SIMULATION
# =============================================================================

def demo_simulation(model: FactorModelData, rng: np.random.Generator, factory: DistributionFactory) -> Dict[str, Any]:
    """Demonstrate Monte Carlo simulation."""
    print_section("2. RETURNS SIMULATION")
    
    n_periods = 2000
    print(f"Simulating {n_periods} periods of returns...")
    
    # Create simulator
    simulator = ReturnsSimulator(model, rng=rng)
    
    # Define Innovation Distributions
    # We'll use Student's t (df=5) for fat tails
    t_dist = factory.create("student_t", df=5)
    
    results = simulator.simulate(
        n_periods=n_periods,
        factor_samplers=[t_dist] * model.k,
        idio_samplers=[t_dist] * model.p
    )
    
    returns = results["security_returns"]
    
    # Stats
    print(f"\n  Returns shape: {returns.shape}")
    print(f"  Mean return: {np.mean(returns):.4f}")
    print(f"  Std return: {np.std(returns):.4f}")
    print(f"  Min return: {np.min(returns):.4f}")
    print(f"  Max return: {np.max(returns):.4f}")
    
    # Kurtosis check
    kurt = scipy.stats.kurtosis(returns.flatten())
    print(f"  Excess kurtosis: {kurt:.2f} (>0 indicates fat tails)")
    
    return results


# =============================================================================
# 3. SVD EXTRACTION
# =============================================================================

def demo_svd_extraction(returns: np.ndarray) -> FactorModelData:
    """Demonstrate extracting factors via SVD."""
    print_section("3. SVD DECOMPOSITION")
    
    T, p = returns.shape
    print(f"Extracting factors from returns ({T} periods, {p} assets)...")
    
    # 1. Determine k automatically
    k_suggested = select_k_by_variance(returns, target_explained=0.80)
    print(f"\n  Automatic k selection (80% variance): k={k_suggested}")
    
    # 2. Extract 3 factors (for comparison with generative model)
    k_extract = 3
    model = svd_decomposition(returns, k=k_extract)
    
    print_model_summary(model, "SVD-Extracted Model")
    
    # Check transforms (Safely!)
    print("\n  Pre-computed transforms available:")
    
    ft_str = "None"
    if model.factor_transform is not None:
        ft_str = 'Diagonal' if model.factor_transform.is_diagonal else 'Dense'
        
    it_str = "None"
    if model.idio_transform is not None:
        it_str = 'Diagonal' if model.idio_transform.is_diagonal else 'Dense'
        
    print(f"    Factor transform: {ft_str}")
    print(f"    Idio transform:   {it_str}")
    
    return model


# =============================================================================
# 4. VALIDATION
# =============================================================================

def demo_validation(model: FactorModelData, returns: np.ndarray) -> None:
    """Demonstrate covariance validation."""
    print_section("4. COVARIANCE VALIDATION")
    
    validator = CovarianceValidator(model)
    result = validator.compare(returns)
    
    print("Comparing Model-Implied Covariance vs Empirical Covariance:")
    print(f"  Frobenius Error: {result.frobenius_error:.4f}")
    print(f"  Mean Abs Error:  {result.mean_absolute_error:.6f}")
    print(f"  Max Abs Error:   {result.max_absolute_error:.6f}")
    
    # Check if fit is acceptable (heuristic)
    if result.mean_absolute_error < 0.01:
        print("\n  >> Model Fit: EXCELLENT")
    else:
        print("\n  >> Model Fit: ACCEPTABLE (Simulated noise expected)")


# =============================================================================
# 5. OPTIMIZATION
# =============================================================================

def demo_optimization(model: FactorModelData) -> None:
    """Demonstrate portfolio optimization."""
    print_section("5. PORTFOLIO OPTIMIZATION")
    
    print("Solving Minimum Variance Portfolio (Long Only)...")
    
    try:
        # Use the convenience function
        result = minimum_variance_portfolio(
            model, 
            long_only=True, 
            max_weight=0.10  # 10% max weight per asset
        )
        
        if result.solved:
            print(f"\n  Optimization Status: {result.status}")
            print(f"  Portfolio Risk:      {result.risk:.2%}")
            
            # Show top weights
            w = result.weights
            top_indices = np.argsort(w)[::-1][:5]
            print("\n  Top 5 Asset Weights:")
            for idx in top_indices:
                print(f"    Asset {idx:<3}: {w[idx]:.1%}")
                
            print(f"\n  Sum of weights: {np.sum(w):.4f}")
        else:
            print("\n  Optimization failed to converge.")
            
    except ImportError:
        print("\n  CVXPY not installed. Skipping optimization demo.")
    except Exception as e:
        print(f"\n  Optimization error: {e}")


# =============================================================================
# 6. PCA vs SVD COMPARISON
# =============================================================================

def demo_pca_comparison(returns: np.ndarray) -> None:
    """Compare numerical results of PCA vs SVD."""
    print_section("6. PCA vs SVD COMPARISON")
    
    k = 3
    print(f"Comparing decomposition methods for k={k}...")
    
    # SVD
    print("  Running SVD...")
    model_svd = svd_decomposition(returns, k=k)
    svd_vals = np.diag(model_svd.F)
    
    # PCA
    print("  Running PCA on Covariance Matrix...")
    cov = np.cov(returns, rowvar=False)
    _, F_pca = pca_decomposition(cov, k=k)
    pca_vals = np.diag(F_pca)
    
    # Compare Eigenvalues (Factor Variances)
    print("\n  Factor Variances (Eigenvalues):")
    print(f"    SVD: {svd_vals}")
    print(f"    PCA: {pca_vals}")
    
    if np.allclose(svd_vals, pca_vals, rtol=1e-5):
        print("\n  >> SUCCESS: Eigenvalues match perfectly!")
    else:
        max_diff = np.max(np.abs(svd_vals - pca_vals))
        print(f"\n  Maximum difference: {max_diff:.2e}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print(" FACTOR LAB DEMONSTRATION")
    print(" A Python Library for Factor Model Construction and Simulation")
    print("="*60)
    
    # Setup: Create a seeded RNG for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    factory = DistributionFactory(rng=rng)
    
    print(f"\nUsing random seed: {seed} (for reproducibility)")
    
    # Run demonstrations
    model = demo_generative_model(rng, factory)
    results = demo_simulation(model, rng, factory)
    extracted_model = demo_svd_extraction(results["security_returns"])
    demo_validation(extracted_model, results["security_returns"])
    demo_optimization(extracted_model)
    demo_pca_comparison(results["security_returns"])
    
    # Summary
    print_section("SUMMARY")
    print("""
    This demonstration covered:
    
    1. Creating synthetic factor models with controlled properties
    2. Simulating returns with custom innovation distributions
    3. Extracting factors from returns via SVD
    4. Validating model fit against empirical covariance
    5. Optimizing portfolios with various constraint sets
    6. Comparing PCA and SVD decomposition methods
    
    For more information, see the documentation and API reference.
    """)


if __name__ == "__main__":
    main()