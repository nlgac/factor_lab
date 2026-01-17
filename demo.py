#!/usr/bin/env python3
"""
demo.py - Comprehensive Demonstration of Factor Lab

This script demonstrates the complete factor_lab workflow:
1. Generative Model Creation - Build synthetic factor models
2. Returns Simulation - Generate Monte Carlo samples
3. SVD Decomposition - Extract factors from returns data
4. Covariance Validation - Verify model accuracy
5. Portfolio Optimization - Find minimum variance portfolios

Usage:
    python demo.py

Each section is self-contained with explanatory comments.
"""

import numpy as np
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
    print(f"  Factor variances: {np.diag(model.F)[:3]}...")
    print(f"  Explained variance: {compute_explained_variance(model):.1%}")


def demo_generative_model(rng: np.random.Generator, factory: DistributionFactory) -> FactorModelData:
    """
    Demonstrate creating a synthetic factor model.
    
    This shows how to use DataSampler to create factor models with
    specified statistical properties for testing and simulation.
    """
    print_section("1. GENERATIVE MODEL CREATION")
    
    p_assets = 100  # 100 stocks
    k_factors = 3   # Market, Value, Momentum
    
    print(f"\nCreating a {k_factors}-factor model for {p_assets} assets...")
    
    # Create the DataSampler
    sampler = DataSampler(p=p_assets, k=k_factors, rng=rng)
    
    # Configure with specific distributions for each component:
    # - beta: Factor loadings (how much each asset is exposed to each factor)
    # - factor_vol: Volatility of each factor
    # - idio_vol: Asset-specific (idiosyncratic) volatility
    
    model = sampler.configure(
        # Each factor has different loading distribution
        beta=[
            factory.create("normal", mean=1.0, std=0.3),   # Market: avg beta=1
            factory.create("normal", mean=0.0, std=0.5),   # Value: centered at 0
            factory.create("normal", mean=0.0, std=0.5),   # Momentum: centered at 0
        ],
        # Factor volatilities (annualized)
        factor_vol=[
            factory.create("constant", value=0.20),  # Market: 20% vol
            factory.create("constant", value=0.12),  # Value: 12% vol
            factory.create("constant", value=0.10),  # Momentum: 10% vol
        ],
        # Idiosyncratic volatility varies by asset
        idio_vol=factory.create("uniform", low=0.15, high=0.30)
    ).generate()
    
    print_model_summary(model, "Generative Model")
    
    # Show some loadings
    print(f"\n  Sample loadings (first 5 assets):")
    print(f"    Market: {model.B[0, :5].round(3)}")
    print(f"    Value:  {model.B[1, :5].round(3)}")
    print(f"    Momentum: {model.B[2, :5].round(3)}")
    
    return model


def demo_simulation(
    model: FactorModelData, 
    rng: np.random.Generator, 
    factory: DistributionFactory
) -> Dict[str, np.ndarray]:
    """
    Demonstrate Monte Carlo simulation of returns.
    
    This shows how to generate synthetic returns that match the
    covariance structure implied by the factor model.
    """
    print_section("2. RETURNS SIMULATION")
    
    n_periods = 2000  # ~8 years of daily data
    
    print(f"\nSimulating {n_periods} periods of returns...")
    
    # Create simulator
    simulator = ReturnsSimulator(model, rng=rng)
    
    # You can use different distributions for innovations:
    # - Normal: Standard assumption
    # - Student's t: Fat tails (more realistic for financial returns)
    # - Custom: Any distribution you define
    
    # Here we use Student's t for factor innovations (fat tails)
    # and normal for idiosyncratic innovations
    
    results = simulator.simulate(
        n_periods=n_periods,
        factor_samplers=[factory.create("student_t", df=5)] * model.k,
        idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p,
        sample_log_rows=5  # Log first 5 rows for debugging
    )
    
    returns = results["security_returns"]
    
    print(f"\n  Returns shape: {returns.shape}")
    print(f"  Mean return: {returns.mean():.4f}")
    print(f"  Std return: {returns.std():.4f}")
    print(f"  Min return: {returns.min():.4f}")
    print(f"  Max return: {returns.max():.4f}")
    
    # Check some basic statistics
    from scipy.stats import kurtosis
    k = kurtosis(returns.flatten())
    print(f"  Excess kurtosis: {k:.2f} (>0 indicates fat tails)")
    
    return results


def demo_svd_extraction(returns: np.ndarray) -> FactorModelData:
    """
    Demonstrate factor extraction via SVD.
    
    This shows how to extract a factor model from historical returns
    data. SVD is more numerically stable than PCA on the covariance
    matrix.
    """
    print_section("3. SVD DECOMPOSITION")
    
    print(f"\nExtracting factors from returns ({returns.shape[0]} periods, {returns.shape[1]} assets)...")
    
    # First, determine how many factors to use
    # Option 1: Use domain knowledge (we know we used 3 factors)
    # Option 2: Use explained variance target
    
    k_auto = select_k_by_variance(returns, target_explained=0.80)
    print(f"\n  Automatic k selection (80% variance): k={k_auto}")
    
    # Extract with k=3 (matching our generative model)
    k = 3
    model = svd_decomposition(returns, k=k)
    
    print_model_summary(model, "SVD-Extracted Model")
    
    # The SVD model includes pre-computed transforms for efficient simulation
    print(f"\n  Pre-computed transforms available:")
    print(f"    Factor transform: {'Diagonal' if model.factor_transform.is_diagonal else 'Dense'}")
    print(f"    Idio transform: {'Diagonal' if model.idio_transform.is_diagonal else 'Dense'}")
    
    return model


def demo_validation(model: FactorModelData, returns: np.ndarray) -> None:
    """
    Demonstrate covariance validation.
    
    This shows how to verify that a factor model accurately captures
    the covariance structure of the underlying returns.
    """
    print_section("4. COVARIANCE VALIDATION")
    
    print(f"\nValidating model against empirical covariance...")
    
    validator = CovarianceValidator(model)
    validation = validator.compare(returns)
    
    print(f"\n  Frobenius error: {validation.frobenius_error:.4f}")
    print(f"  Mean absolute error: {validation.mean_absolute_error:.6f}")
    print(f"  Max absolute error: {validation.max_absolute_error:.6f}")
    print(f"  Explained variance ratio: {validation.explained_variance_ratio:.1%}")
    
    # Interpretation
    if validation.frobenius_error < 0.5:
        print("\n  ✓ Excellent model fit!")
    elif validation.frobenius_error < 1.0:
        print("\n  ✓ Good model fit")
    else:
        print("\n  ⚠ Consider using more factors or checking data quality")


def demo_optimization(model: FactorModelData) -> None:
    """
    Demonstrate portfolio optimization.
    
    This shows how to find minimum variance portfolios with various
    constraint sets using the factor model covariance structure.
    """
    print_section("5. PORTFOLIO OPTIMIZATION")
    
    # ----------------------------------------------------------------------
    # Example 1: Simple long-only minimum variance
    # ----------------------------------------------------------------------
    print("\n--- Example 1: Long-Only Minimum Variance ---")
    
    result = minimum_variance_portfolio(model, long_only=True)
    
    print(f"  Solved: {result.solved}")
    print(f"  Portfolio Risk: {result.risk:.2%}")
    print(f"  Non-zero weights: {np.sum(result.weights > 1e-4)}")
    print(f"  Max weight: {result.weights.max():.2%}")
    
    # ----------------------------------------------------------------------
    # Example 2: Diversified portfolio (max 2% per asset)
    # ----------------------------------------------------------------------
    print("\n--- Example 2: Diversified (Max 2% per Asset) ---")
    
    result = minimum_variance_portfolio(model, long_only=True, max_weight=0.02)
    
    print(f"  Solved: {result.solved}")
    print(f"  Portfolio Risk: {result.risk:.2%}")
    print(f"  Non-zero weights: {np.sum(result.weights > 1e-4)}")
    print(f"  Max weight: {result.weights.max():.2%}")
    
    # ----------------------------------------------------------------------
    # Example 3: Custom constraints using ScenarioBuilder
    # ----------------------------------------------------------------------
    print("\n--- Example 3: Custom Constraints ---")
    
    # Build a complex scenario using the fluent API
    scenario = (ScenarioBuilder(model.p)
        .create(
            name="130/30 Style",
            description="Allow limited shorting with leverage constraint"
        )
        .add_fully_invested()
        .add_box_constraints(low=-0.05, high=0.10)  # -5% to +10% per asset
        .build())
    
    print(f"  Scenario: {scenario.name}")
    print(f"  Equality constraints: {scenario.n_equality()}")
    print(f"  Inequality constraints: {scenario.n_inequality()}")
    
    # Create optimizer and apply scenario
    optimizer = FactorOptimizer(model)
    optimizer.apply_scenario(scenario)
    
    result = optimizer.solve()
    
    print(f"  Solved: {result.solved}")
    print(f"  Portfolio Risk: {result.risk:.2%}")
    print(f"  Sum of weights: {result.weights.sum():.4f}")
    print(f"  Short positions: {np.sum(result.weights < -1e-4)}")
    print(f"  Min weight: {result.weights.min():.2%}")
    print(f"  Max weight: {result.weights.max():.2%}")
    
    # Show top 5 positions
    top_5 = np.argsort(result.weights)[-5:][::-1]
    print(f"\n  Top 5 holdings:")
    for i in top_5:
        print(f"    Asset {i}: {result.weights[i]:.2%}")


def demo_pca_comparison(returns: np.ndarray) -> None:
    """
    Demonstrate PCA vs SVD comparison.
    
    This shows that PCA and SVD give equivalent results when applied
    correctly, but SVD is numerically more stable.
    """
    print_section("6. PCA vs SVD COMPARISON")
    
    k = 3
    
    # SVD on returns
    model_svd = svd_decomposition(returns, k=k)
    
    # PCA on sample covariance
    sample_cov = np.cov(returns, rowvar=False)
    B_pca, F_pca = pca_decomposition(sample_cov, k=k)
    
    print(f"\nComparing eigenvalues (factor variances):")
    print(f"  SVD: {np.diag(model_svd.F).round(6)}")
    print(f"  PCA: {np.diag(F_pca).round(6)}")
    
    # Check if they match
    svd_vals = np.sort(np.diag(model_svd.F))[::-1]
    pca_vals = np.sort(np.diag(F_pca))[::-1]
    
    if np.allclose(svd_vals, pca_vals, rtol=1e-10):
        print("\n  ✓ Eigenvalues match perfectly!")
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
