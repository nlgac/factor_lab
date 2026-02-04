#!/usr/bin/env python3
"""
perturbation_study.py - Factor Model Perturbation Analysis
============================================================

Compares estimation error from finite samples to rotational perturbation.

EXPERIMENTAL DESIGN:
--------------------
1. Create true factor model with loadings B
2. Apply small rotation to get B_perturbed (distance = d_ref)
3. Simulate returns from B_perturbed
4. Estimate B_sample from subsets of returns
5. Compute estimation error: d(B_perturbed, B_sample)
6. Compare estimation error to reference perturbation d_ref

KEY QUESTION:
-------------
Is the estimation error from finite samples comparable to 
a controlled rotational perturbation of the factor loadings?

Usage:
    python perturbation_study.py <spec.json> [--output {minimal,results,subsets,all}]
    
    Arguments:
        spec.json              Configuration file (required)
        
    Options:
        --output OUTPUT        Controls which files are generated (default: minimal)
                               minimal  - Core visualizations only (scatter + histograms)
                               results  - minimal + perturbation_results.npz + CSV files
                               subsets  - minimal + histograms_subsets.png
                               all      - All outputs (everything)
    
    Examples:
        python perturbation_study.py config.json
        python perturbation_study.py config.json --output all
        python perturbation_study.py config.json --output results
    
Output Files:
    Core (always created with --output minimal or higher):
        - scatter_stiefel.png: Stiefel distance scatter plot
        - scatter_grassmannian.png: Grassmannian distance scatter plot
        - histogram_stiefel_distances.png: Estimation error distribution (Stiefel)
        - histogram_grassmannian_distances.png: Estimation error distribution (Grassmannian)
        - histogram_full.png: Full sample returns distribution
    
    Optional (created with --output results or all):
        - perturbation_results.npz: Complete results (all arrays)
        - full_sample_stats.csv: Statistical summary
        - distances.csv: Distance measurements
    
    Optional (created with --output subsets or all):
        - histograms_subsets.png: 100 subset histograms (small multiples)
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
import scipy.stats

# Import from factor_lab
from factor_lab.types import FactorModelData, svd_decomposition
from factor_lab.analyses.manifold import (
    compute_grassmannian_distance,
    compute_procrustes_distance,
    compute_chordal_distance
)


@dataclass
class PerturbationSpec:
    """
    Specification for perturbation study.
    
    This class defines all parameters for the perturbation study and automatically
    adjusts factor_variances to match k_factors in __post_init__.
    
    Parameters
    ----------
    p_assets : int
        Number of assets in the portfolio
    k_factors : int
        Number of factors in the factor model
    n_total : int, default=6300
        Total number of time periods to simulate
    n_subset : int, default=63
        Size of each subset for estimation (n_total should be divisible by n_subset)
    perturbation_size : float, default=0.1
        Size of perturbation (epsilon parameter for orthogonal transformation)
    factor_variances : List[float] or None, default=None
        Variances for each factor. This list is automatically adjusted in __post_init__:
        - If len < k_factors: Extended by repeating last variance
        - If len > k_factors: Extra variances kept but only first k_factors used
        - If None: Defaults used in create_factor_model (decreasing variances)
    idio_variance : float, default=0.01
        Idiosyncratic variance (same for all assets)
    loading_mean : float, default=0.0
        Mean of factor loadings distribution
    loading_std : float, default=1.0
        Standard deviation of factor loadings distribution
    random_seed : int, default=42
        Random seed for reproducibility
    
    Examples
    --------
    # Example 1: Fewer variances than factors (will extend)
    spec = PerturbationSpec(
        p_assets=500,
        k_factors=5,
        factor_variances=[0.04, 0.02, 0.01]
    )
    # Result: factor_variances = [0.04, 0.02, 0.01, 0.01, 0.01]
    
    # Example 2: More variances than factors (will truncate)
    spec = PerturbationSpec(
        p_assets=500,
        k_factors=2,
        factor_variances=[0.04, 0.02, 0.01, 0.005]
    )
    # Result: factor_variances = [0.04, 0.02]
    
    # Example 3: Exact match (no change)
    spec = PerturbationSpec(
        p_assets=500,
        k_factors=3,
        factor_variances=[0.04, 0.02, 0.01]
    )
    # Result: factor_variances = [0.04, 0.02, 0.01]
    
    # Example 4: None (uses defaults)
    spec = PerturbationSpec(
        p_assets=500,
        k_factors=3,
        factor_variances=None
    )
    # Result: Uses [0.18^2/1, 0.18^2/2, 0.18^2/3]
    """
    p_assets: int
    k_factors: int
    n_total: int = 6300
    n_subset: int = 63
    perturbation_size: float = 0.1
    factor_variances: List[float] = None
    idio_variance: float = 0.01
    loading_mean: float = 0.0
    loading_std: float = 1.0
    random_seed: int = 42
    
    def __post_init__(self):
        """
        Extend factor_variances if needed.
        
        If factor_variances has fewer elements than k_factors,
        extend it by repeating the last variance.
        
        If factor_variances has MORE elements than k_factors,
        they are kept in the spec but only the first k_factors
        are used when constructing the F matrix.
        
        Examples
        --------
        Extension:
            k_factors = 5
            factor_variances = [0.0324, 0.01, 0.0025]
            → Extended to: [0.0324, 0.01, 0.0025, 0.0025, 0.0025]
        
        No truncation (extra variances kept):
            k_factors = 2
            factor_variances = [0.04, 0.02, 0.01, 0.005]
            → Kept as: [0.04, 0.02, 0.01, 0.005]
            → Only first 2 used in F matrix
        """
        if self.factor_variances is not None:
            n_variances = len(self.factor_variances)
            
            if n_variances < self.k_factors:
                # Extend by repeating the last variance
                last_variance = self.factor_variances[-1]
                n_needed = self.k_factors - n_variances
                self.factor_variances = self.factor_variances + [last_variance] * n_needed
                print(f"   ℹ️  Extended factor_variances from {n_variances} to {self.k_factors} values")
                print(f"      (repeated last variance {last_variance} for remaining {n_needed} factors)")
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load from JSON file."""
        with open(filepath) as f:
            config = json.load(f)
        return cls(**config)


def create_factor_model(spec: PerturbationSpec, rng: np.random.Generator) -> FactorModelData:
    """
    Create factor model from specification.
    
    Returns B (k, p), F (k, k), D (p, p)
    """
    print(f"\n🏗️  CREATING FACTOR MODEL")
    print(f"   p = {spec.p_assets} assets")
    print(f"   k = {spec.k_factors} factors")
    
    # Factor loadings B (k, p)
    B = rng.normal(
        spec.loading_mean, 
        spec.loading_std, 
        (spec.k_factors, spec.p_assets)
    )
    
    # Factor covariance F (k, k)
    if spec.factor_variances is None:
        # Default: decreasing variances following a 1/i pattern
        # This gives factors decreasing importance: strong first factor, weaker subsequent ones
        variances = np.array([0.18**2 / (i+1) for i in range(spec.k_factors)])
    else:
        # Use provided variances
        # __post_init__ extends if too few, but keeps extras if too many
        # So we slice to ensure we only use first k_factors elements
        variances = np.array(spec.factor_variances[:spec.k_factors])
    
    # Create diagonal covariance matrix for factors
    # F[i,i] = variance of factor i, F[i,j] = 0 for i≠j (uncorrelated factors)
    F = np.diag(variances)
    
    # Idiosyncratic covariance D (p, p)
    D = np.eye(spec.p_assets) * spec.idio_variance
    
    print(f"   ✓ B: {B.shape}")
    print(f"   ✓ F: {F.shape}, variances: {variances}")
    print(f"   ✓ D: {D.shape}, diagonal: {spec.idio_variance}")
    
    return FactorModelData(B=B, F=F, D=D)


def generate_small_orthogonal(p: int, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate small orthogonal transformation O ∈ O(p) close to identity.
    
    Method: O = exp(ε * A) where A is skew-symmetric
    
    Parameters
    ----------
    p : int
        Dimension
    epsilon : float
        Perturbation size (e.g., 0.1)
    rng : Generator
        Random number generator
        
    Returns
    -------
    O : array (p, p)
        Orthogonal matrix close to I
    """
    print(f"\n🔄 GENERATING SMALL ORTHOGONAL PERTURBATION")
    print(f"   Dimension: p = {p}")
    print(f"   Size: ε = {epsilon}")
    
    # Generate random skew-symmetric matrix
    A_raw = rng.standard_normal((p, p))
    A = (A_raw - A_raw.T) / 2  # Make skew-symmetric
    
    # Scale by epsilon
    A_scaled = epsilon * A
    
    # Compute O = exp(A_scaled)
    O = expm(A_scaled)
    
    # Verify orthogonality
    orthogonality_error = np.linalg.norm(O.T @ O - np.eye(p), 'fro')
    distance_from_identity = np.linalg.norm(O - np.eye(p), 'fro')
    
    print(f"   ✓ Orthogonality error: {orthogonality_error:.2e}")
    print(f"   ✓ ||O - I||_F: {distance_from_identity:.4f}")
    
    return O


def simulate_returns(model: FactorModelData, n_periods: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate security returns from factor model.
    
    r_t = B' f_t + ε_t
    
    Returns
    -------
    returns : array (n_periods, p)
    """
    k, p = model.B.shape
    
    print(f"\n📈 SIMULATING RETURNS")
    print(f"   Periods: {n_periods}")
    print(f"   Assets: {p}")
    
    # Factor returns: f_t ~ N(0, F)
    # Use Cholesky for covariance
    F_chol = np.linalg.cholesky(model.F)
    factors = rng.standard_normal((n_periods, k)) @ F_chol.T
    
    # Idiosyncratic returns: ε_t ~ N(0, D)
    D_diag = np.diag(model.D)
    D_std = np.sqrt(D_diag)
    idio = rng.standard_normal((n_periods, p)) * D_std
    
    # Total returns: r_t = B' f_t + ε_t
    returns = factors @ model.B + idio
    
    print(f"   ✓ Returns shape: {returns.shape}")
    print(f"   ✓ Mean: {returns.mean():.6f}")
    print(f"   ✓ Std: {returns.std():.6f}")
    
    return returns


def compute_distances(B_true: np.ndarray, B_test: np.ndarray) -> Tuple[float, float]:
    """
    Compute Procrustes and Grassmannian distances using factor_lab functions.
    
    Both distances use orthonormalized bases (standard for manifold geometry).
    
    Parameters
    ----------
    B_true : array (k, p)
        True loadings
    B_test : array (k, p)
        Test loadings
        
    Returns
    -------
    d_procrustes : float
        Procrustes distance on Stiefel manifold (orthonormalized frames)
    d_grassmannian : float
        Grassmannian distance (subspace distance)
        
    Notes
    -----
    Both distances use orthonormal bases:
    - Procrustes: Measures frame alignment on Stiefel manifold
    - Grassmannian: Measures subspace distance via principal angles
    
    For small perturbations, these distances are often similar because:
        d_Procrustes^2 ≈ 2k - 2*sum(cos(θ_i)) ≈ sum(θ_i^2) = d_Grassmann^2
    
    This is the expected mathematical behavior, not a bug.
    """
    # Use factor_lab's standard implementations
    # Both internally orthonormalize (correct for manifold geometry)
    proc_result = compute_procrustes_distance(B_true, B_test)
    d_proc = proc_result['distance']
    
    d_grass, _ = compute_grassmannian_distance(B_true, B_test)
    
    return d_proc, d_grass


def run_perturbation_study(spec: PerturbationSpec) -> Dict:
    """
    Run complete perturbation study.
    
    EXPERIMENTAL DESIGN:
    --------------------
    This study compares two sources of error in factor model estimation:
    
    1. PERTURBATION ERROR (controlled, known):
       - Start with true loadings B
       - Apply small rotation: B_perturbed = O @ B (where O ≈ I)
       - Distance: d_ref = d(B, B_perturbed)
       - This is a KNOWN, CONTROLLED error
    
    2. ESTIMATION ERROR (stochastic, unknown):
       - Simulate returns from B_perturbed
       - Extract subsets of returns
       - Estimate B_sample via PCA/SVD
       - Distance: d_est = d(B_perturbed, B_sample)
       - This is UNKNOWN estimation error from finite samples
    
    KEY COMPARISON:
    ---------------
    We compare d_est (estimation error) to d_ref (perturbation error):
    
    - If d_est ≈ d_ref: Estimation error ~ Perturbation size
      → Finite sample error is comparable to model uncertainty
    
    - If d_est > d_ref: Estimation error dominates
      → Need more data or simpler model
    
    - If d_est < d_ref: Good recovery
      → Model is well-estimated despite perturbation
    
    WORKFLOW:
    ---------
    1. Create factor model (B, F, D)
    2. Generate perturbation: B_perturbed = O @ B
    3. Compute reference distance: d_ref = d(B, B_perturbed)
    4. Simulate returns from B_perturbed (not B!)
    5. For each subset:
       a. Extract B_sample via PCA
       b. Compute estimation error: d_est = d(B_perturbed, B_sample)
    6. Visualize distribution of d_est vs. reference d_ref
    
    Returns
    -------
    dict
        Complete results including:
        - B: Original loadings (k, p)
        - B_perturbed: Perturbed loadings (k, p)
        - distances: Array (n_subsets, 4) with:
          [d_est_stiefel, d_ref_stiefel, d_est_grass, d_ref_grass]
        - returns: Simulated returns (n_total, p)
        - sample_models: List of estimated models
        - And more...
    """
    print("="*70)
    print("  FACTOR MODEL PERTURBATION STUDY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Assets (p):           {spec.p_assets}")
    print(f"  Factors (k):          {spec.k_factors}")
    print(f"  Total periods:        {spec.n_total}")
    print(f"  Subset size:          {spec.n_subset}")
    print(f"  Number of subsets:    {spec.n_total // spec.n_subset}")
    print(f"  Perturbation size:    {spec.perturbation_size}")
    print(f"  Random seed:          {spec.random_seed}")
    
    # Initialize RNG
    rng = np.random.default_rng(spec.random_seed)
    
    # Step 1: Create factor model
    model = create_factor_model(spec, rng)
    B, F, D = model.B, model.F, model.D
    
    # Step 2: Generate small orthogonal transformation
    O = generate_small_orthogonal(spec.p_assets, spec.perturbation_size, rng)
    
    # Step 3: Perturbed loadings
    print(f"\n🔀 CREATING PERTURBED LOADINGS")
    B_perturbed = O @ B.T  # (p, k)
    B_perturbed = B_perturbed.T  # (k, p)
    print(f"   ✓ B_perturbed: {B_perturbed.shape}")
    
    # Compute baseline perturbation distances
    # This is the REFERENCE distance: how far did we perturb the true model?
    # We'll compare estimation errors to this reference distance.
    d_pert_stiefel, d_pert_grass = compute_distances(B, B_perturbed)
    print(f"   ✓ Procrustes distance (B → B_perturbed): {d_pert_stiefel:.6f}")
    print(f"   ✓ Grassmannian distance (B → B_perturbed): {d_pert_grass:.6f}")
    
    # Create perturbed model for simulation
    # CRITICAL: We simulate from B_perturbed, not B
    # This is the "true" model we're trying to recover via estimation
    model_perturbed = FactorModelData(B=B_perturbed, F=F, D=D)
    
    # Step 4: Simulate returns from PERTURBED model
    # All returns come from model_perturbed (not the original model)
    # Estimation error is: how well can we recover B_perturbed from finite samples?
    returns = simulate_returns(model_perturbed, spec.n_total, rng)
    
    # Compute statistics for full sample
    full_stats = pd.DataFrame(returns).describe()
    
    # Step 5: Split into subsets
    n_subsets = spec.n_total // spec.n_subset
    print(f"\n✂️  SPLITTING INTO {n_subsets} SUBSETS")
    
    returns_subsets = []
    for i in range(n_subsets):
        start = i * spec.n_subset
        end = start + spec.n_subset
        returns_subsets.append(returns[start:end])
    
    # Step 6: Process each subset
    print(f"\n🔬 ANALYZING {n_subsets} SUBSETS")
    
    distances = np.zeros((n_subsets, 4))  # [d_sample_stiefel, d_pert_stiefel, d_sample_grass, d_pert_grass]
    sample_models = []
    subset_stats = []
    
    for i, returns_subset in enumerate(returns_subsets):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Processing subset {i+1}/{n_subsets}...")
        
        # Estimate factor model via SVD
        model_sample = svd_decomposition(returns_subset, spec.k_factors)
        sample_models.append(model_sample)
        
        # Compute distances: B_perturbed → B_sample
        # CRITICAL CHANGE: We compare to B_perturbed (not B)
        # Question: Can we recover B_perturbed from finite samples?
        # The estimation error is d(B_perturbed, B_sample)
        # We compare this to the perturbation distance d(B, B_perturbed)
        d_sample_stiefel, d_sample_grass = compute_distances(B_perturbed, model_sample.B)
        
        # Store distances
        # Column 0: Stiefel distance from B_perturbed to B_sample (estimation error)
        # Column 1: Stiefel distance from B to B_perturbed (reference perturbation)
        # Column 2: Grassmannian distance from B_perturbed to B_sample (estimation error)
        # Column 3: Grassmannian distance from B to B_perturbed (reference perturbation)
        distances[i, 0] = d_sample_stiefel  # Estimation error (Stiefel)
        distances[i, 1] = d_pert_stiefel     # Reference perturbation (Stiefel)
        distances[i, 2] = d_sample_grass     # Estimation error (Grassmannian)
        distances[i, 3] = d_pert_grass       # Reference perturbation (Grassmannian)
        
        # Compute subset statistics
        subset_stats.append(pd.DataFrame(returns_subset).describe())
    
    print(f"   ✓ Complete!")
    
    # Summary statistics
    print(f"\n📊 DISTANCE STATISTICS")
    print(f"\n   Procrustes Distance:")
    print(f"      Estimation Error (mean):  {distances[:, 0].mean():.6f}")
    print(f"      Estimation Error (std):   {distances[:, 0].std():.6f}")
    print(f"      Reference Perturbation:   {distances[0, 1]:.6f}")
    print(f"      Ratio (Error/Perturbation): {distances[:, 0].mean() / distances[0, 1]:.3f}x")
    print(f"\n   Grassmannian Distance:")
    print(f"      Estimation Error (mean):  {distances[:, 2].mean():.6f}")
    print(f"      Estimation Error (std):   {distances[:, 2].std():.6f}")
    print(f"      Reference Perturbation:   {distances[0, 3]:.6f}")
    print(f"      Ratio (Error/Perturbation): {distances[:, 2].mean() / distances[0, 3]:.3f}x")
    
    # Return all results
    return {
        'B': B,
        'F': F,
        'D': D,
        'O': O,
        'B_perturbed': B_perturbed,
        'returns': returns,
        'distances': distances,
        'sample_models': sample_models,
        'full_stats': full_stats,
        'subset_stats': subset_stats,
        'spec': spec
    }


def create_scatter_plots(distances: np.ndarray, output_dir: Path):
    """
    Create scatter plots comparing sample vs perturbation distances.
    
    Parameters
    ----------
    distances : array (n_subsets, 4)
        [d_sample_stiefel, d_pert_stiefel, d_sample_grass, d_pert_grass]
    """
    print(f"\n📊 CREATING SCATTER PLOTS")
    
    # Extract distances
    d_sample_stiefel = distances[:, 0]
    d_pert_stiefel = distances[:, 1]
    d_sample_grass = distances[:, 2]
    d_pert_grass = distances[:, 3]
    
    # Procrustes distance scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(d_sample_stiefel, d_pert_stiefel, alpha=0.6, s=50)
    plt.axline((0, 0), slope=1, color='red', linestyle='--', 
               label='y = x (equal distance)', linewidth=2)
    plt.xlabel('Sample Distance (Procrustes)', fontsize=12)
    plt.ylabel('Perturbation Distance (Procrustes)', fontsize=12)
    plt.title('Procrustes Distance: Sampling Error vs. Rotational Perturbation', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname_stiefel = output_dir / 'scatter_stiefel.png'
    plt.savefig(fname_stiefel, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fname_stiefel}")
    plt.close()
    
    # Grassmannian distance scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(d_sample_grass, d_pert_grass, alpha=0.6, s=50, color='green')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', 
               label='y = x (equal distance)', linewidth=2)
    plt.xlabel('Sample Distance (Grassmannian)', fontsize=12)
    plt.ylabel('Perturbation Distance (Grassmannian)', fontsize=12)
    plt.title('Grassmannian Distance: Sampling Error vs. Rotational Perturbation', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname_grass = output_dir / 'scatter_grassmannian.png'
    plt.savefig(fname_grass, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fname_grass}")
    plt.close()


def create_distance_histograms(distances: np.ndarray, output_dir: Path):
    """
    Create histograms of estimation errors with reference perturbation distance.
    
    This visualization answers the key question:
    "Is the estimation error from finite samples comparable to 
     the controlled perturbation distance?"
    
    Parameters
    ----------
    distances : array (n_subsets, 4)
        Column 0: Stiefel distance B_perturbed → B_sample (estimation error)
        Column 1: Stiefel distance B → B_perturbed (reference perturbation)
        Column 2: Grassmannian distance B_perturbed → B_sample (estimation error)
        Column 3: Grassmannian distance B → B_perturbed (reference perturbation)
        
    Output
    ------
    Creates two PNG files:
    - histogram_stiefel_distances.png: Stiefel distance histogram
    - histogram_grassmannian_distances.png: Grassmannian distance histogram
    
    Each histogram shows:
    - Distribution of estimation errors (blue bars)
    - Red vertical line at reference perturbation distance
    - Statistics (mean, std, min, max)
    
    Interpretation:
    - If red line is near the center of the distribution:
      → Estimation error ≈ Perturbation distance
      → Finite sample error comparable to model perturbation
    - If red line is to the left:
      → Estimation error > Perturbation distance
      → Sampling uncertainty dominates
    - If red line is to the right:
      → Estimation error < Perturbation distance
      → Good recovery despite perturbation
    """
    print(f"\n📊 CREATING DISTANCE HISTOGRAMS")
    
    # Extract distances
    d_sample_stiefel = distances[:, 0]      # Estimation errors (Stiefel)
    d_pert_stiefel = distances[0, 1]        # Reference perturbation (constant)
    d_sample_grass = distances[:, 2]        # Estimation errors (Grassmannian)
    d_pert_grass = distances[0, 3]          # Reference perturbation (constant)
    
    # VERIFICATION: Print reference distances (should be different!)
    print(f"   Reference Distances (Red Lines):")
    print(f"      Procrustes:  {d_pert_stiefel:.6f}")
    print(f"      Grassmannian:        {d_pert_grass:.6f}")
    print(f"      Difference:          {abs(d_pert_stiefel - d_pert_grass):.6f}")
    print(f"      Ratio (Procrustes/Grass): {d_pert_stiefel / d_pert_grass:.3f}")
    print(f"   Note: For small perturbations, these may be similar (expected behavior).")
    
    # ============================================================
    # Stiefel Distance Histogram
    # ============================================================
    plt.figure(figsize=(12, 7))
    
    # Create histogram of estimation errors
    counts, bins, patches = plt.hist(
        d_sample_stiefel, 
        bins=30, 
        alpha=0.7, 
        color='steelblue',
        edgecolor='black',
        linewidth=1.2,
        label='Estimation Error Distribution'
    )
    
    # Add vertical line for reference perturbation distance
    plt.axvline(
        d_pert_stiefel, 
        color='red', 
        linestyle='--', 
        linewidth=3,
        label=f'Reference $\\Delta_{{\\mathrm{{Procrustes}}}}$(B,B_pert) = {d_pert_stiefel:.6f}',
        zorder=10
    )
    
    # Labels and title
    plt.xlabel('Procrustes Distance (B_perturbed → B_sample)', fontsize=13)
    plt.ylabel('Frequency (Number of Subsets)', fontsize=13)
    plt.title(
        'Estimation Error Distribution vs. Reference Perturbation\n'
        '(Procrustes Distance on Stiefel Manifold)',
        fontsize=15, 
        fontweight='bold'
    )
    
    # Add statistics box
    mean_err = d_sample_stiefel.mean()
    std_err = d_sample_stiefel.std()
    min_err = d_sample_stiefel.min()
    max_err = d_sample_stiefel.max()
    
    stats_text = (
        f'Estimation Error Statistics:\n'
        f'Mean:  {mean_err:.6f}\n'
        f'Std:   {std_err:.6f}\n'
        f'Min:   {min_err:.6f}\n'
        f'Max:   {max_err:.6f}\n'
        f'\n'
        f'Reference Perturbation:\n'
        f'Distance: {d_pert_stiefel:.6f}\n'
        f'\n'
        f'Ratio (Mean Error / Perturbation):\n'
        f'{mean_err / d_pert_stiefel:.3f}x'
    )
    
    plt.text(
        0.98, 0.97, stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    # Legend
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    fname_stiefel = output_dir / 'histogram_stiefel_distances.png'
    plt.savefig(fname_stiefel, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fname_stiefel}")
    plt.close()
    
    # ============================================================
    # Grassmannian Distance Histogram
    # ============================================================
    plt.figure(figsize=(12, 7))
    
    # Create histogram of estimation errors
    counts, bins, patches = plt.hist(
        d_sample_grass, 
        bins=30, 
        alpha=0.7, 
        color='seagreen',
        edgecolor='black',
        linewidth=1.2,
        label='Estimation Error Distribution'
    )
    
    # Add vertical line for reference perturbation distance
    plt.axvline(
        d_pert_grass, 
        color='red', 
        linestyle='--', 
        linewidth=3,
        label=f'Reference $\\Delta_{{\\mathrm{{Grassmannian}}}}$(B,B_pert) = {d_pert_grass:.6f}',
        zorder=10
    )
    
    # Labels and title
    plt.xlabel('Grassmannian Distance (B_perturbed → B_sample)', fontsize=13)
    plt.ylabel('Frequency (Number of Subsets)', fontsize=13)
    plt.title(
        'Estimation Error Distribution vs. Reference Perturbation\n'
        '(Grassmannian Subspace Distance)',
        fontsize=15, 
        fontweight='bold'
    )
    
    # Add statistics box
    mean_err = d_sample_grass.mean()
    std_err = d_sample_grass.std()
    min_err = d_sample_grass.min()
    max_err = d_sample_grass.max()
    
    stats_text = (
        f'Estimation Error Statistics:\n'
        f'Mean:  {mean_err:.6f}\n'
        f'Std:   {std_err:.6f}\n'
        f'Min:   {min_err:.6f}\n'
        f'Max:   {max_err:.6f}\n'
        f'\n'
        f'Reference Perturbation:\n'
        f'Distance: {d_pert_grass:.6f}\n'
        f'\n'
        f'Ratio (Mean Error / Perturbation):\n'
        f'{mean_err / d_pert_grass:.3f}x'
    )
    
    plt.text(
        0.98, 0.97, stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        family='monospace'
    )
    
    # Legend
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    fname_grass = output_dir / 'histogram_grassmannian_distances.png'
    plt.savefig(fname_grass, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fname_grass}")
    plt.close()


def create_histogram_full(returns: np.ndarray, output_dir: Path):
    """Create histogram of full sample."""
    print(f"\n📊 CREATING FULL SAMPLE HISTOGRAM")
    
    plt.figure(figsize=(12, 6))
    
    # Flatten to 1D
    returns_flat = returns.flatten()
    
    plt.hist(returns_flat, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Full Sample Returns Distribution (n={len(returns_flat)})', 
              fontsize=14, fontweight='bold')
    
    # Add statistics
    mean = returns_flat.mean()
    std = returns_flat.std()
    skew = scipy.stats.skew(returns_flat)
    kurt = scipy.stats.kurtosis(returns_flat)
    
    stats_text = f'Mean: {mean:.4f}\nStd: {std:.4f}\nSkew: {skew:.4f}\nKurt: {kurt:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = output_dir / 'histogram_full.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fname}")
    plt.close()


def create_histograms_subsets(returns_subsets: List[np.ndarray], output_dir: Path):
    """
    Create 25x4 panel of 100 histograms.
    
    Parameters
    ----------
    returns_subsets : list of arrays
        Each array is (n_subset, p)
    """
    print(f"\n📊 CREATING SUBSET HISTOGRAMS (100 panels)")
    
    n_subsets = len(returns_subsets)
    
    # 25 rows x 4 columns = 100 panels
    fig, axes = plt.subplots(25, 4, figsize=(16, 60))
    axes = axes.flatten()
    
    for i, returns_subset in enumerate(returns_subsets):
        ax = axes[i]
        
        # Flatten
        data = returns_subset.flatten()
        
        # Histogram
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'Subset {i+1}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean = data.mean()
        ax.axvline(mean, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.suptitle('Returns Distribution for 100 Subsets (63 periods each)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fname = output_dir / 'histograms_subsets.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {fname}")
    plt.close()


def save_results(results: Dict, output_dir: Path):
    """
    Save all results to NPZ file.
    
    Includes:
    - B, F, D (factor model)
    - O (orthogonal perturbation)
    - B_perturbed
    - distances (100, 4)
    - sample_models (100 × (B, F, D))
    - statistics
    """
    print(f"\n💾 SAVING RESULTS")
    
    # Prepare sample models arrays
    n_subsets = len(results['sample_models'])
    k, p = results['B'].shape
    
    B_samples = np.zeros((n_subsets, k, p))
    F_samples = np.zeros((n_subsets, k, k))
    D_samples = np.zeros((n_subsets, p, p))
    
    for i, model in enumerate(results['sample_models']):
        B_samples[i] = model.B
        F_samples[i] = model.F
        D_samples[i] = model.D
    
    # Build save dictionary
    save_dict = {
        # Factor model
        'B_true': results['B'],
        'F_true': results['F'],
        'D_true': results['D'],
        
        # Perturbation
        'O': results['O'],
        'B_perturbed': results['B_perturbed'],
        
        # Returns
        'returns_full': results['returns'],
        
        # Distances
        'distances': results['distances'],
        
        # Sample models
        'B_samples': B_samples,
        'F_samples': F_samples,
        'D_samples': D_samples,
        
        # Metadata
        'p_assets': results['spec'].p_assets,
        'k_factors': results['spec'].k_factors,
        'n_total': results['spec'].n_total,
        'n_subset': results['spec'].n_subset,
        'perturbation_size': results['spec'].perturbation_size,
    }
    
    # Save
    fname = output_dir / 'perturbation_results.npz'
    np.savez_compressed(fname, **save_dict)
    
    print(f"   ✓ Saved: {fname}")
    print(f"\n   📊 File contains {len(save_dict)} arrays:")
    for key, val in sorted(save_dict.items()):
        if isinstance(val, np.ndarray):
            print(f"      • {key:25s} {str(val.shape):20s}")
        else:
            print(f"      • {key:25s} scalar: {val}")
    
    # Save statistics as CSV
    fname_stats = output_dir / 'full_sample_stats.csv'
    results['full_stats'].to_csv(fname_stats)
    print(f"   ✓ Saved: {fname_stats}")
    
    # Save distance summary
    distances_df = pd.DataFrame(
        results['distances'],
        columns=['d_sample_stiefel', 'd_pert_stiefel', 
                 'd_sample_grass', 'd_pert_grass']
    )
    fname_dist = output_dir / 'distances.csv'
    distances_df.to_csv(fname_dist, index=False)
    print(f"   ✓ Saved: {fname_dist}")


def main():
    """
    Main entry point with command-line argument parsing.
    
    Supports selective output generation to save time on large studies.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Factor Model Perturbation Study - Compare estimation error to controlled perturbation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perturbation_study.py config.json
  python perturbation_study.py config.json --output all
  python perturbation_study.py config.json --output results
  python perturbation_study.py config.json --output subsets

Output Levels:
  minimal  - Core visualizations only (scatter plots + histograms) [default]
  results  - Core + NPZ file + CSV files (for detailed analysis)
  subsets  - Core + subset histograms (for quality checks)
  all      - Everything (use for complete documentation)
        """
    )
    
    parser.add_argument(
        'spec_file',
        type=str,
        help='JSON configuration file (required)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        choices=['minimal', 'results', 'subsets', 'all'],
        default='minimal',
        help='Controls which output files are generated (default: minimal)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate spec file exists
    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"ERROR: Configuration file not found: {args.spec_file}")
        print("\nExample configuration file format:")
        print("""{
    "p_assets": 500,
    "k_factors": 3,
    "n_total": 6300,
    "n_subset": 63,
    "perturbation_size": 0.1,
    "factor_variances": [0.0324, 0.01, 0.0025],
    "idio_variance": 0.01,
    "loading_mean": 0.0,
    "loading_std": 1.0,
    "random_seed": 42
}

Notes on factor_variances:
- If fewer values than k_factors: Last value is repeated
  Example: k_factors=5, [0.04, 0.01] → [0.04, 0.01, 0.01, 0.01, 0.01]
  
- If more values than k_factors: Truncated to k_factors
  Example: k_factors=2, [0.04, 0.02, 0.01] → [0.04, 0.02]
  
- If omitted: Uses default decreasing variances
  Example: k_factors=3 → [0.0324, 0.0162, 0.0108]
""")
        sys.exit(1)
    
    # Load specification
    spec = PerturbationSpec.from_json(args.spec_file)
    
    # Display output mode
    print("\n" + "="*70)
    print(f"  OUTPUT MODE: {args.output.upper()}")
    print("="*70)
    if args.output == 'minimal':
        print("  Creating core visualizations only (fastest)")
    elif args.output == 'results':
        print("  Creating core + results files (NPZ + CSV)")
    elif args.output == 'subsets':
        print("  Creating core + subset histograms")
    elif args.output == 'all':
        print("  Creating all outputs (complete documentation)")
    print()
    
    # Create output directory
    output_dir = Path('perturbation_output')
    output_dir.mkdir(exist_ok=True)
    
    # Run study
    results = run_perturbation_study(spec)
    
    # ======================================================================
    # CORE VISUALIZATIONS (always created)
    # ======================================================================
    print(f"\n📊 CREATING CORE VISUALIZATIONS")
    
    create_scatter_plots(results['distances'], output_dir)
    create_distance_histograms(results['distances'], output_dir)
    create_histogram_full(results['returns'], output_dir)
    
    # ======================================================================
    # OPTIONAL: Subset histograms (time-consuming for many subsets)
    # ======================================================================
    if args.output in ['subsets', 'all']:
        print(f"\n📊 CREATING SUBSET HISTOGRAMS (--output {args.output})")
        n_subsets = spec.n_total // spec.n_subset
        returns_subsets = [
            results['returns'][i*spec.n_subset:(i+1)*spec.n_subset]
            for i in range(n_subsets)
        ]
        create_histograms_subsets(returns_subsets, output_dir)
    else:
        print(f"\n⊘  SKIPPING subset histograms (use --output subsets or --output all to include)")
    
    # ======================================================================
    # OPTIONAL: Results files (large NPZ + CSV files)
    # ======================================================================
    if args.output in ['results', 'all']:
        print(f"\n💾 SAVING RESULTS FILES (--output {args.output})")
        save_results(results, output_dir)
    else:
        print(f"\n⊘  SKIPPING results files (use --output results or --output all to include)")
    
    # ======================================================================
    # COMPLETION SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("  ✅ PERTURBATION STUDY COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Output mode: {args.output}")
    print("\nFiles created:")
    
    # Always created (core)
    print("\n  Core Visualizations:")
    print("    • scatter_stiefel.png              - Stiefel distance scatter")
    print("    • scatter_grassmannian.png         - Grassmannian distance scatter")
    print("    • histogram_stiefel_distances.png  - Stiefel error distribution")
    print("    • histogram_grassmannian_distances.png - Grassmannian error distribution")
    print("    • histogram_full.png               - Full sample histogram")
    
    # Conditionally created
    if args.output in ['subsets', 'all']:
        print("\n  Subset Diagnostics:")
        print("    • histograms_subsets.png           - 100 subset histograms")
    
    if args.output in ['results', 'all']:
        print("\n  Results Files:")
        print("    • perturbation_results.npz         - Complete results (all arrays)")
        print("    • full_sample_stats.csv            - Sample statistics")
        print("    • distances.csv                    - Distance measurements")
    
    # Suggestions
    if args.output == 'minimal':
        print("\n  💡 Tip: Use --output all for complete documentation")
        print("          Use --output results to save arrays for further analysis")
    
    print()


if __name__ == "__main__":
    main()
