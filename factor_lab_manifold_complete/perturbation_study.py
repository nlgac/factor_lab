#!/usr/bin/env python3
"""
perturbation_study.py - Factor Model Perturbation Analysis
============================================================

Compares estimation error from finite samples to rotational perturbation.

Usage:
    python perturbation_study.py perturbation_spec.json
    
Output:
    - perturbation_results.npz: Complete results
    - scatter_stiefel.png: Stiefel distance scatter plot
    - scatter_grassmannian.png: Grassmannian distance scatter plot
    - histogram_full.png: Full sample histogram
    - histograms_subsets.png: 100 subset histograms
"""

import sys
import json
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
    """Specification for perturbation study."""
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
        # Default: decreasing variances
        variances = np.array([0.18**2 / (i+1) for i in range(spec.k_factors)])
    else:
        variances = np.array(spec.factor_variances[:spec.k_factors])
    
    F = np.diag(variances)
    
    # Idiosyncratic covariance D (p, p)
    D = np.eye(spec.p_assets) * spec.idio_variance
    
    print(f"   ✓ B: {B.shape}")
    print(f"   ✓ F: {F.shape}, variances: {variances}")
    print(f"   ✓ D: {D.shape}, diagonal: {spec.idio_variance}")
    
    return FactorModelData(
        factor_loadings=B,
        factor_covariance=F,
        idio_covariance=D
    )


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
    Compute Stiefel (Procrustes) and Grassmannian distances.
    
    Parameters
    ----------
    B_true : array (k, p)
        True loadings
    B_test : array (k, p)
        Test loadings
        
    Returns
    -------
    d_stiefel : float
        Procrustes distance (optimal alignment on Stiefel)
    d_grassmannian : float
        Grassmannian distance (subspace distance)
    """
    # Orthonormalize both
    Q_true, _ = np.linalg.qr(B_true.T, mode='economic')
    Q_test, _ = np.linalg.qr(B_test.T, mode='economic')
    
    # Grassmannian distance
    d_grass, _ = compute_grassmannian_distance(Q_true.T, Q_test.T)
    
    # Procrustes distance (Stiefel with optimal rotation)
    proc_result = compute_procrustes_distance(Q_true.T, Q_test.T)
    d_proc = proc_result['distance']
    
    return d_proc, d_grass


def run_perturbation_study(spec: PerturbationSpec) -> Dict:
    """
    Run complete perturbation study.
    
    Returns dictionary with all results.
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
    d_pert_stiefel, d_pert_grass = compute_distances(B, B_perturbed)
    print(f"   ✓ Stiefel distance (B → B_perturbed): {d_pert_stiefel:.6f}")
    print(f"   ✓ Grassmannian distance (B → B_perturbed): {d_pert_grass:.6f}")
    
    # Step 4: Simulate returns
    returns = simulate_returns(model, spec.n_total, rng)
    
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
        
        # Compute distances: B → B_sample
        d_sample_stiefel, d_sample_grass = compute_distances(B, model_sample.B)
        
        # Store distances
        distances[i, 0] = d_sample_stiefel  # d_sample_stiefel
        distances[i, 1] = d_pert_stiefel     # d_perturbed_stiefel (constant)
        distances[i, 2] = d_sample_grass     # d_sample_grassmannian
        distances[i, 3] = d_pert_grass       # d_perturbed_grassmannian (constant)
        
        # Compute subset statistics
        subset_stats.append(pd.DataFrame(returns_subset).describe())
    
    print(f"   ✓ Complete!")
    
    # Summary statistics
    print(f"\n📊 DISTANCE STATISTICS")
    print(f"\n   Stiefel Distance:")
    print(f"      Sample mean:       {distances[:, 0].mean():.6f}")
    print(f"      Sample std:        {distances[:, 0].std():.6f}")
    print(f"      Perturbation:      {distances[0, 1]:.6f}")
    print(f"\n   Grassmannian Distance:")
    print(f"      Sample mean:       {distances[:, 2].mean():.6f}")
    print(f"      Sample std:        {distances[:, 2].std():.6f}")
    print(f"      Perturbation:      {distances[0, 3]:.6f}")
    
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
    
    # Stiefel distance scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(d_sample_stiefel, d_pert_stiefel, alpha=0.6, s=50)
    plt.axline((0, 0), slope=1, color='red', linestyle='--', 
               label='y = x (equal distance)', linewidth=2)
    plt.xlabel('Sample Distance (Stiefel)', fontsize=12)
    plt.ylabel('Perturbation Distance (Stiefel)', fontsize=12)
    plt.title('Stiefel Distance: Sampling Error vs. Rotational Perturbation', 
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
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python perturbation_study.py <spec.json>")
        print("\nExample spec.json:")
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
}""")
        sys.exit(1)
    
    # Load specification
    spec_file = sys.argv[1]
    spec = PerturbationSpec.from_json(spec_file)
    
    # Create output directory
    output_dir = Path('perturbation_output')
    output_dir.mkdir(exist_ok=True)
    
    # Run study
    results = run_perturbation_study(spec)
    
    # Create visualizations
    create_scatter_plots(results['distances'], output_dir)
    create_histogram_full(results['returns'], output_dir)
    
    # Create subset histograms
    n_subsets = spec.n_total // spec.n_subset
    returns_subsets = [
        results['returns'][i*spec.n_subset:(i+1)*spec.n_subset]
        for i in range(n_subsets)
    ]
    create_histograms_subsets(returns_subsets, output_dir)
    
    # Save results
    save_results(results, output_dir)
    
    print("\n" + "="*70)
    print("  ✅ PERTURBATION STUDY COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}/")
    print("\nFiles created:")
    print("  • perturbation_results.npz    - Complete results")
    print("  • scatter_stiefel.png         - Stiefel distance scatter")
    print("  • scatter_grassmannian.png    - Grassmannian distance scatter")
    print("  • histogram_full.png          - Full sample histogram")
    print("  • histograms_subsets.png      - 100 subset histograms")
    print("  • full_sample_stats.csv       - Sample statistics")
    print("  • distances.csv               - Distance measurements")
    print()


if __name__ == "__main__":
    main()
