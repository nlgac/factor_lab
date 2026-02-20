#!/usr/bin/env python3
"""
perturbation_study.py - Factor Model Perturbation Analysis (NEW API)
=====================================================================

UPDATED to use FlexibleReturnsSimulator with proper covariance handling.

Compares estimation error from finite samples to rotational perturbation.

Usage:
    python perturbation_study.py perturbation_spec.json
    
Output:
    - perturbation_results.npz: Complete results
    - scatter_stiefel.png: Stiefel distance scatter plot
    - scatter_grassmannian.png: Grassmannian distance scatter plot
    - histogram_full.png: Full sample histogram
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

# Import from factor_lab (NEW API)
from factor_lab import (
    FactorModelData,
    svd_decomposition,
    FlexibleReturnsSimulator,
    create_sampler
)
from factor_lab.analyses.manifold import (
    compute_grassmannian_distance,
    compute_procrustes_distance,
    compute_chordal_distance,
    orthonormalize
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
    
    Returns FactorModelData with B (k, p), F (k, k), D (p, p)
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
    
    return FactorModelData(B=B, F=F, D=D)


def simulate_returns(
    model: FactorModelData, 
    n_periods: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate returns using NEW FlexibleReturnsSimulator API.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model
    n_periods : int
        Number of periods to simulate
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    returns : np.ndarray, shape (n_periods, p)
        Simulated security returns
    """
    # Create factory for samplers
    factory = lambda name, **p: create_sampler(name, rng, **p)
    
    # Use NEW FlexibleReturnsSimulator API
    simulator = FlexibleReturnsSimulator(rng=rng)
    
    results = simulator.simulate(
        model=model,
        n_periods=n_periods,
        factor_return_samplers=factory("normal", loc=0, scale=1),
        idio_return_sampler=factory("normal", loc=0, scale=1)
    )
    
    return results['security_returns']


def perturb_loading_matrix(B: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random rotation perturbation to loading matrix.
    
    Generates rotation matrix R = exp(epsilon * A) where A is skew-symmetric,
    then returns R @ B.
    
    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Original loading matrix
    epsilon : float
        Perturbation size
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    B_perturbed : np.ndarray, shape (k, p)
        Perturbed loading matrix
    """
    k = B.shape[0]
    
    # Generate random skew-symmetric matrix A
    M = rng.normal(0, 1, (k, k))
    A = (M - M.T) / 2  # Make skew-symmetric
    
    # Compute rotation via matrix exponential
    R = expm(epsilon * A)
    
    # Apply rotation
    B_perturbed = R @ B
    
    return B_perturbed


def compute_all_distances(B1: np.ndarray, B2: np.ndarray) -> Dict[str, float]:
    """
    Compute all manifold distances between two loading matrices.
    
    Parameters
    ----------
    B1, B2 : np.ndarray, shape (k, p)
        Loading matrices to compare
    
    Returns
    -------
    distances : dict
        Dictionary with keys:
        - 'grassmannian': Grassmannian distance
        - 'procrustes': Procrustes distance
        - 'chordal': Chordal distance
    """
    # Orthonormalize
    U1 = orthonormalize(B1.T).T  # (k, p)
    U2 = orthonormalize(B2.T).T  # (k, p)
    
    # Compute distances - these functions return dicts with 'distance' key
    grass_result = compute_grassmannian_distance(U1.T, U2.T)
    proc_result = compute_procrustes_distance(U1.T, U2.T)
    chord_result = compute_chordal_distance(U1.T, U2.T)
    
    # Extract the distance value from whatever is returned
    def extract_distance(result):
        if isinstance(result, dict):
            return float(result.get('distance', result.get('dist_grassmannian', result.get('dist_procrustes', result.get('dist_chordal', 0)))))
        elif isinstance(result, (tuple, list)):
            return float(result[0])
        else:
            return float(result)
    
    return {
        'grassmannian': extract_distance(grass_result),
        'procrustes': extract_distance(proc_result),
        'chordal': extract_distance(chord_result),
    }


def run_perturbation_study(spec: PerturbationSpec) -> Dict[str, np.ndarray]:
    """
    Run complete perturbation study.
    
    Compares:
    1. Distance from sampling error (estimate vs true)
    2. Distance from rotational perturbation
    
    Returns
    -------
    results : dict
        Complete results dictionary
    """
    print("\n" + "="*70)
    print("  PERTURBATION STUDY")
    print("="*70)
    
    rng = np.random.default_rng(spec.random_seed)
    
    # Create true model
    model = create_factor_model(spec, rng)
    B_true = model.B
    
    print(f"\n📊 SIMULATION SETUP")
    print(f"   Total samples: {spec.n_total}")
    print(f"   Subset size: {spec.n_subset}")
    print(f"   Number of subsets: {spec.n_total // spec.n_subset}")
    print(f"   Perturbation size: {spec.perturbation_size}")
    
    # Part 1: Sampling error study
    print(f"\n🎲 PART 1: Sampling Error")
    print(f"   Simulating {spec.n_total} observations...")
    
    returns_full = simulate_returns(model, spec.n_total, rng)
    
    print(f"   Extracting model from full sample...")
    model_full = svd_decomposition(returns_full, k=spec.k_factors)
    B_sample_full = model_full.B
    
    # Compute distances for full sample
    dist_full = compute_all_distances(B_true, B_sample_full)
    
    print(f"   ✓ Full sample distances:")
    print(f"      Grassmannian: {dist_full['grassmannian']:.6f}")
    print(f"      Procrustes: {dist_full['procrustes']:.6f}")
    print(f"      Chordal: {dist_full['chordal']:.6f}")
    
    # Subset analysis
    print(f"\n   Computing subset estimates...")
    n_subsets = spec.n_total // spec.n_subset
    
    grassmann_samples = []
    procrustes_samples = []
    chordal_samples = []
    
    for i in range(n_subsets):
        start = i * spec.n_subset
        end = start + spec.n_subset
        returns_subset = returns_full[start:end, :]
        
        model_subset = svd_decomposition(returns_subset, k=spec.k_factors)
        B_subset = model_subset.B
        
        dist_subset = compute_all_distances(B_true, B_subset)
        
        grassmann_samples.append(dist_subset['grassmannian'])
        procrustes_samples.append(dist_subset['procrustes'])
        chordal_samples.append(dist_subset['chordal'])
        
        if (i + 1) % 10 == 0:
            print(f"      Processed {i+1}/{n_subsets} subsets...")
    
    print(f"   ✓ Processed {n_subsets} subsets")
    
    # Part 2: Perturbation study
    print(f"\n🔄 PART 2: Rotational Perturbation")
    print(f"   Generating {n_subsets} perturbed models...")
    
    grassmann_perturb = []
    procrustes_perturb = []
    chordal_perturb = []
    
    for i in range(n_subsets):
        B_perturbed = perturb_loading_matrix(B_true, spec.perturbation_size, rng)
        
        dist_perturb = compute_all_distances(B_true, B_perturbed)
        
        grassmann_perturb.append(dist_perturb['grassmannian'])
        procrustes_perturb.append(dist_perturb['procrustes'])
        chordal_perturb.append(dist_perturb['chordal'])
        
        if (i + 1) % 10 == 0:
            print(f"      Generated {i+1}/{n_subsets} perturbations...")
    
    print(f"   ✓ Generated {n_subsets} perturbations")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print(f"\n   Sampling Error (n={spec.n_subset}):")
    print(f"      Grassmannian: {np.mean(grassmann_samples):.6f} ± {np.std(grassmann_samples):.6f}")
    print(f"      Procrustes:   {np.mean(procrustes_samples):.6f} ± {np.std(procrustes_samples):.6f}")
    print(f"      Chordal:      {np.mean(chordal_samples):.6f} ± {np.std(chordal_samples):.6f}")
    
    print(f"\n   Rotational Perturbation (ε={spec.perturbation_size}):")
    print(f"      Grassmannian: {np.mean(grassmann_perturb):.6f} ± {np.std(grassmann_perturb):.6f}")
    print(f"      Procrustes:   {np.mean(procrustes_perturb):.6f} ± {np.std(procrustes_perturb):.6f}")
    print(f"      Chordal:      {np.mean(chordal_perturb):.6f} ± {np.std(chordal_perturb):.6f}")
    
    # Return complete results
    return {
        # Models
        'B_true': B_true,
        'B_sample_full': B_sample_full,
        'F_true': model.F,
        'D_true': model.D,
        
        # Full sample distances
        'dist_grassmannian_full': dist_full['grassmannian'],
        'dist_procrustes_full': dist_full['procrustes'],
        'dist_chordal_full': dist_full['chordal'],
        
        # Sampling error distributions
        'grassmannian_samples': np.array(grassmann_samples),
        'procrustes_samples': np.array(procrustes_samples),
        'chordal_samples': np.array(chordal_samples),
        
        # Perturbation distributions
        'grassmannian_perturb': np.array(grassmann_perturb),
        'procrustes_perturb': np.array(procrustes_perturb),
        'chordal_perturb': np.array(chordal_perturb),
        
        # Parameters
        'n_total': spec.n_total,
        'n_subset': spec.n_subset,
        'perturbation_size': spec.perturbation_size,
        'p_assets': spec.p_assets,
        'k_factors': spec.k_factors,
    }


def create_visualizations(results: Dict[str, np.ndarray], output_dir: Path = Path("output")):
    """
    Create visualization plots.
    
    Parameters
    ----------
    results : dict
        Results from run_perturbation_study()
    output_dir : Path
        Output directory for plots
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 CREATING VISUALIZATIONS")
    
    # Scatter plots comparing sampling vs perturbation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grassmannian distance
    ax = axes[0]
    ax.scatter(
        results['grassmannian_samples'],
        results['grassmannian_perturb'],
        alpha=0.5,
        s=30
    )
    ax.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax.set_xlabel('Grassmannian Distance (Sampling Error)', fontsize=12)
    ax.set_ylabel('Grassmannian Distance (Perturbation)', fontsize=12)
    ax.set_title('Grassmannian Distance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Procrustes distance
    ax = axes[1]
    ax.scatter(
        results['procrustes_samples'],
        results['procrustes_perturb'],
        alpha=0.5,
        s=30
    )
    ax.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax.set_xlabel('Procrustes Distance (Sampling Error)', fontsize=12)
    ax.set_ylabel('Procrustes Distance (Perturbation)', fontsize=12)
    ax.set_title('Procrustes Distance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'scatter_comparison.png'}")
    plt.close()
    
    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ('grassmannian_samples', 'grassmannian_perturb', 'Grassmannian'),
        ('procrustes_samples', 'procrustes_perturb', 'Procrustes'),
        ('chordal_samples', 'chordal_perturb', 'Chordal'),
    ]
    
    for i, (sample_key, perturb_key, title) in enumerate(metrics):
        # Sampling error histogram
        ax = axes[0, i]
        ax.hist(results[sample_key], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{title} Distance', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{title} - Sampling Error', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Perturbation histogram
        ax = axes[1, i]
        ax.hist(results[perturb_key], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel(f'{title} Distance', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{title} - Perturbation', fontsize=12)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'histograms_all.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'histograms_all.png'}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Perturbation study (NEW API)"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="JSON configuration file"
    )
    args = parser.parse_args()
    
    # Load specification
    spec = PerturbationSpec.from_json(args.config_file)
    
    # Run study
    results = run_perturbation_study(spec)
    
    # Save results
    output_file = "perturbation_results.npz"
    np.savez_compressed(output_file, **results)
    print(f"\n💾 SAVED: {output_file}")
    
    # Create visualizations
    create_visualizations(results)
    
    print(f"\n✅ COMPLETE!\n")


if __name__ == "__main__":
    main()
