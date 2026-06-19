#!/usr/bin/env python3
"""
Factor model perturbation study.

Compares estimation error from finite samples to rotational perturbation:
- Create factor model B; perturb by O to get B_perturbed (target)
- For each (n, θ): simulate n_simulations datasets of n returns; PCA → distances
- Histograms: distribution of d(estimate, target); vertical line at d(ground truth, target)

Usage:
    python perturbation_study.py <spec.json>

Output: 3 distance types × len(distributions) histogram files (12 subplots each)
        e.g. histogram_procrustes_gaussian.png, histogram_procrustes_t.png, ...
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from factor_lab.types import FactorModelData, svd_decomposition
from factor_lab.analyses.manifold import (
    compute_grassmannian_distance,
    compute_procrustes_distance,
    compute_chordal_distance,
)


@dataclass
class PerturbationSpec:
    """Specification for grid experiment."""

    p_assets: int
    k_factors: int
    n_values: List[int]
    theta_values: List[float]
    n_simulations: int = 100
    distributions: Optional[List[str]] = None
    factor_variances: Optional[List[float]] = None
    idio_variance: float = 0.01
    loading_mean: float = 0.0
    loading_std: float = 1.0
    t_df: float = 4.0
    random_seed: int = 42

    def __post_init__(self):
        if self.distributions is None:
            self.distributions = ["gaussian", "t"]
        if self.factor_variances is not None and len(self.factor_variances) < self.k_factors:
            last = self.factor_variances[-1]
            self.factor_variances = self.factor_variances + [last] * (
                self.k_factors - len(self.factor_variances)
            )

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath) as f:
            cfg = json.load(f)
        # Only pass fields the spec needs
        fields = {
            "p_assets", "k_factors", "n_values", "theta_values", "n_simulations",
            "distributions", "factor_variances", "idio_variance", "loading_mean",
            "loading_std", "t_df", "random_seed"
        }
        return cls(**{k: cfg[k] for k in fields if k in cfg})


def create_factor_model(spec: PerturbationSpec, rng: np.random.Generator) -> FactorModelData:
    """Build factor model B (k,p), F (k,k), D (p,p)."""
    B = rng.normal(spec.loading_mean, spec.loading_std, (spec.k_factors, spec.p_assets))
    if spec.factor_variances is None:
        variances = np.array([0.18**2 / (i + 1) for i in range(spec.k_factors)])
    else:
        variances = np.array(spec.factor_variances[: spec.k_factors])
    F = np.diag(variances)
    D = np.eye(spec.p_assets) * spec.idio_variance
    return FactorModelData(B=B, F=F, D=D)


def generate_small_orthogonal(p: int, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """
    Orthogonal O ∈ O(p) close to identity: O = exp(ε * A) with A skew-symmetric.
    Single random draw A_raw, then A = (A_raw - A_raw.T) / 2.
    """
    A_raw = rng.standard_normal((p, p))
    A = (A_raw - A_raw.T) / 2
    A_scaled = epsilon * A
    O = expm(A_scaled)
    # Orthogonality check
    orth_err = np.linalg.norm(O.T @ O - np.eye(p), "fro")
    dist_id = np.linalg.norm(O - np.eye(p), "fro")
    print(f"   Orthogonality error: {orth_err:.2e}, ||O-I||_F: {dist_id:.4f}")
    return O


def _standardized_t(rng: np.random.Generator, shape: Tuple[int, ...], df: float) -> np.ndarray:
    """t-variates scaled to unit variance."""
    raw = rng.standard_t(df, size=shape)
    return raw * np.sqrt((df - 2) / df) if df > 2 else raw


def simulate_returns(
    model: FactorModelData,
    n_periods: int,
    rng: np.random.Generator,
    distribution: str = "gaussian",
    t_df: float = 4.0,
) -> np.ndarray:
    """r_t = B' f_t + ε_t. distribution: 'gaussian' or 't'."""
    k, p = model.B.shape
    if distribution == "gaussian":
        z_fac = rng.standard_normal((n_periods, k))
        z_idio = rng.standard_normal((n_periods, p))
    elif distribution == "t":
        z_fac = _standardized_t(rng, (n_periods, k), t_df)
        z_idio = _standardized_t(rng, (n_periods, p), t_df)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    F_chol = np.linalg.cholesky(model.F)
    factors = z_fac @ F_chol.T
    D_std = np.sqrt(np.diag(model.D))
    idio = z_idio * D_std
    return factors @ model.B + idio


def compute_distances(B_true: np.ndarray, B_test: np.ndarray) -> Tuple[float, float, float]:
    """Procrustes, Grassmannian, Chordal distances."""
    proc = compute_procrustes_distance(B_true, B_test)["distance"]
    grass, _ = compute_grassmannian_distance(B_true, B_test)
    chord = compute_chordal_distance(B_true, B_test)
    return proc, grass, chord


def run_grid_experiment(spec: PerturbationSpec, distribution: str = "gaussian") -> Dict:
    """Run (n, θ) grid for one distribution. Returns grid_results, n_values, theta_values."""
    rng = np.random.default_rng(spec.random_seed)
    model = create_factor_model(spec, rng)
    B, F, D = model.B, model.F, model.D
    k = spec.k_factors

    grid_results = {}
    total = len(spec.n_values) * len(spec.theta_values)
    cell = 0

    for theta in spec.theta_values:
        O = generate_small_orthogonal(spec.p_assets, theta, rng)
        B_perturbed = (O @ B.T).T
        model_perturbed = FactorModelData(B=B_perturbed, F=F, D=D)
        d_ref_proc, d_ref_grass, d_ref_chord = compute_distances(B, B_perturbed)

        for n in spec.n_values:
            cell += 1
            print(f"   Cell {cell}/{total}: n={n}, θ={theta}")

            # Match original design: one long series of n_simulations*n, then split into
            # n_simulations consecutive blocks of n (same as legacy "subset" logic).
            n_periods = spec.n_simulations * n
            returns_long = simulate_returns(
                model_perturbed, n_periods, rng,
                distribution=distribution, t_df=spec.t_df,
            )
            est_proc, est_grass, est_chord = [], [], []
            for i in range(spec.n_simulations):
                block = returns_long[i * n : (i + 1) * n]
                model_sample = svd_decomposition(block, k)
                dp, dg, dc = compute_distances(B_perturbed, model_sample.B)
                est_proc.append(dp)
                est_grass.append(dg)
                est_chord.append(dc)

            grid_results[(n, theta)] = {
                "procrustes": {"estimates": est_proc, "ref": d_ref_proc},
                "grassmannian": {"estimates": est_grass, "ref": d_ref_grass},
                "chordal": {"estimates": est_chord, "ref": d_ref_chord},
            }

    return {
        "grid_results": grid_results,
        "n_values": spec.n_values,
        "theta_values": spec.theta_values,
    }


def create_grid_histograms(
    results_by_distribution: Dict[str, Dict],
    n_values: List[int],
    theta_values: List[float],
    output_dir: Path,
) -> None:
    """Write histogram_procrustes_{dist}.png etc. (12 subplots each)."""
    n_rows, n_cols = len(n_values), len(theta_values)
    figsize = (4 * n_cols, 3 * n_rows)
    # (distance name, color) — name used for data key and filename
    configs = [
        ("procrustes", "steelblue"),
        ("grassmannian", "seagreen"),
        ("chordal", "orange"),
    ]

    for name, color in configs:
        # Shared x and y scale across Gaussian and t for this distance type
        all_vals = []
        for grid_results in results_by_distribution.values():
            for n in n_values:
                for theta in theta_values:
                    data = grid_results[(n, theta)][name]
                    all_vals.extend(data["estimates"])
                    all_vals.append(data["ref"])
        all_vals = np.array(all_vals)
        x_min, x_max = all_vals.min(), all_vals.max()
        bins = np.linspace(x_min, x_max, 61)

        y_max = 0
        for grid_results in results_by_distribution.values():
            for n in n_values:
                for theta in theta_values:
                    counts, _ = np.histogram(grid_results[(n, theta)][name]["estimates"], bins=bins)
                    y_max = max(y_max, counts.max())
        y_max = y_max * 1.05 if y_max > 0 else 1

        for dist_name, grid_results in results_by_distribution.items():
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
            axes = np.atleast_2d(axes)

            for i, n in enumerate(n_values):
                for j, theta in enumerate(theta_values):
                    ax = axes[i, j]
                    data = grid_results[(n, theta)][name]
                    ax.hist(data["estimates"], bins=bins, alpha=0.7, color=color, edgecolor="black")
                    ax.axvline(data["ref"], color="red", linestyle="--", lw=2, label="GT→target")
                    ax.set_ylim(0, y_max)
                    ax.set_title(f"n={n}, θ={theta}", fontsize=10)
                    ax.tick_params(labelsize=8)
                    ax.legend(fontsize=7, loc="upper right")
                    ax.grid(True, alpha=0.3)

            fig.suptitle(f"{name.title()} Distance ({dist_name})", fontsize=14, fontweight="bold", y=1.01)
            plt.tight_layout()
            fname = output_dir / f"histogram_{name}_{dist_name}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"   ✓ {fname}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Factor model perturbation study (grid mode)")
    parser.add_argument("spec_file", help="JSON spec (p_assets, k_factors, n_values, theta_values, distributions, ...)")
    args = parser.parse_args()

    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"ERROR: File not found: {args.spec_file}")
        sys.exit(1)

    spec = PerturbationSpec.from_json(args.spec_file)
    dists = spec.distributions or ["gaussian", "t"]

    output_dir = Path("perturbation_output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  PERTURBATION STUDY (grid mode)")
    print("=" * 60)
    print(f"  p={spec.p_assets}, k={spec.k_factors}")
    print(f"  n_values={spec.n_values}, theta_values={spec.theta_values}")
    print(f"  distributions={dists}, n_simulations={spec.n_simulations}")
    print()

    results_by_dist = {}
    n_vals = theta_vals = None
    for d in dists:
        print(f"\n🔄 {d}" + (f" (df={spec.t_df})" if d == "t" else ""))
        out = run_grid_experiment(spec, distribution=d)
        results_by_dist[d] = out["grid_results"]
        n_vals, theta_vals = out["n_values"], out["theta_values"]

    print(f"\n📊 Histograms ({len(dists) * 3} files)")
    create_grid_histograms(results_by_dist, n_vals, theta_vals, output_dir)

    print("\n✅ Done. Output: perturbation_output/")
    for d in dists:
        print(f"   histogram_procrustes_{d}.png")
        print(f"   histogram_grassmannian_{d}.png")
        print(f"   histogram_chordal_{d}.png")


if __name__ == "__main__":
    main()
