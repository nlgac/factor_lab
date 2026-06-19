"""
large_sample_perturbation_study.py
===================================
Compares two sources of distance on the Grassmannian / Stiefel manifold:

  1. **Sampling error** – distance between the *true* loading subspace and
     the subspace estimated from a finite sample via SVD.  Varies with
     subsample size p.

  2. **Geodesic perturbation** – distance between the true loading subspace
     and a frame obtained by travelling exactly epsilon along a random
     geodesic on the Stiefel manifold.  Controlled precisely via epsilon.

The central question: for a given epsilon, how does the deliberately-induced
perturbation compare (in all three manifold metrics) with the sampling noise
that arises naturally at subsample size p?

Usage
-----
    python large_sample_perturbation_study.py <config.json>

Outputs (written to a folder named after the config file):
    summary.csv                          – tidy stats table
    distance_comparison_<metric>.png     – histogram grids
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
import scipy.stats
from tqdm import tqdm

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

# Separate RNG for perturbation directions so that changing the simulation
# seed does not alter which geodesic directions are sampled (and vice versa).
direction_rng = np.random.default_rng(12345)


@dataclass
class SuperSetSpec:
    """
    Full specification for one perturbation study run.

    All fields have sensible defaults so that a minimal JSON config only
    needs to override what differs from the baseline.

    Attributes
    ----------
    p_assets : int
        Total number of assets in the *true* model.
    subsample_sizes : list of int
        Asset counts at which sampling error is measured.  Each value must
        be <= p_assets.
    k_factors : int
        Number of latent factors.
    n_windows : int
        Number of independent return windows to simulate.  Each window
        contributes one sampling-error observation.
    window_size : int
        Number of time steps (observations) in each window.
    factor_variances : list of float or None
        Diagonal of the factor covariance matrix F.  If None, defaults to
        [0.18**2 / (i+1) for i in range(k_factors)] (geometrically decreasing).
    factor_loadings_distribution : str
        'normal' or 'heavy-tailed' (Student-t with t_df degrees of freedom).
    specific_variance_type : str
        'homoskedastic' (all assets share idio_variance) or
        'heteroskedastic' (per-asset variance drawn from Uniform[0.5, 2.0]
        scaled by idio_variance).
    idio_variance : float
        Idiosyncratic variance level (mean variance for heteroskedastic case).
    loading_mean : float
        Mean of the loading distribution.
    loading_std : float
        Std dev of the loading distribution.
    perturbation_epsilons : list of float
        Geodesic distances at which perturbation frames are generated.
    random_seed : int
        Seed for the main RNG (model + simulation).  The perturbation
        direction RNG uses a fixed separate seed (12345).
    t_df : int
        Degrees of freedom for the Student-t loading distribution
        (used when factor_loadings_distribution == 'heavy-tailed').
    """
    p_assets: int = 5000
    subsample_sizes: List[int] = (100, 500, 1000, 5000)
    k_factors: int = 2
    n_windows: int = 100
    window_size: int = 63
    factor_variances: List[float] = None
    factor_loadings_distribution: str = "normal"        # or "heavy-tailed"
    specific_variance_type: str = "homoskedastic"       # or "heteroskedastic"
    idio_variance: float = 0.01
    loading_mean: float = 0.0
    loading_std: float = 1.0
    perturbation_epsilons: List[float] = (0.01, 0.05, 0.1)
    random_seed: int = 42
    t_df: int = 4   # degrees of freedom for heavy-tailed factor loadings

    @classmethod
    def from_json(cls, filepath: str):
        """
        Load a SuperSetSpec from a JSON file.

        Unknown keys in the JSON are silently ignored so that experiment
        configs can carry extra metadata (e.g. notes, t_df) without
        breaking older versions of this class.
        """
        import dataclasses
        with open(filepath) as f:
            config = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})


def create_factor_model(spec: SuperSetSpec, rng: np.random.Generator) -> FactorModelData:
    """
    Build the *true* factor model from a SuperSetSpec.

    Constructs:
      B  (k, p)  – factor loading matrix
      F  (k, k)  – diagonal factor covariance
      D  (p, p)  – diagonal idiosyncratic covariance

    The implied asset covariance is Sigma = B.T @ F @ B + D.

    Parameters
    ----------
    spec : SuperSetSpec
    rng  : np.random.Generator
        Main simulation RNG (consumed here for B and D; F is deterministic).

    Returns
    -------
    FactorModelData
    """
    print(f"\n🏗️  CREATING FACTOR MODEL")
    print(f"   p = {spec.p_assets} assets")
    print(f"   k = {spec.k_factors} factors")

    # --- Factor loadings B (k, p) ---
    if spec.factor_loadings_distribution == "normal":
        B = rng.normal(spec.loading_mean, spec.loading_std,
                       (spec.k_factors, spec.p_assets))
        
    elif spec.factor_loadings_distribution == "heavy-tailed":
        # Scale t draws so the marginal std matches loading_std
        scale = spec.loading_std * np.sqrt((spec.t_df - 2) / spec.t_df)
        B = (rng.standard_t(df=spec.t_df, size=(spec.k_factors, spec.p_assets))
             * scale + spec.loading_mean)

    # --- normalize rows of B to be unit length
    B = orthonormalize(B).T  # (k, p) with orthonormal rows (factors uncorrelated at unit variance)
    
    # --- Factor covariance F (k, k) ---
    if spec.factor_variances is None:
        # Geometrically decreasing variances as a sensible default
        variances = np.array([0.18**2 / (i + 1) for i in range(spec.k_factors)])
    else:
        variances = np.array(spec.factor_variances[:spec.k_factors])
    F = np.diag(variances)

    # --- Idiosyncratic covariance D (p, p) ---
    if spec.specific_variance_type == "homoskedastic":
        D = np.eye(spec.p_assets) * spec.idio_variance
    elif spec.specific_variance_type == "heteroskedastic":
        # Per-asset variance ~ Uniform[0.5, 2.0] * idio_variance
        D = np.diag(rng.uniform(0.5, 2.0, spec.p_assets)) * spec.idio_variance

    print(f"   ✓ B: {B.shape}")
    print(f"   ✓ F: {F.shape}, variances: {variances}")
    print(f"   ✓ D: {D.shape}, diagonal: {spec.idio_variance}")

    return FactorModelData(B=B, F=F, D=D)


def simulate_returns(
    model: FactorModelData,
    n_periods: int,
    window_size: int,
    rng: np.random.Generator
) -> list:
    """
    Simulate n_periods independent windows of security returns.

    Each window has shape (window_size, p).  Returns are generated by the
    standard factor model:
        r_t = B.T @ f_t + e_t
    where f_t ~ N(0, I_k) and e_t ~ N(0, I_p) (the factor and idiosyncratic
    covariance scaling is absorbed into B and D respectively).

    Parameters
    ----------
    model       : FactorModelData
    n_periods   : int   – number of windows
    window_size : int   – time steps per window
    rng         : np.random.Generator

    Returns
    -------
    list of np.ndarray, each shape (window_size, p)
    """
    factory = lambda name, **p: create_sampler(name, rng, **p)
    simulator = FlexibleReturnsSimulator(rng=rng)

    results = []
    for _ in range(n_periods):
        results.append(simulator.simulate(
            model=model,
            n_periods=window_size,
            factor_return_samplers=factory("normal", loc=0, scale=1),
            idio_return_sampler=factory("normal", loc=0, scale=1)
        )['security_returns'])

    return results


def perturb_loading_matrix(B: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a random *rotation* perturbation to a loading matrix.

    Generates a rotation R = exp(epsilon * A) where A is a random
    skew-symmetric k×k matrix, then returns R @ B.

    Note: this perturbs the *row space* of B by rotating the k-dimensional
    factor directions.  It is NOT a geodesic perturbation on the Stiefel
    manifold; use construct_epsilon_distance_perturbation for that.

    Parameters
    ----------
    B       : (k, p)  loading matrix
    epsilon : float   rotation magnitude
    rng     : np.random.Generator

    Returns
    -------
    B_perturbed : (k, p)
    """
    k = B.shape[0]
    M = rng.normal(0, 1, (k, k))
    A = (M - M.T) / 2          # skew-symmetric: exp(A) is orthogonal
    R = expm(epsilon * A)
    return R @ B


def compute_all_distances(B1: np.ndarray, B2: np.ndarray) -> Dict[str, float]:
    """
    Compute the three standard manifold distances between two loading matrices.

    Both matrices are first orthonormalized (their column spaces are the
    objects being compared, not the raw matrices).

    Parameters
    ----------
    B1, B2 : (k, p)  loading matrices

    Returns
    -------
    dict with keys:
        'grassmannian' – principal-angle based subspace distance
        'procrustes'   – optimal-rotation alignment distance
        'chordal'      – projection-matrix Frobenius distance
    """
    # B1, B2 are (k, p) — functions internally orthonormalize via B.T
    grass_result = compute_grassmannian_distance(B1, B2)
    proc_result  = compute_procrustes_distance(B1, B2)
    chord_result = compute_chordal_distance(B1, B2)

    def extract_distance(result):
        """Pull a scalar float out of whatever the distance function returns."""
        if isinstance(result, dict):
            return float(result.get('distance',
                         result.get('dist_grassmannian',
                         result.get('dist_procrustes',
                         result.get('dist_chordal', 0)))))
        elif isinstance(result, (tuple, list)):
            return float(result[0])
        else:
            return float(result)

    return {
        'grassmannian': extract_distance(grass_result),
        'procrustes':   extract_distance(proc_result),
        'chordal':      extract_distance(chord_result),
    }


def run_perturbation_study(spec: SuperSetSpec) -> Dict[str, dict]:
    """
    Run the full perturbation study and return raw distance collections.

    For each subsample size p and each window t:
      - Estimate B from returns[:p] via SVD  →  record sample-truth distances
      - For each epsilon, draw 20 random geodesic perturbations of B_true
        at distance epsilon  →  record truth-perturb and sample-perturb distances

    Parameters
    ----------
    spec : SuperSetSpec

    Returns
    -------
    dict with three keys:

    'sample_truth_distance_results'
        {p: {'grassmann_sampling': [...], 'procrustes_sampling': [...],
             'chordal_sampling': [...]}}
        One value per window (length = n_windows).

    'truth_perturb_distance_results'
        {(eps, p): {'grassmann_perturb': [...], ...}}
        n_windows * 20 values per (eps, p) pair.

    'sample_perturb_distance_results'
        {(eps, p): {'grassmann_sampling': [...], ...}}
        n_windows * 20 values per (eps, p) pair.
    """
    print("\n" + "=" * 70)
    print("  PERTURBATION STUDY")
    print("=" * 70)

    rng = np.random.default_rng(spec.random_seed)

    model  = create_factor_model(spec, rng)
    B_true = model.B   # (k, p_assets) – the ground-truth loading matrix
    k = spec.k_factors
    T = spec.window_size
    # Simulate all windows up front so the same returns are reused across p values
    all_returns = simulate_returns(model, n_periods=spec.n_windows,
                                   window_size=spec.window_size, rng=rng)

    # --- Result containers ---

    # Distance from B_true to its geodesic perturbation at distance eps
    truth_perturb_distance_results = {
        (eps, p): {"grassmann_perturb": [], "procrustes_perturb": [], "chordal_perturb": []}
        for eps in spec.perturbation_epsilons
        for p   in spec.subsample_sizes
    }

    # Distance from B_true to the SVD estimate from p-asset returns
    sample_truth_distance_results = {
        p: {"grassmann_sampling": [], "procrustes_sampling": [], "chordal_sampling": []}
        for p in spec.subsample_sizes
    }

    # Distance from the SVD estimate to a geodesic perturbation of B_true
    sample_perturb_distance_results = {
        (eps, p): {"grassmann_perturb": [], "procrustes_perturb": [],
                   "chordal_perturb": []}
        for eps in spec.perturbation_epsilons
        for p   in spec.subsample_sizes
    }

    for p in spec.subsample_sizes:
        print(f"\n📊 Sample size: p = {p}")
        
        # Get True model of size p
        B_true_p = B_true[:, :p]  # (k, p) — true loadings for this subsample
        F_original = model.F[:p,:p]
        Model_Covar = B_true_p.T @ F_original @ B_true_p

        # SVD decomposition to get true orthonormal basis
        _, s, Vt = np.linalg.svd(Model_Covar, full_matrices=False)
            
        # Extract top k components
        F_star = (s[:k] ** 2) 
        C = Vt[:k, :]  # Shape: (k, p)
            
        # Sign normalization: ensure each factor has positive mean
        for i in range(k):
            if C[i, :].mean() < 0:
                C[i, :] *= -1

        for t in range(spec.n_windows):
            # Use only the first p assets from this window's returns
            returns = all_returns[t][:, :p]


            
            

            # SVD estimate of the factor loading subspace from finite sample
            model_estimated = svd_decomposition(returns, k=spec.k_factors)
            B_estimated = model_estimated.B  # (k, p)

            # --- Sampling error: distance between true and estimated subspace ---
            dist_sampling_error = compute_all_distances(B_true_p, B_estimated)
            sample_truth_distance_results[p]["grassmann_sampling"].append(dist_sampling_error['grassmannian'])
            sample_truth_distance_results[p]["procrustes_sampling"].append(dist_sampling_error['procrustes'])
            sample_truth_distance_results[p]["chordal_sampling"].append(dist_sampling_error['chordal'])

            # --- Perturbation: 20 random geodesic directions per epsilon ---
            for eps in spec.perturbation_epsilons:
                for _ in range(20):
                    # Perturb the p-asset slice so all frames live in R^p
                    random_frame = construct_epsilon_distance_perturbation(
                        eps, C.T, direction_rng
                    )  # (k, p)

                    # Distance from true subspace to the perturbed frame
                    dists_truth = compute_all_distances(C, random_frame)
                    truth_perturb_distance_results[(eps, p)]["grassmann_perturb"].append(dists_truth['grassmannian'])
                    truth_perturb_distance_results[(eps, p)]["procrustes_perturb"].append(dists_truth['procrustes'])
                    truth_perturb_distance_results[(eps, p)]["chordal_perturb"].append(dists_truth['chordal'])

                    # Distance from the SVD estimate to the perturbed frame
                    dist_ests = compute_all_distances(B_estimated, random_frame)
                    sample_perturb_distance_results[(eps, p)]["grassmann_perturb"].append(dist_ests['grassmannian'])
                    sample_perturb_distance_results[(eps, p)]["procrustes_perturb"].append(dist_ests['procrustes'])
                    sample_perturb_distance_results[(eps, p)]["chordal_perturb"].append(dist_ests['chordal'])

    return {
        "sample_truth_distance_results":   sample_truth_distance_results,
        "truth_perturb_distance_results":  truth_perturb_distance_results,
        "sample_perturb_distance_results": sample_perturb_distance_results,
    }


# Mapping from (metric display name, sampling key, perturbation key)
# Used by both analyze_results and plot_distance_comparison.
METRICS = [
    ("grassmannian", "grassmann_sampling", "grassmann_perturb"),
    ("procrustes",   "procrustes_sampling", "procrustes_perturb"),
    ("chordal",      "chordal_sampling",    "chordal_perturb"),
]


def analyze_results(out_dict: dict, spec: SuperSetSpec) -> pd.DataFrame:
    """
    Summarise run_perturbation_study output into a tidy DataFrame.

    Each row is one (metric, p, eps) combination with columns:
        sampling_mean, sampling_median, sampling_var, sampling_min, sampling_max
            – across n_windows samples
        perturb_mean, perturb_median, perturb_var, perturb_min, perturb_max
            – across n_windows * 20 draws

    Parameters
    ----------
    out_dict : dict   output of run_perturbation_study
    spec     : SuperSetSpec

    Returns
    -------
    pd.DataFrame
    """
    sample_truth  = out_dict["sample_truth_distance_results"]
    truth_perturb = out_dict["truth_perturb_distance_results"]

    rows = []
    for metric_name, samp_key, perturb_key in METRICS:
        for p in spec.subsample_sizes:
            s = np.array(sample_truth[p][samp_key])        # shape (n_windows,)
            for eps in spec.perturbation_epsilons:
                d = np.array(truth_perturb[(eps, p)][perturb_key])  # shape (n_windows * 20,)
                rows.append({
                    "metric":          metric_name,
                    "p":               p,
                    "eps":             eps,
                    "sampling_mean":   s.mean(),
                    "sampling_median": np.median(s),
                    "sampling_var":    s.var(),
                    "sampling_min":    float(s.min()),
                    "sampling_max":    float(s.max()),
                    "perturb_mean":    d.mean(),
                    "perturb_median":  np.median(d),
                    "perturb_var":     d.var(),
                    "perturb_min":     float(d.min()),
                    "perturb_max":     float(d.max()),
                })

    return pd.DataFrame(rows)




def construct_epsilon_distance_perturbation(
    t: float,
    B: np.ndarray,
    direction_rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a new k-frame on the Stiefel manifold at geodesic distance t from B.

    Shoots a geodesic from B in a uniformly random tangent direction, using
    the closed-form geodesic formula from Edelman, Arias & Smith (1998),
    Theorem 4, eq. 15.  The geodesic stays on V(p,k) and travels exactly
    distance t (in the Riemannian metric inherited from the embedding in R^{p×k}).

    Parameters
    ----------
    t             : geodesic distance to travel (controls perturbation magnitude)
    B             : base frame; (p, k) expected (p assets, k factors).
                    If passed as (k, p) it is transposed automatically.
    direction_rng : RNG for the random tangent direction

    Returns
    -------
    gamma_t : (k, p)  new frame, geodesic distance t from B
    """
    p, k = B.shape
    # Normalise to (p, k) — the geodesic formula requires the long axis first
    if k > p:
        B = B.T
        p, k = B.shape

    # --- Step 1: sample a random unit tangent vector at B ----------------
    Z = direction_rng.standard_normal(B.shape)          # (p, k) unconstrained

    # Project onto T_B V(p,k): enforce B.T @ Delta + Delta.T @ B = 0
    Delta = Z - B @ ((B.T @ Z + Z.T @ B) / 2)          # (p, k)
    Delta = Delta / np.linalg.norm(Delta, 'fro')        # normalise to unit length

    # --- Step 2: decompose Delta relative to B ---------------------------
    A = B.T @ Delta                                     # (k, k) "within B" part
    A = (A - A.T) / 2                                   # enforce skew-symmetry

    # Normal component: directions orthogonal to B's column space
    # Equivalent to (I - B @ B.T) @ Delta but avoids the p×p outer product
    Delta_perp = Delta - B @ A                          # (p, k)
    Q, R = np.linalg.qr(Delta_perp, mode='reduced')    # Q: (p,k), R: (k,k)

    # --- Step 3: build the (2k × 2k) geodesic ODE block matrix -----------
    # This small matrix captures all the non-trivial dynamics; p only
    # enters at the final reconstruction step through B and Q.
    M = np.block([[A,             -R.T         ],
                  [R,  np.zeros((k, k))         ]])     # (2k, 2k)

    # --- Step 4: solve the geodesic ODE via matrix exponential -----------
    E   = expm(t * M)                                   # (2k, 2k)
    M_t = E[:k, :k]                                     # (k, k) within-B rotation
    N_t = E[k:, :k]                                     # (k, k) away-from-B motion

    # --- Step 5: assemble gamma(t) = B @ M(t) + Q @ N(t) ----------------
    return (B @ M_t + Q @ N_t).T                        # return as (k, p)


def main():
    import argparse
    import distance_compare_plots as dcp

    parser = argparse.ArgumentParser(description="Large-sample perturbation study")
    parser.add_argument("config_file", type=str, help="JSON configuration file")
    args = parser.parse_args()

    config_path = Path(args.config_file)
    spec = SuperSetSpec.from_json(config_path)
    

    # Output folder is named after the config file (without extension)
    output_dir = config_path.parent / config_path.stem
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    results = run_perturbation_study(spec)

    # Save tidy summary table
    df = analyze_results(results, spec)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    print(df.to_string(index=False))

    # Save histogram plots (non-interactive backend for CLI use)
    plt.switch_backend("Agg")
    subsample_sizes   = spec.subsample_sizes
    perturbation_epsilons = spec.perturbation_epsilons
    dcp.distance_histograms_shared_axes(
        results,
        subsample_sizes,
        perturbation_epsilons,
        output_dir=output_dir,
    )
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()