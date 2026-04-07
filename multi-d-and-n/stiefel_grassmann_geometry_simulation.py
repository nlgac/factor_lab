"""
High-Dimensional Grassmannian Distance Simulation

Studies how Grassmannian distances behave in high-dimensional PCA with a spiked
covariance model. Compares distances between:
  - True subspace vs sample estimate (estimation error)
  - Sample estimate vs synthetic targets at prescribed distances

Key Insight: This studies Gr(k,p) geometry (subspaces), NOT St(p,k) (frames).
The distance is the L2 norm of principal angles: d = √(θ₁² + ... + θₖ²).

For Stiefel manifold distances (which include frame rotation), see StiefelDistance.

Example Usage:
    >>> config = SimulationConfig(
    ...     ps=[100, 500, 1000],
    ...     ns=[63, 126],
    ...     radii=[0.1, 0.3, 0.5],
    ...     eigenvalues={2: (9.0, 4.0)},  # 2D subspace
    ...     noise_std=1.0,
    ...     n_reps=10
    ... )
    >>> results = run_simulation(config)
    >>> results.save("output/")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Protocol, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from numpy.linalg import svd
from scipy.linalg import eigh, expm, logm, qr


# ==============================================================================
# Logging Configuration with loguru
# ==============================================================================

# Remove default handler
logger.remove()

# Add console handler with detailed format
logger.add("app.log",
    # sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="TRACE",
    colorize=True,
)
# logger.add("app.log") 

import sys

def set_logging_level(level: str = "TRACE", log_file: str = "app.log") -> None:
    """
    Set the logging level and configure both console and file outputs.
    
    Parameters
    ----------
    level : str
        Logging level: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    log_file: str
        The path to the log file where outputs should be saved.
    """
    # Remove all existing handlers to prevent duplicates
    logger.remove()
    
    # Define a clean format
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # 1. Add Console Handler
    logger.add(
        sys.stderr, 
        format=log_format,
        level=level.upper(),
        colorize=True
    )
    
    # 2. Add File Handler (logs to the specified file)
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=level.upper(),
            colorize=False, # Color codes look messy in raw text files
            mode="w"        # "w" overwrites previous runs. Use "a" to append.
        )
        
    logger.info(f"Logging level set to {level.upper()}")


def trace_calls(func: Callable) -> Callable:
    """
    Decorator to trace function calls with parameters and return values.
    
    Use @trace_calls on methods to log:
    - Entry with all parameters
    - Exit with return value
    - Execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name and class if it's a method
        func_name = func.__name__
        if args and hasattr(args[0].__class__, func_name):
            class_name = args[0].__class__.__name__
            full_name = f"{class_name}.{func_name}"
            # Skip 'self' in args
            call_args = args[1:]
        else:
            full_name = func_name
            call_args = args
        
        # Format arguments
        args_str = ", ".join([repr(a)[:100] if not isinstance(a, np.ndarray) else f"array{a.shape}" for a in call_args])
        kwargs_str = ", ".join([f"{k}={repr(v)[:100] if not isinstance(v, np.ndarray) else f'array{v.shape}'}" for k, v in kwargs.items()])
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"→ CALL {full_name}({params})")
        
        t_start = time.time()
        try:
            result = func(*args, **kwargs)
            t_elapsed = time.time() - t_start
            
            # Format return value
            if isinstance(result, tuple) and len(result) <= 3:
                result_str = f"({', '.join([f'array{r.shape}' if isinstance(r, np.ndarray) else repr(r)[:50] for r in result])})"
            elif isinstance(result, np.ndarray):
                result_str = f"array{result.shape}"
            elif result is None:
                result_str = "None"
            else:
                result_str = repr(result)[:100]
            
            logger.debug(f"← RETURN {full_name} = {result_str} [{t_elapsed*1000:.2f}ms]")
            return result
            
        except Exception as e:
            t_elapsed = time.time() - t_start
            logger.error(f"✗ EXCEPTION in {full_name}: {e} [{t_elapsed*1000:.2f}ms]")
            raise
    
    return wrapper


# ==============================================================================
# Distance Metrics (Strategy Pattern for extensibility)
# ==============================================================================


class DistanceMetric(Protocol):
    """Protocol for distance metrics on manifolds."""
    
    @property
    def name(self) -> str:
        """Human-readable name for the metric."""
        ...
    
    def distance(self, overlap: np.ndarray) -> float:
        """Compute distance from overlap matrix M = U1^T @ U2."""
        ...
    
    def sample_target(
        self,
        radius: float,
        overlap_truth: np.ndarray,
        overlap_complement: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate target at prescribed distance from truth."""
        ...


class GrassmannDistance:
    """
    Grassmannian distance: d = ||θ|| where θ are principal angles.
    
    This measures the angle between k-dimensional subspaces in R^p.
    Ignores rotations within the subspaces (projects to Gr(k,p)).
    
    For k=1: This is the spherical geodesic distance on S^{p-1}.
    For k≥2: This is the Grassmann geodesic distance on Gr(k,p).
    """
    
    def __init__(self):
        logger.trace("GrassmannDistance.__init__()")
        logger.debug("Initialized GrassmannDistance metric")
    
    @property
    def name(self) -> str:
        return "grassmann"
    
    @trace_calls
    def distance(self, overlap: np.ndarray) -> float:
        """
        Distance from overlap M = U1^T @ U2.
        
        Uses SVD to extract principal angles: M = U @ Σ @ V^T
        where σᵢ = cos(θᵢ). Distance is d = √(Σθᵢ²).
        
        For k=1: Returns arccos(M[0,0]) (spherical distance).
        """
        overlap = np.asarray(overlap)
        k = overlap.shape[0]
        
        logger.trace(f"overlap.shape = {overlap.shape}")
        logger.trace(f"overlap =\n{overlap}")
        
        # Special case: k=1 (sphere)
        if overlap.shape == (1, 1):
            logger.debug(f"k=1 case: spherical distance")
            cos_angle = np.clip(overlap[0, 0], -1.0, 1.0)
            logger.trace(f"cos_angle = {cos_angle}")
            dist = float(np.arccos(cos_angle))
            logger.debug(f"spherical distance = {dist:.6f}")
            return dist
        
        # General case: k≥2 (Grassmannian)
        logger.debug(f"k={k} case: calling np.linalg.svd(overlap, compute_uv=False)")
        singular_values = svd(overlap, compute_uv=False)
        logger.trace(f"singular_values = {singular_values}")
        
        singular_values = np.clip(singular_values, -1.0, 1.0)
        logger.trace(f"clipped singular_values = {singular_values}")
        
        logger.debug(f"computing principal_angles = arccos(singular_values)")
        principal_angles = np.arccos(singular_values)
        logger.trace(f"principal_angles = {principal_angles}")
        
        logger.debug(f"computing distance = np.linalg.norm(principal_angles)")
        dist = float(np.linalg.norm(principal_angles))
        logger.debug(f"Grassmann distance = {dist:.6f} (angles={principal_angles})")
        
        return dist
    
    @trace_calls
    def sample_target(
        self,
        radius: float,
        overlap_truth: np.ndarray,
        overlap_complement: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate target V such that d(truth, V) = radius (Grassmann distance).
        
        Uses geodesic formula:
            V^T @ U_hat = cos(θ) @ R^T @ (U_truth^T @ U_hat) + sin(θ) @ (G^T @ U_hat)
        
        where θ are principal angles summing to radius (in L2 norm).
        """
        k = overlap_truth.shape[0]
        
        logger.debug(f"Sampling target at radius={radius:.6f} for k={k}")
        logger.trace(f"overlap_truth.shape = {overlap_truth.shape}")
        logger.trace(f"overlap_complement.shape = {overlap_complement.shape}")
        
        # Special case: k=1 (sphere)
        if k == 1:
            logger.debug("k=1 case: spherical geodesic")
            theta = radius
            logger.trace(f"theta = {theta}")
            cos_part = np.cos(theta) * overlap_truth[0, 0]
            sin_part = np.sin(theta) * overlap_complement[0, 0]
            logger.trace(f"cos_part = {cos_part}, sin_part = {sin_part}")
            result = np.array([[cos_part + sin_part]])
            logger.debug(f"returning array{result.shape}")
            return result
        
        # General case: k≥2
        logger.debug(f"k={k} case: sampling random principal angles")
        logger.trace(f"calling rng.normal(size={k})")
        weights = np.abs(rng.normal(size=k))
        logger.trace(f"weights (raw) = {weights}")
        
        logger.trace(f"normalizing: weights /= np.linalg.norm(weights)")
        weights /= np.linalg.norm(weights)
        logger.trace(f"weights (normalized) = {weights}")
        
        principal_angles = radius * weights
        logger.debug(f"principal_angles = {principal_angles} (||θ|| = {np.linalg.norm(principal_angles):.6f})")
        
        # Random rotation in k-dimensional subspace
        logger.debug(f"generating random rotation matrix")
        logger.trace(f"calling rng.normal(size=({k}, {k}))")
        R_raw = rng.normal(size=(k, k))
        logger.trace(f"calling scipy.linalg.qr(R_raw)")
        R, _ = qr(R_raw)
        logger.trace(f"R.shape = {R.shape}, det(R) = {np.linalg.det(R):.6f}")
        
        if np.linalg.det(R) < 0:
            logger.trace("det(R) < 0, flipping first column")
            R[:, 0] *= -1
        
        # Geodesic formula: cos(θ)·R^T·S_u + sin(θ)·S_g
        logger.debug("computing geodesic formula")
        logger.trace(f"calling np.diag(np.cos(principal_angles))")
        cos_diag = np.diag(np.cos(principal_angles))
        logger.trace(f"cos_diag.shape = {cos_diag.shape}")
        
        logger.trace(f"computing cos_part = cos_diag @ (R.T @ overlap_truth)")
        cos_part = cos_diag @ (R.T @ overlap_truth)
        logger.trace(f"cos_part.shape = {cos_part.shape}")
        
        logger.trace(f"calling np.diag(np.sin(principal_angles))")
        sin_diag = np.diag(np.sin(principal_angles))
        logger.trace(f"computing sin_part = sin_diag @ overlap_complement")
        sin_part = sin_diag @ overlap_complement
        logger.trace(f"sin_part.shape = {sin_part.shape}")
        
        logger.trace(f"computing result = cos_part + sin_part")
        result = cos_part + sin_part
        logger.debug(f"returning target overlap, shape={result.shape}")
        
        return result


class StiefelCanonicalDistance:
    """
    Stiefel canonical distance: d = √(½||A₁₁||² + ||A₂₁||²).
    
    This measures distance between k-frames in R^p, INCLUDING rotation
    within the k-dimensional subspace (SO(k) fiber).
    
    Uses the 2k×2k matrix logarithm trick for efficiency.
    
    Note: This is MORE information than Grassmannian distance.
    Two frames differing only by rotation have:
      - Stiefel distance > 0
      - Grassmannian distance = 0
    """
    
    @property
    def name(self) -> str:
        return "stiefel-canonical"
    
    def distance(self, overlap: np.ndarray) -> float:
        """
        Compute Stiefel canonical distance using 2k×2k reduction.
        
        This is Gemini's method from our earlier discussion.
        """
        k = overlap.shape[0]
        
        # For direct frame-to-frame, we'd need full frames U1, U2
        # Here we only have overlap, so this is a placeholder
        # In practice, you'd pass frames and use:
        # return self._distance_from_frames(U1, U2)
        
        raise NotImplementedError(
            "Stiefel distance requires full frames, not just overlap. "
            "Use distance_from_frames(U1, U2) instead."
        )
    
    def distance_from_frames(self, U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Compute Stiefel distance using 2k×2k block method (Gemini's approach).
        
        Parameters
        ----------
        U1, U2 : ndarray (p, k)
            Orthonormal frames
        
        Returns
        -------
        distance : float
            Canonical geodesic distance
        """
        k = U1.shape[1]
        
        # Projection and residual
        M = U1.T @ U2
        R_residual = U2 - U1 @ M
        _, R = qr(R_residual, mode='economic')
        
        # Build 2k×2k rotation matrix
        G = np.zeros((2 * k, 2 * k))
        G[:k, :k] = M
        G[k:, :k] = R
        G[:k, k:] = -R.T
        G[k:, k:] = M
        
        # Matrix logarithm
        Delta = np.real(logm(G))
        Delta = 0.5 * (Delta - Delta.T)  # Enforce skew-symmetry
        
        Delta11 = Delta[:k, :k]  # Vertical (rotation)
        Delta21 = Delta[k:, :k]  # Horizontal (tilt)
        
        # Canonical metric: ½||A₁₁||² + ||A₂₁||²
        distance_sq = 0.5 * np.linalg.norm(Delta11, 'fro')**2 + \
                     np.linalg.norm(Delta21, 'fro')**2
        
        return np.sqrt(distance_sq)


# ==============================================================================
# Spiked Covariance Model
# ==============================================================================


@dataclass
class SpikeCovarianceModel:
    """
    Spiked covariance model for high-dimensional PCA.
    
    Population covariance: Σ = U @ Λ @ U^T + σ²·I
    where Λ = diag(λ₁, ..., λₖ) with λᵢ >> σ².
    
    Parameters
    ----------
    p : int
        Ambient dimension
    eigenvalues : tuple[float, ...]
        Signal eigenvalues (k of them)
    noise_std : float
        Noise standard deviation σ
    
    Attributes
    ----------
    k : int
        Latent dimension (number of factors)
    signal_eigenvalues : ndarray
        λ₁, ..., λₖ
    """
    
    p: int
    eigenvalues: tuple[float, ...]
    noise_std: float = 1.0
    
    def __post_init__(self):
        self.k = len(self.eigenvalues)
        self.signal_eigenvalues = np.array(self.eigenvalues)
        
        # Validate
        if self.p < 2 * self.k:
            error_msg = f"Require p ≥ 2k, got p={self.p}, k={self.k}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if np.any(self.signal_eigenvalues <= self.noise_std**2):
            error_msg = f"All eigenvalues must exceed noise variance {self.noise_std**2}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(
            f"Created SpikeCovarianceModel: p={self.p}, k={self.k}, "
            f"eigenvalues={self.eigenvalues}, σ={self.noise_std}"
        )
        logger.debug(f"  Signal-to-noise ratio (λ₁/σ²): {self.signal_eigenvalues[0]/self.noise_std**2:.2f}")
    
    @trace_calls
    def sample_overlaps(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate overlaps U_truth^T @ U_hat and G^T @ U_hat without full matrix.
        
        By rotational invariance, fix truth to first k standard basis vectors.
        Generate only the k×n overlaps needed for distance computations.
        """
        logger.debug(f"SpikeCovarianceModel.sample_overlaps(n={n}, p={self.p}, k={self.k})")
        t_start = time.time()
        
        # Signal scales (after removing noise variance)
        logger.debug("computing signal_scales = sqrt(eigenvalues - noise_std²)")
        signal_scales = np.sqrt(self.signal_eigenvalues - self.noise_std**2)
        logger.trace(f"signal_scales = {signal_scales}")
        
        # Generate components
        logger.debug(f"generating F = rng.normal(size=({self.k}, {n}))")
        F = rng.normal(size=(self.k, n))
        logger.trace(f"F.shape = {F.shape}")
        
        logger.debug(f"generating E_truth = rng.normal(size=({self.k}, {n}))")
        E_truth = rng.normal(size=(self.k, n))
        
        logger.debug(f"generating E_complement = rng.normal(size=({self.k}, {n}))")
        E_complement = rng.normal(size=(self.k, n))
        
        # Build sample Gram matrix components
        logger.debug("computing A = signal_scales[:, None] * F + noise_std * E_truth")
        A = signal_scales[:, None] * F + self.noise_std * E_truth
        logger.trace(f"A.shape = {A.shape}")
        
        logger.debug("computing B = noise_std * E_complement")
        B = self.noise_std * E_complement
        logger.trace(f"B.shape = {B.shape}")
        
        # Residual from (p-2k) dimensions
        residual_df = self.p - 2 * self.k
        logger.debug(f"computing residual_gram via _sample_wishart_gram(df={residual_df}, dim={n})")
        residual_gram = self.noise_std**2 * self._sample_wishart_gram(
            df=residual_df,
            dim=n,
            rng=rng,
        )
        logger.trace(f"residual_gram.shape = {residual_gram.shape}")
        
        # Full Gram matrix
        logger.debug("computing gram = A.T @ A + B.T @ B + residual_gram")
        gram = A.T @ A + B.T @ B + residual_gram
        logger.trace(f"gram.shape = {gram.shape}")
        
        # Extract top k eigenvectors
        n_obs = gram.shape[0]
        logger.debug(f"calling scipy.linalg.eigh(gram, subset_by_index=[{n_obs - self.k}, {n_obs - 1}])")
        evals, V = eigh(gram, subset_by_index=[n_obs - self.k, n_obs - 1])
        logger.trace(f"evals.shape = {evals.shape}, V.shape = {V.shape}")
        
        logger.debug("reversing order: evals[::-1], V[:, ::-1]")
        evals = np.clip(evals[::-1], 0.0, None)
        singular_values = np.sqrt(evals)
        V = V[:, ::-1]
        
        logger.debug(f"top eigenvalues = {evals}")
        logger.trace(f"singular_values = {singular_values}")
        
        # Compute overlaps
        logger.debug("computing inv_s = 1.0 / singular_values")
        inv_s = np.where(singular_values > 0, 1.0 / singular_values, 1.0)
        logger.trace(f"inv_s = {inv_s}")
        
        logger.debug("computing overlap_truth = (A @ V) * inv_s")
        overlap_truth = (A @ V) * inv_s
        logger.trace(f"overlap_truth.shape = {overlap_truth.shape}")
        
        logger.debug("computing overlap_complement = (B @ V) * inv_s")
        overlap_complement = (B @ V) * inv_s
        logger.trace(f"overlap_complement.shape = {overlap_complement.shape}")
        
        # Sign convention for k=1
        if self.k == 1 and overlap_truth[0, 0] < 0.0:
            logger.trace("applying sign convention for k=1: overlap *= -1")
            overlap_truth *= -1.0
            overlap_complement *= -1.0
        
        t_elapsed = time.time() - t_start
        logger.debug(f"sample_overlaps completed in {t_elapsed*1000:.2f}ms")
        
        return overlap_truth, overlap_complement
    
    @staticmethod
    def _sample_wishart_gram(
        df: int,
        dim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample Z^T @ Z with Z ~ N(0,I)_{df×dim} using Bartlett decomposition.
        
        For df ≥ dim: Uses Bartlett decomposition (exact, efficient).
        For df < dim: Falls back to Z^T @ Z (still exact, less efficient).
        """
        if df <= 0:
            return np.zeros((dim, dim))
        
        if df < dim:
            # Fall back to direct sampling
            Z = rng.normal(size=(df, dim))
            return Z.T @ Z
        
        # Bartlett decomposition (Wishart efficient sampling)
        A = np.zeros((dim, dim))
        
        # Diagonal: χ² samples
        chi_dfs = np.arange(df, df - dim, -1)
        A[np.diag_indices(dim)] = np.sqrt(rng.chisquare(df=chi_dfs))
        
        # Lower triangle: N(0,1) samples
        tril_idx = np.tril_indices(dim, k=-1)
        A[tril_idx] = rng.normal(size=len(tril_idx[0]))
        
        return A @ A.T


# ==============================================================================
# Simulation Configuration and Results
# ==============================================================================


@dataclass
class SimulationConfig:
    """
    Configuration for distance simulation experiments.
    
    Example:
        >>> config = SimulationConfig(
        ...     ps=[100, 500, 1000],
        ...     ns=[63, 126],
        ...     radii=[0.1, 0.5],
        ...     eigenvalues={2: (9.0, 4.0)},
        ...     n_reps=10,
        ... )
    """
    
    # Grid parameters
    ps: list[int]  # Ambient dimensions to test
    ns: list[int]  # Sample sizes to test
    radii: list[float]  # Target distances to test
    
    # Models (keyed by latent dimension k)
    eigenvalues: dict[int, tuple[float, ...]]
    
    # Experimental design
    noise_std: float = 1.0
    n_reps: int = 12  # Statistical replications
    n_targets_per_rep: int = 8  # Targets per sample
    seed: int = 20260403
    
    # Which metric to use
    metric: DistanceMetric = field(default_factory=GrassmannDistance)


@dataclass
class SimulationResults:
    """Results from distance simulation."""
    
    long_df: pd.DataFrame  # All individual measurements
    summary_df: pd.DataFrame  # Grouped statistics
    config: SimulationConfig
    
    def save(self, output_dir: Path | str) -> None:
        """Save results and generate plots."""
        output_dir = Path(output_dir)
        
        logger.info("\n" + "="*70)
        logger.info(f"Saving results to: {output_dir}")
        logger.info("="*70)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Save data
        t_start = time.time()
        
        csv_long = output_dir / "distances_all.csv"
        self.long_df.to_csv(csv_long, index=False)
        logger.info(f"  ✓ Saved {csv_long.name} ({len(self.long_df):,} rows)")
        
        csv_summary = output_dir / "distances_summary.csv"
        self.summary_df.to_csv(csv_summary, index=False)
        logger.info(f"  ✓ Saved {csv_summary.name} ({len(self.summary_df)} rows)")
        
        t_csv = time.time() - t_start
        logger.debug(f"    CSV files saved in {t_csv:.3f}s")
        
        # Generate plots
        logger.info("\nGenerating plots...")
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        logger.info(f"Created figure directory: {figure_dir}")
        
        t_plot_start = time.time()
        for k in self.config.eigenvalues.keys():
            plot_file = figure_dir / f"dimension_{k}.png"
            logger.info(f"  Creating plot for dimension k={k}...")
            self._plot_dimension(k, plot_file)
            logger.info(f"    ✓ Saved {plot_file.name}")
        
        t_plot = time.time() - t_plot_start
        logger.info(f"  Plots generated in {t_plot:.3f}s")
        
        # Write README
        logger.info("\nWriting README...")
        self._write_readme(output_dir)
        logger.info(f"  ✓ Saved README.txt")
        
        t_total = time.time() - t_start
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ All results saved successfully in {t_total:.3f}s")
        logger.info("="*70)
        logger.info(f"\nOutput directory: {output_dir.absolute()}")
        logger.info(f"  - distances_all.csv ({len(self.long_df):,} measurements)")
        logger.info(f"  - distances_summary.csv ({len(self.summary_df)} groups)")
        logger.info(f"  - figures/ ({len(self.config.eigenvalues)} plots)")
        logger.info(f"  - README.txt")
    
    def _plot_dimension(self, k: int, save_path: Path) -> None:
        """Create faceted plot for one latent dimension."""
        plot_df = self.long_df[
            (self.long_df["dimension"] == k) &
            (self.long_df["distance_type"].isin(["sample-target", "sample-truth"]))
        ].copy()
        
        if plot_df.empty:
            return
        
        # Ordering
        radius_order = [f"r={r:.1f}" for r in sorted(plot_df["radius"].unique())]
        n_order = [f"n={n}" for n in sorted(plot_df["n"].unique())]
        p_order = sorted(plot_df["p"].unique())
        
        # Create plot
        sns.set_theme(style="whitegrid", context="paper")
        g = sns.catplot(
            data=plot_df,
            kind="box",
            x="p",
            y="distance",
            hue="distance_type",
            col="radius_label",
            row="n_label",
            col_order=radius_order,
            row_order=n_order,
            hue_order=["sample-target", "sample-truth"],
            sharey=True,
            height=3.0,
            aspect=1.1,
            linewidth=0.8,
            showfliers=False,
        )
        
        # Add reference lines at target radius
        radius_map = {f"r={r:.1f}": r for r in sorted(plot_df["radius"].unique())}
        for axes_row in g.axes:
            for label, ax in zip(radius_order, axes_row):
                ax.axhline(radius_map[label], ls="--", lw=1.2, color="black", alpha=0.7)
                ax.set_xlabel("Ambient dimension (p)")
                ax.set_ylabel("Distance")
        
        # Title
        metric_name = "Sphere" if k == 1 else "Grassmann"
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.fig.suptitle(
            f"{k}D {metric_name}: Distance vs (p, n, radius)",
            fontsize=14,
        )
        g.fig.subplots_adjust(top=0.90)
        
        if g._legend:
            g._legend.set_title("")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(g.fig)
    
    def _write_readme(self, output_dir: Path) -> None:
        """Write configuration and file descriptions."""
        lines = [
            "Grassmannian Distance Simulation Results",
            "=" * 50,
            "",
            "Configuration:",
            f"  Ambient dimensions (p): {self.config.ps}",
            f"  Sample sizes (n): {self.config.ns}",
            f"  Target radii: {self.config.radii}",
            f"  Models: {self.config.eigenvalues}",
            f"  Noise std (σ): {self.config.noise_std}",
            f"  Replications: {self.config.n_reps}",
            f"  Metric: {self.config.metric.name}",
            "",
            "Files:",
            "  distances_all.csv - All simulated distances",
            "  distances_summary.csv - Grouped statistics",
            "  figures/ - Faceted plots by dimension",
            "",
            "Distance Types:",
            "  sample-truth: Estimation error (U_hat vs U_true)",
            "  sample-target: Sample to synthetic target",
            "  truth-target: True distance (should equal radius)",
        ]
        
        (output_dir / "README.txt").write_text("\n".join(lines))


# ==============================================================================
# Main Simulation Runner
# ==============================================================================


@trace_calls
def run_simulation(config: SimulationConfig) -> SimulationResults:
    """
    Run distance simulation across parameter grid.
    
    For each (k, p, n, radius) combination:
      1. Generate sample estimate U_hat from spiked model
      2. Measure distance from U_hat to truth (estimation error)
      3. Generate synthetic targets at distance 'radius' from truth
      4. Measure distances from U_hat to targets
    
    Parameters
    ----------
    config : SimulationConfig
        Experimental configuration
    
    Returns
    -------
    results : SimulationResults
        Long and summary DataFrames with distances
    
    Example
    -------
    >>> config = SimulationConfig(
    ...     ps=[100, 500],
    ...     ns=[63],
    ...     radii=[0.1, 0.5],
    ...     eigenvalues={2: (9.0, 4.0)},
    ...     n_reps=5,
    ... )
    >>> results = run_simulation(config)
    >>> results.save("output/")
    """
    logger.info("="*70)
    logger.info("Starting Grassmannian Distance Simulation")
    logger.info("="*70)
    
    t_sim_start = time.time()
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Ambient dimensions (p): {config.ps}")
    logger.info(f"  Sample sizes (n): {config.ns}")
    logger.info(f"  Target radii: {config.radii}")
    logger.info(f"  Models: {config.eigenvalues}")
    logger.info(f"  Noise std (σ): {config.noise_std}")
    logger.info(f"  Replications: {config.n_reps}")
    logger.info(f"  Targets per rep: {config.n_targets_per_rep}")
    logger.info(f"  Metric: {config.metric.name}")
    logger.info(f"  Random seed: {config.seed}")
    
    # Calculate grid size
    total_combinations = (
        len(config.eigenvalues) *
        len(config.ps) *
        len(config.ns) *
        len(config.radii)
    )
    total_iterations = total_combinations * config.n_reps
    
    logger.info(f"\nGrid:")
    logger.info(f"  Parameter combinations: {total_combinations}")
    logger.info(f"  Total iterations: {total_iterations}")
    logger.info(f"  Expected measurements: ~{total_iterations * (1 + 2*config.n_targets_per_rep):,}")
    
    rng = np.random.default_rng(config.seed)
    records: list[dict] = []
    
    iteration = 0
    
    # Iterate over parameter grid
    logger.info("\n" + "="*70)
    logger.info("Beginning parameter grid iteration")
    logger.info("="*70)
    
    for k, eigenvalues in config.eigenvalues.items():
        logger.info(f"\n>>> Dimension k={k}: eigenvalues={eigenvalues}")
        
        for p in config.ps:
            logger.info(f"  >>> Ambient dimension p={p}")
            
            for n in config.ns:
                logger.info(f"    >>> Sample size n={n}")
                
                for radius in config.radii:
                    logger.info(f"      >>> Target radius r={radius:.3f}")
                    
                    # Create model
                    t_model_start = time.time()
                    model = SpikeCovarianceModel(
                        p=p,
                        eigenvalues=eigenvalues,
                        noise_std=config.noise_std,
                    )
                    t_model = time.time() - t_model_start
                    logger.debug(f"        Model created in {t_model*1000:.2f}ms")
                    
                    # Statistical replications
                    for rep in range(config.n_reps):
                        iteration += 1
                        progress_pct = 100 * iteration / total_iterations
                        
                        logger.info(
                            f"        Rep {rep+1}/{config.n_reps} "
                            f"[{iteration}/{total_iterations}, {progress_pct:.1f}%]"
                        )
                        
                        # Generate sample estimate
                        t_sample_start = time.time()
                        overlap_truth, overlap_complement = model.sample_overlaps(
                            n=n,
                            rng=rng,
                        )
                        t_sample = time.time() - t_sample_start
                        
                        # Distance: sample to truth (estimation error)
                        dist_sample_truth = config.metric.distance(overlap_truth)
                        logger.debug(
                            f"          Sample→Truth distance: {dist_sample_truth:.6f} "
                            f"(sampled in {t_sample*1000:.2f}ms)"
                        )
                        
                        records.append({
                            "dimension": k,
                            "p": p,
                            "n": n,
                            "radius": radius,
                            "rep": rep,
                            "distance_type": "sample-truth",
                            "distance": dist_sample_truth,
                            "metric": config.metric.name,
                        })
                        
                        # Generate targets at prescribed distance from truth
                        logger.debug(
                            f"          Generating {config.n_targets_per_rep} targets..."
                        )
                        
                        for target_idx in range(config.n_targets_per_rep):
                            
                            # Sample target
                            overlap_target = config.metric.sample_target(
                                radius=radius,
                                overlap_truth=overlap_truth,
                                overlap_complement=overlap_complement,
                                rng=rng,
                            )
                            
                            # Distance: sample to target
                            dist_sample_target = config.metric.distance(overlap_target)
                            
                            if target_idx == 0:  # Log first target
                                logger.debug(
                                    f"          Sample→Target[0] distance: "
                                    f"{dist_sample_target:.6f}"
                                )
                            
                            records.append({
                                "dimension": k,
                                "p": p,
                                "n": n,
                                "radius": radius,
                                "rep": rep,
                                "distance_type": "sample-target",
                                "distance": dist_sample_target,
                                "metric": config.metric.name,
                            })
                            
                            # Ground truth: radius (for reference)
                            records.append({
                                "dimension": k,
                                "p": p,
                                "n": n,
                                "radius": radius,
                                "rep": rep,
                                "distance_type": "truth-target",
                                "distance": radius,
                                "metric": config.metric.name,
                            })
    
    # Create DataFrames
    logger.info("\n" + "="*70)
    logger.info("Processing results...")
    logger.info("="*70)
    
    t_df_start = time.time()
    long_df = pd.DataFrame.from_records(records)
    long_df["radius_label"] = long_df["radius"].map(lambda x: f"r={x:.1f}")
    long_df["n_label"] = long_df["n"].map(lambda x: f"n={x}")
    
    logger.info(f"Created long DataFrame: {len(long_df):,} rows")
    
    # Summary statistics
    summary_df = (
        long_df
        .groupby(["dimension", "p", "n", "radius", "distance_type", "metric"], 
                 as_index=False)
        ["distance"]
        .agg([
            ("count", "count"),
            ("mean", "mean"),
            ("std", "std"),
            ("median", "median"),
            ("q25", lambda x: np.quantile(x, 0.25)),
            ("q75", lambda x: np.quantile(x, 0.75)),
            ("min", "min"),
            ("max", "max"),
        ])
        .reset_index()
    )
    
    t_df = time.time() - t_df_start
    logger.info(f"Created summary DataFrame: {len(summary_df)} rows ({t_df*1000:.2f}ms)")
    
    # Create results object
    results = SimulationResults(
        long_df=long_df,
        summary_df=summary_df,
        config=config,
    )
    
    t_sim_total = time.time() - t_sim_start
    
    logger.info("\n" + "="*70)
    logger.info("Simulation Complete!")
    logger.info("="*70)
    logger.info(f"Total time: {t_sim_total:.2f}s")
    logger.info(f"Measurements collected: {len(long_df):,}")
    logger.info(f"Average time per iteration: {t_sim_total/total_iterations*1000:.2f}ms")
    logger.info("="*70)
    
    return results


# ==============================================================================
# Example Usage
# ==============================================================================


def main() -> None:
    """Run example simulation."""
    
    # Set logging level (use TRACE for maximum detail, DEBUG for method calls, INFO for progress)
    set_logging_level("TRACE")
    
    logger.info("="*70)
    logger.info(" GRASSMANNIAN DISTANCE SIMULATION - EXAMPLE RUN")
    logger.info("="*70)
    
    # Configuration
    config = SimulationConfig(
        ps=[100, 500, 1000, 2000, 5000, 10000],
        ns=[63, 126],
        radii=[0.1, 0.3, 0.5, 0.7, 0.9],
        eigenvalues={
            1: (9.0,),           # 1D subspace (sphere)
            2: (9.0, 4.0),       # 2D subspace (Grassmannian)
            3: (9.0, 4.0, 2.0),  # 3D subspace
        },
        noise_std=1.0,
        n_reps=12,
        n_targets_per_rep=8,
        seed=20260403,
        metric=GrassmannDistance(),
    )
    
    # Run simulation
    logger.info("\n" + "="*70)
    logger.info(" Starting main simulation run...")
    logger.info("="*70)
    logger.info(f"Grid: {len(config.ps)} × {len(config.ns)} × {len(config.radii)} "
          f"× {len(config.eigenvalues)} = "
          f"{len(config.ps) * len(config.ns) * len(config.radii) * len(config.eigenvalues)} combinations")
    
    results = run_simulation(config)
    
    # Save
    output_dir = Path("grassmann_simulation_output")
    results.save(output_dir)
    
    # Display sample
    logger.info("\n" + "="*70)
    logger.info(" Sample Results Summary")
    logger.info("="*70)
    print("\n" + results.summary_df.head(10).to_string(index=False))
    
    logger.info("\n" + "█"*70)
    logger.info(" SIMULATION COMPLETE!")
    logger.info("█"*70)


if __name__ == "__main__":
    main()
