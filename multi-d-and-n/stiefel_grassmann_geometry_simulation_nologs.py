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
from loguru import
from numpy.linalg import svd
from scipy.linalg import eigh, expm, logm, qr


# ==============================================================================
# Logging Configuration with loguru
# ==============================================================================

# Remove default handler


# Add console handler with detailed format

    # sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="TRACE",
    colorize=True,
)
#

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

    
    # Define a clean format
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # 1. Add Console Handler

        sys.stderr, 
        format=log_format,
        level=level.upper(),
        colorize=True
    )
    
    # 2. Add File Handler (logs to the specified file)
    if log_file:

            log_file,
            format=log_format,
            level=level.upper(),
            colorize=False, # Color codes look messy in raw text files
            mode="w"        # "w" overwrites previous runs. Use "a" to append.
        )
        



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
            

            return result
            
        except Exception as e:
            t_elapsed = time.time() - t_start

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
        


        
        # Special case: k=1 (sphere)
        if overlap.shape == (1, 1):

            cos_angle = np.clip(overlap[0, 0], -1.0, 1.0)

            dist = float(np.arccos(cos_angle))

            return dist
        
        # General case: k≥2 (Grassmannian)

        singular_values = svd(overlap, compute_uv=False)

        
        singular_values = np.clip(singular_values, -1.0, 1.0)

        

        principal_angles = np.arccos(singular_values)

        

        dist = float(np.linalg.norm(principal_angles))

        
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
        



        
        # Special case: k=1 (sphere)
        if k == 1:

            theta = radius

            cos_part = np.cos(theta) * overlap_truth[0, 0]
            sin_part = np.sin(theta) * overlap_complement[0, 0]

            result = np.array([[cos_part + sin_part]])

            return result
        
        # General case: k≥2


        weights = np.abs(rng.normal(size=k))

        

        weights /= np.linalg.norm(weights)

        
        principal_angles = radius * weights

        
        # Random rotation in k-dimensional subspace


        R_raw = rng.normal(size=(k, k))

        R, _ = qr(R_raw)

        
        if np.linalg.det(R) < 0:

            R[:, 0] *= -1
        
        # Geodesic formula: cos(θ)·R^T·S_u + sin(θ)·S_g


        cos_diag = np.diag(np.cos(principal_angles))

        

        cos_part = cos_diag @ (R.T @ overlap_truth)

        

        sin_diag = np.diag(np.sin(principal_angles))

        sin_part = sin_diag @ overlap_complement

        

        result = cos_part + sin_part

        
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

            raise ValueError(error_msg)
        
        if np.any(self.signal_eigenvalues <= self.noise_std**2):
            error_msg = f"All eigenvalues must exceed noise variance {self.noise_std**2}"

            raise ValueError(error_msg)
        

            f"Created SpikeCovarianceModel: p={self.p}, k={self.k}, "
            f"eigenvalues={self.eigenvalues}, σ={self.noise_std}"
        )

    
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

        t_start = time.time()
        
        # Signal scales (after removing noise variance)

        signal_scales = np.sqrt(self.signal_eigenvalues - self.noise_std**2)

        
        # Generate components

        F = rng.normal(size=(self.k, n))

        

        E_truth = rng.normal(size=(self.k, n))
        

        E_complement = rng.normal(size=(self.k, n))
        
        # Build sample Gram matrix components

        A = signal_scales[:, None] * F + self.noise_std * E_truth

        

        B = self.noise_std * E_complement

        
        # Residual from (p-2k) dimensions
        residual_df = self.p - 2 * self.k

        residual_gram = self.noise_std**2 * self._sample_wishart_gram(
            df=residual_df,
            dim=n,
            rng=rng,
        )

        
        # Full Gram matrix

        gram = A.T @ A + B.T @ B + residual_gram

        
        # Extract top k eigenvectors
        n_obs = gram.shape[0]

        evals, V = eigh(gram, subset_by_index=[n_obs - self.k, n_obs - 1])

        

        evals = np.clip(evals[::-1], 0.0, None)
        singular_values = np.sqrt(evals)
        V = V[:, ::-1]
        


        
        # Compute overlaps

        inv_s = np.where(singular_values > 0, 1.0 / singular_values, 1.0)

        

        overlap_truth = (A @ V) * inv_s

        

        overlap_complement = (B @ V) * inv_s

        
        # Sign convention for k=1
        if self.k == 1 and overlap_truth[0, 0] < 0.0:

            overlap_truth *= -1.0
            overlap_complement *= -1.0
        
        t_elapsed = time.time() - t_start

        
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
        



        
        output_dir.mkdir(parents=True, exist_ok=True)

        
        # Save data
        t_start = time.time()
        
        csv_long = output_dir / "distances_all.csv"
        self.long_df.to_csv(csv_long, index=False)

        
        csv_summary = output_dir / "distances_summary.csv"
        self.summary_df.to_csv(csv_summary, index=False)

        
        t_csv = time.time() - t_start

        
        # Generate plots

        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)

        
        t_plot_start = time.time()
        for k in self.config.eigenvalues.keys():
            plot_file = figure_dir / f"dimension_{k}.png"

            self._plot_dimension(k, plot_file)

        
        t_plot = time.time() - t_plot_start

        
        # Write README

        self._write_readme(output_dir)

        
        t_total = time.time() - t_start
        








    
    # ------------------------------------------------------------------
    # Plotting constants — centralised so every helper stays in sync.
    # Override these on the class (or per-instance) to restyle globally.
    # ------------------------------------------------------------------
    _PLOT_DISTANCE_TYPES: tuple[str, ...] = ("sample-target", "sample-truth")
    """Distance types rendered in every faceted plot."""

    _PLOT_STYLE: dict = dict(style="whitegrid", context="paper")
    """Seaborn theme kwargs forwarded to sns.set_theme()."""

    _CATPLOT_KW: dict = dict(
        kind="box",
        x="p",
        y="distance",
        hue="distance_type",
        col="radius_label",
        row="n_label",
        sharey=True,
        height=3.0,
        aspect=1.1,
        linewidth=0.8,
        showfliers=False,
    )
    """Base catplot keyword arguments.  Per-call overrides are merged on top."""

    _REFLINE_STYLE: dict = dict(ls="--", lw=1.2, color="black", alpha=0.7)
    """Style of the horizontal reference line drawn at the target radius."""

    _SUPTITLE_FONTSIZE: int = 14
    _SAVE_DPI: int = 220

    # ------------------------------------------------------------------
    # Public entry point called by save()
    # ------------------------------------------------------------------

    def _plot_dimension(self, k: int, save_path: Path) -> None:
        """Produce and save the faceted distance plot for latent dimension *k*.

        The figure has one row per sample size *n* and one column per target
        radius.  Each panel shows box plots of ``sample-target`` and
        ``sample-truth`` distances across ambient dimensions *p*, with a
        dashed horizontal reference line at the nominal target radius.

        Parameters
        ----------
        k:
            Latent dimension (key in ``SimulationConfig.eigenvalues``).
        save_path:
            Destination file (PNG).  Parent directories are created as needed.

        Example
        -------
        Call indirectly via :meth:`save`::

            results.save("output/")            # writes figures/dimension_2.png

        Or directly for a one-off re-render::

            from pathlib import Path
            results._plot_dimension(k=2, save_path=Path("fig_k2.png"))

        To add a new distance type to the plot, append its label to
        ``SimulationResults._PLOT_DISTANCE_TYPES`` before calling ``save``::

            SimulationResults._PLOT_DISTANCE_TYPES = (
                "sample-target", "sample-truth", "truth-target"
            )
        """
        plot_df = self._filter_plot_data(k)
        if plot_df.empty:
            return

        col_order, row_order = self._derive_facet_orders(plot_df)
        radius_map = self._build_radius_map(plot_df)

        sns.set_theme(**self._PLOT_STYLE)
        g = self._build_catplot(plot_df, col_order, row_order)
        self._annotate_axes(g, col_order, radius_map)
        self._set_figure_titles(g, k, col_order, row_order)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=self._SAVE_DPI, bbox_inches="tight")
        plt.close(g.fig)

    # ------------------------------------------------------------------
    # Private helpers — each has one clear responsibility
    # ------------------------------------------------------------------

    def _filter_plot_data(self, k: int) -> pd.DataFrame:
        """Subset ``long_df`` to the rows relevant for a single-dimension plot.

        Keeps only rows where ``dimension == k`` and ``distance_type`` is in
        ``_PLOT_DISTANCE_TYPES``.  Returns a copy so downstream mutations are safe.

        Parameters
        ----------
        k:
            Latent dimension to filter on.
        """
        mask = (self.long_df["dimension"] == k) & (
            self.long_df["distance_type"].isin(self._PLOT_DISTANCE_TYPES)
        )
        return self.long_df[mask].copy()

    @staticmethod
    def _derive_facet_orders(plot_df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Return sorted column (radius) and row (n) label orders for the FacetGrid.

        Sorting is numerical — derived from the underlying float/int values so
        that lexicographic accidents (e.g. "r=0.9" < "r=0.10") cannot occur.

        Parameters
        ----------
        plot_df:
            Filtered data frame that contains ``radius`` and ``n`` columns.

        Returns
        -------
        col_order:
            Labels ``["r=0.1", "r=0.3", ...]`` in ascending radius order.
        row_order:
            Labels ``["n=63", "n=126", ...]`` in ascending n order.
        """
        col_order = [f"r={r:.1f}" for r in sorted(plot_df["radius"].unique())]
        row_order = [f"n={n}"    for n in sorted(plot_df["n"].unique())]
        return col_order, row_order

    @staticmethod
    def _build_radius_map(plot_df: pd.DataFrame) -> dict[str, float]:
        """Map each radius label (e.g. ``"r=0.3"``) to its float value.

        Used to position the horizontal reference line in each column.

        Parameters
        ----------
        plot_df:
            Filtered data frame that contains the ``radius`` column.
        """
        return {f"r={r:.1f}": r for r in sorted(plot_df["radius"].unique())}

    def _build_catplot(
        self,
        plot_df: pd.DataFrame,
        col_order: list[str],
        row_order: list[str],
    ) -> sns.FacetGrid:
        """Construct the seaborn FacetGrid from ``_CATPLOT_KW`` plus call-specific orders.

        Merging happens via ``{**self._CATPLOT_KW, ...}`` so that subclasses can
        override any default by reassigning ``_CATPLOT_KW`` without touching this
        method.

        Parameters
        ----------
        plot_df:
            Filtered data frame ready for plotting.
        col_order, row_order:
            Sorted facet labels from :meth:`_derive_facet_orders`.
        """
        return sns.catplot(
            data=plot_df,
            **{
                **self._CATPLOT_KW,
                "col_order": col_order,
                "row_order": row_order,
                "hue_order": list(self._PLOT_DISTANCE_TYPES),
            },
        )

    def _annotate_axes(
        self,
        g: sns.FacetGrid,
        col_order: list[str],
        radius_map: dict[str, float],
    ) -> None:
        """Add per-panel reference lines and axis labels to every subplot.

        Iterates over the 2-D ``g.axes`` array (rows × columns) and, for each
        panel, draws a dashed horizontal line at the nominal target radius for
        that column.

        Parameters
        ----------
        g:
            The FacetGrid returned by :meth:`_build_catplot`.
        col_order:
            Column labels in left-to-right order; used to look up the radius
            value for each column position.
        radius_map:
            Label-to-float mapping from :meth:`_build_radius_map`.
        """
        for axes_row in g.axes:
            for label, ax in zip(col_order, axes_row):
                ax.axhline(radius_map[label], **self._REFLINE_STYLE)
                ax.set_xlabel("Ambient dimension (p)")
                ax.set_ylabel("Distance")

    def _set_figure_titles(
        self,
        g: sns.FacetGrid,
        k: int,
        col_order: list[str],
        row_order: list[str],
    ) -> None:
        """Set panel titles, figure suptitle, and strip the legend title.

        The suptitle follows the convention ``"{k}D {geometry}: Distance vs (p, n, radius)"``.
        For ``k == 1`` the geometry label is ``"Sphere"``; for ``k ≥ 2`` it is
        ``"Grassmann"``.  Extend this mapping here if new geometries are added.

        Parameters
        ----------
        g:
            The annotated FacetGrid.
        k:
            Latent dimension — determines the geometry label.
        col_order, row_order:
            Passed only to let callers confirm consistency; not used directly.
        """
        geometry = "Sphere" if k == 1 else "Grassmann"
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.fig.suptitle(
            f"{k}D {geometry}: Distance vs (p, n, radius)",
            fontsize=self._SUPTITLE_FONTSIZE,
        )
        g.fig.subplots_adjust(top=0.90)
        if g._legend:
            g._legend.set_title("")
    
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



    
    t_sim_start = time.time()
    
    # Log configuration










    
    # Calculate grid size
    total_combinations = (
        len(config.eigenvalues) *
        len(config.ps) *
        len(config.ns) *
        len(config.radii)
    )
    total_iterations = total_combinations * config.n_reps
    




    
    rng = np.random.default_rng(config.seed)
    records: list[dict] = []
    
    iteration = 0
    
    # Iterate over parameter grid



    
    for k, eigenvalues in config.eigenvalues.items():

        
        for p in config.ps:

            
            for n in config.ns:

                
                for radius in config.radii:

                    
                    # Create model
                    t_model_start = time.time()
                    model = SpikeCovarianceModel(
                        p=p,
                        eigenvalues=eigenvalues,
                        noise_std=config.noise_std,
                    )
                    t_model = time.time() - t_model_start

                    
                    # Statistical replications
                    for rep in range(config.n_reps):
                        iteration += 1
                        progress_pct = 100 * iteration / total_iterations
                        

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



    
    t_df_start = time.time()
    long_df = pd.DataFrame.from_records(records)
    long_df["radius_label"] = long_df["radius"].map(lambda x: f"r={x:.1f}")
    long_df["n_label"] = long_df["n"].map(lambda x: f"n={x}")
    

    
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

    
    # Create results object
    results = SimulationResults(
        long_df=long_df,
        summary_df=summary_df,
        config=config,
    )
    
    t_sim_total = time.time() - t_sim_start
    







    
    return results


# ==============================================================================
# Example Usage
# ==============================================================================


def main() -> None:
    """Run example simulation."""
    
    # Set logging level (use TRACE for maximum detail, DEBUG for method calls, INFO for progress)
    set_logging_level("TRACE")
    



    
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




          f"× {len(config.eigenvalues)} = "
          f"{len(config.ps) * len(config.ns) * len(config.radii) * len(config.eigenvalues)} combinations")
    
    results = run_simulation(config)
    
    # Save
    output_dir = Path("grassmann_simulation_output")
    results.save(output_dir)
    
    # Display sample



    print("\n" + results.summary_df.head(10).to_string(index=False))
    





if __name__ == "__main__":
    main()
