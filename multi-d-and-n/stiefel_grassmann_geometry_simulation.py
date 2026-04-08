"""
High-Dimensional Grassmannian Distance Simulation
==================================================
Studies how Grassmannian distances behave in high-dimensional PCA under a
spiked covariance model.  For each parameter combination (k, p, n, radius)
the simulation measures:

  - **sample-truth**:  estimation error d(Û, U_true) on Gr(k,p)
  - **sample-target**: d(Û, V) where V is a synthetic point at the prescribed
                       geodesic radius from U_true
  - **truth-target**:  ground-truth radius (constant; sanity check)

Key geometry note
-----------------
All distances live on Gr(k,p) — the Grassmann manifold of k-dimensional
subspaces of ℝᵖ.  The distance is the L² norm of principal angles:

    d = √(θ₁² + … + θₖ²)

For k=1 this reduces to the spherical geodesic on S^{p-1}.
For Stiefel manifold distances (which penalise rotation within the subspace),
see `StiefelCanonicalDistance`.

Quick start
-----------
    >>> config = SimulationConfig(
    ...     ps=[100, 500, 1000],
    ...     ns=[63, 126],
    ...     radii=[0.1, 0.3, 0.5],
    ...     eigenvalues={2: (9.0, 4.0)},   # 2-D subspace
    ...     n_reps=10,
    ... )
    >>> results = run_simulation(config)
    >>> results.save("output/")

Extending the metric
--------------------
Pass any object satisfying the `DistanceMetric` Protocol as `config.metric`.
The two built-in implementations are `GrassmannDistance` (default) and
`StiefelCanonicalDistance`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from numpy.linalg import svd
from scipy.linalg import eigh, logm, qr


# ==============================================================================
# Logging
# ==============================================================================

# Loguru ships with a default stderr handler at level DEBUG.  Remove it so
# `configure_logging` can install a clean pair of handlers from scratch.
logger.remove()

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Install console (and optionally file) log handlers.

    Call once at program startup, before any simulation code runs.

    Parameters
    ----------
    level:
        Loguru level string: ``"TRACE"``, ``"DEBUG"``, ``"INFO"`` (default),
        ``"SUCCESS"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.
        Use ``"TRACE"`` for maximum verbosity during debugging; ``"INFO"``
        for normal production runs.
    log_file:
        If provided, a second handler writes to this path (mode ``"w"``).
        Omit or pass ``None`` for console-only output.

    Example
    -------
        >>> configure_logging("DEBUG", log_file="run.log")
    """
    logger.remove()  # Clear any handlers already added in this session.
    logger.add(sys.stderr, format=_LOG_FORMAT, level=level.upper(), colorize=True)
    if log_file:
        logger.add(log_file, format=_LOG_FORMAT, level=level.upper(),
                   colorize=False, mode="w")


def _fmt_arg(v: object) -> str:
    """Format a single function argument for trace logging."""
    return f"array{v.shape}" if isinstance(v, np.ndarray) else repr(v)[:80]


def trace_calls(func: Callable) -> Callable:
    """Decorator: log entry, exit, and wall-clock time of any function/method.

    Attaches at TRACE level so it produces no output at the default INFO level.
    Apply to computationally significant methods to aid profiling and debugging.

    Example
    -------
        >>> @trace_calls
        ... def my_fn(x: int) -> int:
        ...     return x * 2
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Detect methods: first positional arg is `self`.
        is_method = args and hasattr(type(args[0]), func.__name__)
        prefix = f"{type(args[0]).__name__}." if is_method else ""
        call_args = args[1:] if is_method else args

        params = ", ".join(
            [*(_fmt_arg(a) for a in call_args),
             *(f"{k}={_fmt_arg(v)}" for k, v in kwargs.items())]
        )
        logger.trace("→ {}{}({})", prefix, func.__name__, params)
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.trace("✗ {}{} raised {} in {:.3f}s",
                         prefix, func.__name__, type(exc).__name__,
                         time.perf_counter() - t0)
            raise
        logger.trace("← {}{} → {} in {:.3f}s",
                     prefix, func.__name__, _fmt_arg(result),
                     time.perf_counter() - t0)
        return result
    return wrapper


# ==============================================================================
# Distance Metric Protocol and Implementations
# ==============================================================================

class DistanceMetric(Protocol):
    """Structural protocol for geodesic distance metrics on matrix manifolds.

    Implementors work entirely in the *overlap* representation: a k×k matrix
    M = U₁ᵀ U₂ whose singular values are the cosines of the principal angles
    between the two subspaces.  This avoids materialising the full p×p matrices.

    To plug in a custom metric::

        class MyMetric:
            name = "my-metric"

            def distance(self, overlap: np.ndarray) -> float:
                ...

            def sample_target(self, radius, overlap_truth,
                               overlap_complement, rng) -> np.ndarray:
                ...

        config = SimulationConfig(..., metric=MyMetric())
    """

    @property
    def name(self) -> str:
        """Identifier written into the ``metric`` column of result DataFrames."""
        ...

    def distance(self, overlap: np.ndarray) -> float:
        """Geodesic distance computed from the k×k overlap M = U₁ᵀ U₂."""
        ...

    def sample_target(
        self,
        radius: float,
        overlap_truth: np.ndarray,
        overlap_complement: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return overlap M′ = V_targetᵀ Û for a random target V at distance *radius* from U_true."""
        ...


class GrassmannDistance:
    """Geodesic distance on Gr(k,p): d = ‖θ‖ where θ are principal angles.

    The principal angles θ₁,…,θₖ ∈ [0, π/2] are recovered from the singular
    values σᵢ = cos(θᵢ) of the overlap matrix M = U₁ᵀ U₂.

    - k=1: reduces to the spherical arc-length on S^{p-1}.
    - k≥2: full Grassmann geodesic; rotation within the subspace is ignored.

    This is the **default** metric for `SimulationConfig`.
    """

    name: str = "grassmann"

    @trace_calls
    def distance(self, overlap: np.ndarray) -> float:
        """Compute d(U₁, U₂) = ‖arccos(σ(M))‖ from overlap M = U₁ᵀ U₂.

        Parameters
        ----------
        overlap:
            k×k matrix M = U₁ᵀ U₂.  For k=1 a (1,1) array suffices.
        """
        overlap = np.asarray(overlap)
        # k=1 fast path: no SVD needed.
        if overlap.shape == (1, 1):
            return float(np.arccos(np.clip(overlap[0, 0], -1.0, 1.0)))
        sv = np.clip(svd(overlap, compute_uv=False), -1.0, 1.0)
        return float(np.linalg.norm(np.arccos(sv)))

    @trace_calls
    def sample_target(
        self,
        radius: float,
        overlap_truth: np.ndarray,
        overlap_complement: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate overlap M′ = V_targetᵀ Û for a random V at Grassmann distance *radius* from U_true.

        Uses the geodesic formula

            M′ = diag(cos θ) · Rᵀ · M_truth  +  diag(sin θ) · M_complement

        where θ are principal angles with ‖θ‖ = radius and R ∈ SO(k) is a
        random rotation injecting directional diversity across replications.

        Parameters
        ----------
        radius:
            Target geodesic distance from U_true to V.
        overlap_truth:
            k×n matrix M_truth = U_trueᵀ Û.
        overlap_complement:
            k×n matrix M_comp = Gᵀ Û (complement basis overlaps).
        rng:
            NumPy Generator for reproducibility.
        """
        k = overlap_truth.shape[0]

        # k=1 fast path: single principal angle equals radius exactly.
        if k == 1:
            result = (np.cos(radius) * overlap_truth[0, 0]
                      + np.sin(radius) * overlap_complement[0, 0])
            return np.array([[result]])

        # Distribute radius across k principal angles proportionally to
        # half-normal weights, giving isotropy over the angle simplex.
        weights = np.abs(rng.normal(size=k))
        weights /= np.linalg.norm(weights)
        theta = radius * weights                    # shape (k,)

        # Random SO(k) rotation for directional diversity.
        R, _ = qr(rng.normal(size=(k, k)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1                          # Ensure det = +1.

        cos_part = np.diag(np.cos(theta)) @ (R.T @ overlap_truth)
        sin_part = np.diag(np.sin(theta)) @ overlap_complement
        return cos_part + sin_part


class StiefelCanonicalDistance:
    """Geodesic distance on St(p,k) with the canonical metric (β=½).

    Unlike `GrassmannDistance`, this penalises SO(k) rotations *within* the
    subspace.  Two frames that span the same subspace but differ by a rotation
    have Stiefel distance > 0 but Grassmann distance = 0.

    The canonical metric norm of a tangent vector Δ = U·A + U_⊥·B is

        ‖Δ‖² = ½‖A‖_F² + ‖B‖_F²

    Implementation uses the 2k×2k matrix-log trick (Edelman et al. 1998),
    which reduces complexity from O(p²k) to O(k³) — constant in p.

    Notes
    -----
    `distance()` is intentionally **not** implemented: the simulation pipeline
    passes only overlap matrices, which are insufficient for Stiefel distances.
    Use `distance_from_frames(U1, U2)` directly when full frames are available.
    """

    name: str = "stiefel-canonical"

    def distance(self, overlap: np.ndarray) -> float:
        raise NotImplementedError(
            "Stiefel distance requires full frames, not just the overlap matrix. "
            "Call distance_from_frames(U1, U2) directly."
        )

    def sample_target(self, radius, overlap_truth, overlap_complement, rng):
        raise NotImplementedError(
            "Target sampling on St(p,k) is not yet implemented. "
            "Use GrassmannDistance for the simulation pipeline."
        )

    def distance_from_frames(self, U1: np.ndarray, U2: np.ndarray) -> float:
        """Canonical geodesic distance via the 2k×2k matrix-log reduction.

        Parameters
        ----------
        U1, U2:
            Orthonormal frames, shape (p, k).

        Returns
        -------
        float
            Canonical geodesic distance √(½‖Δ₁₁‖_F² + ‖Δ₂₁‖_F²) where
            Δ = logm(G) is the skew-symmetrised log of the 2k×2k rotation G.

        Example
        -------
            >>> metric = StiefelCanonicalDistance()
            >>> d = metric.distance_from_frames(U1, U2)
        """
        k = U1.shape[1]
        M = U1.T @ U2
        _, R = qr(U2 - U1 @ M, mode="economic")

        # Build 2k×2k orthogonal matrix G encoding the geodesic endpoint.
        G = np.block([[M, -R.T], [R, M]])

        Delta = np.real(logm(G))
        Delta = 0.5 * (Delta - Delta.T)            # Enforce skew-symmetry.

        dist_sq = (0.5 * np.linalg.norm(Delta[:k, :k], "fro") ** 2
                   + np.linalg.norm(Delta[k:, :k], "fro") ** 2)
        return float(np.sqrt(dist_sq))


# ==============================================================================
# Spiked Covariance Model
# ==============================================================================

@dataclass
class SpikeCovarianceModel:
    """Spiked covariance model Σ = UΛUᵀ + σ²I for high-dimensional PCA.

    Population eigenstructure
    -------------------------
    Σ = U diag(λ₁,…,λₖ) Uᵀ + σ²I,   λᵢ > σ² for all i.

    By rotational invariance the truth U can be fixed to the first k standard
    basis vectors, so the simulation works entirely with k×n overlap matrices
    instead of p×p full covariances — O(kn) cost, independent of p.

    Parameters
    ----------
    p:
        Ambient dimension.  Must satisfy p ≥ 2k.
    eigenvalues:
        Signal eigenvalues (λ₁,…,λₖ), each strictly greater than σ².
    noise_std:
        Noise standard deviation σ (default 1.0).

    Raises
    ------
    ValueError
        If p < 2k or any eigenvalue ≤ σ².
    """

    p: int
    eigenvalues: tuple[float, ...]
    noise_std: float = 1.0

    def __post_init__(self) -> None:
        self.k: int = len(self.eigenvalues)
        self.signal_eigenvalues: np.ndarray = np.asarray(self.eigenvalues)
        if self.p < 2 * self.k:
            raise ValueError(f"Require p ≥ 2k; got p={self.p}, k={self.k}.")
        if np.any(self.signal_eigenvalues <= self.noise_std ** 2):
            raise ValueError(
                f"All signal eigenvalues must exceed noise variance σ²="
                f"{self.noise_std**2:.4g}."
            )
        logger.debug("SpikeCovarianceModel: p={}, k={}, λ={}, σ={}",
                     self.p, self.k, self.eigenvalues, self.noise_std)

    @trace_calls
    def sample_overlaps(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw k×n overlap matrices M_truth = U_trueᵀ Û and M_comp = Gᵀ Û.

        Algorithm
        ---------
        Fix U_true = [e₁ … eₖ].  The sample Gram matrix splits as

            XᵀX = AᵀA + BᵀB + residual,

        where A (k×n) captures signal + truth-direction noise, B (k×n) captures
        complement-direction noise, and the residual (n×n) is Wishart(σ²I, p-2k).
        The top-k eigenvectors V of XᵀX define Û via the thin SVD, and overlaps
        follow from projecting A and B onto V.

        Parameters
        ----------
        n:
            Number of observations.
        rng:
            NumPy Generator.

        Returns
        -------
        overlap_truth:
            k×n matrix M_truth = U_trueᵀ Û.
        overlap_complement:
            k×n matrix M_comp = Gᵀ Û (complement basis overlaps).
        """
        sig = np.sqrt(self.signal_eigenvalues - self.noise_std ** 2)  # (k,)
        F = rng.normal(size=(self.k, n))

        A = sig[:, None] * F + self.noise_std * rng.normal(size=(self.k, n))
        B = self.noise_std * rng.normal(size=(self.k, n))
        residual = self.noise_std ** 2 * _sample_wishart_gram(
            df=self.p - 2 * self.k, dim=n, rng=rng
        )

        gram = A.T @ A + B.T @ B + residual       # n×n symmetric

        # Extract top-k eigenvectors of gram (ascending order from eigh).
        evals, V = eigh(gram, subset_by_index=[n - self.k, n - 1])
        evals = np.clip(evals[::-1], 0.0, None)
        V = V[:, ::-1]
        inv_s = np.where(evals > 0, 1.0 / np.sqrt(evals), 1.0)  # (k,)

        overlap_truth = (A @ V) * inv_s
        overlap_complement = (B @ V) * inv_s

        # For k=1, fix sign so overlap_truth ≥ 0 (canonical hemisphere choice).
        if self.k == 1 and overlap_truth[0, 0] < 0.0:
            overlap_truth *= -1.0
            overlap_complement *= -1.0

        return overlap_truth, overlap_complement


def _sample_wishart_gram(df: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ZᵀZ with Z ~ N(0,I)_{df×dim} via Bartlett decomposition.

    This is a module-level helper (not a method) because it is a pure numerical
    utility with no dependence on model state.

    - df ≤ 0  → zero matrix (degenerate case).
    - df < dim → direct ZᵀZ (exact; Bartlett requires df ≥ dim).
    - df ≥ dim → Bartlett lower-triangular Cholesky factor (O(dim²) memory,
                  avoids forming the df×dim matrix).

    Parameters
    ----------
    df:
        Wishart degrees of freedom (number of rows of Z).
    dim:
        Dimension of the resulting dim×dim Gram matrix.
    rng:
        NumPy Generator.
    """
    if df <= 0:
        return np.zeros((dim, dim))
    if df < dim:
        Z = rng.normal(size=(df, dim))
        return Z.T @ Z
    # Bartlett: lower-triangular Cholesky factor of a Wishart(I_dim, df) sample.
    L = np.zeros((dim, dim))
    L[np.diag_indices(dim)] = np.sqrt(rng.chisquare(df=np.arange(df, df - dim, -1)))
    L[np.tril_indices(dim, k=-1)] = rng.normal(size=dim * (dim - 1) // 2)
    return L @ L.T


# ==============================================================================
# Simulation Configuration and Results
# ==============================================================================

@dataclass
class SimulationConfig:
    """Full specification of a distance-simulation experiment.

    Parameters
    ----------
    ps:
        Ambient dimensions to sweep (e.g. ``[100, 500, 1000]``).
    ns:
        Sample sizes to sweep (e.g. ``[63, 126]``).
    radii:
        Target geodesic radii (e.g. ``[0.1, 0.3, 0.5]``).
    eigenvalues:
        Dict mapping latent dimension k → signal eigenvalue tuple.
        E.g. ``{1: (9.0,), 2: (9.0, 4.0)}`` runs both k=1 and k=2.
    noise_std:
        Noise standard deviation σ (default 1.0).
    n_reps:
        Statistical replications per (k, p, n, radius) cell.
    n_targets_per_rep:
        Synthetic targets drawn per replication (averaged over to reduce
        Monte Carlo noise in the sample-target distribution).
    seed:
        Global RNG seed for reproducibility.
    metric:
        Distance metric satisfying the `DistanceMetric` Protocol.
        Defaults to `GrassmannDistance`.

    Example
    -------
        >>> config = SimulationConfig(
        ...     ps=[100, 500, 1000],
        ...     ns=[63, 126],
        ...     radii=[0.1, 0.5],
        ...     eigenvalues={2: (9.0, 4.0)},
        ...     n_reps=10,
        ... )
    """

    ps: list[int]
    ns: list[int]
    radii: list[float]
    eigenvalues: dict[int, tuple[float, ...]]
    noise_std: float = 1.0
    n_reps: int = 12
    n_targets_per_rep: int = 8
    seed: int = 20260403
    metric: DistanceMetric = field(default_factory=GrassmannDistance)

    @property
    def n_combinations(self) -> int:
        """Total number of (k, p, n, radius) grid cells."""
        return len(self.eigenvalues) * len(self.ps) * len(self.ns) * len(self.radii)


@dataclass
class SimulationResults:
    """Simulation outputs: tidy DataFrames and publication-ready plots.

    Attributes
    ----------
    long_df:
        One row per distance measurement.  Columns: ``dimension``, ``p``,
        ``n``, ``radius``, ``rep``, ``distance_type``, ``distance``,
        ``metric``, ``radius_label``, ``n_label``.
    summary_df:
        Per-cell descriptive statistics (count, mean, std, median, q25, q75,
        min, max).
    config:
        The `SimulationConfig` that produced these results.

    Example
    -------
        >>> results.save("output/")          # CSV + figures + README
        >>> results.summary_df.head()
    """

    long_df: pd.DataFrame
    summary_df: pd.DataFrame
    config: SimulationConfig

    # ------------------------------------------------------------------
    # Plotting configuration — override at class or instance level to
    # restyle all figures without touching method bodies.
    # ------------------------------------------------------------------
    _PLOT_DISTANCE_TYPES: tuple[str, ...] = ("sample-target", "sample-truth")
    _PLOT_STYLE:  dict = field(default_factory=lambda: dict(style="whitegrid", context="paper"))
    _CATPLOT_KW:  dict = field(default_factory=lambda: dict(
        kind="boxen", x="p", y="distance", hue="distance_type",
        col="radius_label", row="n_label",
        sharey=True, height=3.0, aspect=1.1, linewidth=0.8, showfliers=False,
    ))
    _REFLINE_STYLE: dict = field(default_factory=lambda: dict(
        ls="--", lw=1.2, color="black", alpha=0.7
    ))
    _SUPTITLE_FONTSIZE: int = 14
    _SAVE_DPI: int = 220

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, output_dir: Path | str) -> None:
        """Persist results to *output_dir*: CSVs, per-dimension figures, README.

        Directory structure created::

            output_dir/
              distances_all.csv       — long-format individual measurements
              distances_summary.csv   — grouped descriptive statistics
              figures/
                dimension_1.png       — faceted plot for k=1
                dimension_2.png       — faceted plot for k=2
                …
              README.txt

        Parameters
        ----------
        output_dir:
            Destination directory; created if absent.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)

        self.long_df.to_csv(output_dir / "distances_all.csv", index=False)
        self.summary_df.to_csv(output_dir / "distances_summary.csv", index=False)
        logger.info("Saved CSVs to {}", output_dir)

        for k in self.config.eigenvalues:
            path = figure_dir / f"dimension_{k}.png"
            self._plot_dimension(k, path)
            logger.info("Saved figure: {}", path)

        self._write_readme(output_dir)
        logger.info("Saved README to {}", output_dir)

    # ------------------------------------------------------------------
    # Plotting — one public entry point; private helpers below
    # ------------------------------------------------------------------

    def _plot_dimension(self, k: int, save_path: Path) -> None:
        """Build and save the faceted distance plot for latent dimension *k*.

        Layout: rows = sample sizes n, columns = target radii.  Each panel
        shows box-plots of ``sample-target`` and ``sample-truth`` distances
        across ambient dimensions p, with a dashed reference line at the
        nominal radius.

        Parameters
        ----------
        k:
            Latent dimension (key in `SimulationConfig.eigenvalues`).
        save_path:
            Destination PNG.  Parent directories created as needed.

        Example
        -------
        Re-render a single figure after tweaking `_CATPLOT_KW`::

            SimulationResults._CATPLOT_KW["kind"] = "violin"
            results._plot_dimension(k=2, save_path=Path("fig_k2.png"))

        Add ``"truth-target"`` to the rendered distance types::

            SimulationResults._PLOT_DISTANCE_TYPES = (
                "sample-target", "sample-truth", "truth-target"
            )
        """
        plot_df = self.long_df[
            (self.long_df["dimension"] == k)
            & self.long_df["distance_type"].isin(self._PLOT_DISTANCE_TYPES)
        ].copy()
        if plot_df.empty:
            logger.warning("No data for dimension k={}; skipping plot.", k)
            return

        col_order = [f"r={r:.1f}" for r in sorted(plot_df["radius"].unique())]
        row_order = [f"n={n}"    for n in sorted(plot_df["n"].unique())]
        radius_map = {label: float(label[2:]) for label in col_order}

        sns.set_theme(**self._PLOT_STYLE)
        g = sns.catplot(
            data=plot_df,
            **{**self._CATPLOT_KW,
               "col_order": col_order,
               "row_order": row_order,
               "hue_order": list(self._PLOT_DISTANCE_TYPES)},
        )
        for axes_row in g.axes:
            for label, ax in zip(col_order, axes_row):
                ax.axhline(radius_map[label], **self._REFLINE_STYLE)
                ax.set_xlabel("Ambient dimension (p)")
                ax.set_ylabel("Distance")

        geometry = "Sphere" if k == 1 else "Grassmann"
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.fig.suptitle(f"{k}D {geometry}: Distance vs (p, n, radius)",
                       fontsize=self._SUPTITLE_FONTSIZE)
        g.fig.subplots_adjust(top=0.90)
        if g._legend:
            g._legend.set_title("")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=self._SAVE_DPI, bbox_inches="tight")
        plt.close(g.fig)

    def _write_readme(self, output_dir: Path) -> None:
        """Write a plain-text summary of config and output file descriptions."""
        cfg = self.config
        lines = [
            "Grassmannian Distance Simulation Results",
            "=" * 50,
            "",
            "Configuration:",
            f"  Ambient dimensions (p) : {cfg.ps}",
            f"  Sample sizes (n)        : {cfg.ns}",
            f"  Target radii            : {cfg.radii}",
            f"  Models (k → eigenvalues): {cfg.eigenvalues}",
            f"  Noise std (σ)           : {cfg.noise_std}",
            f"  Replications            : {cfg.n_reps}",
            f"  Targets per rep         : {cfg.n_targets_per_rep}",
            f"  Metric                  : {cfg.metric.name}",
            "",
            "Files:",
            "  distances_all.csv     — all simulated distances (long format)",
            "  distances_summary.csv — grouped descriptive statistics",
            "  figures/              — faceted PNG plots by latent dimension",
            "",
            "Distance types:",
            "  sample-truth  : estimation error d(Û, U_true)",
            "  sample-target : d(Û, V) for synthetic target V",
            "  truth-target  : nominal radius (constant; sanity check)",
        ]
        (output_dir / "README.txt").write_text("\n".join(lines))


# ==============================================================================
# Simulation Runner
# ==============================================================================

def run_simulation(config: SimulationConfig) -> SimulationResults:
    """Execute the full distance simulation across the parameter grid.

    For each (k, p, n, radius) cell and each replication:

    1. Draw overlaps (M_truth, M_comp) from `SpikeCovarianceModel`.
    2. Record ``sample-truth`` distance d(Û, U_true).
    3. Draw ``n_targets_per_rep`` synthetic targets at distance *radius* from
       U_true; record each as a ``sample-target`` measurement.
    4. Record ``truth-target = radius`` as a ground-truth sanity check.

    Parameters
    ----------
    config:
        Full experiment specification.

    Returns
    -------
    SimulationResults
        Contains ``long_df`` (all measurements) and ``summary_df`` (statistics).

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
    t0 = time.perf_counter()
    total = config.n_combinations * config.n_reps
    logger.info(
        "Starting simulation: {} combinations × {} reps = {} iterations",
        config.n_combinations, config.n_reps, total,
    )

    rng = np.random.default_rng(config.seed)
    records: list[dict] = []
    iteration = 0

    for k, eigenvalues in config.eigenvalues.items():
        for p in config.ps:
            # Build model once per (k, p) cell; it is independent of n and radius.
            model = SpikeCovarianceModel(p=p, eigenvalues=eigenvalues,
                                         noise_std=config.noise_std)
            for n in config.ns:
                for radius in config.radii:
                    for rep in range(config.n_reps):
                        iteration += 1
                        logger.debug(
                            "[{}/{}] k={} p={} n={} r={} rep={}",
                            iteration, total, k, p, n, radius, rep,
                        )
                        overlap_truth, overlap_complement = model.sample_overlaps(
                            n=n, rng=rng
                        )
                        base = dict(dimension=k, p=p, n=n, radius=radius,
                                    rep=rep, metric=config.metric.name)

                        records.append({**base,
                                        "distance_type": "sample-truth",
                                        "distance": config.metric.distance(overlap_truth)})

                        for _ in range(config.n_targets_per_rep):
                            overlap_target = config.metric.sample_target(
                                radius=radius,
                                overlap_truth=overlap_truth,
                                overlap_complement=overlap_complement,
                                rng=rng,
                            )
                            records.append({**base,
                                            "distance_type": "sample-target",
                                            "distance": config.metric.distance(overlap_target)})
                            records.append({**base,
                                            "distance_type": "truth-target",
                                            "distance": radius})

    long_df = pd.DataFrame.from_records(records)
    long_df["radius_label"] = long_df["radius"].map(lambda x: f"r={x:.1f}")
    long_df["n_label"]      = long_df["n"].map(lambda x: f"n={x}")

    summary_df = (
        long_df
        .groupby(["dimension", "p", "n", "radius", "distance_type", "metric"],
                 as_index=False)["distance"]
        .agg([("count",  "count"),
              ("mean",   "mean"),
              ("std",    "std"),
              ("median", "median"),
              ("q25",    lambda x: np.quantile(x, 0.25)),
              ("q75",    lambda x: np.quantile(x, 0.75)),
              ("min",    "min"),
              ("max",    "max")])
        .reset_index()
    )

    logger.info("Simulation complete in {:.1f}s", time.perf_counter() - t0)
    return SimulationResults(long_df=long_df, summary_df=summary_df, config=config)


# ==============================================================================
# Entry Point
# ==============================================================================

def main() -> None:
    """Run the reference simulation and save outputs to ``grassmann_output/``.

    Adjust ``configure_logging`` level:
    - ``"TRACE"`` — full per-call traces (very verbose).
    - ``"DEBUG"`` — per-iteration progress.
    - ``"INFO"``  — high-level progress only (default).
    """
    configure_logging("INFO")

    config = SimulationConfig(
        ps=[100, 500, 1000, 2000, 5000, 10000],
        ns=[63, 126],
        radii=[0.1, 0.3, 0.5, 0.7, 0.9],
        eigenvalues={
            1: (9.0,),            # k=1: sphere S^{p-1}
            2: (9.0, 4.0),        # k=2: Grassmannian Gr(2,p)
            3: (9.0, 4.0, 2.0),   # k=3: Grassmannian Gr(3,p)
        },
        noise_std=1.0,
        n_reps=12,
        n_targets_per_rep=8,
        seed=20260403,
        metric=GrassmannDistance(),
    )
    logger.info(
        "Grid: {} ps × {} ns × {} radii × {} ks = {} combinations",
        len(config.ps), len(config.ns), len(config.radii),
        len(config.eigenvalues), config.n_combinations,
    )

    results = run_simulation(config)
    results.save(Path("grassmann_output"))
    print(results.summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
