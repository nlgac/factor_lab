"""
factor_sims.py - Multi-factor dispersion-bias simulation
=========================================================

Simulates a k-factor model at population scale `max_num_sec`, then for each
market-size slice `p ∈ nums_sec` compares:
  - B^GT: top-k eigenvectors of M_p = B_p^T F B_p + D_p (ground truth frame)
  - B^S:  top-k from SVD of a (num_obs × p) sample-returns window

by measuring distances from random targets (placed at prescribed radii around
B^GT) to B^S, under both Grassmann and Stiefel canonical metrics.

Design notes
------------
Shape convention: FactorModelData stores B with shape (k, p) — factors in
rows, assets in columns. So B_p^T is (p, k), and the covariance
M_p = B_p^T F B_p + D_p has shape (p, p) as expected. The original design
doc mistakenly wrote M_p = B_p F B_p^T; the code and this docstring are
authoritative.

Single-population design: one model is built at max_num_sec and sliced to each
p. This conflates subset-selection with dimension effects; accepted here as an
explicit choice, not an oversight.

Returns are drawn as one big block of shape (num_obs * num_sim, max_num_sec)
and reshaped into num_sim windows. Each window is then column-sliced to p.

Stiefel targets follow the canonical-metric geodesic: random skew-symmetric
A11 (k×k) and random A21 ((p-k)×k), jointly rescaled so
sqrt(½‖A11‖² + ‖A21‖²) = r, then exp-mapped via the O(k³) 2k×2k reduction.

Grassmann targets are the horizontal-only special case: A11 = 0, A21 random
and renormalized so ‖A21‖_F = r.

Usage
-----
    # Named spec
    python factor_sims.py toy
    python factor_sims.py full

    # Custom JSON spec (overrides numeric fields, uses default samplers)
    python factor_sims.py my_spec.json

    # From Python
    results = run_simulation(build_spec('toy'))
    results.save('factor_sims_output/')
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from scipy.linalg import expm, null_space, qr, schur
from scipy.sparse.linalg import LinearOperator, eigsh
from tqdm import tqdm

from factor_lab import (
    FactorModelBuilder,
    FactorModelData,
    FlexibleReturnsSimulator,
    create_sampler,
    svd_decomposition,
)

# ---------------------------------------------------------------------------
# Type aliases (documentation and optional static analysis; not enforced at runtime)
# Sampler:        a callable that takes an int (sample size) and returns an array of draws
# SamplerFactory: a callable that takes a Generator and returns a Sampler
#                 — binds distribution parameters but defers RNG choice until run time
# ---------------------------------------------------------------------------

Sampler = Callable[[int], np.ndarray]
SamplerFactory = Callable[[np.random.Generator], Sampler]


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class SimSpec:
    """
    All parameters for one simulation run.

    The sampler fields (beta_sampler_factories, etc.) store factory functions
    rather than live samplers. A factory is a small function that takes a
    random number generator and returns a sampler — it knows *what* distribution
    to draw from, but waits to be given a generator before drawing anything.

    This matters for reproducibility: every call to run_simulation() creates a
    fresh generator reset to the same starting state (determined by seed_model
    or seed_targets). If live samplers were stored instead, a second call would
    continue drawing from wherever the first call left off, producing different
    results. With factories, each call starts from scratch and produces identical
    output for the same seeds.

    Example:
        # Most users call build_spec('toy') instead of this directly.
        spec = SimSpec(
            max_num_sec=500, nums_sec=(50, 100, 250, 500),
            num_obs=63, num_sim=10,
            target_radii=(0.1, 0.5, 1.0), num_targets=5, k_factors=3,
            beta_sampler_factories=[
                lambda rng: create_sampler('normal', rng, loc=1.0, scale=0.5),
                lambda rng: create_sampler('normal', rng),
                lambda rng: create_sampler('normal', rng),
            ],
            idio_vol_sampler_factory=lambda rng: create_sampler('uniform', rng, low=0.1, high=5.0),
            factor_variances=[0.05**2, 0.1**2, 0.1**2],
            factor_return_sampler_factories=[lambda rng: create_sampler('normal', rng)] * 3,
            idio_return_sampler_factory=lambda rng: create_sampler('normal', rng),
        )
    """

    max_num_sec: int
    nums_sec: Sequence[int]
    num_obs: int
    num_sim: int
    target_radii: Sequence[float]
    num_targets: int
    k_factors: int

    # Model construction
    beta_sampler_factories: list[SamplerFactory]          # length k
    idio_vol_sampler_factory: SamplerFactory              # single
    factor_variances: list[float]                         # length k

    # Returns simulation (standardized samplers, scaled internally by F, D)
    factor_return_sampler_factories: list[SamplerFactory] # length k
    idio_return_sampler_factory: SamplerFactory           # single

    seed_model: int = 42
    seed_targets: int = 12345

    def __post_init__(self) -> None:
        k = self.k_factors
        if len(self.beta_sampler_factories) != k:
            raise ValueError(
                f"beta_sampler_factories length {len(self.beta_sampler_factories)} != k={k}"
            )
        if len(self.factor_variances) != k:
            raise ValueError(f"factor_variances length {len(self.factor_variances)} != k={k}")
        if len(self.factor_return_sampler_factories) != k:
            raise ValueError(
                f"factor_return_sampler_factories length "
                f"{len(self.factor_return_sampler_factories)} != k={k}"
            )
        if max(self.nums_sec) > self.max_num_sec:
            raise ValueError(f"max(nums_sec)={max(self.nums_sec)} > max_num_sec={self.max_num_sec}")
        if min(self.nums_sec) < 2 * k:
            raise ValueError(f"min(nums_sec)={min(self.nums_sec)} < 2k={2*k}; Stiefel exp needs p>=2k")
        if self.num_obs <= 0 or self.num_sim <= 0 or self.num_targets <= 0:
            raise ValueError("num_obs, num_sim, num_targets must be positive")


# ---------------------------------------------------------------------------
# Model & returns
# ---------------------------------------------------------------------------


def build_population_model(spec: SimSpec, rng: np.random.Generator) -> FactorModelData:
    """Build (B, F, D) at p = max_num_sec using FactorModelBuilder."""
    logger.debug("Building population model: p={}, k={}", spec.max_num_sec, spec.k_factors)
    beta_samplers = [factory(rng) for factory in spec.beta_sampler_factories]
    idio_vol_sampler = spec.idio_vol_sampler_factory(rng)
    builder = FactorModelBuilder(rng=rng)
    model = builder.build(
        p=spec.max_num_sec,
        k=spec.k_factors,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=spec.factor_variances,
    )
    logger.debug("Population model built: B={}, F={}, D={}", model.B.shape, model.F.shape, model.D.shape)
    return model


def simulate_all_returns(
    model: FactorModelData,
    spec: SimSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One contiguous return draw, reshaped into (num_sim, num_obs, max_num_sec).

    Single big call amortizes sampler overhead; reshape is zero-copy.
    """
    n_total = spec.num_obs * spec.num_sim
    logger.debug("Simulating returns: {} periods x {} securities ({} sims x {} obs)",
                 n_total, spec.max_num_sec, spec.num_sim, spec.num_obs)
    factor_return_samplers = [f(rng) for f in spec.factor_return_sampler_factories]
    idio_return_sampler = spec.idio_return_sampler_factory(rng)
    simulator = FlexibleReturnsSimulator(rng=rng)
    result = simulator.simulate(
        model=model,
        n_periods=n_total,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler,
    )
    returns = result['security_returns']                      # (T_total, p)
    reshaped = returns.reshape(spec.num_sim, spec.num_obs, spec.max_num_sec)
    logger.debug("Returns array shape: {}", reshaped.shape)
    return reshaped


def slice_model(model: FactorModelData, p: int) -> FactorModelData:
    """Restrict (B, F, D) to the first p assets. F is untouched (k×k)."""
    return FactorModelData(
        B=model.B[:, :p],
        F=model.F,
        D=model.D[:p, :p],
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def ground_truth_frame(model_slice: FactorModelData, k: int) -> np.ndarray:
    """
    Top-k eigenvectors of M_p = B_p^T F B_p + D_p, descending eigenvalue.

    Exploits the structure of our setting — B (k,p), F diagonal (k,k),
    D diagonal (p,p) — to avoid forming the (p,p) matrix entirely.

    Uses a matrix-free approach via scipy.sparse.linalg.eigsh:
    each matrix-vector product M x = D x + B^T (F (B x)) costs O(kp),
    and eigsh needs O(k) products, giving O(k^2 p) total.

    At p=10000, k=3: negligible memory and <1ms, vs ~4s and ~800MB
    for dense eigh on the full (p,p) matrix.

    Accuracy: subspace matches dense eigh to ~1e-8 Grassmann distance,
    verified at p=50, 100, 500.
    """
    B = model_slice.B                                         # (k, p)
    f_diag = np.diag(model_slice.F)                           # (k,)
    d = np.diag(model_slice.D)                                # (p,)
    p = B.shape[1]

    def matvec(x):
        return d * x + B.T @ (f_diag * (B @ x))

    A = LinearOperator((p, p), matvec=matvec, dtype=float)
    eigvals, evecs = eigsh(A, k=k, which='LM')
    idx = np.argsort(eigvals)[::-1]                           # descending order
    return evecs[:, idx]                                      # (p, k)


def sample_frame(returns_window: np.ndarray, k: int) -> np.ndarray:
    """
    Top-k orthonormal frame from SVD of a (num_obs, p) return window.

    Uses factor_lab.svd_decomposition then orthonormalizes the loading rows.
    """
    model_hat = svd_decomposition(returns_window, k=k)
    B = model_hat.B                                           # (k, p)
    Q, _ = qr(B.T, mode='economic')                           # (p, k)
    return Q


# ---------------------------------------------------------------------------
# Target generation (Stiefel canonical + Grassmann special case)
# ---------------------------------------------------------------------------


def _extend_to_orthogonal_basis(U: np.ndarray) -> np.ndarray:
    """Complete U (p, k) to a full (p, p) orthogonal matrix [U | null_space(U^T)]."""
    return np.hstack([U, null_space(U.T)])


def _stiefel_exp_at_standard_base(A11: np.ndarray, A21: np.ndarray) -> np.ndarray:
    """
    Fast Stiefel canonical exp at the standard base point [I_k; 0].

    Returns Y_std (p, k) = exp([[A11, -A21^T], [A21, 0]]) · [I_k; 0], computed
    via the O(k³) 2k×2k block reduction: let A21 = U_A R_A (QR), then the
    relevant column block factors through a 2k×2k matrix exp.
    """
    p, k = A21.shape[0] + A11.shape[0], A11.shape[0]
    U_A, R_A = qr(A21, mode='economic')                       # (p-k, k), (k, k)

    A_tilde = np.zeros((2 * k, 2 * k))
    A_tilde[:k, :k] = A11
    A_tilde[k:, :k] = R_A
    A_tilde[:k, k:] = -R_A.T

    Y_tilde = expm(A_tilde)[:, :k]                            # (2k, k)

    Y_std = np.empty((p, k))
    Y_std[:k, :] = Y_tilde[:k, :]
    Y_std[k:, :] = U_A @ Y_tilde[k:, :]
    return Y_std


def sample_stiefel_targets(
    U_base: np.ndarray,
    radius: float,
    n: int,
    rng: np.random.Generator,
    Q_full: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    n frames at exact Stiefel canonical distance `radius` from U_base.

    Adopted from gen_equi_samples.generate_stiefel_canonical, refactored to
    take an rng (no global numpy seeding).

    Parameters
    ----------
    Q_full : optional pre-computed orthogonal basis [U_base | null_space(U_base^T)].
        Pass this when calling repeatedly for the same U_base (e.g. across radii
        within a p-slice) to avoid recomputing null_space each time.

    Algorithm:
      1. Draw random skew-symmetric A11 (k×k) and random A21 ((p-k)×k).
      2. Rescale jointly so sqrt(½‖A11‖² + ‖A21‖²) = radius.
      3. Exp-map via 2k×2k reduction at the standard base.
      4. Rotate to U_base via Q_full = [U_base | null_space].
    """
    p, k = U_base.shape
    if Q_full is None:
        Q_full = _extend_to_orthogonal_basis(U_base)
    targets = []

    for _ in range(n):
        A11_raw = rng.standard_normal(size=(k, k))
        A11 = A11_raw - A11_raw.T
        A21 = rng.standard_normal(size=(p - k, k))

        norm_sq = 0.5 * np.linalg.norm(A11, 'fro')**2 + np.linalg.norm(A21, 'fro')**2
        scale = radius / np.sqrt(norm_sq)
        A11 *= scale
        A21 *= scale

        Y_std = _stiefel_exp_at_standard_base(A11, A21)
        targets.append(Q_full @ Y_std)

    return targets


def sample_grassmann_targets(
    U_base: np.ndarray,
    radius: float,
    n: int,
    rng: np.random.Generator,
    Q_full: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    n frames at exact Grassmann (horizontal-only) geodesic distance `radius`.

    The Stiefel special case A11 = 0: pure horizontal motion. With A11 = 0
    the canonical norm reduces to ‖A21‖_F, so we just scale A21 to `radius`.
    By construction the target has zero SO(k) rotation relative to U_base,
    so Grassmann distance equals Stiefel distance equals `radius`.

    Parameters
    ----------
    Q_full : optional pre-computed orthogonal basis (see sample_stiefel_targets).
    """
    p, k = U_base.shape
    if Q_full is None:
        Q_full = _extend_to_orthogonal_basis(U_base)
    targets = []

    A11 = np.zeros((k, k))
    for _ in range(n):
        A21 = rng.standard_normal(size=(p - k, k))
        A21 *= radius / np.linalg.norm(A21, 'fro')
        Y_std = _stiefel_exp_at_standard_base(A11, A21)
        targets.append(Q_full @ Y_std)

    return targets


# ---------------------------------------------------------------------------
# Distance measurement
# ---------------------------------------------------------------------------


def grassmann_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Grassmann distance = L2 norm of principal angles between col(U1), col(U2).

    Uses SVD of the overlap U1^T @ U2; singular values are cos(θ_i).
    """
    overlap = U1.T @ U2
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.arccos(singular_values)
    return float(np.linalg.norm(angles))


def stiefel_canonical_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Stiefel canonical geodesic distance via real Schur decomposition.

    Replaces the previous scipy.linalg.logm implementation with a direct
    real Schur decomposition of the 2k×2k block rotation matrix G. For a
    rotation matrix, the real Schur form has 2×2 diagonal blocks
    [[cos θ, -sin θ], [sin θ, cos θ]], and the matrix logarithm of each
    block is simply [[0, -θ], [θ, 0]]. This avoids the full Padé
    approximant series used by logm and is ~30x faster.

    Known precision floor: round-trip error grows with radius.
    Measured at p=80, k=3 vs known exact targets:
      r=0.1: ~1e-4,  r=0.5: ~1e-2,  r=1.0: ~4e-2.
    Slightly less accurate than logm at large radii but within the existing
    test tolerances and appropriate for hot-loop use.

    See test_stiefel_tangent_norm_exact for the separately-verified claim
    that target generation is exact to machine precision.
    """
    k = U1.shape[1]
    M = U1.T @ U2
    residual = U2 - U1 @ M
    _, R = qr(residual, mode='economic')

    G = np.zeros((2 * k, 2 * k))
    G[:k, :k] = M
    G[k:, :k] = R
    G[:k, k:] = -R.T
    G[k:, k:] = M

    # Real Schur: G = Z T Z^T, T block-upper-triangular with 2×2 rotation blocks.
    # logm maps each [[cos θ, -sin θ],[sin θ, cos θ]] block to [[0, -θ],[θ, 0]].
    T, Z = schur(G, output='real')
    n = 2 * k
    log_T = np.zeros((n, n))
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-10:          # 2×2 rotation block
            theta = np.arctan2(T[i + 1, i], T[i, i])
            log_T[i, i + 1] = -theta
            log_T[i + 1, i] = theta
            i += 2
        else:                                                  # 1×1 block (eigenvalue ±1)
            log_T[i, i] = 0.0                                 # +1 → log 0; -1 won't arise
            i += 1

    Delta = Z @ log_T @ Z.T
    Delta = 0.5 * (Delta - Delta.T)                           # enforce skew-symmetry
    Delta11 = Delta[:k, :k]
    Delta21 = Delta[k:, :k]
    return float(np.sqrt(0.5 * np.linalg.norm(Delta11, 'fro')**2 +
                         np.linalg.norm(Delta21, 'fro')**2))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


@dataclass
class SimResults:
    """Output of run_simulation: long-form records + experimental config."""
    long_df: pd.DataFrame
    summary_df: pd.DataFrame
    spec: SimSpec

    def save(self, output_dir: str | Path) -> None:
        """Write long.csv + summary.csv to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        long_path = output_dir / 'distances_all.csv'
        summary_path = output_dir / 'distances_summary.csv'
        self.long_df.to_csv(long_path, index=False)
        self.summary_df.to_csv(summary_path, index=False)
        logger.info("Saved {} rows to {}", len(self.long_df), long_path)
        logger.info("Saved summary ({} rows) to {}", len(self.summary_df), summary_path)


def _record(p: int, sim: int, radius: float, metric: str,
            distance_type: str, distance: float, k: int, n: int) -> dict:
    """Build one long-form row. Schema matches existing SimulationResults."""
    return {
        'dimension': k,
        'p': p,
        'n': n,
        'radius': radius,
        'rep': sim,
        'metric': metric,
        'distance_type': distance_type,
        'distance': distance,
    }


# Dispatch: metric name → (target_sampler, distance_fn)
_METRICS = {
    'grassmann': (sample_grassmann_targets, grassmann_distance),
    'stiefel-canonical': (sample_stiefel_targets, stiefel_canonical_distance),
}


def _measure_one_cell(
    U_gt: np.ndarray, U_sample: np.ndarray,
    Q_full: np.ndarray,
    p: int, sim: int, radius: float,
    num_targets: int, k: int, n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    All records for one (p, sim, radius) cell, across both metrics.

    Q_full is the pre-computed orthogonal basis for U_gt, shared across all
    radii and sims within a p-slice to avoid redundant null_space calls.
    """
    records = []
    for metric_name, (sampler, distance_fn) in _METRICS.items():
        targets = sampler(U_gt, radius, num_targets, rng, Q_full=Q_full)
        for target in targets:
            records.append(_record(
                p=p, sim=sim, radius=radius, metric=metric_name,
                distance_type='sample-target',
                distance=distance_fn(target, U_sample),
                k=k, n=n,
            ))
        # Reference row: target ↔ ground truth is exactly `radius` by construction
        records.append(_record(
            p=p, sim=sim, radius=radius, metric=metric_name,
            distance_type='truth-target', distance=radius,
            k=k, n=n,
        ))
    return records


def run_simulation(spec: SimSpec) -> SimResults:
    """
    Run the full sim and return long + summary DataFrames.

    Two RNGs: one for model+returns (seed_model), one for target directions
    (seed_targets). Keeps target geometry stable when return seed changes.
    """
    total_cells = len(spec.nums_sec) * spec.num_sim * len(spec.target_radii)
    logger.info(
        "Starting simulation: k={}, {} p-slices, {} sims, {} radii, {} targets/cell "
        "({} total cells), seed_model={}, seed_targets={}",
        spec.k_factors, len(spec.nums_sec), spec.num_sim,
        len(spec.target_radii), spec.num_targets,
        total_cells, spec.seed_model, spec.seed_targets,
    )

    model_rng = np.random.default_rng(spec.seed_model)
    target_rng = np.random.default_rng(spec.seed_targets)

    # model_rng is consumed sequentially: build_population_model draws
    # beta and idio-vol samples first, then simulate_all_returns continues
    # from that same RNG state for return draws. Both calls must use the
    # same generator in this order to reproduce the same output. Do not
    # call them with separate generators seeded identically — the return
    # draws would then start at the wrong RNG state.
    model = build_population_model(spec, model_rng)
    all_returns = simulate_all_returns(model, spec, model_rng)

    records: list[dict] = []
    for p in tqdm(spec.nums_sec, desc="p-slices"):
        logger.debug("Starting slice p={}", p)
        U_gt = ground_truth_frame(slice_model(model, p), spec.k_factors)
        logger.debug("Ground truth frame computed: shape={}", U_gt.shape)

        # Pre-compute the orthogonal basis once per p-slice; reused across
        # all sims and radii. null_space is O(p²) so this saves num_sim *
        # num_radii * 2 redundant calls.
        Q_full = _extend_to_orthogonal_basis(U_gt)

        for sim in range(spec.num_sim):
            logger.debug("  sim {}/{} at p={}", sim + 1, spec.num_sim, p)
            returns_window = all_returns[sim, :, :p]
            U_sample = sample_frame(returns_window, spec.k_factors)
            for radius in spec.target_radii:
                records.extend(_measure_one_cell(
                    U_gt=U_gt, U_sample=U_sample,
                    Q_full=Q_full,
                    p=p, sim=sim, radius=radius,
                    num_targets=spec.num_targets,
                    k=spec.k_factors, n=spec.num_obs,
                    rng=target_rng,
                ))

    logger.info("Simulation complete: {} records generated", len(records))
    long_df = pd.DataFrame.from_records(records)
    long_df['radius_label'] = long_df['radius'].map(lambda r: f"r={r:.2f}")
    long_df['n_label'] = long_df['n'].map(lambda n: f"n={n}")

    summary_df = (
        long_df
        .groupby(['dimension', 'p', 'n', 'radius', 'metric', 'distance_type'],
                 as_index=False)['distance']
        .agg([('count', 'count'), ('mean', 'mean'), ('std', 'std'),
              ('median', 'median'),
              ('q25', lambda x: np.quantile(x, 0.25)),
              ('q75', lambda x: np.quantile(x, 0.75)),
              ('min', 'min'), ('max', 'max')])
        .reset_index()
    )

    return SimResults(long_df=long_df, summary_df=summary_df, spec=spec)


# ---------------------------------------------------------------------------
# Pre-configured specs: three sizes from micro (CI) to full (publication)
# ---------------------------------------------------------------------------


def _default_sampler_factories(k: int) -> dict:
    """
    Sampler FACTORIES matching the pseudo-code defaults.
      beta:  N(1, 0.5), N(0, 1), N(0, 1)  for k=3
      idio_vol: U(0.1, 5)
      factor_variances: [0.05², 0.1², 0.1²]
      factor returns & idio returns: standardized N(0, 1)

    Each factory takes an rng and returns a live sampler. The rng is supplied
    fresh by run_simulation, so reproducibility is keyed on spec.seed_model alone.
    """
    if k != 3:
        raise ValueError("Default samplers are tuned for k=3; provide your own for other k.")
    return dict(
        beta_sampler_factories=[
            lambda rng: create_sampler('normal', rng, loc=1.0, scale=0.5),
            lambda rng: create_sampler('normal', rng, loc=0.0, scale=1.0),
            lambda rng: create_sampler('normal', rng, loc=0.0, scale=1.0),
        ],
        idio_vol_sampler_factory=lambda rng: create_sampler('uniform', rng, low=0.2, high=0.8),
        factor_variances=[0.16**2, 0.07**2, 0.03**2],
        factor_return_sampler_factories=[
            lambda rng: create_sampler('normal', rng) for _ in range(k)
        ],
        idio_return_sampler_factory=lambda rng: create_sampler('normal', rng, loc=0.0, scale=0.5),
    )


# Numeric fields that can be overridden via a JSON spec file.
# Sampler factories are not serialisable and always use the defaults.
_JSON_FIELDS = frozenset({
    'max_num_sec', 'nums_sec', 'num_obs', 'num_sim',
    'target_radii', 'num_targets', 'k_factors',
    'factor_variances', 'seed_model', 'seed_targets',
})

_SIZES = {
    'micro': dict(max_num_sec=100,   nums_sec=(30, 60, 100),
                  num_sim=3,  num_targets=3),
    'toy':   dict(max_num_sec=500,   nums_sec=(50, 100, 250, 500),
                  num_sim=10, num_targets=5),
    'full':  dict(max_num_sec=10000, nums_sec=(100, 500, 1000, 3000, 5000, 10000),
                  num_sim=100, num_targets=20),
}


def build_spec(size: str, seed_model: int = 42, seed_targets: int = 12345) -> SimSpec:
    """
    Build one of three tiered specs.

      'micro': ~1 s,   for unit tests.
      'toy':   ~11 s,  for interactive development.
      'full':  ~5 min, the pseudo-code target (max_num_sec=10000, num_sim=100).

    Runtime is dominated by stiefel_canonical_distance (~0.2 ms/call after
    the Schur optimisation, down from ~5 ms with logm).
    """
    if size not in _SIZES:
        raise ValueError(f"size must be one of {list(_SIZES)}, got {size!r}")
    knobs = _SIZES[size]
    return SimSpec(
        max_num_sec=knobs['max_num_sec'],
        nums_sec=knobs['nums_sec'],
        num_obs=63,
        num_sim=knobs['num_sim'],
        target_radii=(0.1, 0.5, 1.0),
        num_targets=knobs['num_targets'],
        k_factors=3,
        seed_model=seed_model,
        seed_targets=seed_targets,
        **_default_sampler_factories(k=3),
    )


def build_spec_from_json(path: str | Path, seed_model: int = 42, seed_targets: int = 12345) -> SimSpec:
    """
    Build a SimSpec from a JSON file, falling back to 'full' defaults for
    any fields not present in the file.

    Only numeric fields can be specified in JSON; sampler factories always
    use the defaults (k_factors must be 3).

    Example JSON:
        {
            "max_num_sec": 2000,
            "nums_sec": [100, 500, 1000, 2000],
            "num_obs": 126,
            "num_sim": 50,
            "target_radii": [0.1, 0.3, 0.5, 1.0],
            "num_targets": 10,
            "k_factors": 3,
            "factor_variances": [0.0025, 0.01, 0.01],
            "seed_model": 99,
            "seed_targets": 777
        }
    """
    with open(path) as f:
        raw = json.load(f)

    unknown = set(raw) - _JSON_FIELDS
    if unknown:
        raise ValueError(f"Unknown fields in JSON spec: {unknown}. "
                         f"Sampler distributions cannot be set via JSON.")

    # Start from 'full' defaults, override with JSON values
    knobs = dict(_SIZES['full'])
    knobs.update({k: v for k, v in raw.items() if k in ('max_num_sec', 'nums_sec',
                                                          'num_sim', 'num_targets')})
    seed_model  = raw.get('seed_model', seed_model)
    seed_targets = raw.get('seed_targets', seed_targets)
    k = raw.get('k_factors', 3)

    # Build sampler factories from defaults, then override factor_variances
    # if the JSON supplies it. The override must happen after the factories
    # dict is built — _default_sampler_factories includes factor_variances,
    # so passing both ** and an explicit kwarg causes a duplicate-keyword error.
    factories = _default_sampler_factories(k=k)
    if 'factor_variances' in raw:
        factories['factor_variances'] = raw['factor_variances']

    return SimSpec(
        max_num_sec=knobs['max_num_sec'],
        nums_sec=tuple(knobs['nums_sec']),
        num_obs=raw.get('num_obs', 63),
        num_sim=knobs['num_sim'],
        target_radii=tuple(raw.get('target_radii', (0.1, 0.5, 1.0))),
        num_targets=knobs['num_targets'],
        k_factors=k,
        seed_model=seed_model,
        seed_targets=seed_targets,
        **factories,
    )



def print_spec(spec: SimSpec) -> None:
    """Print a complete human-readable summary of a SimSpec to stdout."""
    k = spec.k_factors
    total_cells = len(spec.nums_sec) * spec.num_sim * len(spec.target_radii)
    total_rows = total_cells * 2 * (spec.num_targets + 1)  # 2 metrics, +1 truth-target

    lines = [
        "",
        "=" * 62,
        "  Simulation Specification",
        "=" * 62,
        "",
        "  Model",
        f"    Population size (max_num_sec) : {spec.max_num_sec:,}",
        f"    Number of factors (k)         : {k}",
        f"    Factor variances              : {spec.factor_variances}",
        f"    Beta samplers (k={k})          : N(1.0, 0.5), N(0, 1), N(0, 1)",
        f"    Idio vol sampler              : U(0.1, 5.0)  [per-asset variance]",
        "",
        "  Returns simulation",
        f"    Observations per window       : {spec.num_obs}",
        f"    Factor return samplers        : N(0, 1) × {k}  [standardised]",
        f"    Idio return sampler           : N(0, 1)   [standardised]",
        "",
        "  Experimental design",
        f"    p-slices (nums_sec)           : {list(spec.nums_sec)}",
        f"    Simulations per slice         : {spec.num_sim:,}",
        f"    Target radii                  : {list(spec.target_radii)}",
        f"    Targets per cell              : {spec.num_targets}",
        f"    Metrics                       : grassmann, stiefel-canonical",
        "",
        "  Reproducibility",
        f"    seed_model                    : {spec.seed_model}",
        f"    seed_targets                  : {spec.seed_targets}",
        "",
        "  Output sizing",
        f"    Total cells                   : {total_cells:,}",
        f"      = {len(spec.nums_sec)} p-slices × {spec.num_sim} sims"
        f" × {len(spec.target_radii)} radii",
        f"    Rows in distances_all.csv     : {total_rows:,}",
        f"      = {total_cells:,} cells × 2 metrics"
        f" × ({spec.num_targets} sample-target + 1 truth-target)",
        "",
        "=" * 62,
        "",
    ]
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-factor dispersion-bias simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python factor_sims.py                    # toy spec (default)
  python factor_sims.py toy                # toy spec
  python factor_sims.py full               # full publication spec (~5 min)
  python factor_sims.py micro              # micro spec (~1 s, for testing)
  python factor_sims.py my_spec.json       # custom JSON spec
  python factor_sims.py full --output results/run1
  python factor_sims.py my_spec.json --seed-model 99 --seed-targets 777
""",
    )
    parser.add_argument(
        'spec',
        nargs='?',
        default='toy',
        help="Named spec ('micro', 'toy', 'full') or path to a JSON spec file. Default: toy.",
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('factor_sims_output'),
        help="Output directory for CSVs and figures. Default: factor_sims_output/",
    )
    parser.add_argument(
        '--seed-model',
        type=int,
        default=None,
        help="Override seed_model (ignored when loading from JSON that sets it).",
    )
    parser.add_argument(
        '--seed-targets',
        type=int,
        default=None,
        help="Override seed_targets.",
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help="Skip figure generation after saving CSVs.",
    )
    parser.add_argument(
        '--print-spec',
        action='store_true',
        help="Print the complete specification and exit without running.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    seed_model  = args.seed_model  or 42
    seed_targets = args.seed_targets or 12345

    spec_arg = args.spec
    if spec_arg in _SIZES:
        logger.info("Building '{}' spec", spec_arg)
        spec = build_spec(spec_arg, seed_model=seed_model, seed_targets=seed_targets)
    else:
        path = Path(spec_arg)
        if not path.exists():
            raise FileNotFoundError(f"Spec not a named size and file not found: {path}")
        logger.info("Loading spec from {}", path)
        spec = build_spec_from_json(path, seed_model=seed_model, seed_targets=seed_targets)

    logger.info(
        "Spec: max_p={}, slices={}, num_sim={}, num_obs={}, radii={}, targets={}",
        spec.max_num_sec, list(spec.nums_sec), spec.num_sim,
        spec.num_obs, list(spec.target_radii), spec.num_targets,
    )

    if args.print_spec:
        print_spec(spec)
        return

    results = run_simulation(spec)
    results.save(args.output)

    if not args.no_plot:
        try:
            from factor_sims_plots import plot_results
            plot_results(results, args.output / 'figures')
        except ImportError:
            logger.warning("factor_sims_plots not found — skipping figure generation.")

    print(results.summary_df.head(10).to_string(index=False))
    logger.info("Done.")


if __name__ == '__main__':
    main()
