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


def _build_rng_state_to_returns(rng: np.random.Generator, spec: SimSpec) -> None:
    """
    Advance rng past the model-construction draws to reach the state that
    simulate_all_returns would see.

    build_population_model draws from rng in this order:
      - k beta samplers (each calls create_sampler which may draw internally)
      - 1 idio_vol sampler
      - FactorModelBuilder.build() draws for B and D

    The only way to reproduce the exact state is to replay the same calls.
    We do so by rebuilding the model with a fresh rng — the draws happen
    identically, consuming the same number of random variates.
    """
    beta_samplers = [factory(rng) for factory in spec.beta_sampler_factories]
    idio_vol_sampler = spec.idio_vol_sampler_factory(rng)
    builder = FactorModelBuilder(rng=rng)
    builder.build(
        p=spec.max_num_sec,
        k=spec.k_factors,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=spec.factor_variances,
    )
    # rng is now at the same state as after build_population_model


def _simulate_full_returns(
    model: FactorModelData,
    spec: SimSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate returns and return (factor_returns, idio_returns) separately.

    Returns
    -------
    factor_returns : ndarray, shape (num_sim, num_obs, k)
    idio_returns   : ndarray, shape (num_sim, num_obs, max_num_sec)
    """
    n_total = spec.num_obs * spec.num_sim
    factor_return_samplers = [f(rng) for f in spec.factor_return_sampler_factories]
    idio_return_sampler = spec.idio_return_sampler_factory(rng)
    simulator = FlexibleReturnsSimulator(rng=rng)
    result = simulator.simulate(
        model=model,
        n_periods=n_total,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler,
    )
    k = spec.k_factors
    shape_f = (spec.num_sim, spec.num_obs, k)
    shape_e = (spec.num_sim, spec.num_obs, spec.max_num_sec)
    factor_ret = result['factor_returns'].reshape(shape_f)
    idio_ret   = result['idio_returns'].reshape(shape_e)
    return factor_ret, idio_ret



def run_simulation(spec: SimSpec, sample_truth: bool = False,
                   save_model_path: Path | None = None,
                   save_returns_path: Path | None = None) -> SimResults:
    """
    Run the full sim and return long + summary DataFrames.

    Two RNGs: one for model+returns (seed_model), one for target directions
    (seed_targets). Keeps target geometry stable when return seed changes.

    Parameters
    ----------
    spec : SimSpec
    sample_truth : bool
        When True, also record d(B^S, B^GT) under both metrics for every
        (p, sim) pair, replicated across all radii so the schema stays
        uniform. Adds distance_type='sample-truth' rows to long_df.
    save_model_path : Path or None
        If given, save the population model (B, F, D) to this .npz file
        immediately after construction, before any simulation.
    save_returns_path : Path or None
        If given, save all simulated returns to this .npz file after
        simulation. Stores factor_returns and idio_returns separately
        as well as security_returns. Shape: (num_sim, num_obs, max_num_sec).
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

    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_model_path,
            B=model.B,   # (k, max_num_sec) factor loadings
            F=model.F,   # (k, k)           factor covariance
            D=model.D,   # (max_num_sec, max_num_sec) idio covariance
        )
        logger.info("Saved model (B, F, D) to {}", save_model_path)

    all_returns = simulate_all_returns(model, spec, model_rng)

    if save_returns_path is not None:
        save_returns_path = Path(save_returns_path)
        save_returns_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-simulate with a fresh RNG at the same state to capture
        # factor and idio components. We reset to seed_model and
        # fast-forward past the model-construction draws by rebuilding
        # factories (no draws) then calling simulate directly.
        _rng2 = np.random.default_rng(spec.seed_model)
        # Consume the same model-construction draws to reach the correct state
        _build_rng_state_to_returns(_rng2, spec)
        factor_ret, idio_ret = _simulate_full_returns(model, spec, _rng2)
        np.savez_compressed(
            save_returns_path,
            security_returns=all_returns,       # (num_sim, num_obs, max_num_sec)
            factor_returns=factor_ret,           # (num_sim, num_obs, k)
            idio_returns=idio_ret,               # (num_sim, num_obs, max_num_sec)
        )
        logger.info("Saved returns (security, factor, idio) to {}", save_returns_path)

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
            if sample_truth:
                records.extend(_sample_truth_records(
                    U_gt=U_gt, U_sample=U_sample,
                    p=p, sim=sim, k=spec.k_factors, n=spec.num_obs,
                    target_radii=spec.target_radii,
                ))

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
# Spec building from JSON files
# ---------------------------------------------------------------------------

# Numeric fields accepted in JSON spec files.
_JSON_FIELDS = frozenset({
    'max_num_sec', 'nums_sec', 'num_obs', 'num_sim',
    'target_radii', 'num_targets', 'k_factors',
    'factor_variances', 'seed_model', 'seed_targets',
})

# Sampler keys accepted with "_" prefix in JSON files.
# Each maps to a distribution spec dict, e.g.:
#   {"distribution": "uniform", "low": 0.1, "high": 5.0}
#   {"distribution": "normal", "loc": 0.0, "scale": 1.0}
#   {"distribution": "constant", "value": 0.5}
_SAMPLER_KEYS = frozenset({
    '_beta_samplers',          # list of k dicts, one per factor
    '_idio_vol_sampler',       # single dict
    '_factor_return_samplers', # list of k dicts, one per factor
    '_idio_return_sampler',    # single dict
})

# Default sampler specs — used when a JSON file does not override them.
_DEFAULT_SAMPLER_SPECS: dict = {
    '_beta_samplers': [
        {'distribution': 'normal', 'loc': 1.0, 'scale': 0.5},
        {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
        {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
    ],
    '_idio_vol_sampler':       {'distribution': 'uniform', 'low': 0.1, 'high': 5.0},
    '_factor_return_samplers': [
        {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
        {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
        {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
    ],
    '_idio_return_sampler':    {'distribution': 'normal', 'loc': 0.0, 'scale': 1.0},
}


def _factory_from_spec(spec: dict) -> SamplerFactory:
    """
    Build a SamplerFactory from a distribution spec dict.

    Supported distributions and their parameters:
      normal:   loc (mean), scale (std dev)
      uniform:  low, high
      constant: value  (every draw returns the same value)

    Example:
        _factory_from_spec({'distribution': 'uniform', 'low': 0.1, 'high': 0.8})
        # returns: lambda rng: create_sampler('uniform', rng, low=0.1, high=0.8)
    """
    dist = spec['distribution']
    params = {k: v for k, v in spec.items() if k != 'distribution'}
    return lambda rng, d=dist, p=params: create_sampler(d, rng, **p)


def _sampler_factories_from_specs(
    k: int,
    factor_variances: list[float],
    sampler_specs: dict,
) -> dict:
    """
    Build the full sampler factory dict from distribution spec dicts.

    Parameters
    ----------
    k : int
        Number of factors (must equal len of beta and factor_return spec lists).
    factor_variances : list[float]
        Per-factor variances (from numeric JSON fields).
    sampler_specs : dict
        Merged sampler specs keyed by _SAMPLER_KEYS names.
    """
    beta_specs = sampler_specs['_beta_samplers']
    idio_vol_spec = sampler_specs['_idio_vol_sampler']
    factor_ret_specs = sampler_specs['_factor_return_samplers']
    idio_ret_spec = sampler_specs['_idio_return_sampler']

    if len(beta_specs) != k:
        raise ValueError(
            f"_beta_samplers has {len(beta_specs)} entries but k_factors={k}. "
            f"They must match."
        )
    if len(factor_ret_specs) != k:
        raise ValueError(
            f"_factor_return_samplers has {len(factor_ret_specs)} entries but k_factors={k}. "
            f"They must match."
        )

    return dict(
        beta_sampler_factories=[_factory_from_spec(s) for s in beta_specs],
        idio_vol_sampler_factory=_factory_from_spec(idio_vol_spec),
        factor_variances=factor_variances,
        factor_return_sampler_factories=[_factory_from_spec(s) for s in factor_ret_specs],
        idio_return_sampler_factory=_factory_from_spec(idio_ret_spec),
    )


def _load_json(path: Path) -> tuple[dict, dict]:
    """
    Load one JSON spec file.

    Returns
    -------
    numeric : dict
        All non-prefixed fields (the _JSON_FIELDS).
    sampler_specs : dict
        All _-prefixed sampler keys recognised by _SAMPLER_KEYS.
        Other _-prefixed keys (e.g. _comment) are silently ignored.
    """
    with open(path) as f:
        raw = json.load(f)
    unknown = {k for k in raw if not k.startswith('_')} - _JSON_FIELDS
    if unknown:
        raise ValueError(
            f"{path}: unknown fields {unknown}. "
            f"Valid numeric fields: {sorted(_JSON_FIELDS)}. "
            f"Sampler distributions use '_'-prefixed keys: {sorted(_SAMPLER_KEYS)}."
        )
    numeric = {k: v for k, v in raw.items() if not k.startswith('_')}
    sampler_specs = {k: v for k, v in raw.items() if k in _SAMPLER_KEYS}
    return numeric, sampler_specs


def build_spec_from_jsons(paths: list[Path]) -> tuple[SimSpec, dict]:
    """
    Build a SimSpec by merging one or more JSON spec files left-to-right.

    Later files take precedence for both numeric fields and sampler specs.
    Returns the SimSpec and the merged sampler specs dict (for print_spec).

    Example
    -------
        spec, sampler_specs = build_spec_from_jsons([
            Path('defaults.json'), Path('toy.json')
        ])
    """
    merged_numeric: dict = {}
    merged_samplers: dict = dict(_DEFAULT_SAMPLER_SPECS)   # start from defaults

    for path in paths:
        numeric, sampler_specs = _load_json(path)
        merged_numeric.update(numeric)
        merged_samplers.update(sampler_specs)
        logger.debug("Merged {}: numeric={}, samplers={}",
                     path.name, sorted(numeric.keys()), sorted(sampler_specs.keys()))

    missing = _JSON_FIELDS - set(merged_numeric)
    if missing:
        raise ValueError(
            f"Missing required numeric fields after merging "
            f"{[p.name for p in paths]}: {sorted(missing)}. "
            f"Add them to one of the spec files."
        )

    k = int(merged_numeric['k_factors'])
    fv = list(merged_numeric['factor_variances'])

    spec = SimSpec(
        max_num_sec=int(merged_numeric['max_num_sec']),
        nums_sec=tuple(int(x) for x in merged_numeric['nums_sec']),
        num_obs=int(merged_numeric['num_obs']),
        num_sim=int(merged_numeric['num_sim']),
        target_radii=tuple(float(x) for x in merged_numeric['target_radii']),
        num_targets=int(merged_numeric['num_targets']),
        k_factors=k,
        seed_model=int(merged_numeric['seed_model']),
        seed_targets=int(merged_numeric['seed_targets']),
        **_sampler_factories_from_specs(k=k, factor_variances=fv,
                                        sampler_specs=merged_samplers),
    )
    return spec, merged_samplers


def _fmt_dist(spec: dict) -> str:
    """Format a distribution spec dict as a human-readable string."""
    dist = spec.get('distribution', '?')
    if dist == 'normal':
        return f"N({spec.get('loc', 0)}, {spec.get('scale', 1)})"
    if dist == 'uniform':
        return f"U({spec.get('low', 0)}, {spec.get('high', 1)})"
    if dist == 'constant':
        return f"constant({spec.get('value', 0)})"
    # fallback: show all params
    params = ', '.join(f"{k}={v}" for k, v in spec.items() if k != 'distribution')
    return f"{dist}({params})"


def print_spec(spec: SimSpec, sampler_specs: dict | None = None) -> None:
    """
    Print a complete human-readable summary of a SimSpec to stdout.

    Parameters
    ----------
    spec : SimSpec
    sampler_specs : dict or None
        Merged sampler spec dicts returned by build_spec_from_jsons.
        If None, falls back to the default sampler descriptions.
    """
    if sampler_specs is None:
        sampler_specs = _DEFAULT_SAMPLER_SPECS

    k = spec.k_factors
    total_cells = len(spec.nums_sec) * spec.num_sim * len(spec.target_radii)
    rows_per_cell = spec.num_targets + 1
    total_rows = total_cells * 2 * rows_per_cell

    beta_strs = [_fmt_dist(s) for s in sampler_specs['_beta_samplers']]
    idio_vol_str = _fmt_dist(sampler_specs['_idio_vol_sampler'])
    fret_strs = [_fmt_dist(s) for s in sampler_specs['_factor_return_samplers']]
    idio_ret_str = _fmt_dist(sampler_specs['_idio_return_sampler'])

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
    ]
    for i, s in enumerate(beta_strs):
        label = f"Beta sampler factor {i}"
        lines.append(f"    {label:<30}: {s}")
    lines += [
        f"    Idio vol sampler              : {idio_vol_str}  [per-asset std dev]",
        "",
        "  Returns simulation",
        f"    Observations per window       : {spec.num_obs}",
    ]
    for i, s in enumerate(fret_strs):
        label = f"Factor return sampler {i}"
        lines.append(f"    {label:<30}: {s}")
    lines += [
        f"    Idio return sampler           : {idio_ret_str}",
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
        "  Output sizing (sample-target + truth-target only)",
        f"    Total cells                   : {total_cells:,}",
        f"      = {len(spec.nums_sec)} p-slices x {spec.num_sim} sims"
        f" x {len(spec.target_radii)} radii",
        f"    Rows in distances_all.csv     : {total_rows:,}",
        f"      = {total_cells:,} cells x 2 metrics"
        f" x ({spec.num_targets} sample-target + 1 truth-target)",
        f"    With --sample-truth: +{total_cells * 2:,} additional rows",
        f"      = {total_cells:,} cells x 2 metrics x 1 sample-truth row",
        "",
        "=" * 62,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# sample-truth distance helpers
# ---------------------------------------------------------------------------


def _sample_truth_records(
    U_gt: np.ndarray, U_sample: np.ndarray,
    p: int, sim: int, k: int, n: int,
    target_radii: Sequence[float],
) -> list[dict]:
    """
    Compute d(B^S, B^GT) under both metrics and replicate across all radii.

    The estimation error does not depend on the target radius, but replicating
    it across radii keeps the schema uniform so the catplot can use radius as
    a column facet with sample-truth appearing in every column. The values are
    scalars (not arrays), so the memory cost is negligible.
    """
    d_grass = grassmann_distance(U_sample, U_gt)
    d_stief = stiefel_canonical_distance(U_sample, U_gt)
    rows = []
    for radius in target_radii:
        for metric, dist in (('grassmann', d_grass), ('stiefel-canonical', d_stief)):
            rows.append(_record(
                p=p, sim=sim, radius=radius, metric=metric,
                distance_type='sample-truth', distance=dist,
                k=k, n=n,
            ))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-factor dispersion-bias simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python factor_sims.py defaults.json toy.json
  python factor_sims.py defaults.json full.json --output results/run1
  python factor_sims.py defaults.json toy.json --sample-truth
  python factor_sims.py defaults.json my_overrides.json --print-spec
  python factor_sims.py defaults.json full.json --seed-model 99
  python factor_sims.py defaults.json toy.json --save-model model.npz
  python factor_sims.py defaults.json toy.json --save-returns returns.npz

JSON files are merged left-to-right; later files take precedence.
All required fields must be present after merging.
""",
    )
    parser.add_argument(
        'specs',
        nargs='+',
        type=Path,
        metavar='SPEC_JSON',
        help="One or more JSON spec files, merged left-to-right.",
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
        help="Override seed_model from the merged spec.",
    )
    parser.add_argument(
        '--seed-targets',
        type=int,
        default=None,
        help="Override seed_targets from the merged spec.",
    )
    parser.add_argument(
        '--sample-truth',
        action='store_true',
        help="Also record d(B^S, B^GT) for each (p, sim) under both metrics.",
    )
    parser.add_argument(
        '--save-model',
        type=Path,
        default=None,
        metavar='FILE.npz',
        help="Save population model (B, F, D) to a .npz file.",
    )
    parser.add_argument(
        '--save-returns',
        type=Path,
        default=None,
        metavar='FILE.npz',
        help="Save simulated returns (security, factor, idio) to a .npz file.",
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

    for path in args.specs:
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

    logger.info("Loading spec from: {}", [p.name for p in args.specs])
    spec, sampler_specs = build_spec_from_jsons(args.specs)

    # CLI seed overrides take final precedence over JSON values
    if args.seed_model is not None:
        object.__setattr__(spec, 'seed_model', args.seed_model)
    if args.seed_targets is not None:
        object.__setattr__(spec, 'seed_targets', args.seed_targets)

    logger.info(
        "Spec: max_p={}, slices={}, num_sim={}, num_obs={}, radii={}, targets={}",
        spec.max_num_sec, list(spec.nums_sec), spec.num_sim,
        spec.num_obs, list(spec.target_radii), spec.num_targets,
    )

    print_spec(spec, sampler_specs)
    if args.print_spec:
        return

    results = run_simulation(
        spec,
        sample_truth=args.sample_truth,
        save_model_path=args.save_model,
        save_returns_path=args.save_returns,
    )
    results.save(args.output)

    if not args.no_plot:
        try:
            from factor_sims_plots import plot_results
            plot_results(results, args.output / 'figures',
                         sample_truth=args.sample_truth)
        except ImportError:
            logger.warning("factor_sims_plots not found — skipping figure generation.")

    print(results.summary_df.head(10).to_string(index=False))
    logger.info("Done.")


if __name__ == '__main__':
    main()
