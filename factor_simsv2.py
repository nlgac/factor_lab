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
    results = run_simulation(build_toy_spec())
    results.save('factor_sims_output/')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm, qr, null_space

from factor_lab import (
    FactorModelData,
    FactorModelBuilder,
    FlexibleReturnsSimulator,
    create_sampler,
    svd_decomposition,
)


Sampler = Callable[[int], np.ndarray]
SamplerFactory = Callable[[np.random.Generator], Sampler]
# A SamplerFactory binds distribution parameters but defers RNG choice.
# run_simulation calls each factory with a fresh RNG seeded by spec.seed_model,
# so two back-to-back run_simulation(spec) calls produce identical output.


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class SimSpec:
    """
    All knobs for one simulation run.

    Sampler fields hold FACTORIES, not live samplers. A factory takes an
    np.random.Generator and returns a callable (n) -> ndarray. This lets
    run_simulation rebuild samplers from a fresh RNG on each call, so
    reproducibility is driven entirely by the two seeds in this dataclass.

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
    beta_sampler_factories: list[SamplerFactory]        # length k
    idio_vol_sampler_factory: SamplerFactory            # single
    factor_variances: list[float]                       # length k

    # Returns simulation (standardized samplers, scaled internally by F, D)
    factor_return_sampler_factories: list[SamplerFactory]   # length k
    idio_return_sampler_factory: SamplerFactory             # single

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
    beta_samplers = [factory(rng) for factory in spec.beta_sampler_factories]
    idio_vol_sampler = spec.idio_vol_sampler_factory(rng)
    builder = FactorModelBuilder(rng=rng)
    return builder.build(
        p=spec.max_num_sec,
        k=spec.k_factors,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=spec.factor_variances,
    )


def simulate_all_returns(
    model: FactorModelData,
    spec: SimSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One contiguous return draw, reshaped into (num_sim, num_obs, max_num_sec).

    Single big call amortizes sampler overhead; reshape is zero-copy.
    """
    factor_return_samplers = [f(rng) for f in spec.factor_return_sampler_factories]
    idio_return_sampler = spec.idio_return_sampler_factory(rng)
    simulator = FlexibleReturnsSimulator(rng=rng)
    result = simulator.simulate(
        model=model,
        n_periods=spec.num_obs * spec.num_sim,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler,
    )
    returns = result['security_returns']                    # (T_total, p)
    return returns.reshape(spec.num_sim, spec.num_obs, spec.max_num_sec)


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

    TODO: at full scale (p=10000) forming M_p is 800MB and eigh is ~seconds.
    The top-k subspace of a rank-k-plus-diagonal matrix has a cheaper
    factored form (when D = σ²I, it's exactly span(B_p)). For now, dense.
    """
    M = model_slice.implied_covariance()                    # (p, p)
    p = M.shape[0]
    _, evecs = eigh(M, subset_by_index=[p - k, p - 1])
    return evecs[:, ::-1]                                   # (p, k), top-first


def sample_frame(returns_window: np.ndarray, k: int) -> np.ndarray:
    """
    Top-k orthonormal frame from SVD of a (num_obs, p) return window.

    Uses factor_lab.svd_decomposition then orthonormalizes the loading rows.
    """
    model_hat = svd_decomposition(returns_window, k=k)
    B = model_hat.B                                         # (k, p)
    Q, _ = qr(B.T, mode='economic')                         # (p, k)
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
    U_A, R_A = qr(A21, mode='economic')                     # (p-k, k), (k, k)

    A_tilde = np.zeros((2 * k, 2 * k))
    A_tilde[:k, :k] = A11
    A_tilde[k:, :k] = R_A
    A_tilde[:k, k:] = -R_A.T

    Y_tilde = expm(A_tilde)[:, :k]                          # (2k, k)

    Y_std = np.empty((p, k))
    Y_std[:k, :] = Y_tilde[:k, :]
    Y_std[k:, :] = U_A @ Y_tilde[k:, :]
    return Y_std


def sample_stiefel_targets(
    U_base: np.ndarray,
    radius: float,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    n frames at exact Stiefel canonical distance `radius` from U_base.

    Adopted from gen_equi_samples.generate_stiefel_canonical, refactored to
    take an rng (no global numpy seeding).

    Algorithm:
      1. Draw random skew-symmetric A11 (k×k) and random A21 ((p-k)×k).
      2. Rescale jointly so sqrt(½‖A11‖² + ‖A21‖²) = radius.
      3. Exp-map via 2k×2k reduction at the standard base.
      4. Rotate to U_base via Q_full = [U_base | null_space].
    """
    p, k = U_base.shape
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
) -> list[np.ndarray]:
    """
    n frames at exact Grassmann (horizontal-only) geodesic distance `radius`.

    The Stiefel special case A11 = 0: pure horizontal motion. With A11 = 0
    the canonical norm reduces to ‖A21‖_F, so we just scale A21 to `radius`.
    By construction the target has zero SO(k) rotation relative to U_base,
    so Grassmann distance equals Stiefel distance equals `radius`.
    """
    p, k = U_base.shape
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
    Stiefel canonical geodesic distance via 2k×2k block logarithm.

    Called in the hot loop for every stiefel-canonical sample-target
    distance measurement. scipy.linalg.logm dominates run_simulation
    runtime at ~5 ms per call for k=3 (~60% of toy/full wall time).
    See "Known follow-ups" in the KT doc for a potential logm-free
    replacement via polar decomposition of the overlap.

    Known precision floor: round-trip error grows with radius.
    Measured at p=80, k=3: ~5e-5 at r=0.1, ~4e-3 at r=0.5, ~2.5e-2
    at r=1.0. The forward generator (target generation) is exact to
    machine precision; this function is the lossy verification path.
    See test_stiefel_tangent_norm_exact vs test_stiefel_target_radius_exact.
    """
    from scipy.linalg import logm
    k = U1.shape[1]
    M = U1.T @ U2
    residual = U2 - U1 @ M
    _, R = qr(residual, mode='economic')

    G = np.zeros((2 * k, 2 * k))
    G[:k, :k] = M
    G[k:, :k] = R
    G[:k, k:] = -R.T
    G[k:, k:] = M

    Delta = np.real(logm(G))
    Delta = 0.5 * (Delta - Delta.T)                         # enforce skew-sym
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
        self.long_df.to_csv(output_dir / 'distances_all.csv', index=False)
        self.summary_df.to_csv(output_dir / 'distances_summary.csv', index=False)


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
    p: int, sim: int, radius: float,
    num_targets: int, k: int, n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """All records for one (p, sim, radius) cell, across both metrics."""
    records = []
    for metric_name, (sampler, distance_fn) in _METRICS.items():
        targets = sampler(U_gt, radius, num_targets, rng)
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
    for p in spec.nums_sec:
        U_gt = ground_truth_frame(slice_model(model, p), spec.k_factors)
        for sim in range(spec.num_sim):
            returns_window = all_returns[sim, :, :p]
            U_sample = sample_frame(returns_window, spec.k_factors)
            for radius in spec.target_radii:
                records.extend(_measure_one_cell(
                    U_gt=U_gt, U_sample=U_sample,
                    p=p, sim=sim, radius=radius,
                    num_targets=spec.num_targets,
                    k=spec.k_factors, n=spec.num_obs,
                    rng=target_rng,
                ))

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
        raise ValueError("Default samplers are tuned for k=3; provide your own otherwise.")
    return dict(
        beta_sampler_factories=[
            lambda rng: create_sampler('normal', rng, loc=1.0, scale=0.5),
            lambda rng: create_sampler('normal', rng, loc=0.0, scale=1.0),
            lambda rng: create_sampler('normal', rng, loc=0.0, scale=1.0),
        ],
        idio_vol_sampler_factory=lambda rng: create_sampler('uniform', rng, low=0.1, high=5.0),
        factor_variances=[0.05**2, 0.1**2, 0.1**2],
        factor_return_sampler_factories=[
            lambda rng: create_sampler('normal', rng) for _ in range(k)
        ],
        idio_return_sampler_factory=lambda rng: create_sampler('normal', rng),
    )


def build_spec(size: str, seed_model: int = 42, seed_targets: int = 12345) -> SimSpec:
    """
    Build one of three tiered specs.

      'micro': ~0.1 s, for unit tests.
      'toy':   ~1-2 s, for interactive development.
      'full':  the pseudo-code target (max_num_sec=10000, num_sim=100).
    """
    sizes = {
        'micro': dict(max_num_sec=100,  nums_sec=(30, 60, 100),
                      num_sim=3, num_targets=3),
        'toy':   dict(max_num_sec=500,  nums_sec=(50, 100, 250, 500),
                      num_sim=10, num_targets=5),
        'full':  dict(max_num_sec=10000,
                      nums_sec=(100, 500, 1000, 3000, 5000, 10000),
                      num_sim=100, num_targets=20),
    }
    if size not in sizes:
        raise ValueError(f"size must be one of {list(sizes)}, got {size!r}")
    knobs = sizes[size]

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


def main() -> None:
    """CLI entry point: run the toy spec and save to factor_sims_output/."""
    spec = build_spec('toy')
    print(f"Running toy spec: {spec.max_num_sec=}, {len(spec.nums_sec)} slices, "
          f"{spec.num_sim} sims, {spec.num_targets} targets/cell")
    results = run_simulation(spec)
    out_dir = Path('factor_sims_output')
    results.save(out_dir)
    print(f"Wrote {len(results.long_df)} rows to {out_dir}/")
    print(results.summary_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
