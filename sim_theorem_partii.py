"""
sim_theorem_partii.py
=====================
Numerical verification of the Theorem (Multifactor Dispersion Bias),
Part (ii), Equation (6) from:

    "Multifactor Dispersion Bias with Per-Column Prevalence: A Unified Treatment"
    §4, diagonal-Gram case (G∞ = I_k).

The claim: for k=3 factors with G∞ = I_k (diagonal Gram), conditional
on F and almost surely as p → ∞,

    sin²∠(hⱼ, b̄ⱼ)  →  δ²/(nρⱼ+δ²)  +  nρⱼ/(nρⱼ+δ²) · sin²∠(ŵⱼ, eⱼ)
                         ──────────────    ──────────────────────────────
                              floor               rotation

where ρⱼ and ŵⱼ are the j-th eigenvalue/eigenvector of
D̂ = C^{1/2}(F^T F/n)C^{1/2}, and eⱼ is the j-th standard basis vector.

Setup
-----
- Loadings: B[j,:] has i.i.d. N(0, cⱼ) entries, independent across j.
  Prevalence ‖B[j,:]‖²/p → cⱼ (Assumption 1); unit-loading Gram G∞ = I_k.
- Factor returns: F is n×k, columns drawn i.i.d. N(0, Σ_F), Σ_F = diag(σⱼ²).
- Noise: idiosyncratic entries are i.i.d. N(0, δ²).
- Population loading directions b̄ⱼ: top-k eigenvectors of Σ₀ = BΣ_F B^T/p.
- The model is drawn once per (n, p) cell and held fixed; F and Z are
  redrawn each replication (simulating the conditional-on-F regime).

Outputs
-------
By default outputs go to ``results/MM-DD_run_NN/`` (NN sequential per date):
- sim_thmptii.parquet               — raw per-rep records
- fig_theorem1_convergence_v2.png   — gap sin²∠−RHS vs p, for each n and factor
- fig_theorem1_scatter_v2.png       — sin²∠ vs RHS scatter at p=p_values[-2]
- fig_theorem1_components_v2.png    — floor and rotation convergence separately

Override the directory with ``--out PATH`` (CLI) or ``"output_path": "..."``
(JSON spec).

Usage
-----
    python sim_theorem_partii.py                            # built-in defaults
    python sim_theorem_partii.py sim_thmptii_spec.json
    python sim_theorem_partii.py sim_thmptii_spec.json --plot
    python sim_theorem_partii.py sim_thmptii_spec.json --plot-save --out results.parquet
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from factor_lab.model_builder import FactorModelBuilder
from factor_lab.flexible_simulator import ReturnsSimulator
from factor_lab.distributions import create_sampler
from factor_lab.analysis import SimulationContext
from factor_lab.analyses.spectral import compute_true_eigenvalues
from factor_lab.analyses import compute_sine_alignment, register_manifold_distance

# ── Experiment specification ──────────────────────────────────────────────────


@dataclass
class SimSpec:
    """Specification for the Eq. (6) Part-(ii) simulation.

    Loaded from a JSON file. Keys starting with ``_`` are treated as comments
    and ignored. Sampler fields use the same ``{"distribution": name, ...}``
    shape consumed by :func:`factor_lab.distributions.create_sampler`.

    The defaults below reproduce the original hardcoded experiment:
    Assumption 3 requires c₁σ₁² > c₂σ₂² > c₃σ₃². With prevalences inherited
    from the loading scales (cⱼ = scaleⱼ²) and σ² = [.04, .02, .01], the
    effective spikes are dⱼ = cⱼσⱼ² = [.040, .016, .006] ✓.
    """

    k_factors: int = 3
    n_values: list[int] = field(default_factory=lambda: [30, 60, 120])
    p_values: list[int] = field(
        default_factory=lambda: [200, 500, 1000, 2000, 5000, 10_000]
    )
    n_reps: int = 300
    random_seed: int = 20260511

    # Factor return variances σⱼ², length k.
    factor_variances: list[float] = field(
        default_factory=lambda: [0.04, 0.02, 0.01]
    )

    # Per-factor loading samplers (broadcast scalar OK). For the diagonal-Gram
    # case use independent zero-mean draws with scaleⱼ = √cⱼ so ‖βⱼ‖²/p → cⱼ
    # and the off-diagonal Gram entries vanish.
    beta_samplers: Union[list[dict], dict] = field(
        default_factory=lambda: [
            {"distribution": "normal", "loc": 0.0, "scale": 1.0},
            {"distribution": "normal", "loc": 0.0, "scale": float(np.sqrt(0.8))},
            {"distribution": "normal", "loc": 0.0, "scale": float(np.sqrt(0.6))},
        ]
    )
    # Idiosyncratic vol sampler; "constant" with value √δ² reproduces the
    # uniform-δ noise model in the theorem statement.
    idio_vol_sampler: dict = field(
        default_factory=lambda: {"distribution": "constant", "value": 1.0}
    )
    # Per-rep factor return sampler (broadcast scalar OK).
    factor_return_sampler: Union[list[dict], dict] = field(
        default_factory=lambda: {"distribution": "normal", "loc": 0.0, "scale": 1.0}
    )
    # Per-rep idiosyncratic return sampler.
    idio_return_sampler: dict = field(
        default_factory=lambda: {"distribution": "normal", "loc": 0.0, "scale": 1.0}
    )

    # Optional overrides for CLI behavior. CLI flags take precedence when set.
    output_path: Optional[str] = None
    plot_mode: Optional[str] = None   # None | "plot" | "plot-save"

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "SimSpec":
        with open(filepath) as f:
            config = json.load(f)
        # Drop "_..."  commentary keys.
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        return cls(**config)


def _make_one_sampler(spec: dict, rng: np.random.Generator):
    """Materialize a single sampler from a {"distribution": name, ...} dict."""
    params = {k: v for k, v in spec.items() if k != "distribution"}
    return create_sampler(spec["distribution"], rng, **params)


def _make_samplers(
    spec: Union[list[dict], dict], rng: np.random.Generator, k: int
):
    """List-or-broadcast sampler resolution matching FactorModelBuilder.build."""
    if isinstance(spec, list):
        if len(spec) != k:
            raise ValueError(
                f"Expected {k} per-factor samplers, got {len(spec)}: {spec!r}"
            )
        return [_make_one_sampler(s, rng) for s in spec]
    return _make_one_sampler(spec, rng)


def _next_run_dir(base: Path) -> Path:
    """Allocate and return ``{base}/results/MM-DD_run_NN`` with NN sequential per date.

    Scans existing siblings matching the date prefix, picks ``max(NN)+1``, and
    creates the directory. NN is zero-padded to 2 digits (01, 02, …, 99, 100).

    Example:
        # On 2026-05-19, with results/05-19_run_01 and 05-19_run_02 present,
        _next_run_dir(Path('.'))  # → Path('results/05-19_run_03'), created.
    """
    today = datetime.now().strftime("%m-%d")
    results_root = base / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(today)}_run_(\d+)$")
    used = [
        int(m.group(1))
        for d in results_root.iterdir() if d.is_dir()
        for m in [pat.match(d.name)] if m
    ]
    next_num = max(used, default=0) + 1
    run_dir = results_root / f"{today}_run_{next_num:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

# ── SimulationAnalysis implementations ───────────────────────────────────────


class SineAlignmentAnalysis:
    """
    Observed LHS of Equation (6): sin²∠(hⱼ, b̄ⱼ) for each factor j.

    hⱼ:  j-th top left singular vector of Y (estimated loading direction,
          computed via the n×n Gram trick — uncentered, matching the theorem).
    b̄ⱼ: j-th population loading direction (unit eigenvector of Σ₀ = BΣ_F B^T/p),
          injected at construction so ARPACK runs once per (n, p) cell.

    Example:
        _, b_pop = compute_true_eigenvalues(model, K)
        analysis = SineAlignmentAnalysis(b_pop)
        result = analysis.analyze(context)
        # {"sin2_j": array shape (K,), "dist_sine": float}
    """

    def __init__(self, b_pop: np.ndarray):
        self.b_pop = b_pop   # (k, p), rows are population loading directions b̄ⱼ

    def analyze(self, context: SimulationContext) -> dict:
        k = context.k
        Y = context.security_returns.T   # (p, n)
        # Top-k left SVs of Y via the n×n Gram Y^T Y (uncentered).
        # Cost O(p·n²) vs O(p²·n) for the full SVD.
        G = Y.T @ Y
        vals, vecs = np.linalg.eigh(G)
        idx = np.argsort(vals)[::-1][:k]
        s = np.sqrt(np.maximum(vals[idx], 0.0))
        H = (Y @ vecs[:, idx]) / np.where(s > 1e-14, s, 1.0)   # (p, k)
        # H.T has shape (k, p); b_pop has shape (k, p)
        sin2, dist = compute_sine_alignment(self.b_pop, H.T)
        return {"sin2_j": sin2, "dist_sine": dist}


class Eq6RHSAnalysis:
    """
    Predicted RHS of Equation (6), Part (ii): floor + weight × rotation for each j.

    Uses factor returns F from the context and empirical prevalences cⱼ = ‖B[j,:]‖²/p
    from the model loadings. Computes D̂ = C^{1/2}(F^T F/n)C^{1/2}.

    δ² is taken from ``context.model.D`` as the mean of its diagonal (D already
    holds variances — the idio_vol_sampler outputs vols which FactorModelBuilder
    squares internally).

    Example:
        analysis = Eq6RHSAnalysis()
        result = analysis.analyze(context)
        # keys: "rhs", "floor", "rotation", "rhos", "delta2"
    """

    def analyze(self, context: SimulationContext) -> dict:
        k, n = context.k, context.T
        F = context.factor_returns.T                     # (k, n)
        c_half = np.sqrt((context.model.B ** 2).mean(axis=1))   # √cⱼ
        D_hat = (c_half[:, None] * (F @ F.T / n)) * c_half[None, :]
        vals, vecs = np.linalg.eigh(D_hat)
        idx = np.argsort(vals)[::-1]
        rhos = vals[idx]
        W = vecs[:, idx]
        delta2 = float(np.diag(context.model.D).mean())
        floor = delta2 / (n * rhos + delta2)
        weight = n * rhos / (n * rhos + delta2)
        # sin²∠(ŵⱼ, eⱼ) = 1 − (ŵⱼ)ⱼ²; squaring the diagonal removes sign ambiguity.
        rotation = 1.0 - np.diag(W) ** 2
        return {"rhs": floor + weight * rotation, "floor": floor,
                "rotation": rotation, "rhos": rhos, "delta2": delta2}


# ── Model construction ────────────────────────────────────────────────────────


def build_model(spec: SimSpec, p: int, rng: np.random.Generator):
    """Build a k-factor model from the spec for the given p.

    Loading samplers, idio-vol sampler, and factor variances all come from
    ``spec``. For the diagonal-Gram case (G∞ = I_k), loadings should be
    independent zero-mean draws with scaleⱼ = √cⱼ so ‖βⱼ‖²/p → cⱼ and
    off-diagonal unit-loading Gram entries vanish.

    Example:
        model = build_model(SimSpec(), p=1000, rng=np.random.default_rng(0))
        # model.B.shape == (3, 1000), model.F == diag(spec.factor_variances)
    """
    return FactorModelBuilder(rng=rng).build(
        p=p,
        k=spec.k_factors,
        beta_samplers=_make_samplers(spec.beta_samplers, rng, spec.k_factors),
        idio_vol_sampler=_make_one_sampler(spec.idio_vol_sampler, rng),
        factor_variances=list(spec.factor_variances),
    )


# ── Simulation helpers ────────────────────────────────────────────────────────


def _rep_records(
    k: int, n: int, p: int, lhs_res: dict, rhs_res: dict
) -> list[dict]:
    """Flatten one replication's analysis results into k per-factor records."""
    gap = lhs_res["sin2_j"] - rhs_res["rhs"]
    return [
        {"n": n, "p": p, "j": j + 1,
         "sin2_j":   float(lhs_res["sin2_j"][j]),
         "rhs":      float(rhs_res["rhs"][j]),
         "gap":      float(gap[j]),
         "floor":    float(rhs_res["floor"][j]),
         "rotation": float(rhs_res["rotation"][j]),
         "rho":      float(rhs_res["rhos"][j])}
        for j in range(k)
    ]


# ── Main simulation ───────────────────────────────────────────────────────────


def simulate(spec: SimSpec) -> pd.DataFrame:
    # Register sine distance once per process so ManifoldDistanceAnalysis
    # picks it up when used alongside this simulation.  The registration is
    # guarded to be idempotent; the centering difference vs. uncentered Y
    # used here means this value is diagnostic only.
    from factor_lab.analyses.manifold import _EXTRA_DISTANCES
    if 'dist_sine' not in _EXTRA_DISTANCES:
        register_manifold_distance(
            'dist_sine',
            lambda bt, be: compute_sine_alignment(bt, be)[1],
        )

    rng_master = np.random.default_rng(spec.random_seed)
    simulator = ReturnsSimulator()   # stateless; all draws go through samplers
    rhs_analysis = Eq6RHSAnalysis()
    records: list[dict] = []

    for n in spec.n_values:
        logger.info("Starting n = {}", n)
        for p in tqdm(spec.p_values, desc=f"n={n}", unit="p"):

            # Build model once per (n, p) cell — fresh β each time.
            model = build_model(spec, p, rng_master)

            # Population directions computed once here; ARPACK skips the rep loop.
            _, b_pop = compute_true_eigenvalues(model, spec.k_factors)
            lhs_analysis = SineAlignmentAnalysis(b_pop)

            logger.debug("n={}, p={}: c={}",
                         n, p, list((model.B ** 2).mean(axis=1)))

            # Independent seed per rep; master rng advances only here.
            rep_seeds = rng_master.integers(0, 2 ** 31, size=spec.n_reps)
            for r in range(spec.n_reps):
                rep_rng = np.random.default_rng(int(rep_seeds[r]))
                factor_samplers = _make_samplers(
                    spec.factor_return_sampler, rep_rng, spec.k_factors
                )
                idio_sampler = _make_one_sampler(spec.idio_return_sampler, rep_rng)
                sim_out = simulator.simulate(
                    model=model, n_periods=n,
                    factor_return_samplers=factor_samplers,
                    idio_return_sampler=idio_sampler,
                )
                context = SimulationContext(
                    model=model,
                    security_returns=sim_out["security_returns"],
                    factor_returns=sim_out["factor_returns"],
                    idio_returns=sim_out["idio_returns"],
                )
                records.extend(
                    _rep_records(spec.k_factors, n, p,
                                 lhs_analysis.analyze(context),
                                 rhs_analysis.analyze(context))
                )

    return pd.DataFrame(records)


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact RMSE table: RMSE of (sin²∠ − RHS) by (n, p, j)."""
    tbl = (
        df.groupby(["n", "p", "j"])["gap"]
        .apply(lambda g: np.sqrt((g ** 2).mean()))
        .rename("RMSE")
        .reset_index()
        .pivot(index=["n", "p"], columns="j", values="RMSE")
    )
    tbl.columns = [f"j={c}" for c in tbl.columns]
    print("\nRMSE of (sin²∠ − RHS)  [smaller is better; should → 0 as p grows]\n")
    print(tbl.to_string(float_format="{:.5f}".format))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Multifactor Dispersion Bias simulation (Theorem, Part ii, Eq. (6)).",
    )
    parser.add_argument(
        "config_file", type=str, nargs="?", default=None,
        help="JSON spec file with experiment parameters (see SimSpec). "
             "If omitted, uses the built-in defaults (the original experiment).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path for the .parquet file. Overrides spec.output_path; "
             "defaults to sim_thmptii.parquet next to this script.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plot", action="store_true",
        help="Generate figures from in-memory results; skip saving the parquet.",
    )
    mode.add_argument(
        "--plot-save", action="store_true",
        help="Save the parquet and generate figures.",
    )
    args = parser.parse_args()

    if args.config_file is None:
        logger.info("No config file given; using built-in SimSpec defaults.")
        spec = SimSpec()
    else:
        spec = SimSpec.from_json(args.config_file)

    # Resolve output path: CLI --out > spec.output_path > auto-allocated run dir.
    # Default places results in {ROOT}/results/MM-DD_run_NN/sim_thmptii.parquet
    # with NN sequential within each date.
    if args.out is not None:
        parquet_path = args.out.with_suffix(".parquet")
    elif spec.output_path is not None:
        parquet_path = Path(spec.output_path).with_suffix(".parquet")
    else:
        run_dir = _next_run_dir(ROOT)
        parquet_path = run_dir / "sim_thmptii.parquet"
        logger.info("Auto-allocated run directory: {}", run_dir)

    # Resolve plot mode: CLI flags > spec.plot_mode > none.
    if args.plot:
        plot_mode = "plot"
    elif args.plot_save:
        plot_mode = "plot-save"
    else:
        plot_mode = spec.plot_mode   # may be None

    logger.info(
        "Simulation: k={}, n={}, p={}, reps={}, seed={}",
        spec.k_factors, spec.n_values, spec.p_values, spec.n_reps, spec.random_seed,
    )
    logger.info(
        "σ²={}, idio_vol_sampler={}",
        list(spec.factor_variances), spec.idio_vol_sampler,
    )

    df = simulate(spec)

    if plot_mode != "plot":
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved {} rows to {}", len(df), parquet_path)

    if plot_mode in ("plot", "plot-save"):
        from fl_graphics import plot_all
        plot_all(df, out_dir=parquet_path.parent)

    print_summary(df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
