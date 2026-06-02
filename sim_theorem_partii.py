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

Architecture
------------
This script is the dispersion-bias-specific *theorem checker*. The generic,
dispersion-agnostic mechanics (sampler resolution, return generation, analysis
dispatch, run-dir allocation) live in ``fl_orchestration`` and are imported
here. The four pipeline stages are decoupled:

    build_model        — Stage 1: construct one (B, F, D) model for a (n, p) cell
    simulate_returns   — Stages 2–4: sample one replication's returns  (imported)
    run_analyses       — dispatch a list of analyses over a context     (imported)
    run_cell           — drive one (n, p) cell: model → reps → records
    simulate           — orchestrate cells into a tidy DataFrame

``run_cell`` is the sole owner of the master-RNG draw order, which the
verification depends on: per cell, ``build_model`` draws first, then the per-rep
seeds are drawn, then each rep uses an independent child generator. Reordering
that stream changes every downstream number.

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

Notebook idiom
--------------
    from sim_theorem_partii import SimSpec, simulate, SineAlignmentAnalysis, Eq6RHSAnalysis
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.model_builder import FactorModelBuilder
from factor_lab.analysis import SimulationContext
from factor_lab.analyses.spectral import compute_true_eigenvalues
from factor_lab.analyses import compute_sine_alignment, register_manifold_distance

# Generic orchestration mechanics. Re-exported under their historical private
# names so existing call sites and the test suite (sim._make_one_sampler, etc.)
# keep resolving after the move to fl_orchestration.
from fl_orchestration import (
    make_one_sampler as _make_one_sampler,
    make_samplers as _make_samplers,
    next_run_dir as _next_run_dir,
    simulate_returns,
    run_analyses,
)

__all__ = [
    "SimSpec",
    "ModelSpec",
    "ExperimentSpec",
    "SineAlignmentAnalysis",
    "Eq6RHSAnalysis",
    "build_model",
    "run_cell",
    "simulate",
    "print_summary",
    "main",
    "simulate_returns",
    "run_analyses",
    "compute_sine_alignment",
]

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
        # Explicit utf-8: JSON is utf-8 by spec, and our shipped specs use
        # σ/β/δ in _comment fields — on Windows the default encoding (cp1252)
        # can't decode those bytes.
        with open(filepath, encoding="utf-8") as f:
            config = json.load(f)
        # Drop "_..."  commentary keys.
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        return cls(**config)

    @classmethod
    def from_split(cls, model: "ModelSpec", experiment: "ExperimentSpec") -> "SimSpec":
        """Compose a runtime SimSpec from a (model, experiment) pair.

        The model owns what defines the factor model (k, variances, loading and
        idio-vol samplers); the experiment owns the sweep and return process
        (n/p grids, reps, seed, return samplers, output). This keeps the runtime
        object — what :func:`simulate` consumes — byte-for-byte equivalent to a
        single unified SimSpec, so the verification path is unchanged.
        """
        return cls(
            k_factors=model.k_factors,
            factor_variances=list(model.factor_variances),
            beta_samplers=model.beta_samplers,
            idio_vol_sampler=model.idio_vol_sampler,
            n_values=list(experiment.n_values),
            p_values=list(experiment.p_values),
            n_reps=experiment.n_reps,
            random_seed=experiment.random_seed,
            factor_return_sampler=experiment.factor_return_sampler,
            idio_return_sampler=experiment.idio_return_sampler,
            output_path=experiment.output_path,
            plot_mode=experiment.plot_mode,
        )

    @classmethod
    def from_experiment_json(cls, filepath: Union[str, Path]) -> "SimSpec":
        """Load an experiment-spec JSON and resolve its model reference into a SimSpec.

        The experiment JSON carries a ``"model"`` field that is either a path to
        a model-spec JSON (resolved relative to the experiment file) or an inline
        model-spec object. Everything else on the experiment file is the sweep /
        return process.
        """
        experiment = ExperimentSpec.from_json(filepath)
        model = experiment.resolve_model(base_dir=Path(filepath).resolve().parent)
        return cls.from_split(model, experiment)


def _drop_comment_keys(config: dict) -> dict:
    """Drop ``_``-prefixed commentary keys (same convention as SimSpec.from_json)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


@dataclass
class ModelSpec:
    """The factor-model half of the split config: what defines (B, F, D).

    Field semantics are identical to the matching fields on :class:`SimSpec`.
    A model spec is reusable across many experiments — fix it once, vary the
    return process / sweep in different experiment specs against it.
    """

    k_factors: int = 3
    factor_variances: list[float] = field(
        default_factory=lambda: [0.04, 0.02, 0.01]
    )
    beta_samplers: Union[list[dict], dict] = field(
        default_factory=lambda: [
            {"distribution": "normal", "loc": 0.0, "scale": 1.0},
            {"distribution": "normal", "loc": 0.0, "scale": float(np.sqrt(0.8))},
            {"distribution": "normal", "loc": 0.0, "scale": float(np.sqrt(0.6))},
        ]
    )
    idio_vol_sampler: dict = field(
        default_factory=lambda: {"distribution": "constant", "value": 1.0}
    )

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "ModelSpec":
        with open(filepath, encoding="utf-8") as f:
            return cls(**_drop_comment_keys(json.load(f)))


@dataclass
class ExperimentSpec:
    """The experiment half of the split config: the sweep and return process.

    ``model`` references the factor-model half: either a path to a ModelSpec
    JSON (resolved relative to the experiment file) or an inline model-spec
    object. When omitted, the ModelSpec defaults are used.
    """

    model: Union[str, dict, None] = None
    n_values: list[int] = field(default_factory=lambda: [30, 60, 120])
    p_values: list[int] = field(
        default_factory=lambda: [200, 500, 1000, 2000, 5000, 10_000]
    )
    n_reps: int = 300
    random_seed: int = 20260511
    factor_return_sampler: Union[list[dict], dict] = field(
        default_factory=lambda: {"distribution": "normal", "loc": 0.0, "scale": 1.0}
    )
    idio_return_sampler: dict = field(
        default_factory=lambda: {"distribution": "normal", "loc": 0.0, "scale": 1.0}
    )
    output_path: Optional[str] = None
    plot_mode: Optional[str] = None

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "ExperimentSpec":
        with open(filepath, encoding="utf-8") as f:
            return cls(**_drop_comment_keys(json.load(f)))

    def resolve_model(self, base_dir: Path) -> ModelSpec:
        """Resolve the ``model`` reference into a ModelSpec.

        - ``None``  → ModelSpec defaults.
        - ``dict``  → inline ModelSpec(**dict).
        - ``str``   → path to a model-spec JSON, relative to ``base_dir`` if not
                      absolute.
        """
        if self.model is None:
            return ModelSpec()
        if isinstance(self.model, dict):
            return ModelSpec(**_drop_comment_keys(self.model))
        model_path = Path(self.model)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        return ModelSpec.from_json(model_path)


def _register_sine_distance() -> None:
    """Register the diagnostic ``dist_sine`` manifold distance, once per process.

    Idempotent: a guard checks the registry first so repeated ``simulate`` calls
    (and the test suite) do not double-register. The centering difference vs. the
    uncentered Y used in the LHS means this value is diagnostic only.
    """
    from factor_lab.analyses.manifold import _EXTRA_DISTANCES
    if 'dist_sine' not in _EXTRA_DISTANCES:
        register_manifold_distance(
            'dist_sine',
            lambda bt, be: compute_sine_alignment(bt, be)[1],
        )


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


# ── Model construction (Stage 1) ──────────────────────────────────────────────


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


def run_cell(spec: SimSpec, n: int, p: int, rng_master: np.random.Generator) -> list[dict]:
    """Drive one (n, p) cell of the verification and return its per-factor records.

    Owns the master-RNG draw order, which the verification's reproducibility
    depends on:

        1. ``build_model`` draws β (and idio vols) from ``rng_master`` — a
           p-dependent number of draws.
        2. ``compute_true_eigenvalues`` runs ARPACK on the fixed model and draws
           nothing from ``rng_master``.
        3. The n_reps per-rep seeds are drawn from ``rng_master``.
        4. Each rep uses an independent child generator seeded from (3).

    The model is rebuilt with fresh β every cell — the conditional-on-F regime
    the Part-(ii) claim is stated under. This per-cell-fresh contract is specific
    to the *verification*; reusing one model across cells (now possible via
    ``simulate_returns``) would be a different experiment and must not be wired in
    here.
    """
    # (1) fresh model for this cell.
    model = build_model(spec, p, rng_master)
    # (2) population directions once per cell; ARPACK stays out of the rep loop.
    _, b_pop = compute_true_eigenvalues(model, spec.k_factors)
    lhs_analysis = SineAlignmentAnalysis(b_pop)
    rhs_analysis = Eq6RHSAnalysis()

    logger.debug("n={}, p={}: c={}", n, p, list((model.B ** 2).mean(axis=1)))

    # (3) independent seed per rep; master rng advances only here.
    rep_seeds = rng_master.integers(0, 2 ** 31, size=spec.n_reps)
    records: list[dict] = []
    for r in range(spec.n_reps):
        # (4) child generator — isolated from rng_master.
        rep_rng = np.random.default_rng(int(rep_seeds[r]))
        context = simulate_returns(
            model=model, n=n,
            factor_return_spec=spec.factor_return_sampler,
            idio_return_spec=spec.idio_return_sampler,
            k=spec.k_factors, rep_rng=rep_rng,
        )
        res = run_analyses(context, [lhs_analysis, rhs_analysis])
        # LHS/RHS result keys are disjoint, so the merged dict serves as both.
        records.extend(_rep_records(spec.k_factors, n, p, res, res))
    return records


# ── Main simulation ───────────────────────────────────────────────────────────


def simulate(spec: SimSpec) -> pd.DataFrame:
    """Run the full verification sweep and return a tidy per-rep DataFrame.

    Thin orchestrator: register the diagnostic distance, then drive each (n, p)
    cell via :func:`run_cell` in the original loop order, and concatenate.
    """
    _register_sine_distance()

    rng_master = np.random.default_rng(spec.random_seed)
    records: list[dict] = []
    for n in spec.n_values:
        logger.info("Starting n = {}", n)
        for p in tqdm(spec.p_values, desc=f"n={n}", unit="p"):
            records.extend(run_cell(spec, n, p, rng_master))

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
        "--experiment", type=str, default=None,
        help="Experiment-spec JSON (split config). Its 'model' field references "
             "a model-spec JSON by path or inline. Mutually exclusive with the "
             "positional unified spec.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model-spec JSON (split config). Overrides the 'model' reference in "
             "the --experiment file; pair with --experiment.",
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

    if args.experiment is not None:
        # Split config: experiment references model (by path or inline). An
        # explicit --model overrides whatever the experiment file points at.
        experiment = ExperimentSpec.from_json(args.experiment)
        if args.model is not None:
            model = ModelSpec.from_json(args.model)
        else:
            model = experiment.resolve_model(
                base_dir=Path(args.experiment).resolve().parent
            )
        spec = SimSpec.from_split(model, experiment)
        logger.info("Loaded split config: model={}, experiment={}",
                    args.model or experiment.model, args.experiment)
    elif args.config_file is None:
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
