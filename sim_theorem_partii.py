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
This script is *only* the dispersion-bias probe. The machinery is layered:

    fl_orchestration   — stateless seams: sampler resolution, return generation,
                         analysis dispatch, run-dir allocation
    fl_experiment      — the generic engine: ModelSpec, DesignSpec, the
                         Experiment protocol, build_model, and run_experiment
                         (owns the master-RNG draw order + the n×p sweep)
    sim_theorem_partii — THIS file: the SineAlignment / Eq6RHS analyses and the
                         DispersionBiasExperiment that wires them to the engine

The three inputs to the engine are fully decoupled:

    ModelSpec   (the factor model)  — what factor model        [theorem-agnostic]
    DesignSpec  (design_spec.json)  — sweep + return process   [theorem-agnostic]
    Experiment  (this file)         — what to measure          [theorem-specific]

A ``DesignSpec`` carries its model under the ``model`` field, which may be a
path reference, an inline object, or — for a single self-contained file —
model fields written at the JSON top level (folded in by ``DesignSpec.from_json``).
``resolve_model`` turns any of those into a concrete ``ModelSpec``. There is no
separate "unified" spec class: one file shape, one loader.

``run_experiment`` owns the master-RNG draw order the verification depends on:
per (n, p) cell ``build_model`` draws first, then the per-rep seeds are drawn,
then each rep uses an independent child generator. Reordering that stream
changes every downstream number; the Experiment hooks never touch the master
RNG. A different theorem is a new Experiment with no engine change.

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
    python sim_theorem_partii.py                              # built-in defaults
    python sim_theorem_partii.py sim_thmptii_spec.json       # single self-contained file
    python sim_theorem_partii.py sim_thmptii_design.json     # design referencing a model
    python sim_theorem_partii.py sim_thmptii_design.json --model sim_thmptii_model.json
    python sim_theorem_partii.py sim_thmptii_spec.json --plot-save --out results.parquet

Notebook idiom
--------------
    from fl_experiment import ModelSpec, DesignSpec, run_experiment
    from sim_theorem_partii import DispersionBiasExperiment
    df = run_experiment(ModelSpec(), DesignSpec(), DispersionBiasExperiment())
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.analysis import SimulationContext
from factor_lab.analyses.spectral import compute_true_eigenvalues
from factor_lab.analyses import compute_sine_alignment, register_manifold_distance

# Generic engine + data specs. The engine knows nothing about dispersion bias;
# this script supplies the theorem-specific Experiment below.
from fl_experiment import ModelSpec, DesignSpec, build_model, run_experiment

# Generic orchestration mechanics. Re-exported under their historical private
# names so the test suite (sim._make_one_sampler, etc.) keeps resolving.
from fl_orchestration import (
    make_one_sampler as _make_one_sampler,
    make_samplers as _make_samplers,
    next_run_dir as _next_run_dir,
    simulate_returns,
    run_analyses,
)

__all__ = [
    "ModelSpec",
    "DesignSpec",
    "DispersionBiasExperiment",
    "SineAlignmentAnalysis",
    "Eq6RHSAnalysis",
    "build_model",
    "simulate",
    "print_summary",
    "main",
    "run_experiment",
    "simulate_returns",
    "run_analyses",
    "compute_sine_alignment",
]


def _register_sine_distance() -> None:
    """Register the diagnostic ``dist_sine`` manifold distance, once per process.

    Idempotent: a guard checks the registry first so repeated runs (and the test
    suite) do not double-register. The centering difference vs. the uncentered Y
    used in the LHS means this value is diagnostic only.
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


# ── The dispersion-bias probe ─────────────────────────────────────────────────


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


class DispersionBiasExperiment:
    """The theorem-specific :class:`~fl_experiment.Experiment` for Eq. (6) Part (ii).

    Supplies the engine's three hooks:

    - ``setup()`` registers the diagnostic ``dist_sine`` distance once.
    - ``cell_setup(model, n, p)`` computes the population loading directions
      b̄ⱼ once per cell (ARPACK, RNG-free) and returns the LHS/RHS analyses to
      run on every replication.
    - ``record(n, p, merged)`` flattens one rep's merged result into k rows.

    The engine guarantees the per-cell-fresh model draw order; this probe is
    what makes that draw order *mean* the conditional-on-F regime the Part-(ii)
    claim is stated under. A different theorem is a different Experiment, with
    no engine change.

    Notebook idiom::

        from fl_experiment import ModelSpec, DesignSpec, run_experiment
        from sim_theorem_partii import DispersionBiasExperiment
        df = run_experiment(ModelSpec(), DesignSpec(), DispersionBiasExperiment())
    """

    def setup(self) -> None:
        _register_sine_distance()

    def cell_setup(self, model, n: int, p: int):
        # Population directions once per cell; ARPACK stays out of the rep loop.
        _, b_pop = compute_true_eigenvalues(model, model.k)
        return [SineAlignmentAnalysis(b_pop), Eq6RHSAnalysis()]

    def record(self, n: int, p: int, merged: dict) -> list[dict]:
        # LHS/RHS result keys are disjoint, so the merged dict serves as both.
        k = len(merged["sin2_j"])
        return _rep_records(k, n, p, merged, merged)


# ── Single-call driver ────────────────────────────────────────────────────────


def simulate(design: DesignSpec, *, base_dir: Path = ROOT) -> pd.DataFrame:
    """Run the dispersion-bias verification from a :class:`DesignSpec`.

    Convenience wrapper: resolve the design's model (inline, referenced, or
    defaults) and hand both to the generic engine with the dispersion-bias
    probe. Equivalent to::

        model = design.resolve_model(base_dir)
        run_experiment(model, design, DispersionBiasExperiment())

    ``base_dir`` only matters when the design's ``model`` is a relative path;
    it defaults to this script's directory.
    """
    model = design.resolve_model(base_dir)
    return run_experiment(model, design, DispersionBiasExperiment())


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
        help="Design-spec JSON (see DesignSpec). Its model may be inline (top-"
             "level fields or a 'model' object) or a path reference. If omitted, "
             "uses the built-in defaults.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model-spec JSON. Overrides the design's 'model' reference.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path for the .parquet file. Overrides the design output_path; "
             "defaults to an auto-allocated results/MM-DD_run_NN/ directory.",
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

    # Resolve inputs into a (ModelSpec, DesignSpec) pair. The design carries its
    # own model (inline / referenced / defaults); --model overrides it.
    if args.config_file is None:
        logger.info("No config file given; using built-in defaults.")
        design_spec = DesignSpec()
        base_dir = ROOT
    else:
        design_spec = DesignSpec.from_json(args.config_file)
        base_dir = Path(args.config_file).resolve().parent

    if args.model is not None:
        model_spec = ModelSpec.from_json(args.model)
        logger.info("Model overridden by --model={}", args.model)
    else:
        model_spec = design_spec.resolve_model(base_dir)

    # Resolve output path: CLI --out > design output_path > auto-allocated run dir.
    if args.out is not None:
        parquet_path = args.out.with_suffix(".parquet")
    elif design_spec.output_path is not None:
        parquet_path = Path(design_spec.output_path).with_suffix(".parquet")
    else:
        run_dir = _next_run_dir(ROOT)
        parquet_path = run_dir / "sim_thmptii.parquet"
        logger.info("Auto-allocated run directory: {}", run_dir)

    # Resolve plot mode: CLI flags > design plot_mode > none.
    if args.plot:
        plot_mode = "plot"
    elif args.plot_save:
        plot_mode = "plot-save"
    else:
        plot_mode = design_spec.plot_mode   # may be None

    logger.info(
        "Simulation: k={}, n={}, p={}, reps={}, seed={}",
        model_spec.k_factors, design_spec.n_values, design_spec.p_values,
        design_spec.n_reps, design_spec.random_seed,
    )
    logger.info(
        "σ²={}, idio_vol_sampler={}",
        list(model_spec.factor_variances), model_spec.idio_vol_sampler,
    )

    df = run_experiment(model_spec, design_spec, DispersionBiasExperiment())

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
