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

    fl_experiment_setup  — specs (ModelSpec, DesignSpec), the Experiment
                           protocol, build_model, and the stateless seams
                           (sampler resolution, return generation, analysis
                           dispatch, run-dir allocation)
    fl_experiment_runner — the sweep: run_experiment / run_cell, which own the
                           master-RNG draw order + the n×p grid
    sim_theorem_partii   — THIS file: the SineAlignment / Eq6RHS analyses and the
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
    from fl_experiment_setup import ModelSpec, DesignSpec
    from fl_experiment_runner import run_experiment
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

# Engine, in two layers (both theorem-agnostic):
#   fl_experiment_setup  — specs, the Experiment protocol, model construction,
#                          and the stateless seams.
#   fl_experiment_runner — the sweep (run_experiment / run_cell).
# This script supplies the theorem-specific Experiment below.
from fl_experiment_setup import (
    ModelSpec, DesignSpec, build_model, BaseExperiment, register_experiment,
    make_one_sampler as _make_one_sampler,
    make_samplers as _make_samplers,
    next_run_dir as _next_run_dir,
    simulate_returns,
    run_analyses,
)
from fl_experiment_runner import run_experiment

__all__ = [
    "ModelSpec",
    "DesignSpec",
    "DispersionBiasExperiment",
    "SineAlignmentAnalysis",
    "Eq6RHSAnalysis",
    "sin2_angle_to_axis",
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
    suite) do not double-register. This distance is a direction-comparison
    diagnostic only; the verified LHS quantity is ``sin2_j``.
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
          computed via the n×n Gram trick). With ``center=True`` (default) Y is
          row-demeaned first — each asset's time-series mean removed — so h is the
          MLE covariance eigenvector. ``center=False`` recovers the uncentered
          second-moment form (Y^T Y on the raw Y).
    b̄ⱼ: j-th population loading direction (unit eigenvector of Σ₀ = BΣ_F B^T/p),
          injected at construction so ARPACK runs once per (n, p) cell.

    Example:
        _, b_pop = compute_true_eigenvalues(model, K)
        analysis = SineAlignmentAnalysis(b_pop)
        result = analysis.analyze(context)
        # {"sin2_j": array shape (K,), "dist_sine": float}
    """

    def __init__(self, b_pop: np.ndarray, center: bool = True):
        self.b_pop = b_pop   # (k, p), rows are population loading directions b̄ⱼ
        self.center = center

    def analyze(self, context: SimulationContext) -> dict:
        k = context.k
        Y = context.security_returns.T   # (p, n): rows assets, cols periods
        if self.center:
            # Row-demean: subtract each asset's mean over the n periods (MLE cov).
            Y = Y - Y.mean(axis=1, keepdims=True)
        # Top-k left SVs of Y via the n×n Gram Y^T Y.
        # Cost O(p·n²) vs O(p²·n) for the full SVD.
        G = Y.T @ Y
        vals, vecs = np.linalg.eigh(G)
        idx = np.argsort(vals)[::-1][:k]
        s = np.sqrt(np.maximum(vals[idx], 0.0))
        H = (Y @ vecs[:, idx]) / np.where(s > 1e-14, s, 1.0)   # (p, k)
        # H.T has shape (k, p); b_pop has shape (k, p)
        sin2, dist = compute_sine_alignment(self.b_pop, H.T)
        return {"sin2_j": sin2, "dist_sine": dist}


def sin2_angle_to_axis(vectors: np.ndarray) -> np.ndarray:
    """sin²∠(vⱼ, eⱼ) for each column vⱼ against the matching standard basis axis eⱼ.

    Since eⱼ is the j-th canonical basis vector, cos∠(vⱼ, eⱼ) is just vⱼ's j-th
    component over its norm, so this reduces to ``1 − (diag / ‖col‖)²``. The squared
    diagonal removes sign ambiguity (eigenvectors are defined up to ±1), and columns
    are normalized first so non-unit input is handled.

    This is the "rotation" term of the Part (ii) RHS (module docstring): how far the
    j-th signal-Gram eigenvector νⱼ^(n) tilts off the axis eⱼ. Abstracted here as a
    standalone utility so it can be reused wherever a per-column off-axis angle is
    needed (any eigenvector/loadings matrix aligned so column j pairs with axis j).

    Parameters
    ----------
    vectors : (m, k) array
        Columns are the vectors vⱼ (e.g. the eigenvector matrix ``W`` from ``eigh``,
        already ordered so column j pairs with axis eⱼ). ``diagonal`` uses the first
        ``min(m, k)`` columns.

    Returns
    -------
    (min(m, k),) array
        sin²∠(vⱼ, eⱼ) in [0, 1] — 0 when vⱼ is aligned with eⱼ, 1 when orthogonal.
    """
    V = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(V, axis=0)[: min(V.shape)]
    cos2 = (np.diagonal(V) / np.where(norms > 0.0, norms, 1.0)) ** 2
    return 1.0 - cos2


class Eq6RHSAnalysis:
    """
    Predicted RHS of Equation (6), Part (ii): floor + weight × rotation for each j.

    Uses factor returns F from the context and empirical prevalences cⱼ = ‖B[j,:]‖²/p
    from the model loadings. Computes D̂ = C^{1/2}(F_c F_c^T/n)C^{1/2}, where F_c is
    F row-demeaned over time when ``center=True`` (default) — the realized-factor
    analogue of centering Y in the LHS, keeping the paired LHS/RHS consistent.
    ``center=False`` uses the raw F (uncentered second moment), matching a
    ``SineAlignmentAnalysis(center=False)`` LHS. Normalization stays 1/n (the MLE;
    the lost degree of freedom is a minor large-n bias we accept).

    δ² is taken from ``context.model.D`` as the mean of its diagonal (D already
    holds variances — the idio_vol_sampler outputs vols which FactorModelBuilder
    squares internally).

    Example:
        analysis = Eq6RHSAnalysis()
        result = analysis.analyze(context)
        # keys: "rhs", "floor", "rotation", "rhos", "delta2"
    """

    def __init__(self, center: bool = True):
        self.center = center

    def analyze(self, context: SimulationContext) -> dict:
        k, n = context.k, context.T
        F = context.factor_returns.T                     # (k, n)
        if self.center:
            # Row-demean F over time — the realized-factor analogue of centering Y.
            F = F - F.mean(axis=1, keepdims=True)
        c_half = np.sqrt((context.model.B ** 2).mean(axis=1))   # √cⱼ
        D_hat = (c_half[:, None] * (F @ F.T / n)) * c_half[None, :]
        vals, vecs = np.linalg.eigh(D_hat)
        idx = np.argsort(vals)[::-1]
        rhos = vals[idx]
        W = vecs[:, idx]
        delta2 = float(np.diag(context.model.D).mean())
        floor = delta2 / (n * rhos + delta2)
        weight = n * rhos / (n * rhos + delta2)
        # sin²∠(ŵⱼ, eⱼ) — the RHS rotation term (see sin2_angle_to_axis).
        rotation = sin2_angle_to_axis(W)
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


@register_experiment("dispersion_bias")
class DispersionBiasExperiment(BaseExperiment):
    """The theorem-specific :class:`~fl_experiment_setup.Experiment` for Eq. (6) Part (ii).

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

        from fl_experiment_setup import ModelSpec, DesignSpec
        from fl_experiment_runner import run_experiment
        from sim_theorem_partii import DispersionBiasExperiment
        df = run_experiment(ModelSpec(), DesignSpec(), DispersionBiasExperiment())
    """

    def __init__(self, center: bool = True):
        # center=True (default): row-demean Y and F (the MLE covariance estimator);
        # center=False: the uncentered second-moment form.
        self.center = center

    def setup(self) -> None:
        _register_sine_distance()

    def cell_setup(self, model, n: int, p: int):
        # Population directions once per cell; ARPACK stays out of the rep loop.
        _, b_pop = compute_true_eigenvalues(model, model.k)
        return [SineAlignmentAnalysis(b_pop, center=self.center),
                Eq6RHSAnalysis(center=self.center)]

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
        "factor_vols={}, idio_vol_sampler={}",
        list(model_spec.factor_vols), model_spec.idio_vol_sampler,
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
