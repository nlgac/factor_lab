"""
fl_experiment_runner.py
=======================
The *runner* layer of the experiment engine: drive a sweep over an (n, p) grid
and collect a tidy DataFrame, without knowing which theorem is being checked.

It consumes the setup layer (:mod:`fl_experiment_setup`) — the specs, the
``Experiment`` protocol, ``build_model``, and the stateless seams — and owns the
**master-RNG draw order**, which any reproducibility claim depends on:

    per (n, p) cell:
        1. build_model draws β (and idio vols) from the master RNG
        2. Experiment.cell_setup runs on the fixed model — draws nothing
        3. n_reps per-rep seeds are drawn from the master RNG
        4. each rep uses an independent child generator seeded from (3)

Reordering that stream changes every downstream number. ``Experiment`` hooks
must therefore never draw from the master RNG (``cell_setup`` is handed the
already-built model; analyses are deterministic given their context).

Sampling topology (``DesignSpec.sampling``)
-------------------------------------------
- ``"independent"`` (default): the draw order above — a fresh model + returns
  for every (n, p) cell, so each p is statistically independent.
- ``"nested"``: per replicate, draw one superset model + returns at
  ``p_max = max(p_values)`` and take each smaller p as an *asset subset* (the
  first p, by default), so p₁ ⊂ p₂ ⊂ … ⊂ p_max are the same assets. The factor
  realization is shared across p, so a smaller p is an exact slice of the
  superset — not a new sample — giving a clean monotone-in-p convergence curve.
  Output rows carry a ``rep`` column; the replicate, not the row, is the unit of
  statistical independence. The ``Experiment`` is unchanged across both modes —
  only how each cell's (model, returns) is produced differs.

Time nesting (``DesignSpec.nest_time``, nested mode only)
---------------------------------------------------------
By default the nested path redraws the factor/idio returns for each n. Setting
``nest_time=True`` nests the n axis as well: per replicate the returns are drawn
once at ``n_max = max(n_values)`` and each n is the first-n-periods prefix of
that single draw (the time analogue of the p subset), giving a monotone-in-n
curve. It changes the master-RNG draw order, so its numbers differ from the
per-n-redraw default — the same way nested differs from independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.factor_types import FactorModelData
from factor_lab.analysis import SimulationContext
from fl_experiment_setup import (
    ModelSpec,
    DesignSpec,
    Experiment,
    build_model,
    simulate_returns,
    run_analyses,
)

__all__ = [
    "run_experiment",
    "run_cell",
]


# ── Per-cell driver ───────────────────────────────────────────────────────────


def run_cell(
    model_spec: ModelSpec,
    design_spec: DesignSpec,
    experiment: Experiment,
    n: int,
    p: int,
    rng_master: np.random.Generator,
) -> list[dict]:
    """Drive one (n, p) cell and return its records.

    Owns the per-cell master-RNG draw order (see module docstring). The model is
    rebuilt with fresh draws every cell; whether that matters (e.g. the
    conditional-on-F regime of a particular theorem) is the Experiment's concern,
    not the runner's — the runner simply guarantees the documented draw order.
    """
    # (1) fresh model for this cell — draws β / idio vols from the master RNG.
    model = build_model(model_spec, p, rng_master)
    # (2) model-only setup; the Experiment returns the analyses to run. RNG-free.
    analyses = experiment.cell_setup(model, n, p)

    logger.debug("n={}, p={}: c={}", n, p, list((model.B ** 2).mean(axis=1)))

    # (3) independent seed per rep; master rng advances only here.
    rep_seeds = rng_master.integers(0, 2 ** 31, size=design_spec.n_reps)
    records: list[dict] = []
    for r in range(design_spec.n_reps):
        # (4) child generator — isolated from rng_master.
        rep_rng = np.random.default_rng(int(rep_seeds[r]))
        context = simulate_returns(
            model=model, n=n,
            factor_return_spec=design_spec.factor_return_sampler,
            idio_return_spec=design_spec.idio_return_sampler,
            k=model_spec.k_factors, rep_rng=rep_rng,
        )
        merged = run_analyses(context, analyses)
        records.extend(experiment.record(n, p, merged))
    logger.debug("cell n={}, p={} done: {} records", n, p, len(records))
    return records


# ── The sweep ─────────────────────────────────────────────────────────────────


def run_experiment(
    model_spec: ModelSpec,
    design_spec: DesignSpec,
    experiment: Experiment,
    *,
    rng: np.random.Generator = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run a full sweep and return a tidy per-rep DataFrame.

    Parameters
    ----------
    model_spec, design_spec, experiment
        The three halves of a study (model + design from
        :mod:`fl_experiment_setup`; the Experiment from the caller).
    rng
        Master generator. Defaults to ``np.random.default_rng(design_spec.random_seed)``.
        Pass one explicitly to thread a shared stream.
    progress
        Show a per-n tqdm bar over the p grid.

    Notebook idiom::

        from fl_experiment_setup import ModelSpec, DesignSpec
        from fl_experiment_runner import run_experiment
        df = run_experiment(ModelSpec(), DesignSpec(n_values=[60], p_values=[1000],
                                                    n_reps=50), MyExperiment())
    """
    if rng is None:
        rng = np.random.default_rng(design_spec.random_seed)

    setup = getattr(experiment, "setup", None)
    if callable(setup):
        setup()

    logger.info(
        "Running {} sweep: n={}, p={}, reps={}, seed={}",
        design_spec.sampling, design_spec.n_values, design_spec.p_values,
        design_spec.n_reps, design_spec.random_seed,
    )

    if design_spec.sampling == "nested":
        df = _run_nested(model_spec, design_spec, experiment, rng, progress)
    elif design_spec.sampling == "independent":
        records: list[dict] = []
        for n in design_spec.n_values:
            logger.info("Starting n = {}", n)
            p_iter = (tqdm(design_spec.p_values, desc=f"n={n}", unit="p")
                      if progress else design_spec.p_values)
            for p in p_iter:
                records.extend(run_cell(model_spec, design_spec, experiment, n, p, rng))
        df = pd.DataFrame(records)
    else:
        raise ValueError(
            f"Unknown sampling mode {design_spec.sampling!r}; "
            "expected 'independent' or 'nested'."
        )

    logger.info("Sweep complete: {} rows", len(df))
    return df


# ── Nested (monotone-in-p) sampling ───────────────────────────────────────────


def _slice_to_p(context: SimulationContext, p: int) -> SimulationContext:
    """Return a view of ``context`` restricted to its first ``p`` assets.

    Assets are columns of B / rows of Y. Because the factor realization does not
    depend on p, slicing to p assets is an exact subset — the same draw, fewer
    columns — not a new sample. All slices are numpy views (no copy).
    """
    return SimulationContext(
        model=_slice_model_to_p(context.model, p),
        security_returns=context.security_returns[:, :p],
        factor_returns=context.factor_returns,          # shared across p
        idio_returns=context.idio_returns[:, :p],
    )


def _slice_model_to_p(model: FactorModelData, p: int) -> FactorModelData:
    """Return a view of ``model`` restricted to its first ``p`` assets (B, D)."""
    return FactorModelData(B=model.B[:, :p], F=model.F, D=model.D[:p, :p])


def _slice_to_np(context: SimulationContext, n: int, p: int) -> SimulationContext:
    """Return a view of ``context`` restricted to its first ``n`` periods and ``p`` assets.

    Periods are rows (axis 0) of the realized return arrays; assets are columns
    (axis 1) and B's columns. Used by time-nested sampling: the returns are drawn
    once at (n_max, p_max) and each (n, p) cell is the matching prefix block — the
    same draw, fewer rows and columns — not a new sample. All slices are views.
    """
    return SimulationContext(
        model=_slice_model_to_p(context.model, p),
        security_returns=context.security_returns[:n, :p],
        factor_returns=context.factor_returns[:n, :],
        idio_returns=context.idio_returns[:n, :p],
    )


def _run_nested(
    model_spec: ModelSpec,
    design_spec: DesignSpec,
    experiment: Experiment,
    rng_master: np.random.Generator,
    progress: bool,
) -> pd.DataFrame:
    """Nested sampling: per replicate, draw one superset at ``p_max`` and slice.

    Draw order, per replicate (a child generator seeded off the master, so
    replicates are independent and reproducible):

        1. build_model at p_max  — draws β / idio vols (the assets), once.
        2. for each n: simulate_returns at p_max — draws factor returns + Z.
        3. for each p (any order): slice the superset to its first p assets and
           run the Experiment on the slice.

    The model's assets (β, D) are shared across all n and p within a replicate.
    By default the factor/idio returns are redrawn per n (the n axis is *not*
    nested). With ``design_spec.nest_time=True`` the returns are instead drawn
    once per replicate at ``n_max = max(n_values)`` and every n becomes the
    first-n-periods prefix of that single draw, so the n axis is nested too (the
    time analogue of the p nesting) — giving a monotone-in-n curve. Each output
    row is tagged with its ``rep`` index: within a replicate the nested axes are
    correlated, so the replicate — not the row — is the unit of statistical
    independence. Enabling ``nest_time`` changes the master-RNG draw order, so its
    output is not expected to match the per-n-redraw output.

    ``cell_setup`` is called once per (replicate, p) and the returned analyses
    are reused across all n. This assumes the per-cell setup is n-independent —
    true for population-direction work like the dispersion probe (b̄ⱼ depends on
    the model, not the realized sample size) — and avoids redundant ARPACK across
    the n axis. It does not change any output value (cell_setup is RNG-free).

    Unlike the independent path this is a distinct sampling scheme, so its
    output is not expected to match independent-mode output.
    """
    if design_spec.subsample != "prefix":
        raise NotImplementedError(
            f"subsample={design_spec.subsample!r} not implemented; only 'prefix' "
            "(first p assets) is currently supported."
        )

    p_values = list(design_spec.p_values)
    p_max = max(p_values)
    n_values = list(design_spec.n_values)
    n_max = max(n_values)
    nest_time = design_spec.nest_time
    k = model_spec.k_factors
    rep_seeds = rng_master.integers(0, 2 ** 31, size=design_spec.n_reps)

    records: list[dict] = []
    rep_iter = (
        tqdm(range(design_spec.n_reps), desc="replicate", unit="rep")
        if progress else range(design_spec.n_reps)
    )
    n0 = n_values[0]
    for r in rep_iter:
        rep_rng = np.random.default_rng(int(rep_seeds[r]))
        # (1) one superset model for this replicate — shared across all n and p.
        model_full = build_model(model_spec, p_max, rep_rng)
        logger.debug("rep={}: built superset model at p_max={}", r, p_max)
        # (2) per-p analyses once per replicate — model-only, RNG-free, reused
        #     across all n (population directions do not depend on n).
        analyses_by_p = {
            p: experiment.cell_setup(_slice_model_to_p(model_full, p), n0, p)
            for p in p_values
        }

        if nest_time:
            # (3) one returns superset at (n_max, p_max) for this replicate; every
            #     (n, p) cell is the matching first-n-by-first-p prefix block.
            ctx_full = simulate_returns(
                model=model_full, n=n_max,
                factor_return_spec=design_spec.factor_return_sampler,
                idio_return_spec=design_spec.idio_return_sampler,
                k=k, rep_rng=rep_rng,
            )
            for n in n_values:
                for p in p_values:
                    ctx_np = _slice_to_np(ctx_full, n, p)
                    merged = run_analyses(ctx_np, analyses_by_p[p])
                    for row in experiment.record(n, p, merged):
                        row["rep"] = r
                        records.append(row)
        else:
            for n in n_values:
                # (3) one superset of returns at p_max for this (rep, n).
                ctx_full = simulate_returns(
                    model=model_full, n=n,
                    factor_return_spec=design_spec.factor_return_sampler,
                    idio_return_spec=design_spec.idio_return_sampler,
                    k=k, rep_rng=rep_rng,
                )
                # (4) each p is an asset subset of the same draw.
                for p in p_values:
                    ctx_p = _slice_to_p(ctx_full, p)
                    merged = run_analyses(ctx_p, analyses_by_p[p])
                    for row in experiment.record(n, p, merged):
                        row["rep"] = r
                        records.append(row)

    return pd.DataFrame(records)
