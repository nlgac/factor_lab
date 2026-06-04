"""
fl_orchestration.py
===================
Dispersion-agnostic orchestration mechanics shared across simulation scripts.

This module holds the *general* plumbing that any factor-model experiment needs,
deliberately kept free of dispersion-bias-specific concepts (no
``DispersionBiasExperiment``, no ``SineAlignmentAnalysis``/``Eq6RHSAnalysis``).
Those live in the probe script (e.g. ``sim_theorem_partii.py``) so that the
verification stays the property of the checker, not of this layer.

Seams provided
--------------
- Sampler resolution:   ``make_one_sampler`` / ``make_samplers``
- Return generation:    ``simulate_returns``  (model + return specs → context)
- Analysis dispatch:    ``run_analyses``      (context + [analysis, …] → merged dict)
- Output bookkeeping:   ``next_run_dir``      (sequential results/MM-DD_run_NN dir)

A future script reuses these directly, e.g.::

    from fl_orchestration import simulate_returns, run_analyses
    ctx = simulate_returns(model, n, fac_spec, idio_spec, k, rep_rng)
    res = run_analyses(ctx, [my_lhs_analysis, my_rhs_analysis])

None of these helpers touch a *master* RNG: ``simulate_returns`` draws only from
the per-rep generator it is handed, in a fixed order (factor samplers, then idio
sampler), so a caller that preserves its own draw sequence gets bit-identical
output across refactors.
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.distributions import create_sampler
from factor_lab.flexible_simulator import ReturnsSimulator
from factor_lab.analysis import SimulationContext

__all__ = [
    "make_one_sampler",
    "make_samplers",
    "next_run_dir",
    "simulate_returns",
    "run_analyses",
]


# ── Sampler resolution ────────────────────────────────────────────────────────


def make_one_sampler(spec: dict, rng: np.random.Generator):
    """Materialize a single sampler from a ``{"distribution": name, ...}`` dict."""
    params = {k: v for k, v in spec.items() if k != "distribution"}
    return create_sampler(spec["distribution"], rng, **params)


def make_samplers(spec: Union[list[dict], dict], rng: np.random.Generator, k: int):
    """List-or-broadcast sampler resolution matching ``FactorModelBuilder.build``.

    A list spec must have length ``k`` (one sampler per factor); a single dict is
    broadcast. Raises ``ValueError`` on a length mismatch.
    """
    if isinstance(spec, list):
        if len(spec) != k:
            raise ValueError(
                f"Expected {k} per-factor samplers, got {len(spec)}: {spec!r}"
            )
        return [make_one_sampler(s, rng) for s in spec]
    return make_one_sampler(spec, rng)


# ── Output bookkeeping ──────────────────────────────────────────────────────────


def next_run_dir(base: Path) -> Path:
    """Allocate and return ``{base}/results/MM-DD_run_NN`` with NN sequential per date.

    Scans existing siblings matching today's date prefix, picks ``max(NN)+1``,
    and creates the directory. NN is zero-padded to 2 digits (01, 02, …).
    Unrelated directory names are ignored.

    Example:
        # On 2026-05-19, with results/05-19_run_01 and 05-19_run_02 present,
        next_run_dir(Path('.'))  # → Path('results/05-19_run_03'), created.
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


# ── Return generation (Stages 2–4) ──────────────────────────────────────────────


def simulate_returns(
    model,
    n: int,
    factor_return_spec: Union[list[dict], dict],
    idio_return_spec: dict,
    k: int,
    rep_rng: np.random.Generator,
    simulator: ReturnsSimulator = None,
) -> SimulationContext:
    """Sample one replication of returns for ``model`` and wrap it in a context.

    Draws strictly from ``rep_rng`` (the per-rep generator), in the original
    order — factor return samplers first, then the idiosyncratic sampler — so the
    realized draws match the pre-refactor inline loop exactly. ``simulator`` is
    stateless and constructed on demand if not supplied; it consumes no RNG.

    Returns a :class:`SimulationContext` holding the model and the realized
    security / factor / idiosyncratic returns, ready for analysis.
    """
    if simulator is None:
        simulator = ReturnsSimulator()
    factor_samplers = make_samplers(factor_return_spec, rep_rng, k)
    idio_sampler = make_one_sampler(idio_return_spec, rep_rng)
    sim_out = simulator.simulate(
        model=model, n_periods=n,
        factor_return_samplers=factor_samplers,
        idio_return_sampler=idio_sampler,
    )
    return SimulationContext(
        model=model,
        security_returns=sim_out["security_returns"],
        factor_returns=sim_out["factor_returns"],
        idio_returns=sim_out["idio_returns"],
    )


# ── Analysis dispatch ────────────────────────────────────────────────────────────


def run_analyses(context: SimulationContext, analyses: Sequence) -> dict:
    """Run each analysis over ``context`` and merge the result dicts.

    Each element of ``analyses`` must expose ``analyze(context) -> dict``. Results
    are merged left to right; callers are responsible for keeping result keys
    disjoint across analyses (the verification path uses LHS keys ``sin2_j`` /
    ``dist_sine`` and RHS keys ``rhs`` / ``floor`` / ``rotation`` / ``rhos`` /
    ``delta2``, which do not collide).
    """
    merged: dict = {}
    for analysis in analyses:
        merged.update(analysis.analyze(context))
    return merged
