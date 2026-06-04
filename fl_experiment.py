"""
fl_experiment.py
================
Generic, theorem-agnostic experiment engine for factor-model studies.

This is the layer that drives a *sweep* — build a model per (n, p) cell, sample
many replications of returns, run a set of analyses, and collect a tidy
DataFrame — without knowing anything about *which* theorem is being checked.
The theorem-specific part is supplied as an :class:`Experiment`.

Separation of concerns
----------------------
- **What model** (``ModelSpec``): k, factor variances, loading + idio-vol
  samplers. Reusable across many studies. Serialized as ``model_spec.json``.
- **What design / DGP** (``DesignSpec``): n/p grids, reps, seed, the return
  samplers, output. Serialized as ``design_spec.json``; it may reference a
  model spec by path or inline.
- **What question** (``Experiment``): the analyses to run and how to flatten a
  replication into records. This is the only theorem-specific piece; it lives
  in the caller (a notebook or a script like ``sim_theorem_partii.py``), not
  here.

The engine consumes all three via :func:`run_experiment` and is the sole owner
of the master-RNG draw order, which any reproducibility claim depends on:

    per (n, p) cell:
        1. build_model draws β (and idio vols) from the master RNG
        2. Experiment.cell_setup runs on the fixed model — draws nothing
        3. n_reps per-rep seeds are drawn from the master RNG
        4. each rep uses an independent child generator seeded from (3)

Reordering that stream changes every downstream number. ``Experiment`` hooks
must therefore never draw from the master RNG (``cell_setup`` is handed the
already-built model; analyses are deterministic given their context).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.model_builder import FactorModelBuilder
from fl_orchestration import make_one_sampler, make_samplers, simulate_returns, run_analyses

__all__ = [
    "ModelSpec",
    "DesignSpec",
    "Experiment",
    "build_model",
    "run_cell",
    "run_experiment",
]


def _drop_comment_keys(config: dict) -> dict:
    """Drop ``_``-prefixed commentary keys (the shipped-JSON comment convention)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# Fields that define the factor model. When they appear at the top level of a
# design JSON (the "unified single-file" shape), DesignSpec folds them into an
# inline ``model`` so one loader handles every file shape.
_MODEL_FIELDS = ("k_factors", "factor_variances", "beta_samplers", "idio_vol_sampler")


# ── Specs ─────────────────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    """The factor-model half of a study: what defines (B, F, D).

    Reusable across many designs — fix it once, vary the return process / sweep
    in different design specs against it. Sampler fields use the
    ``{"distribution": name, ...}`` shape consumed by
    :func:`factor_lab.distributions.create_sampler`.

    Defaults reproduce the diagonal-Gram baseline: loadings β_j ~ N(0, √c_j)
    with c = [1.0, 0.8, 0.6], factor variances σ² = [.04, .02, .01], constant
    idio vol 1.0 (so δ² = 1 after squaring).
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
class DesignSpec:
    """A full study config: the sweep + return process, plus its model.

    The ``model`` field is the single home for the factor model and accepts
    three forms, so one class covers every file shape:

    - **reference** ``"model": "model_spec.json"`` — a path, resolved relative to
      the design file. The split-config shape (model reusable across designs).
    - **inline**    ``"model": { ...model fields... }`` — an embedded model.
    - **unified**   model fields (``k_factors`` etc.) written at the *top level*
      of the JSON — :meth:`from_json` folds them into an inline ``model``. This
      is the convenient single-file shape.
    - **omitted**   ``model`` absent and no top-level model fields → ModelSpec
      defaults.

    The engine never sees the reference; call :meth:`resolve_model` to get a
    concrete :class:`ModelSpec`, then hand both to
    :func:`run_experiment`.
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
    plot_mode: Optional[str] = None   # None | "plot" | "plot-save"

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "DesignSpec":
        with open(filepath, encoding="utf-8") as f:
            return cls.from_config(_drop_comment_keys(json.load(f)))

    @classmethod
    def from_config(cls, config: dict) -> "DesignSpec":
        """Build from a config dict, folding top-level model fields into ``model``.

        Lets the unified single-file shape (model fields at the top level) load
        through the same path as the split/inline shapes.
        """
        config = dict(config)
        inline = {k: config.pop(k) for k in _MODEL_FIELDS if k in config}
        if inline:
            if config.get("model") not in (None, {}):
                raise ValueError(
                    "Model specified both inline (top-level fields like "
                    f"{sorted(inline)}) and via a 'model' reference; use one form."
                )
            config["model"] = inline
        return cls(**config)

    def resolve_model(self, base_dir: Path) -> ModelSpec:
        """Resolve the ``model`` field into a concrete ModelSpec.

        - ``None``  → ModelSpec defaults.
        - ``dict``  → inline ``ModelSpec(**dict)`` (comment keys dropped).
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


# ── Experiment protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Experiment(Protocol):
    """The theorem-specific half of a study: what to measure.

    An Experiment supplies three hooks the engine calls. None of them may draw
    from the master RNG — ``cell_setup`` receives the already-built model, and
    analyses are deterministic given their context.

    - ``setup()``                — optional one-time process setup (e.g. register
                                    a diagnostic distance). Called once before the
                                    sweep. A default no-op is fine.
    - ``cell_setup(model, n, p)`` — model-only per-cell preparation; returns the
                                    list of analyses to run on every replication
                                    in this cell. Each analysis exposes
                                    ``analyze(context) -> dict``.
    - ``record(n, p, merged)``    — flatten one replication's merged analysis
                                    result dict into a list of row dicts.
    """

    def cell_setup(self, model, n: int, p: int) -> Sequence: ...

    def record(self, n: int, p: int, merged: dict) -> list[dict]: ...


# ── Model construction (Stage 1) ──────────────────────────────────────────────


def build_model(model_spec: ModelSpec, p: int, rng: np.random.Generator):
    """Build a k-factor model from ``model_spec`` for the given p.

    Loading samplers, idio-vol sampler, and factor variances all come from the
    model spec. Draws from ``rng`` — this is the master-RNG draw in step (1) of
    the per-cell order.
    """
    return FactorModelBuilder(rng=rng).build(
        p=p,
        k=model_spec.k_factors,
        beta_samplers=make_samplers(model_spec.beta_samplers, rng, model_spec.k_factors),
        idio_vol_sampler=make_one_sampler(model_spec.idio_vol_sampler, rng),
        factor_variances=list(model_spec.factor_variances),
    )


# ── The engine ────────────────────────────────────────────────────────────────


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
    not the engine's — the engine simply guarantees the documented draw order.
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
    return records


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
        The three halves of a study (see module docstring).
    rng
        Master generator. Defaults to ``np.random.default_rng(design_spec.random_seed)``.
        Pass one explicitly to thread a shared stream.
    progress
        Show a per-n tqdm bar over the p grid.

    Notebook idiom::

        from fl_experiment import ModelSpec, DesignSpec, run_experiment
        df = run_experiment(ModelSpec(), DesignSpec(n_values=[60], p_values=[1000],
                                                    n_reps=50), MyExperiment())
    """
    if rng is None:
        rng = np.random.default_rng(design_spec.random_seed)

    setup = getattr(experiment, "setup", None)
    if callable(setup):
        setup()

    records: list[dict] = []
    for n in design_spec.n_values:
        logger.info("Starting n = {}", n)
        p_iter = tqdm(design_spec.p_values, desc=f"n={n}", unit="p") if progress else design_spec.p_values
        for p in p_iter:
            records.extend(run_cell(model_spec, design_spec, experiment, n, p, rng))

    return pd.DataFrame(records)
