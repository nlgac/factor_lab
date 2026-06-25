"""
fl_experiment_setup.py
======================
The *setup* layer of the experiment engine: what defines and constructs a study,
plus the stateless building-block seams the runner consumes.

This is theorem-agnostic — no dispersion-bias concepts live here. It holds:

- **Specs** (the data):
    - ``ModelSpec``  — what defines the factor model (k, factor vols, loading +
      idio-vol samplers). Serialized as ``model_spec.json``.
    - ``DesignSpec`` — the sweep + return process (n/p grids, reps, seed, return
      samplers, sampling topology, output). Serialized as ``design_spec.json``;
      it carries its model inline, by reference, or folded from the top level.
- **The ``Experiment`` protocol** — the theorem-specific hooks a caller supplies
  (the implementation lives in the probe script, not here).
- **Model construction** — ``build_model`` (Stage 1).
- **Stateless seams** the runner reuses:
    - Sampler resolution:   ``make_one_sampler`` / ``make_samplers``
    - Return generation:    ``simulate_returns``  (model + return specs → context)
    - Analysis dispatch:    ``run_analyses``      (context + [analysis, …] → dict)
    - Output bookkeeping:   ``next_run_dir``      (sequential results/MM-DD_run_NN)

The actual sweep — looping these over an (n, p) grid and owning the master-RNG
draw order — lives in :mod:`fl_experiment_runner`.

None of the seams touch a *master* RNG: ``simulate_returns`` draws only from the
per-rep generator it is handed, in a fixed order (factor samplers, then idio
sampler), so a caller that preserves its own draw sequence gets bit-identical
output across refactors.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.distributions import create_sampler
from factor_lab.model_builder import FactorModelBuilder
from factor_lab.flexible_simulator import ReturnsSimulator
from factor_lab.analysis import SimulationContext

__all__ = [
    "ModelSpec",
    "DesignSpec",
    "Experiment",
    "BaseExperiment",
    "register_experiment",
    "get_experiment",
    "registered_experiments",
    "build_model",
    "make_one_sampler",
    "make_samplers",
    "simulate_returns",
    "run_analyses",
    "next_run_dir",
]


def _drop_comment_keys(config: dict) -> dict:
    """Drop ``_``-prefixed commentary keys (the shipped-JSON comment convention)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# Fields that define the factor model. When they appear at the top level of a
# design JSON (the "unified single-file" shape), DesignSpec folds them into an
# inline ``model`` so one loader handles every file shape.
_MODEL_FIELDS = ("k_factors", "factor_vols", "beta_samplers", "idio_vol_sampler", "units")


# ── Specs ─────────────────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    """The factor-model half of a study: what defines (B, F, D).

    Reusable across many designs — fix it once, vary the return process / sweep
    in different design specs against it. Sampler fields use the
    ``{"distribution": name, ...}`` shape consumed by
    :func:`factor_lab.distributions.create_sampler`.

    ``factor_vols`` and ``idio_vol_sampler`` are both in **volatility** units by
    default — they are squared into the variance matrices F and D when the model
    is built. Set ``units="variance"`` to instead pass *variances*: the values in
    ``factor_vols`` and the ``idio_vol_sampler`` draws are then treated as
    variances (square-rooted at the build boundary, so they land in F / D
    unchanged). ``units="vol"`` (the default) preserves the original behavior.

    Defaults reproduce the diagonal-Gram baseline: a market-like factor 1 with
    loadings β₁ ~ N(1, sd 0.5) and zero-mean unit factors 2,3 (β_j ~ N(0, 1)),
    giving prevalences c = E‖β_j‖²/p = mean² + sd² = [1.25, 1, 1] (off-diagonal
    Gram entries vanish since factors 2,3 are zero-mean → G∞ = I_k still holds).
    Factor vols σ = [.16, .08, .06] (variances [.0256, .0064, .0036]); constant
    idio vol 0.4 (so δ² = 0.16). Spikes d_j = c_j σ_j² = [.032, .0064, .0036]
    satisfy Assumption 3.
    """

    k_factors: int = 3
    factor_vols: list[float] = field(
        default_factory=lambda: [0.16, 0.08, 0.06]
    )
    beta_samplers: Union[list[dict], dict] = field(
        default_factory=lambda: [
            {"distribution": "normal", "loc": 1.0, "scale": 0.5},   # market-like factor 1 (c_1 = 1.25)
            {"distribution": "normal", "loc": 0.0, "scale": 1.0},
            {"distribution": "normal", "loc": 0.0, "scale": 1.0},
        ]
    )
    idio_vol_sampler: dict = field(
        default_factory=lambda: {"distribution": "constant", "value": 0.4}
    )
    # How to interpret factor_vols / idio_vol_sampler values: "vol" (default,
    # squared into F/D) or "variance" (passed straight into F/D — sqrt'd at the
    # build boundary so the downstream squaring round-trips).
    units: str = "vol"

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

    The runner never sees the reference; call :meth:`resolve_model` to get a
    concrete :class:`ModelSpec`, then hand both to
    :func:`fl_experiment_runner.run_experiment`.
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

    # Sampling topology over the p (asset) grid:
    #   "independent" — draw a fresh model + returns for every (n, p) cell (the
    #                   default; each p is statistically independent).
    #   "nested"      — per replicate, draw one superset model + returns at
    #                   p_max = max(p_values), then take each smaller p as an
    #                   asset subset of it, so p₁ ⊂ p₂ ⊂ … ⊂ p_max are the SAME
    #                   assets. Gives a clean monotone-in-p convergence curve.
    sampling: str = "independent"
    # Asset-subsample rule when sampling == "nested". "prefix" takes the first p
    # assets (consecutive from the start). Other rules (random/block) reserved.
    subsample: str = "prefix"
    # Also nest the time (n) axis (requires sampling == "nested"). When True, each
    # replicate draws ONE returns superset at n_max = max(n_values) and every n is
    # the first-n-periods prefix of it, so n₁ ⊂ n₂ ⊂ … ⊂ n_max are the SAME draw —
    # a clean monotone-in-n curve, the time analogue of the p nesting. With False
    # (the default) returns are redrawn independently for each n.
    nest_time: bool = False

    def __post_init__(self) -> None:
        # nest_time piggybacks on the nested sampler's per-replicate superset draw,
        # so it is only meaningful there. Fail loud rather than silently ignore it.
        if self.nest_time and self.sampling != "nested":
            raise ValueError(
                "nest_time=True requires sampling='nested' (it nests the n axis "
                f"on top of the p nesting); got sampling={self.sampling!r}."
            )

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

    An Experiment supplies three hooks the runner calls. None of them may draw
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


# ── Experiment base + registry ────────────────────────────────────────────────

# Required hooks every Experiment must provide (validated at subclass definition
# and at registration time, so a typo fails fast instead of mid-sweep).
_EXPERIMENT_HOOKS = ("cell_setup", "record")


class BaseExperiment:
    """Optional convenience base for :class:`Experiment` implementations.

    Inheriting is **not** required — the runner only needs the ``Experiment``
    Protocol (structural). The base just adds ergonomics:

    - a default no-op ``setup()`` (so probes that need no one-time setup can skip it);
    - ``__init_subclass__`` validation that ``cell_setup`` and ``record`` exist,
      raising ``TypeError`` at *class-definition* time rather than producing an
      ``AttributeError`` partway through a sweep.

    Composition still happens at the analysis level: ``cell_setup`` returns a list
    of analyses (each ``analyze(context) -> dict``), which a new probe mixes and
    matches — reuse existing analyses, add new ones.
    """

    #: set by :func:`register_experiment` when the class is registered by name.
    experiment_name: "Optional[str]" = None

    def setup(self) -> None:  # default no-op; override to register distances etc.
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for hook in _EXPERIMENT_HOOKS:
            if not callable(getattr(cls, hook, None)):
                raise TypeError(
                    f"{cls.__name__} must define {hook}() to be an Experiment."
                )


_EXPERIMENT_REGISTRY: "dict[str, type]" = {}


def register_experiment(name: str):
    """Class decorator: register an Experiment under ``name`` for lookup by string.

    Also fail-fast-validates that the class provides the required hooks, so a
    registry entry is always a usable Experiment. Lets a CLI / config select a
    theorem by name (``get_experiment(name)``); harmless for the single-probe
    case, paying off once there are several.

    Example::

        @register_experiment("dispersion_bias")
        class DispersionBiasExperiment(BaseExperiment): ...
    """
    def deco(cls):
        if not isinstance(cls, type):
            raise TypeError("@register_experiment decorates a class")
        for hook in _EXPERIMENT_HOOKS:
            if not callable(getattr(cls, hook, None)):
                raise TypeError(
                    f"@register_experiment({name!r}): {cls.__name__} must "
                    f"define {hook}()."
                )
        existing = _EXPERIMENT_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"experiment name {name!r} already registered to "
                f"{existing.__name__}."
            )
        _EXPERIMENT_REGISTRY[name] = cls
        cls.experiment_name = name
        return cls
    return deco


def get_experiment(name: str) -> type:
    """Return the Experiment class registered under ``name`` (else ``KeyError``)."""
    try:
        return _EXPERIMENT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no experiment registered as {name!r}; "
            f"known: {registered_experiments()}"
        ) from None


def registered_experiments() -> list[str]:
    """Names of all registered experiments, sorted."""
    return sorted(_EXPERIMENT_REGISTRY)


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


# ── Model construction (Stage 1) ──────────────────────────────────────────────


def build_model(model_spec: ModelSpec, p: int, rng: np.random.Generator):
    """Build a k-factor model from ``model_spec`` for the given p.

    Loading samplers, idio-vol sampler, and factor vols all come from the model
    spec. With ``units="vol"`` (default) ``factor_vols`` are volatilities, squared
    into F, just as the idio-vol sampler's draws are squared into D. With
    ``units="variance"`` the same values are *variances*: a sqrt at this boundary
    converts them to vols, so the downstream squaring lands them in F / D
    unchanged. Draws from ``rng`` — this is the master-RNG draw in step (1) of the
    runner's per-cell order.
    """
    if model_spec.units not in ("vol", "variance"):
        raise ValueError(
            f"units must be 'vol' or 'variance', got {model_spec.units!r}"
        )
    as_variance = model_spec.units == "variance"

    # Factor side. F holds variances either way: with "vol" inputs, square them;
    # with "variance" inputs, pass them straight through (sqrt then square = id).
    if as_variance:
        if any(float(v) < 0 for v in model_spec.factor_vols):
            raise ValueError("factor variances must be non-negative")
        factor_variances = [float(v) for v in model_spec.factor_vols]
    else:
        factor_variances = [float(v) ** 2 for v in model_spec.factor_vols]

    # Idio side. The builder squares the sampler's draws into D. With "variance"
    # inputs the draws are variances, so sqrt them first (round-tripping to the
    # intended variance); clip at 0 to keep the sqrt real.
    idio_sampler = make_one_sampler(model_spec.idio_vol_sampler, rng)
    if as_variance:
        _draw_variance = idio_sampler
        idio_sampler = lambda n: np.sqrt(np.maximum(_draw_variance(n), 0.0))

    model = FactorModelBuilder(rng=rng).build(
        p=p,
        k=model_spec.k_factors,
        beta_samplers=make_samplers(model_spec.beta_samplers, rng, model_spec.k_factors),
        idio_vol_sampler=idio_sampler,
        factor_variances=factor_variances,
    )
    logger.debug("built model: k={}, p={} (units={})", model_spec.k_factors, p, model_spec.units)
    return model


# ── Return generation (Stages 2–4) ────────────────────────────────────────────


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
    # Per-rep stage: TRACE so it stays quiet at INFO/DEBUG but is capturable.
    logger.trace("sampled returns: n={}, p={}", n, model.p)
    return SimulationContext(
        model=model,
        security_returns=sim_out["security_returns"],
        factor_returns=sim_out["factor_returns"],
        idio_returns=sim_out["idio_returns"],
    )


# ── Analysis dispatch ─────────────────────────────────────────────────────────


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
    logger.trace("ran {} analyses -> {} result keys", len(analyses), len(merged))
    return merged


# ── Output bookkeeping ────────────────────────────────────────────────────────


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
    logger.debug("allocated run dir: {}", run_dir)
    return run_dir
