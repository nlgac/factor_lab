"""
integration.py - Integration Layer

Complete pipeline: Build Model → Simulate Returns → Estimate → Analyze.

Usage
-----
>>> from factor_lab.integration import build_simulate_analyze
>>> from factor_lab.distributions import create_sampler
>>> import numpy as np
>>>
>>> rng = np.random.default_rng(42)
>>> factory = lambda name, **p: create_sampler(name, rng, **p)
>>>
>>> results = build_simulate_analyze(
...     p=100, k=2,
...     beta_samplers=factory("normal", loc=0, scale=1),
...     idio_vol_sampler=factory("constant", value=0.03),
...     factor_variances=[0.04, 0.01],
...     n_periods=1000,
...     factor_return_samplers=factory("normal", loc=0, scale=1),
...     idio_return_sampler=factory("normal", loc=0, scale=1),
...     rng=rng,
... )
>>> print(f"Grassmannian: {results['dist_grassmannian']:.6f}")
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .decomposition import svd_decomposition
from .distributions import Sampler
from .factor_types import FactorModelData
from .flexible_simulator import ReturnsSimulator as FlexibleReturnsSimulator
from .model_builder import FactorModelBuilder

log = logging.getLogger(__name__)

__all__ = [
    'build_simulate_analyze',
    'build_simulate_analyze_from_model',
    'create_simulation_context',
    'run_analyses',
]

# ---------------------------------------------------------------------------
# Analysis dispatch table — add new analyses here, nowhere else
# ---------------------------------------------------------------------------

def _make_analysis_registry():
    """Build registry lazily to avoid circular imports at module load."""
    from .analyses import Analyses
    return {
        'manifold':    lambda ctx: Analyses.manifold_distances().analyze(ctx),
        'eigenvalue':  lambda ctx: Analyses.eigenvalue_analysis(k_top=ctx.k).analyze(ctx),
        'eigenvector': lambda ctx: Analyses.eigenvector_comparison(k=ctx.k).analyze(ctx),
    }

_ALL_ANALYSES = ['manifold', 'eigenvalue', 'eigenvector']


def _resolve_analyses(analyses: Optional[List[str]]) -> List[str]:
    if analyses is None:
        return ['manifold']
    if 'all' in analyses:
        return list(_ALL_ANALYSES)
    return analyses


# ---------------------------------------------------------------------------
# Shared pipeline tail
# ---------------------------------------------------------------------------

def _simulate_and_analyze(
    model: FactorModelData,
    n_periods: int,
    factor_return_samplers: Union[Sampler, List[Sampler]],
    idio_return_sampler: Sampler,
    analyses: List[str],
    rng: np.random.Generator,
    timestamp: datetime,
) -> Dict[str, Any]:
    """Shared tail: simulate, estimate, build context, run analyses."""
    log.debug("Simulating %d periods...", n_periods)
    sim_results = FlexibleReturnsSimulator(rng=rng).simulate(
        model=model,
        n_periods=n_periods,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler,
    )

    log.debug("Estimating model via SVD...")
    estimated_model = svd_decomposition(
        sim_results['security_returns'], k=model.k, center=True
    )

    context = create_simulation_context(model, sim_results, timestamp)
    analysis_results = run_analyses(context, analyses)

    return {
        'true_model': model,
        'estimated_model': estimated_model,
        'simulation_results': sim_results,
        'context': context,
        **analysis_results,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_simulate_analyze(
    p: int,
    k: int,
    beta_samplers: Union[Sampler, List[Sampler]],
    idio_vol_sampler: Sampler,
    factor_variances: List[float],
    n_periods: int,
    factor_return_samplers: Union[Sampler, List[Sampler]],
    idio_return_sampler: Sampler,
    analyses: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Complete pipeline: build model, simulate returns, estimate, analyze.

    Parameters
    ----------
    p : int
        Number of assets.
    k : int
        Number of factors.
    beta_samplers : Sampler or list of Sampler
        Distribution(s) for factor loadings.
    idio_vol_sampler : Sampler
        Distribution for idiosyncratic volatilities.
    factor_variances : list of float
        Variance for each factor (diagonal of F).
    n_periods : int
        Number of time periods to simulate.
    factor_return_samplers : Sampler or list of Sampler
        Distribution(s) for factor returns.
    idio_return_sampler : Sampler
        Distribution for idiosyncratic returns.
    analyses : list of str, optional
        Analyses to run: 'manifold', 'eigenvalue', 'eigenvector', or 'all'.
        Defaults to ['manifold'].
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        Keys: 'true_model', 'estimated_model', 'simulation_results',
        'context', 'duration', 'timestamp', plus analysis-specific keys.

    Example
    -------
    >>> results = build_simulate_analyze(
    ...     p=100, k=2,
    ...     beta_samplers=factory("normal"),
    ...     idio_vol_sampler=factory("constant", value=0.03),
    ...     factor_variances=[0.04, 0.01],
    ...     n_periods=500,
    ...     factor_return_samplers=factory("normal"),
    ...     idio_return_sampler=factory("normal"),
    ... )
    """
    start = time.time()
    timestamp = datetime.now()
    analyses = _resolve_analyses(analyses)
    rng = rng or np.random.default_rng()

    log.debug("Building model: p=%d, k=%d", p, k)
    model = FactorModelBuilder(rng=rng).build(
        p=p, k=k,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=factor_variances,
    )

    result = _simulate_and_analyze(
        model, n_periods, factor_return_samplers,
        idio_return_sampler, analyses, rng, timestamp,
    )
    duration = time.time() - start
    log.info("build_simulate_analyze complete in %.2fs", duration)
    return {**result, 'duration': duration, 'timestamp': timestamp}


def build_simulate_analyze_from_model(
    model: FactorModelData,
    n_periods: int,
    factor_return_samplers: Union[Sampler, List[Sampler]],
    idio_return_sampler: Sampler,
    analyses: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Simulate and analyze from an existing model (skips model-building step).

    Useful for comparing return distributions on the same model structure.

    Example
    -------
    >>> results_normal = build_simulate_analyze_from_model(
    ...     model, 1000, factory("normal"), factory("normal")
    ... )
    >>> results_t = build_simulate_analyze_from_model(
    ...     model, 1000, factory("student_t", df=5), factory("student_t", df=7)
    ... )
    """
    start = time.time()
    timestamp = datetime.now()
    analyses = _resolve_analyses(analyses)
    rng = rng or np.random.default_rng()

    result = _simulate_and_analyze(
        model, n_periods, factor_return_samplers,
        idio_return_sampler, analyses, rng, timestamp,
    )
    duration = time.time() - start
    log.info("build_simulate_analyze_from_model complete in %.2fs", duration)
    return {**result, 'duration': duration, 'timestamp': timestamp}


def create_simulation_context(
    model: FactorModelData,
    sim_results: Dict[str, np.ndarray],
    timestamp: Optional[datetime] = None,
):
    """
    Build a SimulationContext from a model and simulation results.

    Example
    -------
    >>> ctx = create_simulation_context(model, sim_results)
    >>> ctx.sample_covariance()
    """
    from .analysis import SimulationContext
    return SimulationContext(
        model=model,
        security_returns=sim_results['security_returns'],
        factor_returns=sim_results['factor_returns'],
        idio_returns=sim_results['idio_returns'],
        timestamp=timestamp or datetime.now(),
        duration=0.0,
    )


def run_analyses(
    context,
    analyses: List[str],
) -> Dict[str, Any]:
    """
    Run named analyses on a SimulationContext and merge their results.

    Parameters
    ----------
    context : SimulationContext
    analyses : list of str
        Any subset of 'manifold', 'eigenvalue', 'eigenvector'.

    Returns
    -------
    dict
        Merged results from all requested analyses.

    Raises
    ------
    ValueError
        If any analysis name is not recognised.

    Example
    -------
    >>> run_analyses(ctx, ['manifold', 'eigenvector'])
    {'dist_grassmannian': ..., 'mean_correlation': ..., ...}
    """
    registry = _make_analysis_registry()
    unknown = [a for a in analyses if a not in registry]
    if unknown:
        raise ValueError(
            f"Unknown analyses: {unknown}. Available: {sorted(registry)}"
        )
    merged: Dict[str, Any] = {}
    for name in analyses:
        log.debug("Running %s analysis...", name)
        merged.update(registry[name](context))
    return merged
