"""
test_integration.py  (tests/analysis/)

Integration tests for the full pipeline:
- build_simulate_analyze and build_simulate_analyze_from_model
- run_analyses dispatch (valid names; unknown name raises ValueError)
- create_simulation_context
- End-to-end workflow
- Context caching
- Custom analyses
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.distributions import create_sampler
from factor_lab.integration import (
    build_simulate_analyze,
    build_simulate_analyze_from_model,
    create_simulation_context,
    run_analyses,
)

# make_simulator is provided as a helper from conftest (not a fixture,
# so we import it directly)
from conftest import make_simulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(k=2, p=30, T=100, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((k, p))
    F = np.diag(rng.uniform(0.01, 0.1, k))
    D = np.diag(np.full(p, 0.01))
    model = FactorModelData(B=B, F=F, D=D)
    results = make_simulator(model, rng).simulate(T)
    return SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns'],
    )


def _factory(rng):
    return lambda name, **p: create_sampler(name, rng, **p)


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

class TestBuildSimulateAnalyze:

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        f = _factory(rng)
        results = build_simulate_analyze(
            p=30, k=2,
            beta_samplers=f("normal"),
            idio_vol_sampler=f("constant", value=0.03),
            factor_variances=[0.04, 0.01],
            n_periods=100,
            factor_return_samplers=f("normal"),
            idio_return_sampler=f("normal"),
            rng=rng,
        )
        for key in ('true_model', 'estimated_model', 'simulation_results',
                    'context', 'duration', 'timestamp', 'dist_grassmannian'):
            assert key in results, f"Missing key: {key}"

    def test_duration_is_non_negative(self):
        rng = np.random.default_rng(1)
        f = _factory(rng)
        results = build_simulate_analyze(
            p=20, k=2,
            beta_samplers=f("normal"),
            idio_vol_sampler=f("constant", value=0.03),
            factor_variances=[0.04, 0.01],
            n_periods=50,
            factor_return_samplers=f("normal"),
            idio_return_sampler=f("normal"),
            rng=rng,
        )
        assert results['duration'] >= 0

    def test_all_analyses(self):
        rng = np.random.default_rng(2)
        f = _factory(rng)
        results = build_simulate_analyze(
            p=30, k=2,
            beta_samplers=f("normal"),
            idio_vol_sampler=f("constant", value=0.03),
            factor_variances=[0.04, 0.01],
            n_periods=100,
            factor_return_samplers=f("normal"),
            idio_return_sampler=f("normal"),
            analyses=['all'],
            rng=rng,
        )
        assert 'dist_grassmannian' in results
        assert 'mean_correlation' in results

    def test_from_model_reuses_structure(self):
        rng = np.random.default_rng(3)
        f = _factory(rng)
        from factor_lab.model_builder import FactorModelBuilder
        model = FactorModelBuilder(rng=rng).build(
            p=20, k=2,
            beta_samplers=f("normal"),
            idio_vol_sampler=f("constant", value=0.03),
            factor_variances=[0.04, 0.01],
        )
        r1 = build_simulate_analyze_from_model(
            model, 80, f("normal"), f("normal"), rng=rng)
        r2 = build_simulate_analyze_from_model(
            model, 80, f("student_t", df=5), f("student_t", df=7), rng=rng)
        assert r1['true_model'] is model
        assert r2['true_model'] is model
        assert r1['dist_grassmannian'] != r2['dist_grassmannian']


# ---------------------------------------------------------------------------
# run_analyses dispatch
# ---------------------------------------------------------------------------

class TestRunAnalyses:

    def test_manifold_only(self):
        ctx = _make_context()
        results = run_analyses(ctx, ['manifold'])
        assert 'dist_grassmannian' in results
        assert 'mean_correlation' not in results

    def test_eigenvalue_only(self):
        ctx = _make_context()
        results = run_analyses(ctx, ['eigenvalue'])
        assert 'dist_grassmannian' not in results

    def test_all_three(self):
        ctx = _make_context()
        results = run_analyses(ctx, ['manifold', 'eigenvalue', 'eigenvector'])
        assert 'dist_grassmannian' in results
        assert 'mean_correlation' in results

    def test_unknown_analysis_raises(self):
        ctx = _make_context()
        with pytest.raises(ValueError, match="Unknown analyses"):
            run_analyses(ctx, ['manifold', 'bogus'])

    def test_empty_list_returns_empty(self):
        ctx = _make_context()
        assert run_analyses(ctx, []) == {}


# ---------------------------------------------------------------------------
# create_simulation_context
# ---------------------------------------------------------------------------

class TestCreateSimulationContext:

    def test_properties(self):
        rng = np.random.default_rng(0)
        k, p, T = 2, 20, 50
        B = rng.standard_normal((k, p))
        model = FactorModelData(B=B, F=np.eye(k) * 0.1, D=np.eye(p) * 0.01)
        sim_results = {
            'security_returns': rng.standard_normal((T, p)),
            'factor_returns':   rng.standard_normal((T, k)),
            'idio_returns':     rng.standard_normal((T, p)),
        }
        ctx = create_simulation_context(model, sim_results)
        assert ctx.T == T
        assert ctx.p == p
        assert ctx.k == k


# ---------------------------------------------------------------------------
# Legacy workflow (FlexibleReturnsSimulator via adapter)
# ---------------------------------------------------------------------------

class TestCompleteWorkflow:

    def test_basic_workflow(self):
        k, p, T = 2, 30, 100
        rng = np.random.default_rng(0)
        B = rng.standard_normal((k, p))
        model = FactorModelData(B=B, F=np.diag([0.1, 0.05]),
                                D=np.diag(np.full(p, 0.01)))
        results = make_simulator(model, rng).simulate(T)
        context = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns'],
        )
        manifold = Analyses.manifold_distances().analyze(context)
        eigen    = Analyses.eigenvalue_analysis(k_top=k).analyze(context)
        eigvec   = Analyses.eigenvector_comparison(k=k).analyze(context)
        assert 'dist_grassmannian' in manifold
        assert 'eigenvalue_rmse' in eigen
        assert 'mean_correlation' in eigvec
        assert 0 <= manifold['dist_grassmannian'] <= 5
        assert 0 <= eigvec['mean_correlation'] <= 1

    def test_multiple_simulations_vary(self):
        k, p, T = 2, 20, 50
        rng0 = np.random.default_rng(99)
        B = rng0.standard_normal((k, p))
        model = FactorModelData(B=B, F=np.diag([0.1, 0.05]),
                                D=np.diag(np.full(p, 0.01)))
        distances = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            results = make_simulator(model, rng).simulate(T)
            ctx = SimulationContext(
                model=model,
                security_returns=results['security_returns'],
                factor_returns=results['factor_returns'],
                idio_returns=results['idio_returns'],
            )
            distances.append(
                Analyses.manifold_distances().analyze(ctx)['dist_grassmannian']
            )
        assert len(set(distances)) > 1

    def test_large_scale(self):
        k, p, T = 5, 100, 200
        rng = np.random.default_rng(0)
        B = rng.standard_normal((k, p))
        F = np.diag(rng.uniform(0.01, 0.1, k))
        model = FactorModelData(B=B, F=F, D=np.diag(np.full(p, 0.01)))
        results = make_simulator(model, rng).simulate(T)
        ctx = SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns'],
        )
        manifold = Analyses.manifold_distances().analyze(ctx)
        eigen    = Analyses.eigenvalue_analysis(k_top=k).analyze(ctx)
        eigvec   = Analyses.eigenvector_comparison(k=k).analyze(ctx)
        assert all(k_ in manifold for k_ in
                   ['dist_grassmannian', 'dist_procrustes', 'dist_chordal'])
        assert 'eigenvalue_rmse' in eigen
        assert 'mean_correlation' in eigvec


# ---------------------------------------------------------------------------
# Context caching
# ---------------------------------------------------------------------------

class TestContextCaching:

    def test_sample_covariance_cached(self):
        ctx = _make_context()
        assert ctx.sample_covariance() is ctx.sample_covariance()

    def test_pca_decomposition_cached_by_k(self):
        ctx = _make_context(k=3, p=30, T=100)
        pca2a = ctx.pca_decomposition(n_components=2)
        pca2b = ctx.pca_decomposition(n_components=2)
        pca3  = ctx.pca_decomposition(n_components=3)
        assert pca2a.B is pca2b.B
        assert pca3.B is not pca2a.B


# ---------------------------------------------------------------------------
# Custom analyses
# ---------------------------------------------------------------------------

class TestCustomAnalyses:

    def test_custom_lambda(self):
        ctx = _make_context()
        custom = Analyses.custom(lambda c: {
            'frobenius_B': float(np.linalg.norm(c.model.B, 'fro')),
            'trace_F':     float(np.trace(c.model.F)),
        })
        results = custom.analyze(ctx)
        assert results['frobenius_B'] > 0
        assert results['trace_F'] > 0

    def test_custom_function(self):
        ctx = _make_context()
        def my_analysis(c):
            pca = c.pca_decomposition(n_components=c.model.k)
            return {'loading_error': float(np.linalg.norm(c.model.B - pca.B, 'fro'))}
        assert Analyses.custom(my_analysis).analyze(ctx)['loading_error'] >= 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_very_small_sample(self):
        ctx = _make_context(k=2, p=10, T=15)
        assert 'dist_grassmannian' in Analyses.manifold_distances().analyze(ctx)

    def test_json_round_trip(self):
        config = {"meta": {"p_assets": 50, "n_periods": 100}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            path = Path(f.name)
        try:
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['meta']['p_assets'] == 50
        finally:
            path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
