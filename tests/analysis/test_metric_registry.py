"""
test_metric_registry.py — Tests for the ManifoldDistanceAnalysis extension point.

Covers register_manifold_distance() and _EXTRA_DISTANCES.
"""

import numpy as np
import pytest

from factor_lab.analyses import manifold as _manifold_module
from factor_lab.analyses.manifold import ManifoldDistanceAnalysis, register_manifold_distance
from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_extra_distances():
    """Save and restore _EXTRA_DISTANCES around every test."""
    original = dict(_manifold_module._EXTRA_DISTANCES)
    yield
    _manifold_module._EXTRA_DISTANCES.clear()
    _manifold_module._EXTRA_DISTANCES.update(original)


def _make_context(k=2, p=20, T=50, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((k, p))
    model = FactorModelData(B=B, F=np.diag([0.1] * k), D=np.diag(np.full(p, 0.01)))
    return SimulationContext(
        model=model,
        security_returns=rng.standard_normal((T, p)),
        factor_returns=rng.standard_normal((T, k)),
        idio_returns=rng.standard_normal((T, p)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegisterManifoldDistance:

    def test_key_appears_in_results(self):
        """Registered key must be present in analyze() output."""
        register_manifold_distance('dist_always_one', lambda bt, be: 1.0)
        results = ManifoldDistanceAnalysis(use_pca_loadings=False).analyze(_make_context())
        assert 'dist_always_one' in results
        assert results['dist_always_one'] == 1.0

    def test_fn_receives_b_true_and_b_estimated(self):
        """Registered fn is called with arrays of shape (k, p)."""
        captured = {}
        def capture_fn(B_true, B_estimated):
            captured['shapes'] = (B_true.shape, B_estimated.shape)
            return 0.0

        register_manifold_distance('dist_capture', capture_fn)
        ManifoldDistanceAnalysis(use_pca_loadings=False).analyze(_make_context(k=3, p=30))

        assert captured['shapes'] == ((3, 30), (3, 30))

    def test_multiple_extras_all_appear(self):
        """All registered extras should appear, each with the correct value."""
        register_manifold_distance('dist_a', lambda bt, be: 1.0)
        register_manifold_distance('dist_b', lambda bt, be: 2.0)
        results = ManifoldDistanceAnalysis(use_pca_loadings=False).analyze(_make_context())
        assert results['dist_a'] == 1.0
        assert results['dist_b'] == 2.0

    def test_builtin_metrics_unaffected(self):
        """Registering an extra must not change the three built-in metric values."""
        ctx = _make_context()
        analysis = ManifoldDistanceAnalysis(use_pca_loadings=False)
        baseline = analysis.analyze(ctx)

        register_manifold_distance('dist_extra', lambda bt, be: 99.0)
        with_extra = analysis.analyze(ctx)

        assert with_extra['dist_grassmannian'] == baseline['dist_grassmannian']
        assert with_extra['dist_procrustes'] == baseline['dist_procrustes']
        assert with_extra['dist_chordal'] == baseline['dist_chordal']

    def test_isolation_between_tests(self):
        """_EXTRA_DISTANCES must be empty at the start of each test."""
        assert _manifold_module._EXTRA_DISTANCES == {}
