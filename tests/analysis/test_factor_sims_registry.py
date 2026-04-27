"""
test_factor_sims_registry.py — Tests for MetricSpec and register_metric().

Verifies that the simulation-pipeline dispatch table is driven entirely by
_METRICS, so a registered spec propagates through _measure_one_cell and
_sample_truth_records without any further edits.
"""

import numpy as np
import pytest
from scipy.linalg import null_space

import factor_sims as _sims_module
from factor_sims import (
    MetricSpec,
    _measure_one_cell,
    _sample_truth_records,
    register_metric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_metrics():
    """Save and restore _METRICS around every test."""
    original = dict(_sims_module._METRICS)
    yield
    _sims_module._METRICS.clear()
    _sims_module._METRICS.update(original)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ortho_frame(p, k, seed=0):
    """Random orthonormal (p, k) frame."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, k + 4)))
    return Q[:, :k]


def _extend_basis(U):
    """Full (p, p) orthogonal basis with U as the first k columns."""
    N = null_space(U.T)          # orthogonal complement, shape (p, p-k)
    return np.hstack([U, N])


def _constant_sampler(target_frame):
    """Returns a sampler that always yields copies of target_frame."""
    def sampler(U_gt, radius, n, rng, Q_full=None):
        return [target_frame.copy() for _ in range(n)]
    return sampler


# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------

class TestMetricSpec:

    def test_frozen(self):
        """MetricSpec must be immutable."""
        spec = MetricSpec('x', lambda a, b: 0.0, lambda *a, **k: [])
        with pytest.raises((AttributeError, TypeError)):
            spec.name = 'y'

    def test_fields_stored(self):
        fn = lambda a, b: 1.0
        sampler = lambda *a, **k: []
        spec = MetricSpec('my-metric', fn, sampler)
        assert spec.name == 'my-metric'
        assert spec.distance_fn is fn
        assert spec.sampler is sampler


# ---------------------------------------------------------------------------
# register_metric
# ---------------------------------------------------------------------------

class TestRegisterMetric:

    def test_adds_to_metrics(self):
        spec = MetricSpec('new', lambda a, b: 0.0, lambda *a, **k: [])
        register_metric(spec)
        assert _sims_module._METRICS['new'] is spec

    def test_overwrites_existing_name(self):
        spec1 = MetricSpec('grassmann', lambda a, b: 1.0, lambda *a, **k: [])
        spec2 = MetricSpec('grassmann', lambda a, b: 2.0, lambda *a, **k: [])
        register_metric(spec1)
        register_metric(spec2)
        assert _sims_module._METRICS['grassmann'] is spec2

    def test_isolation_between_tests(self):
        """_METRICS is restored after each test (fixture sanity check)."""
        assert 'new' not in _sims_module._METRICS


# ---------------------------------------------------------------------------
# _measure_one_cell consumers
# ---------------------------------------------------------------------------

class TestMeasureOneCellConsumesRegistry:

    def test_registered_metric_produces_records(self):
        """Records for a registered metric should appear in _measure_one_cell output."""
        p, k = 20, 2
        U_gt = _ortho_frame(p, k, seed=1)
        U_sample = _ortho_frame(p, k, seed=2)
        Q_full = _extend_basis(U_gt)
        rng = np.random.default_rng(0)

        register_metric(MetricSpec(
            'test-dist',
            lambda a, b: 0.5,
            _constant_sampler(_ortho_frame(p, k, seed=3)),
        ))

        records = _measure_one_cell(
            U_gt=U_gt, U_sample=U_sample, Q_full=Q_full,
            p=p, sim=0, radius=0.5,
            num_targets=2, k=k, n=100, rng=rng,
        )

        metric_names = {r['metric'] for r in records}
        assert 'test-dist' in metric_names

    def test_distance_fn_called_per_target(self):
        """distance_fn must be invoked once per generated target."""
        p, k = 20, 2
        U_gt = _ortho_frame(p, k, seed=1)
        U_sample = _ortho_frame(p, k, seed=2)
        Q_full = _extend_basis(U_gt)
        rng = np.random.default_rng(0)
        call_log = []

        def counting_fn(a, b):
            call_log.append(1)
            return 0.0

        register_metric(MetricSpec(
            'counting',
            counting_fn,
            _constant_sampler(_ortho_frame(p, k, seed=3)),
        ))

        num_targets = 3
        _measure_one_cell(
            U_gt=U_gt, U_sample=U_sample, Q_full=Q_full,
            p=p, sim=0, radius=0.5,
            num_targets=num_targets, k=k, n=100, rng=rng,
        )

        assert len(call_log) == num_targets

    def test_record_schema(self):
        """Every record must carry the expected keys."""
        p, k = 15, 2
        U_gt = _ortho_frame(p, k)
        rng = np.random.default_rng(0)
        records = _measure_one_cell(
            U_gt=U_gt, U_sample=_ortho_frame(p, k, seed=1),
            Q_full=_extend_basis(U_gt),
            p=p, sim=0, radius=0.3,
            num_targets=1, k=k, n=50, rng=rng,
        )
        expected = {'dimension', 'p', 'n', 'radius', 'rep', 'metric', 'distance_type', 'distance'}
        for record in records:
            assert set(record.keys()) == expected


# ---------------------------------------------------------------------------
# _sample_truth_records consumers
# ---------------------------------------------------------------------------

class TestSampleTruthRecordsConsumesRegistry:

    def test_registered_metric_appears(self):
        """Newly registered metric must produce sample-truth rows."""
        p, k = 20, 2
        U_gt = _ortho_frame(p, k)
        U_sample = _ortho_frame(p, k, seed=1)

        register_metric(MetricSpec('truth-test', lambda a, b: 0.42, lambda *a, **kw: []))

        rows = _sample_truth_records(
            U_gt=U_gt, U_sample=U_sample,
            p=p, sim=0, k=k, n=100,
            target_radii=[0.1, 0.5],
        )

        names = {r['metric'] for r in rows}
        assert 'truth-test' in names

    def test_one_row_per_radius_per_metric(self):
        """sample-truth produces exactly one row per (metric, radius)."""
        p, k = 20, 2
        U_gt = _ortho_frame(p, k)
        U_sample = _ortho_frame(p, k, seed=1)
        radii = [0.1, 0.3, 0.5]

        register_metric(MetricSpec('fixed', lambda a, b: 0.777, lambda *a, **kw: []))

        rows = _sample_truth_records(
            U_gt=U_gt, U_sample=U_sample,
            p=p, sim=0, k=k, n=100,
            target_radii=radii,
        )

        fixed_rows = [r for r in rows if r['metric'] == 'fixed']
        assert len(fixed_rows) == len(radii)

    def test_distance_value_matches_fn(self):
        """The recorded distance must equal the value returned by the metric fn."""
        p, k = 20, 2
        U_gt = _ortho_frame(p, k)
        U_sample = _ortho_frame(p, k, seed=1)

        register_metric(MetricSpec('pinned', lambda a, b: 0.777, lambda *a, **kw: []))

        rows = _sample_truth_records(
            U_gt=U_gt, U_sample=U_sample,
            p=p, sim=0, k=k, n=100,
            target_radii=[0.5],
        )

        pinned_rows = [r for r in rows if r['metric'] == 'pinned']
        assert len(pinned_rows) == 1
        assert pinned_rows[0]['distance'] == pytest.approx(0.777)
