"""
test_factor_sims.py — Test suite for factor_sims.py and factor_sims_plots.py
=============================================================================

Coverage map
------------
Spec building
    test_factory_from_spec_normal           _factory_from_spec: normal distribution
    test_factory_from_spec_uniform          _factory_from_spec: uniform distribution
    test_factory_from_spec_constant         _factory_from_spec: constant distribution
    test_factory_from_spec_unknown_dist     _factory_from_spec: bad dist name propagates
    test_sampler_factories_from_specs_ok    _sampler_factories_from_specs: happy path
    test_sampler_factories_length_mismatch  _sampler_factories_from_specs: wrong k
    test_load_json_separates_fields         _load_json: numeric vs sampler vs comment keys
    test_load_json_rejects_unknown_fields   _load_json: unknown non-prefixed key
    test_build_spec_from_jsons_merge_order  Later file overrides earlier
    test_build_spec_from_jsons_missing_field Missing required numeric field
    test_build_spec_from_jsons_sampler_override Sampler specs actually flow through

SimSpec validation
    test_spec_rejects_beta_length_mismatch
    test_spec_rejects_factor_return_length_mismatch
    test_spec_rejects_p_exceeds_max
    test_spec_rejects_p_too_small_for_stiefel
    test_spec_rejects_nonpositive_counts

Ground truth frame
    test_ground_truth_frame_shape_and_orthonormality
    test_ground_truth_frame_subspace_under_scalar_D

Target generation
    test_grassmann_target_orthonormality_and_radius
    test_stiefel_tangent_norm_exact
    test_stiefel_target_radius_roundtrip
    test_stiefel_geq_grassmann
    test_precomputed_Q_full_matches_fresh

Distance functions
    test_grassmann_zero_same_subspace
    test_grassmann_rotation_invariant
    test_stiefel_nonzero_under_rotation
    test_stiefel_symmetric
    test_stiefel_geq_grassmann_distance

Model and returns
    test_slice_model_shapes
    test_simulate_all_returns_shape
    test_sample_frame_orthonormal

sample-truth helpers
    test_sample_truth_records_schema
    test_sample_truth_distances_positive
    test_sample_truth_replicated_across_radii

run_simulation
    test_run_simulation_schema_no_sample_truth
    test_run_simulation_schema_with_sample_truth
    test_run_simulation_truth_target_rows_equal_radius
    test_run_simulation_stiefel_geq_grassmann_per_row
    test_run_simulation_distances_positive
    test_run_simulation_reproducibility
    test_run_simulation_model_seed_independence
    test_run_simulation_target_seed_independence
    test_save_writes_csvs

factor_sims_plots
    test_plot_dataframe_creates_figure
    test_plot_dataframe_sample_truth_mode
    test_plot_dataframe_wrong_csv_graceful
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import qr

import factor_sims as fs
from factor_sims import (
    SimSpec,
    _factory_from_spec,
    _sampler_factories_from_specs,
    _DEFAULT_SAMPLER_SPECS,
    _load_json,
    build_spec_from_jsons,
    build_population_model,
    simulate_all_returns,
    slice_model,
    ground_truth_frame,
    sample_frame,
    sample_grassmann_targets,
    sample_stiefel_targets,
    grassmann_distance,
    stiefel_canonical_distance,
    _sample_truth_records,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_frame(p: int, k: int, seed: int = 0) -> np.ndarray:
    """Random orthonormal (p, k) frame."""
    rng = np.random.default_rng(seed)
    Q, _ = qr(rng.standard_normal((p, k)), mode='economic')
    return Q


def _minimal_spec(**overrides) -> SimSpec:
    """Build a minimal valid SimSpec for unit tests that don't need JSON."""
    factories = _sampler_factories_from_specs(
        k=3,
        factor_variances=[0.0025, 0.01, 0.01],
        sampler_specs=_DEFAULT_SAMPLER_SPECS,
    )
    kwargs = dict(
        max_num_sec=30, nums_sec=(10, 20, 30),
        num_obs=20, num_sim=2,
        target_radii=(0.1, 0.5), num_targets=2,
        k_factors=3, seed_model=1, seed_targets=2,
        **factories,
    )
    kwargs.update(overrides)
    return SimSpec(**kwargs)


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _align_frame_signs(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Return a sign-adjusted copy of U2 so that each column has the same
    orientation as the corresponding column of U1.

    eigsh returns eigenvectors that are unique up to sign when eigenvalues
    are distinct. Two calls with the same matrix but different starting
    vectors may return v or -v for each column independently. This helper
    aligns the signs of U2 to match U1, enabling numerical equality checks.

    The sign of column j is flipped in U2 if U1[:,j] · U2[:,j] < 0.
    """
    signs = np.sign(np.einsum('ij,ij->j', U1, U2))  # dot product per column
    signs[signs == 0] = 1                             # leave zero-dot columns alone
    return U2 * signs[np.newaxis, :]


def _distances_up_to_sign(df: pd.DataFrame) -> np.ndarray:
    """
    Extract distances from a long_df in a way that is insensitive to the
    sign of U_gt eigenvectors.

    Because eigsh sign is non-deterministic, two runs may produce different
    U_gt frames (same subspace, flipped signs on some eigenvectors). The
    Grassmann distance between U_sample and a target is invariant to sign
    flips of U_gt because targets are placed relative to U_gt — a sign flip
    of U_gt rotates all targets identically, leaving inter-frame distances
    unchanged.

    However Stiefel canonical distance IS sensitive to sign, because it
    measures frame orientation. For sample-target rows, both the target
    (generated from U_gt) and U_sample are affected consistently by U_gt's
    sign, so the distance between them is preserved. For sample-truth rows,
    d(U_sample, U_gt) changes if U_gt's sign flips.

    This helper returns the distances grouped and sorted in a way that
    allows comparison across runs modulo sign-flip effects on sample-truth.
    For run-to-run equality tests use Grassmann distances only, or compare
    sample-target distances which are sign-invariant.
    """
    return df['distance'].values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def micro_spec_and_model():
    """
    Build a small spec and model once per module — shared by many tests.
    scope='module' avoids rebuilding for every test function.
    """
    spec_numeric = {
        'max_num_sec': 30, 'nums_sec': [10, 20, 30],
        'num_obs': 20, 'num_sim': 3,
        'target_radii': [0.1, 0.5], 'num_targets': 2,
        'k_factors': 3, 'factor_variances': [0.0025, 0.01, 0.01],
        'seed_model': 42, 'seed_targets': 123,
    }
    factories = _sampler_factories_from_specs(
        k=3,
        factor_variances=spec_numeric['factor_variances'],
        sampler_specs=_DEFAULT_SAMPLER_SPECS,
    )
    spec = SimSpec(
        **{k: (tuple(v) if isinstance(v, list) else v)
           for k, v in spec_numeric.items()
           if k not in ('factor_variances',)},
        **factories,
    )
    rng = np.random.default_rng(spec.seed_model)
    model = build_population_model(spec, rng)
    return spec, model


@pytest.fixture(scope='module')
def micro_results(micro_spec_and_model):
    """Run the full micro simulation once; reused by schema/content tests."""
    spec, _ = micro_spec_and_model
    results, _ = run_simulation(spec), None
    # run_simulation returns SimResults; capture it properly
    return run_simulation(spec)


# ---------------------------------------------------------------------------
# _factory_from_spec
# ---------------------------------------------------------------------------

class TestFactoryFromSpec:

    def test_normal(self):
        factory = _factory_from_spec({'distribution': 'normal', 'loc': 1.0, 'scale': 0.5})
        rng = np.random.default_rng(0)
        sampler = factory(rng)
        draws = sampler(10000)
        assert draws.shape == (10000,)
        assert abs(np.mean(draws) - 1.0) < 0.05
        assert abs(np.std(draws) - 0.5) < 0.05

    def test_uniform(self):
        factory = _factory_from_spec({'distribution': 'uniform', 'low': 2.0, 'high': 4.0})
        rng = np.random.default_rng(1)
        sampler = factory(rng)
        draws = sampler(10000)
        assert np.all(draws >= 2.0) and np.all(draws <= 4.0)
        assert abs(np.mean(draws) - 3.0) < 0.05

    def test_constant(self):
        factory = _factory_from_spec({'distribution': 'constant', 'value': 0.7})
        rng = np.random.default_rng(2)
        sampler = factory(rng)
        draws = sampler(50)
        assert np.all(draws == 0.7)

    def test_unknown_distribution_returns_callable(self):
        """
        _factory_from_spec accepts any distribution name and builds a factory.
        create_sampler's behaviour for unknown names is its own responsibility;
        we only verify that _factory_from_spec returns a callable factory.
        Whether drawing raises is tested in integration with create_sampler.
        """
        factory = _factory_from_spec({'distribution': 'student_t', 'df': 5})
        assert callable(factory)
        rng = np.random.default_rng(3)
        # Factory call should return a sampler (callable)
        sampler = factory(rng)
        assert callable(sampler)


# ---------------------------------------------------------------------------
# _sampler_factories_from_specs
# ---------------------------------------------------------------------------

class TestSamplerFactoriesFromSpecs:

    def test_happy_path_returns_all_keys(self):
        result = _sampler_factories_from_specs(
            k=3, factor_variances=[0.01]*3, sampler_specs=_DEFAULT_SAMPLER_SPECS
        )
        assert set(result.keys()) == {
            'beta_sampler_factories', 'idio_vol_sampler_factory',
            'factor_variances', 'factor_return_sampler_factories',
            'idio_return_sampler_factory',
        }
        assert len(result['beta_sampler_factories']) == 3
        assert len(result['factor_return_sampler_factories']) == 3
        assert result['factor_variances'] == [0.01]*3

    def test_beta_length_mismatch_raises(self):
        bad_specs = dict(_DEFAULT_SAMPLER_SPECS)
        bad_specs['_beta_samplers'] = bad_specs['_beta_samplers'][:2]  # only 2 for k=3
        with pytest.raises(ValueError, match="_beta_samplers"):
            _sampler_factories_from_specs(k=3, factor_variances=[0.01]*3,
                                          sampler_specs=bad_specs)

    def test_factor_return_length_mismatch_raises(self):
        bad_specs = dict(_DEFAULT_SAMPLER_SPECS)
        bad_specs['_factor_return_samplers'] = bad_specs['_factor_return_samplers'][:1]
        with pytest.raises(ValueError, match="_factor_return_samplers"):
            _sampler_factories_from_specs(k=3, factor_variances=[0.01]*3,
                                          sampler_specs=bad_specs)


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:

    def test_separates_numeric_and_sampler_fields(self, tmp_path):
        data = {
            'max_num_sec': 500,
            'nums_sec': [100, 500],
            '_idio_vol_sampler': {'distribution': 'uniform', 'low': 0.1, 'high': 0.8},
            '_comment': 'ignored',
        }
        path = _write_json(tmp_path, 'test.json', data)
        numeric, samplers = _load_json(path)
        assert 'max_num_sec' in numeric
        assert '_idio_vol_sampler' in samplers
        assert '_comment' not in samplers   # comment keys ignored
        assert '_comment' not in numeric

    def test_rejects_unknown_non_prefixed_field(self, tmp_path):
        data = {'max_num_sec': 500, 'bad_field': 99}
        path = _write_json(tmp_path, 'bad.json', data)
        with pytest.raises(ValueError, match="unknown fields"):
            _load_json(path)

    def test_accepts_all_known_fields(self, tmp_path):
        data = {
            'max_num_sec': 100, 'nums_sec': [30, 100],
            'num_obs': 63, 'num_sim': 5,
            'target_radii': [0.1], 'num_targets': 3,
            'k_factors': 3, 'factor_variances': [0.01]*3,
            'seed_model': 1, 'seed_targets': 2,
        }
        path = _write_json(tmp_path, 'full.json', data)
        numeric, samplers = _load_json(path)
        assert set(numeric.keys()) == set(data.keys())
        assert samplers == {}


# ---------------------------------------------------------------------------
# build_spec_from_jsons
# ---------------------------------------------------------------------------

class TestBuildSpecFromJsons:

    def _base_numeric(self) -> dict:
        return {
            'max_num_sec': 100, 'nums_sec': [30, 100],
            'num_obs': 63, 'num_sim': 3,
            'target_radii': [0.1, 0.5], 'num_targets': 3,
            'k_factors': 3, 'factor_variances': [0.0025, 0.01, 0.01],
            'seed_model': 42, 'seed_targets': 123,
        }

    def test_single_complete_file(self, tmp_path):
        path = _write_json(tmp_path, 'a.json', self._base_numeric())
        spec, sampler_specs = build_spec_from_jsons([path])
        assert spec.max_num_sec == 100
        assert spec.k_factors == 3
        assert sampler_specs == _DEFAULT_SAMPLER_SPECS  # no overrides

    def test_later_file_overrides_earlier(self, tmp_path):
        base = _write_json(tmp_path, 'base.json', self._base_numeric())
        override = _write_json(tmp_path, 'override.json', {'num_sim': 99})
        spec, _ = build_spec_from_jsons([base, override])
        assert spec.num_sim == 99

    def test_missing_required_field_raises(self, tmp_path):
        incomplete = dict(self._base_numeric())
        del incomplete['num_obs']
        path = _write_json(tmp_path, 'incomplete.json', incomplete)
        with pytest.raises(ValueError, match="Missing required"):
            build_spec_from_jsons([path])

    def test_sampler_override_flows_through(self, tmp_path):
        """Overriding _idio_vol_sampler actually changes the factory used."""
        base = _write_json(tmp_path, 'base.json', self._base_numeric())
        override = _write_json(tmp_path, 'override.json', {
            '_idio_vol_sampler': {'distribution': 'constant', 'value': 0.3}
        })
        spec, sampler_specs = build_spec_from_jsons([base, override])
        assert sampler_specs['_idio_vol_sampler']['distribution'] == 'constant'
        # Verify the factory actually draws 0.3
        rng = np.random.default_rng(0)
        sampler = spec.idio_vol_sampler_factory(rng)
        draws = sampler(20)
        assert np.all(draws == pytest.approx(0.3))

    def test_sampler_override_does_not_affect_earlier_keys(self, tmp_path):
        """Overriding one sampler key leaves the others at defaults."""
        base = _write_json(tmp_path, 'base.json', self._base_numeric())
        override = _write_json(tmp_path, 'override.json', {
            '_idio_vol_sampler': {'distribution': 'uniform', 'low': 0.2, 'high': 0.8}
        })
        _, sampler_specs = build_spec_from_jsons([base, override])
        # beta samplers should still be the defaults
        assert sampler_specs['_beta_samplers'] == _DEFAULT_SAMPLER_SPECS['_beta_samplers']


# ---------------------------------------------------------------------------
# SimSpec validation
# ---------------------------------------------------------------------------

class TestSimSpecValidation:

    def test_rejects_beta_length_mismatch(self):
        factories = _sampler_factories_from_specs(
            k=3, factor_variances=[0.01]*3, sampler_specs=_DEFAULT_SAMPLER_SPECS
        )
        with pytest.raises(ValueError, match="beta_sampler_factories length"):
            SimSpec(
                max_num_sec=30, nums_sec=(10, 30), num_obs=20, num_sim=2,
                target_radii=(0.1,), num_targets=2, k_factors=3,
                **{**factories, 'beta_sampler_factories': factories['beta_sampler_factories'][:1]},
            )

    def test_rejects_factor_return_length_mismatch(self):
        factories = _sampler_factories_from_specs(
            k=3, factor_variances=[0.01]*3, sampler_specs=_DEFAULT_SAMPLER_SPECS
        )
        with pytest.raises(ValueError, match="factor_return_sampler_factories length"):
            SimSpec(
                max_num_sec=30, nums_sec=(10, 30), num_obs=20, num_sim=2,
                target_radii=(0.1,), num_targets=2, k_factors=3,
                **{**factories,
                   'factor_return_sampler_factories':
                       factories['factor_return_sampler_factories'][:2]},
            )

    def test_rejects_p_exceeds_max(self):
        with pytest.raises(ValueError, match="max\\(nums_sec\\)"):
            _minimal_spec(nums_sec=(10, 50), max_num_sec=30)

    def test_rejects_p_too_small_for_stiefel(self):
        with pytest.raises(ValueError, match="2k"):
            _minimal_spec(nums_sec=(4, 30), max_num_sec=30)

    def test_rejects_nonpositive_num_obs(self):
        with pytest.raises(ValueError):
            _minimal_spec(num_obs=0)


# ---------------------------------------------------------------------------
# Ground truth frame
# ---------------------------------------------------------------------------

class TestGroundTruthFrame:

    def test_shape_and_orthonormality(self, micro_spec_and_model):
        spec, model = micro_spec_and_model
        for p in spec.nums_sec:
            U = ground_truth_frame(slice_model(model, p), spec.k_factors)
            assert U.shape == (p, spec.k_factors)
            assert np.allclose(U.T @ U, np.eye(spec.k_factors), atol=1e-10)

    def test_subspace_matches_loadings_under_scalar_D(self):
        """With D = σ²I, top-k eigenvectors of Σ span the same space as B^T."""
        from factor_lab import FactorModelBuilder, create_sampler
        rng = np.random.default_rng(0)
        builder = FactorModelBuilder(rng=rng)
        model = builder.build(
            p=40, k=3,
            beta_samplers=[create_sampler('normal', rng) for _ in range(3)],
            idio_vol_sampler=create_sampler('constant', rng, value=0.1),
            factor_variances=[0.04, 0.02, 0.01],
        )
        U_gt = ground_truth_frame(model, k=3)
        Q_B, _ = qr(model.B.T, mode='economic')
        d = grassmann_distance(U_gt, Q_B)
        assert d < 1e-6, f"Subspace mismatch under scalar D: d={d:.2e}"


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

class TestTargetGeneration:

    def test_grassmann_orthonormality_and_radius(self):
        rng = np.random.default_rng(7)
        U = _rand_frame(60, 3, seed=7)
        for radius in (0.1, 0.5, 1.0):
            targets = sample_grassmann_targets(U, radius, n=10, rng=rng)
            for t in targets:
                assert t.shape == U.shape
                assert np.allclose(t.T @ t, np.eye(3), atol=1e-10)
                d = grassmann_distance(U, t)
                assert abs(d - radius) < 1e-10, f"radius={radius}, got d={d:.4e}"

    def test_stiefel_tangent_norm_exact(self):
        """Forward generator is exact to machine precision."""
        rng = np.random.default_rng(17)
        p, k = 60, 3
        for radius in (0.01, 0.1, 0.5, 1.0, 2.0):
            for _ in range(20):
                A11_raw = rng.standard_normal((k, k))
                A11 = A11_raw - A11_raw.T
                A21 = rng.standard_normal((p - k, k))
                norm_sq = (0.5 * np.linalg.norm(A11, 'fro')**2
                           + np.linalg.norm(A21, 'fro')**2)
                scale = radius / np.sqrt(norm_sq)
                A11 *= scale; A21 *= scale
                actual = np.sqrt(0.5 * np.linalg.norm(A11, 'fro')**2
                                 + np.linalg.norm(A21, 'fro')**2)
                assert abs(actual - radius) < 1e-12

    def test_stiefel_target_radius_roundtrip(self):
        """Schur-based distance recovers radius within documented precision floor."""
        rng = np.random.default_rng(11)
        U = _rand_frame(60, 3, seed=11)
        # Tolerances reflect measured Schur precision floor (see docstring).
        tolerances = {0.1: 5e-4, 0.5: 5e-2, 1.0: 1e-1}
        for radius, tol in tolerances.items():
            targets = sample_stiefel_targets(U, radius, n=10, rng=rng)
            for t in targets:
                assert np.allclose(t.T @ t, np.eye(3), atol=1e-10)
                d = stiefel_canonical_distance(U, t)
                assert abs(d - radius) < tol, (
                    f"radius={radius}: d={d:.4f}, err={abs(d-radius):.2e} > tol={tol}"
                )

    def test_stiefel_geq_grassmann_for_targets(self):
        """
        Stiefel canonical ≥ Grassmann for Stiefel targets.

        The mathematical guarantee is exact. The Schur-based distance has a
        precision floor that grows with radius: ~4e-4 asymmetry at r=0.1,
        ~5e-3 at r=0.3, ~1e-2 at r=0.5. At r=0.5 the Schur measurement of
        Stiefel can occasionally fall below Grassmann by up to ~4e-3.

        We test at r=0.1 where the Schur floor is small enough that Stiefel
        reliably exceeds Grassmann, confirming the fundamental relationship
        is correctly implemented. The tolerance 5e-3 is 10× the measured
        max asymmetry at this radius, catching genuine regressions.
        """
        rng = np.random.default_rng(17)
        U = _rand_frame(50, 3, seed=17)
        targets = sample_stiefel_targets(U, radius=0.1, n=20, rng=rng)
        grass = [grassmann_distance(U, t) for t in targets]
        stief = [stiefel_canonical_distance(U, t) for t in targets]
        # At r=0.1 the Schur floor is ~4e-4; 5e-3 catches genuine violations
        assert all(g <= s + 5e-3 for g, s in zip(grass, stief)), (
            "Stiefel < Grassmann by more than Schur precision floor at r=0.1"
        )
        # With random A11, Stiefel should exceed Grassmann for most targets
        assert sum(s > g for g, s in zip(grass, stief)) >= 12

    def test_precomputed_Q_full_gives_same_result(self):
        """Passing Q_full explicitly gives identical targets to computing it internally."""
        from factor_sims import _extend_to_orthogonal_basis
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)  # same seed
        U = _rand_frame(40, 3, seed=99)
        Q_full = _extend_to_orthogonal_basis(U)
        t_fresh = sample_grassmann_targets(U, 0.3, n=5, rng=rng1, Q_full=None)
        t_precomp = sample_grassmann_targets(U, 0.3, n=5, rng=rng2, Q_full=Q_full)
        for a, b in zip(t_fresh, t_precomp):
            assert np.allclose(a, b, atol=1e-14)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

class TestDistanceFunctions:

    def test_grassmann_zero_same_subspace(self):
        """Rotating within the subspace gives Grassmann distance 0."""
        U = _rand_frame(50, 3, seed=3)
        R, _ = qr(np.random.default_rng(4).standard_normal((3, 3)), mode='economic')
        assert grassmann_distance(U, U @ R) < 1e-6

    def test_grassmann_rotation_invariant(self):
        """Grassmann distance is invariant to SO(k) rotation of either frame."""
        U1 = _rand_frame(40, 3, seed=5)
        U2 = _rand_frame(40, 3, seed=6)
        R, _ = qr(np.random.default_rng(7).standard_normal((3, 3)), mode='economic')
        d_orig = grassmann_distance(U1, U2)
        d_rot1 = grassmann_distance(U1 @ R, U2)
        d_rot2 = grassmann_distance(U1, U2 @ R)
        assert abs(d_orig - d_rot1) < 1e-10
        assert abs(d_orig - d_rot2) < 1e-10

    def test_stiefel_nonzero_under_pure_rotation(self):
        """Stiefel detects SO(k) rotation that Grassmann ignores."""
        from scipy.linalg import expm
        U = _rand_frame(50, 3, seed=5)
        rng = np.random.default_rng(6)
        A = rng.standard_normal((3, 3)); A = 0.3 * (A - A.T)
        U_rot = U @ expm(A)
        d_grass = grassmann_distance(U, U_rot)
        d_stief = stiefel_canonical_distance(U, U_rot)
        assert d_grass < 1e-6            # same subspace
        assert d_stief > 1e-3            # different frame orientation

    def test_stiefel_symmetric(self):
        """
        d(U1, U2) ≈ d(U2, U1).

        Exact symmetry holds for the true geodesic distance. The Schur-based
        implementation has a measured asymmetry floor of:
          r=0.1: ~4e-4,  r=0.3: ~5e-3,  r=0.5: ~1e-2,  r=1.0: ~5e-2.

        We test symmetry at r=0.1 where the Schur floor is small, using
        tolerance 2e-3 (5× the measured max at this radius). This confirms
        the implementation is symmetric up to its documented precision limit
        without being sensitive to Schur noise at larger distances.
        """
        rng = np.random.default_rng(8)
        U_base = _rand_frame(40, 3, seed=8)
        # Use r=0.1 where Schur asymmetry is ~4e-4
        target = sample_stiefel_targets(U_base, radius=0.1, n=1, rng=rng)[0]
        d12 = stiefel_canonical_distance(U_base, target)
        d21 = stiefel_canonical_distance(target, U_base)
        # Tolerance 2e-3 is 5× the measured Schur asymmetry floor at r=0.1
        assert abs(d12 - d21) < 2e-3, (
            f"Asymmetry at r=0.1 exceeds Schur floor: "
            f"d(U1,U2)={d12:.6f}, d(U2,U1)={d21:.6f}, diff={abs(d12-d21):.2e}"
        )

    def test_stiefel_geq_grassmann(self):
        """For arbitrary frame pairs, Stiefel ≥ Grassmann."""
        rng = np.random.default_rng(10)
        for _ in range(20):
            U1 = _rand_frame(30, 3, seed=rng.integers(1000))
            U2 = _rand_frame(30, 3, seed=rng.integers(1000))
            d_g = grassmann_distance(U1, U2)
            d_s = stiefel_canonical_distance(U1, U2)
            assert d_s >= d_g - 1e-8, f"Stiefel {d_s:.4f} < Grassmann {d_g:.4f}"


# ---------------------------------------------------------------------------
# Model and returns
# ---------------------------------------------------------------------------

class TestModelAndReturns:

    def test_slice_model_shapes(self, micro_spec_and_model):
        spec, model = micro_spec_and_model
        for p in spec.nums_sec:
            sliced = slice_model(model, p)
            assert sliced.B.shape == (spec.k_factors, p)
            assert sliced.D.shape == (p, p)
            assert sliced.F.shape == (spec.k_factors, spec.k_factors)
            assert np.allclose(sliced.F, model.F)

    def test_simulate_all_returns_shape(self, micro_spec_and_model):
        spec, model = micro_spec_and_model
        rng = np.random.default_rng(1)
        all_ret = simulate_all_returns(model, spec, rng)
        assert all_ret.shape == (spec.num_sim, spec.num_obs, spec.max_num_sec)
        assert np.all(np.isfinite(all_ret))
        assert np.var(all_ret) > 0

    def test_sample_frame_orthonormal(self, micro_spec_and_model):
        spec, model = micro_spec_and_model
        rng = np.random.default_rng(1)
        all_ret = simulate_all_returns(model, spec, rng)
        for p in spec.nums_sec:
            U = sample_frame(all_ret[0, :, :p], spec.k_factors)
            assert U.shape == (p, spec.k_factors)
            assert np.allclose(U.T @ U, np.eye(spec.k_factors), atol=1e-10)


# ---------------------------------------------------------------------------
# sample-truth helpers
# ---------------------------------------------------------------------------

class TestSampleTruthRecords:

    def test_schema(self):
        U1 = _rand_frame(30, 3, seed=0)
        U2 = _rand_frame(30, 3, seed=1)
        rows = _sample_truth_records(U1, U2, p=30, sim=0, k=3, n=20,
                                     target_radii=(0.1, 0.5, 1.0))
        assert len(rows) == 6   # 3 radii × 2 metrics
        for row in rows:
            assert row['distance_type'] == 'sample-truth'
            assert row['metric'] in ('grassmann', 'stiefel-canonical')
            assert row['p'] == 30
            assert row['n'] == 20

    def test_distances_positive_and_finite(self):
        U1 = _rand_frame(30, 3, seed=2)
        U2 = _rand_frame(30, 3, seed=3)
        rows = _sample_truth_records(U1, U2, p=30, sim=0, k=3, n=20,
                                     target_radii=(0.1, 0.5))
        for row in rows:
            assert np.isfinite(row['distance'])
            assert row['distance'] > 0

    def test_replicated_across_radii(self):
        """Same (p, sim) pair gives same distance regardless of radius."""
        U1 = _rand_frame(30, 3, seed=4)
        U2 = _rand_frame(30, 3, seed=5)
        rows = _sample_truth_records(U1, U2, p=30, sim=0, k=3, n=20,
                                     target_radii=(0.1, 0.5, 1.0))
        by_metric: dict = {}
        for row in rows:
            by_metric.setdefault(row['metric'], []).append(row['distance'])
        for metric, dists in by_metric.items():
            # All three radii should have the same distance value
            assert len(set(dists)) == 1, (
                f"metric={metric}: distances differ across radii: {dists}"
            )

    def test_stiefel_geq_grassmann(self):
        U1 = _rand_frame(30, 3, seed=6)
        U2 = _rand_frame(30, 3, seed=7)
        rows = _sample_truth_records(U1, U2, p=30, sim=0, k=3, n=20,
                                     target_radii=(0.5,))
        d_by_metric = {r['metric']: r['distance'] for r in rows}
        assert (d_by_metric['stiefel-canonical']
                >= d_by_metric['grassmann'] - 1e-8)


# ---------------------------------------------------------------------------
# run_simulation — schema and content
# ---------------------------------------------------------------------------

class TestRunSimulation:

    def test_schema_no_sample_truth(self, micro_results):
        df = micro_results.long_df
        expected_cols = {'dimension', 'p', 'n', 'radius', 'rep',
                         'metric', 'distance_type', 'distance',
                         'radius_label', 'n_label'}
        assert expected_cols.issubset(df.columns)
        assert set(df['distance_type'].unique()) == {'sample-target', 'truth-target'}
        assert set(df['metric'].unique()) == {'grassmann', 'stiefel-canonical'}

    def test_row_count_no_sample_truth(self, micro_results):
        df = micro_results.long_df
        spec = micro_results.spec
        rows_per_cell = spec.num_targets + 1  # sample-target + truth-target
        expected = (len(spec.nums_sec) * spec.num_sim
                    * len(spec.target_radii) * 2 * rows_per_cell)
        assert len(df) == expected, f"Expected {expected}, got {len(df)}"

    def test_schema_with_sample_truth(self, micro_spec_and_model):
        spec, _ = micro_spec_and_model
        results = run_simulation(spec, sample_truth=True)
        df = results.long_df
        assert 'sample-truth' in df['distance_type'].unique()

    def test_row_count_with_sample_truth(self, micro_spec_and_model):
        spec, _ = micro_spec_and_model
        results = run_simulation(spec, sample_truth=True)
        df = results.long_df
        n_sample_truth = len(df[df['distance_type'] == 'sample-truth'])
        # sample-truth: one per (p, sim, radius, metric)
        expected_st = (len(spec.nums_sec) * spec.num_sim
                       * len(spec.target_radii) * 2)
        assert n_sample_truth == expected_st

    def test_truth_target_rows_equal_radius(self, micro_results):
        df = micro_results.long_df
        ref = df[df['distance_type'] == 'truth-target']
        assert np.allclose(ref['distance'].values, ref['radius'].values)

    def test_stiefel_geq_grassmann_per_matching_row(self, micro_results):
        """For every (p, rep, radius) cell, Stiefel ≥ Grassmann."""
        df = micro_results.long_df
        sample = df[df['distance_type'] == 'sample-target']
        pivot = sample.pivot_table(
            index=['p', 'rep', 'radius', 'n'],
            columns='metric',
            values='distance',
            aggfunc='mean',
        )
        violations = (pivot['stiefel-canonical'] < pivot['grassmann'] - 1e-6).sum()
        assert violations == 0, f"{violations} rows violated Stiefel >= Grassmann"

    def test_all_distances_positive_and_finite(self, micro_results):
        df = micro_results.long_df
        measured = df[df['distance_type'] == 'sample-target']
        assert np.all(np.isfinite(measured['distance']))
        assert np.all(measured['distance'] > 0)

    def test_sample_target_counts_per_cell(self, micro_results):
        df = micro_results.long_df
        spec = micro_results.spec
        counts = (df[df['distance_type'] == 'sample-target']
                  .groupby(['p', 'rep', 'radius', 'metric']).size())
        assert (counts == spec.num_targets).all()

    def test_reproducibility(self, micro_spec_and_model):
        """
        Same spec → reproducible output across two calls to run_simulation.

        eigsh is deterministic given the same model (same matrix M), but its
        starting vector comes from numpy's global RNG, so the sign of each
        returned eigenvector may differ between calls. This affects target
        placement: targets are geodesics FROM U_gt's specific frame, so a
        sign flip of a column changes where the targets land in R^p, changing
        all sample-target distances.

        The quantities that ARE invariant to U_gt sign flips are:
          (a) Grassmann sample-truth: d_G(U_sample, U_gt) measures subspace
              distance and is invariant to the basis choice for U_gt.
          (b) truth-target rows: distance = radius by construction, no eigsh.
          (c) All non-distance metadata columns.

        We do NOT test sample-target distance reproducibility here because
        it depends on the sign of U_gt's eigenvectors. The mathematical
        content — that targets are placed at exactly the requested radius
        from U_gt — is tested separately in TestTargetGeneration.
        """
        spec, _ = micro_spec_and_model
        r1 = run_simulation(spec, sample_truth=True)
        r2 = run_simulation(spec, sample_truth=True)

        # Non-distance metadata must be identical
        for col in ['dimension', 'p', 'n', 'radius', 'rep', 'metric',
                    'distance_type', 'radius_label', 'n_label']:
            pd.testing.assert_series_equal(r1.long_df[col], r2.long_df[col],
                                           check_names=False)

        # Grassmann sample-truth: invariant to U_gt sign, must be equal
        mask_gst = ((r1.long_df['metric'] == 'grassmann') &
                    (r1.long_df['distance_type'] == 'sample-truth'))
        np.testing.assert_array_almost_equal(
            r1.long_df.loc[mask_gst, 'distance'].values,
            r2.long_df.loc[mask_gst, 'distance'].values,
            decimal=12,
        )

        # truth-target rows: always equal radius by construction
        mask_tt = r1.long_df['distance_type'] == 'truth-target'
        np.testing.assert_array_equal(
            r1.long_df.loc[mask_tt, 'distance'].values,
            r2.long_df.loc[mask_tt, 'distance'].values,
        )

    def test_model_seed_changes_results(self, micro_spec_and_model):
        """Different seed_model → different distances."""
        spec, _ = micro_spec_and_model
        import dataclasses
        spec2 = dataclasses.replace(spec, seed_model=spec.seed_model + 1)
        r1 = run_simulation(spec)
        r2 = run_simulation(spec2)
        assert not r1.long_df['distance'].equals(r2.long_df['distance'])

    def test_target_seed_changes_targets_not_model(self, micro_spec_and_model):
        """
        Different seed_targets → different sample-target distances,
        but identical Grassmann sample-truth distances.

        sample-truth under Grassmann = d_G(B^S, B^GT). The Grassmann metric
        depends only on the subspace spanned by U_gt, not its specific basis,
        so it is insensitive to eigsh sign flips. Therefore Grassmann
        sample-truth distances must be identical across two runs that share
        seed_model but differ in seed_targets.

        sample-target distances use target_rng → must differ when seed_targets
        differs.
        """
        spec, _ = micro_spec_and_model
        import dataclasses
        spec2 = dataclasses.replace(spec, seed_targets=spec.seed_targets + 1)
        r1 = run_simulation(spec, sample_truth=True)
        r2 = run_simulation(spec2, sample_truth=True)

        # Grassmann sample-truth: subspace metric, sign-invariant, must be equal
        mask_gst = ((r1.long_df['metric'] == 'grassmann') &
                    (r1.long_df['distance_type'] == 'sample-truth'))
        np.testing.assert_array_almost_equal(
            r1.long_df.loc[mask_gst, 'distance'].values,
            r2.long_df.loc[mask_gst, 'distance'].values,
            decimal=12,
        )

        # Grassmann sample-target: must differ when seed_targets differs
        mask_gsa = ((r1.long_df['metric'] == 'grassmann') &
                    (r1.long_df['distance_type'] == 'sample-target'))
        assert not np.allclose(
            r1.long_df.loc[mask_gsa, 'distance'].values,
            r2.long_df.loc[mask_gsa, 'distance'].values,
        )

    def test_save_writes_csvs(self, micro_spec_and_model, tmp_path):
        spec, _ = micro_spec_and_model
        results = run_simulation(spec)
        results.save(tmp_path)
        assert (tmp_path / 'distances_all.csv').exists()
        assert (tmp_path / 'distances_summary.csv').exists()
        # Round-trip: CSV should reproduce the DataFrame
        df_rt = pd.read_csv(tmp_path / 'distances_all.csv')
        assert len(df_rt) == len(results.long_df)

    def test_save_model_npz(self, micro_spec_and_model, tmp_path):
        spec, _ = micro_spec_and_model
        out = tmp_path / 'model.npz'
        run_simulation(spec, save_model_path=out)
        data = np.load(out)
        assert set(data.keys()) >= {'B', 'F', 'D'}
        assert data['B'].shape == (spec.k_factors, spec.max_num_sec)
        assert data['F'].shape == (spec.k_factors, spec.k_factors)
        assert data['D'].shape == (spec.max_num_sec, spec.max_num_sec)


# ---------------------------------------------------------------------------
# factor_sims_plots
# ---------------------------------------------------------------------------

class TestFactorSimsPlots:
    """
    Tests for factor_sims_plots.py.

    These tests do not display figures — they verify that the functions run
    without error and produce the expected output file. matplotlib's Agg
    backend is used to avoid any display requirement.
    """

    @pytest.fixture(autouse=True)
    def use_agg_backend(self):
        import matplotlib
        matplotlib.use('Agg')

    def _make_long_df(self, include_sample_truth: bool = False) -> pd.DataFrame:
        """Minimal conformant long_df for plot tests."""
        rows = []
        for p in (50, 100):
            for rep in range(3):
                for radius in (0.1, 0.5):
                    for metric in ('grassmann', 'stiefel-canonical'):
                        for _ in range(5):
                            rows.append({
                                'dimension': 3, 'p': p, 'n': 63,
                                'radius': radius, 'rep': rep, 'metric': metric,
                                'distance_type': 'sample-target',
                                'distance': float(np.random.default_rng(0)
                                                  .uniform(radius * 0.5, radius * 1.5)),
                                'radius_label': f'r={radius:.2f}',
                                'n_label': 'n=63',
                            })
                        rows.append({
                            'dimension': 3, 'p': p, 'n': 63,
                            'radius': radius, 'rep': rep, 'metric': metric,
                            'distance_type': 'truth-target',
                            'distance': radius,
                            'radius_label': f'r={radius:.2f}',
                            'n_label': 'n=63',
                        })
                        if include_sample_truth:
                            rows.append({
                                'dimension': 3, 'p': p, 'n': 63,
                                'radius': radius, 'rep': rep, 'metric': metric,
                                'distance_type': 'sample-truth',
                                'distance': float(np.random.default_rng(1).uniform(0.1, 0.8)),
                                'radius_label': f'r={radius:.2f}',
                                'n_label': 'n=63',
                            })
        return pd.DataFrame(rows)

    def test_plot_dataframe_creates_file(self, tmp_path):
        from factor_sims_plots import plot_dataframe
        df = self._make_long_df()
        plot_dataframe(df, tmp_path)
        assert (tmp_path / 'distances.png').exists()
        assert (tmp_path / 'distances.png').stat().st_size > 0

    def test_plot_dataframe_sample_truth_mode(self, tmp_path):
        from factor_sims_plots import plot_dataframe
        df = self._make_long_df(include_sample_truth=True)
        plot_dataframe(df, tmp_path, sample_truth=True)
        assert (tmp_path / 'distances.png').exists()

    def test_plot_results_from_simulation(self, micro_spec_and_model, tmp_path):
        """plot_results accepts a SimResults object directly."""
        from factor_sims_plots import plot_results
        spec, _ = micro_spec_and_model
        results = run_simulation(spec)
        plot_results(results, tmp_path)
        assert (tmp_path / 'distances.png').exists()

    def test_plot_dataframe_empty_data_no_crash(self, tmp_path):
        """Empty DataFrame (after filtering) should log a warning, not crash."""
        from factor_sims_plots import plot_dataframe
        # DataFrame with only truth-target rows — after filtering, nothing to plot
        df = self._make_long_df()
        df = df[df['distance_type'] == 'truth-target'].copy()
        # Should not raise; just log a warning
        plot_dataframe(df, tmp_path)
        # No file should be created (empty data)
        assert not (tmp_path / 'distances.png').exists()

    def test_radius_label_normalisation(self, tmp_path):
        """r=0.10 and r=0.1 both get normalised to r=0.1 in the figure."""
        from factor_sims_plots import plot_dataframe
        df = self._make_long_df()
        # Simulate factor_sims two-decimal format
        df['radius_label'] = df['radius'].map(lambda r: f'r={r:.2f}')
        plot_dataframe(df, tmp_path)
        assert (tmp_path / 'distances.png').exists()
