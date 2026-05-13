"""
tests/test_sim_theorem_partii.py
================================
Unit and smoke tests for sim_theorem_partii.py and fl_graphics.py.

Covers:
  - build_model: shapes, factor covariance, idio variance, prevalence convergence
  - SineAlignmentAnalysis: perfect recovery, shape/range
  - Eq20RHSAnalysis: diagonal-X case (rotation=0), shape/range, floor ≤ rhs
  - _rep_records: structure, semantics, key names
  - simulate(): schema and row count (small grid, mocked)
  - fl_graphics: smoke tests for all three plot functions
  - print_summary: output content
  - compute_sine_alignment: self-alignment, orthogonal rows, shape/range
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_theorem_partii as sim
import fl_graphics as gfx
from factor_lab.analysis import SimulationContext
from factor_lab.factor_types import FactorModelData


# ── Global isolation ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_extra_distances():
    """Prevent simulate() side-effects on _EXTRA_DISTANCES leaking between tests."""
    from factor_lab.analyses import manifold as _m
    original = dict(_m._EXTRA_DISTANCES)
    yield
    _m._EXTRA_DISTANCES.clear()
    _m._EXTRA_DISTANCES.update(original)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(2026)


@pytest.fixture
def mock_context(rng):
    k, p, n = 3, 50, 30
    B = rng.standard_normal((k, p))
    model = FactorModelData(B=B, F=np.diag(sim.SIGMA2), D=np.diag(np.full(p, sim.DELTA2)))
    return SimulationContext(
        model=model,
        security_returns=rng.standard_normal((n, p)),
        factor_returns=rng.standard_normal((n, k)),
        idio_returns=rng.standard_normal((n, p)),
    )


@pytest.fixture
def results_df():
    """Minimal conformant DataFrame for graphics smoke tests.

    Uses three p values (so plot_scatter has a second-largest), three n values
    (so plot_components can filter to n=60), and three factors.
    """
    records = []
    for p in [100, 200, 300]:
        for n in [30, 60, 120]:
            for j in [1, 2, 3]:
                records.append({
                    "n": n, "p": p, "j": j,
                    "sin2_j": 0.5 + 0.01 * j,
                    "rhs":    0.45 + 0.01 * j,
                    "gap":    0.05,
                    "floor":  0.2,
                    "rotation": 0.1,
                    "rho":    1.5,
                })
    return pd.DataFrame(records)


# ── build_model ───────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_shapes(self, rng):
        model = sim.build_model(p=100, rng=rng)
        assert model.B.shape == (sim.K, 100)
        assert model.F.shape == (sim.K, sim.K)
        assert model.D.shape == (100, 100)

    def test_factor_covariance(self, rng):
        np.testing.assert_array_almost_equal(
            np.diag(sim.build_model(200, rng).F), sim.SIGMA2
        )

    def test_idio_variance(self, rng):
        np.testing.assert_allclose(
            np.diag(sim.build_model(200, rng).D), sim.DELTA2, rtol=1e-10
        )

    def test_prevalences_converge(self, rng):
        """At p=5000, empirical ‖B[j,:]‖²/p → TAU2 within 5%."""
        model = sim.build_model(5_000, rng)
        np.testing.assert_allclose(
            (model.B ** 2).mean(axis=1), sim.TAU2, rtol=0.05
        )


# ── SineAlignmentAnalysis ─────────────────────────────────────────────────────

class TestSineAlignmentAnalysis:

    def test_keys(self, mock_context, rng):
        k, p = mock_context.k, mock_context.p
        b_pop = rng.standard_normal((k, p))
        result = sim.SineAlignmentAnalysis(b_pop).analyze(mock_context)
        assert "sin2_j"    in result
        assert "dist_sine" in result

    def test_shape_and_range(self, mock_context, rng):
        k, p = mock_context.k, mock_context.p
        b_pop = rng.standard_normal((k, p))
        result = sim.SineAlignmentAnalysis(b_pop).analyze(mock_context)
        assert result["sin2_j"].shape == (k,)
        assert np.all(result["sin2_j"] >= 0.0)
        assert np.all(result["sin2_j"] <= 1.0)
        assert result["dist_sine"] >= 0.0

    def test_perfect_recovery(self, rng):
        """When estimated directions equal population directions, sin2_j ≈ 0."""
        p, k, n = 60, 3, 15
        b_pop  = np.eye(k, p)
        sigmas = np.array([5.0, 3.0, 1.5])
        V, _   = np.linalg.qr(rng.standard_normal((n, k)))
        Y      = b_pop.T @ np.diag(sigmas) @ V.T
        model  = FactorModelData(B=b_pop, F=np.diag(sigmas ** 2), D=np.eye(p))
        ctx    = SimulationContext(model=model, security_returns=Y.T,
                                   factor_returns=np.zeros((n, k)),
                                   idio_returns=np.zeros((n, p)))
        result = sim.SineAlignmentAnalysis(b_pop).analyze(ctx)
        np.testing.assert_allclose(result["sin2_j"], 0.0, atol=1e-12)


# ── Eq20RHSAnalysis ───────────────────────────────────────────────────────────

class TestEq20RHSAnalysis:

    def _diagonal_context(self, k=3, n=20, p=30):
        """Context where X@X.T/n = diag(SIGMA2[:k]), making D̂ diagonal.

        When D̂ is diagonal its eigenvectors form the identity and
        rotation = 1 − diag(W)² = 0, so rhs == floor exactly.
        """
        sigma2 = sim.SIGMA2[:k]
        tau2   = sim.TAU2[:k]
        X      = np.zeros((k, n))
        for j in range(k):
            X[j, j] = np.sqrt(n * sigma2[j])
        B     = np.tile(np.sqrt(tau2)[:, None], (1, p))
        model = FactorModelData(B=B, F=np.diag(sigma2), D=np.eye(p) * sim.DELTA2)
        return SimulationContext(
            model=model,
            security_returns=np.random.default_rng(0).standard_normal((n, p)),
            factor_returns=X.T,
            idio_returns=np.zeros((n, p)),
        )

    def test_diagonal_X_rotation_is_zero(self):
        result = sim.Eq20RHSAnalysis(sim.DELTA2).analyze(self._diagonal_context())
        np.testing.assert_allclose(result["rotation"], 0.0,            atol=1e-12)
        np.testing.assert_allclose(result["rhs"],      result["floor"], atol=1e-12)

    def test_shape_and_range(self):
        result = sim.Eq20RHSAnalysis(sim.DELTA2).analyze(self._diagonal_context())
        for key in ("rhs", "floor", "rotation"):
            assert result[key].shape == (sim.K,)
        assert np.all(result["rhs"] >= -1e-12)
        assert np.all(result["rhs"] <=  1.0 + 1e-12)

    def test_floor_le_rhs(self, mock_context):
        result = sim.Eq20RHSAnalysis(sim.DELTA2).analyze(mock_context)
        assert np.all(result["floor"] <= result["rhs"] + 1e-12)


# ── _rep_records ──────────────────────────────────────────────────────────────

class TestRepRecords:

    def _make_lhs_rhs(self, k=3, seed=0):
        rng    = np.random.default_rng(seed)
        sin2_j = rng.uniform(0.1, 0.9, k)
        floor  = rng.uniform(0.05, 0.4, k)
        rot    = rng.uniform(0.0, 0.5, k)
        rhos   = np.sort(rng.uniform(0.01, 1.0, k))[::-1]
        lhs = {"sin2_j": sin2_j, "dist_sine": float(np.sqrt(sin2_j.sum()))}
        rhs = {"rhs": floor + (1.0 - floor) * rot, "floor": floor,
               "rotation": rot, "rhos": rhos}
        return lhs, rhs

    def test_returns_k_records(self):
        lhs, rhs = self._make_lhs_rhs()
        assert len(sim._rep_records(30, 500, lhs, rhs)) == sim.K

    def test_required_keys(self):
        lhs, rhs = self._make_lhs_rhs()
        expected = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor", "rotation", "rho"}
        for rec in sim._rep_records(30, 500, lhs, rhs):
            assert set(rec.keys()) == expected

    def test_gap_equals_sin2_minus_rhs(self):
        lhs, rhs = self._make_lhs_rhs()
        for rec in sim._rep_records(30, 500, lhs, rhs):
            assert rec["gap"] == pytest.approx(rec["sin2_j"] - rec["rhs"], abs=1e-14)

    def test_j_is_one_indexed(self):
        lhs, rhs = self._make_lhs_rhs()
        assert [r["j"] for r in sim._rep_records(30, 500, lhs, rhs)] == list(range(1, sim.K + 1))


# ── simulate() integration ────────────────────────────────────────────────────

class TestSimulate:

    @pytest.mark.parametrize("n_reps,expected_rows", [(2, 6)])
    def test_schema_and_row_count(self, n_reps, expected_rows, monkeypatch):
        """Small grid runs without error and produces correct schema."""
        monkeypatch.setattr(sim, "N_VALUES", [30])
        monkeypatch.setattr(sim, "P_VALUES", [100])
        monkeypatch.setattr(sim, "N_REPS",   n_reps)
        df = sim.simulate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == expected_rows   # 1 n × 1 p × 2 reps × 3 factors
        expected_cols = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor", "rotation", "rho"}
        assert expected_cols.issubset(df.columns)
        assert set(df["j"].unique()) == {1, 2, 3}
        assert df["sin2_j"].between(0.0, 1.0).all()
        assert df["rhs"].between(0.0, 1.0).all()


# ── fl_graphics smoke tests ───────────────────────────────────────────────────

class TestGraphics:

    def test_plot_convergence(self, results_df, tmp_path):
        out = tmp_path / "convergence.png"
        gfx.plot_convergence(results_df, out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_scatter(self, results_df, tmp_path):
        out = tmp_path / "scatter.png"
        gfx.plot_scatter(results_df, out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_components(self, results_df, tmp_path):
        out = tmp_path / "components.png"
        gfx.plot_components(results_df, out, n_show=60)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_all(self, results_df, tmp_path):
        gfx.plot_all(results_df, tmp_path, n_show=60)
        assert (tmp_path / "fig_theorem1_convergence_v2.png").exists()
        assert (tmp_path / "fig_theorem1_scatter_v2.png").exists()
        assert (tmp_path / "fig_theorem1_components_v2.png").exists()

    def test_plot_all_infers_n_show(self, results_df, tmp_path):
        """plot_all with n_show=None should infer 60 (median of [30, 60, 120])."""
        gfx.plot_all(results_df, tmp_path)
        assert (tmp_path / "fig_theorem1_components_v2.png").exists()

    def test_load_results_csv(self, results_df, tmp_path):
        csv_path = tmp_path / "results.csv"
        results_df.to_csv(csv_path, index=False)
        loaded = gfx.load_results(csv_path)
        assert list(loaded.columns) == list(results_df.columns)
        assert len(loaded) == len(results_df)

    def test_load_results_parquet(self, results_df, tmp_path):
        pq_path = tmp_path / "results.parquet"
        results_df.to_parquet(pq_path, index=False)
        loaded = gfx.load_results(pq_path)
        assert list(loaded.columns) == list(results_df.columns)
        assert len(loaded) == len(results_df)


# ── print_summary ─────────────────────────────────────────────────────────────

class TestPrintSummary:

    def test_output_content(self, results_df, capsys):
        sim.print_summary(results_df)
        out = capsys.readouterr().out
        assert "RMSE of" in out
        assert "j=1"      in out


# ── compute_sine_alignment ────────────────────────────────────────────────────

class TestComputeSineAlignment:

    def test_self_alignment_is_zero(self):
        rng = np.random.default_rng(0)
        B   = rng.standard_normal((3, 40))
        sin2, dist = sim.compute_sine_alignment(B, B)
        np.testing.assert_allclose(sin2, 0.0, atol=1e-6)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_rows_give_one(self):
        B_true = np.eye(3, 10)
        B_est  = np.roll(B_true, 1, axis=0)
        sin2, _ = sim.compute_sine_alignment(B_true, B_est)
        np.testing.assert_allclose(sin2, 1.0, atol=1e-12)

    def test_output_shapes(self):
        rng  = np.random.default_rng(7)
        B    = rng.standard_normal((3, 50))
        sin2, dist = sim.compute_sine_alignment(B, B)
        assert sin2.shape == (3,)
        assert isinstance(dist, float)

    def test_range(self):
        rng  = np.random.default_rng(99)
        B, C = rng.standard_normal((3, 50)), rng.standard_normal((3, 50))
        sin2, dist = sim.compute_sine_alignment(B, C)
        assert np.all(sin2 >= 0.0)
        assert np.all(sin2 <= 1.0 + 1e-12)
        assert dist >= 0.0
