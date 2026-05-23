"""
tests/test_sim_theorem_partii.py
================================
Unit and smoke tests for sim_theorem_partii.py and fl_graphics.py.

Covers:
  - SimSpec: defaults reproduce the original experiment; from_json round-trip;
    comment-key stripping; CLI override fields default to None.
  - _make_one_sampler / _make_samplers: dict → callable; broadcast vs. list;
    wrong-length raises.
  - build_model: shapes, factor covariance, idio variance (squared from vol),
    prevalence convergence.
  - SineAlignmentAnalysis: perfect recovery, shape/range, keys.
  - Eq6RHSAnalysis: δ² derived from model.D; "delta2" appears in result;
    diagonal-F case (rotation=0); shape/range; floor ≤ rhs.
  - _rep_records: k-parameterized length, structure, gap = sin²−rhs, j is
    1-indexed.
  - _next_run_dir: sequential allocation, ignores unrelated names, NN is two
    digits, directory is created.
  - simulate(): schema and row count for a small spec.
  - fl_graphics: smoke tests for all three plot functions.
  - main() CLI: no config_file → SimSpec(); positional spec → from_json;
    --out > spec.output_path > auto-allocated run dir; --plot skips parquet;
    --plot-save writes both.
  - print_summary: output content.
  - compute_sine_alignment: self-alignment, orthogonal rows, shape/range.
"""

import json
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
def default_spec():
    """The built-in SimSpec — reproduces the original hardcoded experiment."""
    return sim.SimSpec()


@pytest.fixture
def small_spec():
    """A tiny spec good enough for end-to-end smoke tests."""
    return sim.SimSpec(
        k_factors=3,
        n_values=[30],
        p_values=[100],
        n_reps=2,
        random_seed=2026,
    )


@pytest.fixture
def mock_context(rng, default_spec):
    k, p, n = default_spec.k_factors, 50, 30
    B = rng.standard_normal((k, p))
    F = np.diag(default_spec.factor_variances)
    D = np.eye(p)
    return SimulationContext(
        model=FactorModelData(B=B, F=F, D=D),
        security_returns=rng.standard_normal((n, p)),
        factor_returns=rng.standard_normal((n, k)),
        idio_returns=rng.standard_normal((n, p)),
    )


@pytest.fixture
def results_df():
    """Minimal conformant DataFrame for graphics smoke tests."""
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


# ── SimSpec ───────────────────────────────────────────────────────────────────

class TestSimSpec:

    def test_defaults_reproduce_original(self, default_spec):
        s = default_spec
        assert s.k_factors == 3
        assert s.n_values == [30, 60, 120]
        assert s.p_values == [200, 500, 1000, 2000, 5000, 10_000]
        assert s.n_reps == 300
        assert s.random_seed == 20260511
        assert s.factor_variances == [0.04, 0.02, 0.01]
        # Beta sampler scales correspond to c = [1.0, 0.8, 0.6] (i.e. √c).
        scales = [b["scale"] for b in s.beta_samplers]
        np.testing.assert_allclose(np.array(scales) ** 2, [1.0, 0.8, 0.6], rtol=1e-10)
        # Idio sampler is constant vol 1.0 → D's diagonal will be 1.0.
        assert s.idio_vol_sampler == {"distribution": "constant", "value": 1.0}
        # Optional CLI overrides default to None.
        assert s.output_path is None
        assert s.plot_mode is None

    def test_from_json_roundtrip(self, tmp_path, default_spec):
        cfg = {
            "k_factors": 3,
            "n_values": [30, 60, 120],
            "p_values": [200, 500, 1000, 2000, 5000, 10000],
            "n_reps": 300,
            "random_seed": 20260511,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": default_spec.beta_samplers,
            "idio_vol_sampler": default_spec.idio_vol_sampler,
            "factor_return_sampler": default_spec.factor_return_sampler,
            "idio_return_sampler": default_spec.idio_return_sampler,
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(cfg))
        loaded = sim.SimSpec.from_json(path)
        assert loaded.k_factors == default_spec.k_factors
        assert loaded.n_values == default_spec.n_values
        assert loaded.factor_variances == default_spec.factor_variances
        assert loaded.idio_vol_sampler == default_spec.idio_vol_sampler

    def test_from_json_strips_comment_keys(self, tmp_path):
        cfg = {
            "_comment": "this should be ignored",
            "_note": "and this too",
            "k_factors": 2,
            "n_values": [30],
            "p_values": [100],
            "n_reps": 1,
            "random_seed": 0,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [
                {"distribution": "normal", "loc": 0.0, "scale": 1.0},
                {"distribution": "normal", "loc": 0.0, "scale": 1.0},
            ],
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(cfg))
        # Would raise TypeError if "_comment" wasn't stripped (unknown kwarg).
        loaded = sim.SimSpec.from_json(path)
        assert loaded.k_factors == 2

    def test_shipped_spec_files_load(self):
        """Both committed spec files load and produce a valid SimSpec."""
        for name in ("sim_thmptii_spec.json", "sim_thmptii_standard_setup.json"):
            spec = sim.SimSpec.from_json(ROOT / name)
            assert spec.k_factors == 3
            assert len(spec.factor_variances) == 3
            assert len(spec.beta_samplers) == 3


# ── _make_one_sampler / _make_samplers ────────────────────────────────────────

class TestSamplerHelpers:

    def test_make_one_sampler_normal(self, rng):
        s = sim._make_one_sampler(
            {"distribution": "normal", "loc": 10.0, "scale": 0.0}, rng,
        )
        out = s(5)
        np.testing.assert_allclose(out, np.full(5, 10.0))

    def test_make_one_sampler_constant(self, rng):
        s = sim._make_one_sampler({"distribution": "constant", "value": 0.5}, rng)
        out = s(7)
        np.testing.assert_allclose(out, np.full(7, 0.5))

    def test_make_samplers_list(self, rng):
        spec_list = [
            {"distribution": "normal", "loc": 0.0, "scale": 0.0},
            {"distribution": "normal", "loc": 1.0, "scale": 0.0},
            {"distribution": "normal", "loc": 2.0, "scale": 0.0},
        ]
        samplers = sim._make_samplers(spec_list, rng, k=3)
        assert isinstance(samplers, list) and len(samplers) == 3
        for j, s in enumerate(samplers):
            np.testing.assert_allclose(s(4), np.full(4, float(j)))

    def test_make_samplers_broadcast(self, rng):
        """Single dict broadcasts — returned as a single sampler, not a list."""
        s = sim._make_samplers(
            {"distribution": "constant", "value": 0.25}, rng, k=3,
        )
        assert callable(s)
        np.testing.assert_allclose(s(4), np.full(4, 0.25))

    def test_make_samplers_wrong_length_raises(self, rng):
        bad = [{"distribution": "normal"}, {"distribution": "normal"}]
        with pytest.raises(ValueError, match=r"Expected 3 .*samplers"):
            sim._make_samplers(bad, rng, k=3)


# ── build_model ───────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_shapes(self, default_spec, rng):
        model = sim.build_model(default_spec, p=100, rng=rng)
        assert model.B.shape == (default_spec.k_factors, 100)
        assert model.F.shape == (default_spec.k_factors, default_spec.k_factors)
        assert model.D.shape == (100, 100)

    def test_factor_covariance(self, default_spec, rng):
        np.testing.assert_array_almost_equal(
            np.diag(sim.build_model(default_spec, 200, rng).F),
            default_spec.factor_variances,
        )

    def test_idio_variance_is_vol_squared(self, rng):
        """idio_vol_sampler outputs vol; D's diagonal must hold vol² (variances)."""
        spec = sim.SimSpec(
            idio_vol_sampler={"distribution": "constant", "value": 0.5},
        )
        D_diag = np.diag(sim.build_model(spec, p=200, rng=rng).D)
        np.testing.assert_allclose(D_diag, 0.25, rtol=1e-10)

    def test_prevalences_converge(self, default_spec, rng):
        """At p=5000, ‖B[j,:]‖²/p → cⱼ (the squared β scales) within 5%."""
        model = sim.build_model(default_spec, p=5_000, rng=rng)
        expected_c = np.array([b["scale"] for b in default_spec.beta_samplers]) ** 2
        np.testing.assert_allclose(
            (model.B ** 2).mean(axis=1), expected_c, rtol=0.05,
        )


# ── SineAlignmentAnalysis ─────────────────────────────────────────────────────

class TestSineAlignmentAnalysis:

    def test_keys(self, mock_context, rng):
        k, p = mock_context.k, mock_context.p
        b_pop = rng.standard_normal((k, p))
        result = sim.SineAlignmentAnalysis(b_pop).analyze(mock_context)
        assert "sin2_j" in result and "dist_sine" in result

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
        ctx    = SimulationContext(
            model=model, security_returns=Y.T,
            factor_returns=np.zeros((n, k)),
            idio_returns=np.zeros((n, p)),
        )
        result = sim.SineAlignmentAnalysis(b_pop).analyze(ctx)
        np.testing.assert_allclose(result["sin2_j"], 0.0, atol=1e-12)


# ── Eq6RHSAnalysis ────────────────────────────────────────────────────────────

class TestEq6RHSAnalysis:

    def _diagonal_context(self, k=3, n=20, p=30, delta2=1.0):
        """Context where F@F.T/n = diag(σ²), making D̂ diagonal.

        When D̂ is diagonal its eigenvectors are the standard basis eⱼ, so
        sin²∠(ŵⱼ, eⱼ) = rotation = 0 and rhs == floor exactly.
        """
        sigma2 = np.array([0.04, 0.02, 0.01])[:k]
        c2     = np.array([1.0, 0.8, 0.6])[:k]
        F      = np.zeros((k, n))
        for j in range(k):
            F[j, j] = np.sqrt(n * sigma2[j])
        B     = np.tile(np.sqrt(c2)[:, None], (1, p))
        model = FactorModelData(B=B, F=np.diag(sigma2), D=np.eye(p) * delta2)
        return SimulationContext(
            model=model,
            security_returns=np.random.default_rng(0).standard_normal((n, p)),
            factor_returns=F.T,
            idio_returns=np.zeros((n, p)),
        )

    def test_no_args_constructor(self):
        """Eq6RHSAnalysis takes no arguments — δ² comes from context.model.D."""
        a = sim.Eq6RHSAnalysis()
        result = a.analyze(self._diagonal_context())
        assert "delta2" in result

    def test_delta2_from_model_D(self):
        """δ² in result must equal mean(diag(model.D))."""
        ctx = self._diagonal_context(delta2=0.25)
        result = sim.Eq6RHSAnalysis().analyze(ctx)
        assert result["delta2"] == pytest.approx(0.25, abs=1e-12)

    def test_delta2_squares_constant_vol(self):
        """idio_vol = v (sampler) → D's diagonal = v² → δ² in result = v²."""
        spec = sim.SimSpec(
            idio_vol_sampler={"distribution": "constant", "value": 0.5},
        )
        model = sim.build_model(spec, p=30, rng=np.random.default_rng(0))
        n = 20
        ctx = SimulationContext(
            model=model,
            security_returns=np.zeros((n, model.B.shape[1])),
            factor_returns=np.random.default_rng(1).standard_normal((n, spec.k_factors)),
            idio_returns=np.zeros((n, model.B.shape[1])),
        )
        result = sim.Eq6RHSAnalysis().analyze(ctx)
        assert result["delta2"] == pytest.approx(0.25, abs=1e-12)

    def test_diagonal_F_rotation_is_zero(self):
        result = sim.Eq6RHSAnalysis().analyze(self._diagonal_context())
        np.testing.assert_allclose(result["rotation"], 0.0,             atol=1e-12)
        np.testing.assert_allclose(result["rhs"],      result["floor"], atol=1e-12)

    def test_shape_and_range(self, default_spec):
        result = sim.Eq6RHSAnalysis().analyze(self._diagonal_context())
        for key in ("rhs", "floor", "rotation"):
            assert result[key].shape == (default_spec.k_factors,)
        assert np.all(result["rhs"] >= -1e-12)
        assert np.all(result["rhs"] <=  1.0 + 1e-12)

    def test_floor_le_rhs(self, mock_context):
        result = sim.Eq6RHSAnalysis().analyze(mock_context)
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
               "rotation": rot, "rhos": rhos, "delta2": 1.0}
        return lhs, rhs

    def test_returns_k_records(self):
        lhs, rhs = self._make_lhs_rhs(k=3)
        assert len(sim._rep_records(3, 30, 500, lhs, rhs)) == 3

    def test_k_parameterizes_length(self):
        for k in (1, 2, 5):
            lhs, rhs = self._make_lhs_rhs(k=k)
            assert len(sim._rep_records(k, 30, 500, lhs, rhs)) == k

    def test_required_keys(self):
        lhs, rhs = self._make_lhs_rhs()
        expected = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor", "rotation", "rho"}
        for rec in sim._rep_records(3, 30, 500, lhs, rhs):
            assert set(rec.keys()) == expected

    def test_gap_equals_sin2_minus_rhs(self):
        lhs, rhs = self._make_lhs_rhs()
        for rec in sim._rep_records(3, 30, 500, lhs, rhs):
            assert rec["gap"] == pytest.approx(rec["sin2_j"] - rec["rhs"], abs=1e-14)

    def test_j_is_one_indexed(self):
        lhs, rhs = self._make_lhs_rhs()
        assert [r["j"] for r in sim._rep_records(3, 30, 500, lhs, rhs)] == [1, 2, 3]


# ── _next_run_dir ─────────────────────────────────────────────────────────────

class TestNextRunDir:

    def test_first_call_creates_run_01(self, tmp_path, monkeypatch):
        d = sim._next_run_dir(tmp_path)
        assert d.exists() and d.is_dir()
        assert d.name.endswith("_run_01")
        assert d.parent == tmp_path / "results"

    def test_sequential_allocation(self, tmp_path):
        dirs = [sim._next_run_dir(tmp_path) for _ in range(3)]
        nums = [int(d.name.split("_run_")[-1]) for d in dirs]
        assert nums == [1, 2, 3]
        # All distinct directories.
        assert len({str(d) for d in dirs}) == 3

    def test_two_digit_zero_padding(self, tmp_path):
        d = sim._next_run_dir(tmp_path)
        suffix = d.name.split("_run_")[-1]
        assert suffix == "01" and len(suffix) == 2

    def test_ignores_unrelated_names(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        # Things that match the date prefix but not the run_NN suffix:
        for name in ("05-12_01", "run1", "scratch", "05-19_old"):
            (results / name).mkdir()
        d = sim._next_run_dir(tmp_path)
        assert d.name.endswith("_run_01")

    def test_picks_max_plus_one(self, tmp_path, monkeypatch):
        """If 05-19_run_07 already exists, next allocation is 05-19_run_08."""
        from datetime import datetime
        today = datetime.now().strftime("%m-%d")
        results = tmp_path / "results"
        results.mkdir()
        (results / f"{today}_run_07").mkdir()
        d = sim._next_run_dir(tmp_path)
        assert d.name == f"{today}_run_08"


# ── simulate() integration ────────────────────────────────────────────────────

class TestSimulate:

    def test_schema_and_row_count(self, small_spec):
        df = sim.simulate(small_spec)
        assert isinstance(df, pd.DataFrame)
        # 1 n × 1 p × 2 reps × 3 factors = 6
        assert len(df) == 6
        expected_cols = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor",
                         "rotation", "rho"}
        assert expected_cols.issubset(df.columns)
        assert set(df["j"].unique()) == {1, 2, 3}
        assert df["sin2_j"].between(0.0, 1.0).all()
        assert df["rhs"].between(0.0, 1.0).all()

    def test_reproducible_under_same_seed(self, small_spec):
        df1 = sim.simulate(small_spec)
        df2 = sim.simulate(small_spec)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        a = sim.SimSpec(n_values=[30], p_values=[100], n_reps=2, random_seed=1)
        b = sim.SimSpec(n_values=[30], p_values=[100], n_reps=2, random_seed=2)
        df_a = sim.simulate(a)
        df_b = sim.simulate(b)
        # At least some of the sin² values should differ.
        assert not np.allclose(df_a["sin2_j"], df_b["sin2_j"])


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


# ── main() CLI dispatch ───────────────────────────────────────────────────────

class TestMain:

    @pytest.fixture(autouse=True)
    def _patch_simulate(self, monkeypatch, results_df):
        """Replace simulate() with a fast no-op returning the fixture DataFrame."""
        monkeypatch.setattr(sim, "simulate", lambda spec: results_df)

    def test_no_config_uses_defaults(self, monkeypatch, tmp_path, results_df):
        """No positional arg → SimSpec() is used; auto-allocates a run dir."""
        captured = {}

        def fake_simulate(spec):
            captured["spec"] = spec
            return results_df

        monkeypatch.setattr(sim, "ROOT", tmp_path)
        monkeypatch.setattr(sim, "simulate", fake_simulate)
        monkeypatch.setattr(sys, "argv", ["sim_theorem_partii.py"])
        sim.main()
        # SimSpec() defaults were used.
        assert captured["spec"].k_factors == 3
        assert captured["spec"].random_seed == 20260511
        # Auto-allocated results/MM-DD_run_NN exists.
        assert (tmp_path / "results").is_dir()
        run_dirs = list((tmp_path / "results").iterdir())
        assert len(run_dirs) == 1
        # Parquet landed inside it.
        assert any(p.suffix == ".parquet" for p in run_dirs[0].iterdir())

    def test_config_file_loaded(self, monkeypatch, tmp_path, results_df):
        """Positional spec arg routes through SimSpec.from_json."""
        cfg = tmp_path / "spec.json"
        cfg.write_text(json.dumps({
            "k_factors": 3,
            "n_values": [30],
            "p_values": [100],
            "n_reps": 1,
            "random_seed": 999,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": [
                {"distribution": "normal", "loc": 0.0, "scale": 1.0}
            ] * 3,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        captured = {}

        def fake_simulate(spec):
            captured["spec"] = spec
            return results_df

        monkeypatch.setattr(sim, "simulate", fake_simulate)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", str(cfg), "--out", str(out)])
        sim.main()
        assert captured["spec"].random_seed == 999

    def test_cli_out_overrides_spec(self, monkeypatch, tmp_path):
        """--out wins over spec.output_path and over auto-allocation."""
        cfg = tmp_path / "spec.json"
        cfg.write_text(json.dumps({
            "k_factors": 3, "n_values": [30], "p_values": [100],
            "n_reps": 1, "random_seed": 0,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": [{"distribution": "normal"}] * 3,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
            "output_path": str(tmp_path / "spec_default.parquet"),
        }))
        cli_out = tmp_path / "cli_winner.parquet"
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", str(cfg), "--out", str(cli_out)])
        sim.main()
        assert cli_out.exists()
        assert not (tmp_path / "spec_default.parquet").exists()

    def test_spec_output_path_used_when_no_cli(self, monkeypatch, tmp_path):
        """spec.output_path is honored when no --out and no auto-allocation."""
        target = tmp_path / "from_spec.parquet"
        cfg = tmp_path / "spec.json"
        cfg.write_text(json.dumps({
            "k_factors": 3, "n_values": [30], "p_values": [100],
            "n_reps": 1, "random_seed": 0,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": [{"distribution": "normal"}] * 3,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
            "output_path": str(target),
        }))
        monkeypatch.setattr(sys, "argv", ["sim_theorem_partii.py", str(cfg)])
        sim.main()
        assert target.exists()

    def test_plot_skips_parquet(self, monkeypatch, tmp_path):
        out = tmp_path / "out.parquet"
        plot_calls = []
        monkeypatch.setattr(
            "fl_graphics.plot_all", lambda df, out_dir, **kw: plot_calls.append(out_dir),
        )
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", "--out", str(out), "--plot"])
        sim.main()
        assert not out.exists()
        assert len(plot_calls) == 1

    def test_plot_save_writes_parquet_and_plots(self, monkeypatch, tmp_path):
        out = tmp_path / "out.parquet"
        plot_calls = []
        monkeypatch.setattr(
            "fl_graphics.plot_all", lambda df, out_dir, **kw: plot_calls.append(out_dir),
        )
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", "--out", str(out), "--plot-save"])
        sim.main()
        assert out.exists()
        assert len(plot_calls) == 1


# ── print_summary ─────────────────────────────────────────────────────────────

class TestPrintSummary:

    def test_output_content(self, results_df, capsys):
        sim.print_summary(results_df)
        out = capsys.readouterr().out
        assert "RMSE of" in out
        assert "j=1" in out


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
        rng = np.random.default_rng(7)
        B   = rng.standard_normal((3, 50))
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
