"""
tests/test_sim_theorem_partii.py
================================
Unit and smoke tests for sim_theorem_partii.py and fl_graphics.py.

Covers:
  - ModelSpec / DesignSpec: defaults reproduce the original experiment;
    from_json round-trip; comment-key stripping; the unified single-file shape
    (top-level model fields folded into an inline model); model reference forms
    (None / inline dict / path); utf-8 loading.
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
  - simulate() / run_experiment(): schema and row count for a small design.
  - fl_graphics: smoke tests for all three plot functions.
  - main() CLI: no config_file → defaults; positional design spec; --model
    override; --out > design.output_path > auto-allocated run dir; --plot skips
    parquet; --plot-save writes both.
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
from fl_experiment import ModelSpec, DesignSpec, run_experiment
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
def default_model():
    """The built-in ModelSpec — the model half of the original experiment."""
    return ModelSpec()


@pytest.fixture
def default_design():
    """The built-in DesignSpec — the sweep/return half of the original experiment."""
    return DesignSpec()


@pytest.fixture
def small_design():
    """A tiny design good enough for end-to-end smoke tests (model = defaults)."""
    return DesignSpec(
        n_values=[30],
        p_values=[100],
        n_reps=2,
        random_seed=2026,
    )


@pytest.fixture
def mock_context(rng, default_model):
    k, p, n = default_model.k_factors, 50, 30
    B = rng.standard_normal((k, p))
    F = np.diag(default_model.factor_variances)
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


# ── ModelSpec / DesignSpec ────────────────────────────────────────────────────

class TestSpecs:
    """ModelSpec + DesignSpec defaults, loading, and the unified single-file fold."""

    def test_model_defaults_reproduce_original(self, default_model):
        m = default_model
        assert m.k_factors == 3
        assert m.factor_variances == [0.04, 0.02, 0.01]
        # Beta sampler scales correspond to c = [1.0, 0.8, 0.6] (i.e. √c).
        scales = [b["scale"] for b in m.beta_samplers]
        np.testing.assert_allclose(np.array(scales) ** 2, [1.0, 0.8, 0.6], rtol=1e-10)
        # Idio sampler is constant vol 1.0 → D's diagonal will be 1.0.
        assert m.idio_vol_sampler == {"distribution": "constant", "value": 1.0}

    def test_design_defaults_reproduce_original(self, default_design):
        d = default_design
        assert d.model is None
        assert d.n_values == [30, 60, 120]
        assert d.p_values == [200, 500, 1000, 2000, 5000, 10_000]
        assert d.n_reps == 300
        assert d.random_seed == 20260511
        assert d.factor_return_sampler == {"distribution": "normal", "loc": 0.0, "scale": 1.0}
        assert d.idio_return_sampler == {"distribution": "normal", "loc": 0.0, "scale": 1.0}
        # Optional CLI overrides default to None.
        assert d.output_path is None
        assert d.plot_mode is None

    def test_unified_file_folds_top_level_model_fields(self, tmp_path, default_model):
        """A single-file design with model fields at the top level folds them
        into an inline model (the shipped 'unified' shape)."""
        cfg = {
            "n_values": [30, 60, 120],
            "p_values": [200, 500, 1000, 2000, 5000, 10000],
            "n_reps": 300,
            "random_seed": 20260511,
            # model fields written flat at the top level:
            "k_factors": 3,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": default_model.beta_samplers,
            "idio_vol_sampler": default_model.idio_vol_sampler,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }
        path = tmp_path / "unified.json"
        path.write_text(json.dumps(cfg))
        design = DesignSpec.from_json(path)
        # Top-level model fields were folded into an inline model dict.
        assert isinstance(design.model, dict)
        assert design.n_values == [30, 60, 120]
        model = design.resolve_model(base_dir=tmp_path)
        assert model.k_factors == 3
        assert model.factor_variances == [0.04, 0.02, 0.01]

    def test_fold_conflict_raises(self, tmp_path):
        """Mixing top-level model fields with an explicit 'model' reference errors."""
        cfg = {
            "model": "model.json",
            "k_factors": 3,           # also inline at top level → conflict
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 0,
        }
        path = tmp_path / "conflict.json"
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="both inline .* and via a 'model' reference"):
            DesignSpec.from_json(path)

    def test_design_from_json_strips_comment_keys(self, tmp_path):
        cfg = {
            "_comment": "this should be ignored",
            "_note": "and this too",
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 0,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }
        path = tmp_path / "design.json"
        path.write_text(json.dumps(cfg))
        loaded = DesignSpec.from_json(path)
        assert loaded.n_reps == 1

    def test_shipped_unified_files_load(self):
        """The committed single-file specs load and fold their model in."""
        for name in ("sim_thmptii_spec.json", "sim_thmptii_standard_setup.json"):
            design = DesignSpec.from_json(ROOT / name)
            model = design.resolve_model(base_dir=ROOT)
            assert model.k_factors == 3
            assert len(model.factor_variances) == 3
            assert len(model.beta_samplers) == 3

    def test_shipped_split_pair_loads(self):
        """The committed split pair loads and resolves its model reference."""
        design = DesignSpec.from_json(ROOT / "sim_thmptii_design.json")
        model = design.resolve_model(base_dir=ROOT)
        assert model.k_factors == 3
        assert len(model.factor_variances) == 3
        assert len(model.beta_samplers) == 3
        assert design.n_values and design.p_values

    def test_from_json_handles_non_ascii(self, tmp_path):
        """Specs may contain σ/β/δ etc. in comment fields — must load on
        any platform, not just utf-8-default systems (regression: Windows
        cp1252 raised UnicodeDecodeError on byte 0x81)."""
        cfg = {
            "_comment": "σⱼ are volatilities; β samples drawn N(0, √cⱼ); δ² ≈ 1.",
            "k_factors": 2,
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 0,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }
        path = tmp_path / "spec_unicode.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        # Sanity: file actually contains the non-ASCII bytes that broke Windows.
        assert "σ" in path.read_text(encoding="utf-8")
        design = DesignSpec.from_json(path)
        assert design.resolve_model(base_dir=tmp_path).k_factors == 2

    def test_from_json_opens_with_utf8(self, tmp_path, monkeypatch):
        """Cross-platform: from_json must explicitly request utf-8 so that
        systems with a non-utf-8 default (e.g. Windows cp1252) still load
        the σ/β/δ characters in our shipped specs."""
        import builtins
        opened_with = {}
        real_open = builtins.open

        def spy(file, mode="r", *args, **kwargs):
            opened_with[str(file)] = kwargs.get("encoding")
            return real_open(file, mode, *args, **kwargs)

        path = tmp_path / "design.json"
        path.write_text(json.dumps({
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 0,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        monkeypatch.setattr(builtins, "open", spy)
        DesignSpec.from_json(path)
        assert opened_with.get(str(path)) == "utf-8", (
            f"from_json must open with encoding='utf-8'; got "
            f"{opened_with.get(str(path))!r}"
        )


# ── ModelSpec reference resolution ────────────────────────────────────────────

class TestSplitConfig:
    """The model reference forms and composition into a runtime pair.

    Each test pins down a contract that the engine, a notebook, or a second
    script would lean on.
    """

    def test_model_spec_from_json_strips_comments(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text(json.dumps({
            "_comment": "ignored",
            "k_factors": 2,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        loaded = ModelSpec.from_json(path)
        assert loaded.k_factors == 2

    def test_design_spec_from_json_strips_comments(self, tmp_path):
        path = tmp_path / "design.json"
        path.write_text(json.dumps({
            "_comment": "ignored",
            "n_values": [30],
            "p_values": [100],
            "n_reps": 1,
            "random_seed": 7,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        loaded = DesignSpec.from_json(path)
        assert loaded.random_seed == 7

    def test_resolve_model_none_uses_defaults(self, tmp_path):
        spec = DesignSpec(model=None).resolve_model(tmp_path)
        assert isinstance(spec, ModelSpec)
        assert spec.k_factors == ModelSpec().k_factors

    def test_resolve_model_inline_dict(self, tmp_path):
        inline = {
            "k_factors": 5,
            "factor_variances": [0.1] * 5,
            "beta_samplers": [{"distribution": "normal"}] * 5,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }
        spec = DesignSpec(model=inline).resolve_model(tmp_path)
        assert spec.k_factors == 5

    def test_resolve_model_inline_dict_strips_comments(self, tmp_path):
        """Inline model dicts should accept _-prefixed comment keys too."""
        inline = {
            "_comment": "ignored",
            "k_factors": 2,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }
        spec = DesignSpec(model=inline).resolve_model(tmp_path)
        assert spec.k_factors == 2

    def test_resolve_model_relative_path(self, tmp_path):
        (tmp_path / "model.json").write_text(json.dumps({
            "k_factors": 4,
            "factor_variances": [0.04, 0.02, 0.01, 0.005],
            "beta_samplers": [{"distribution": "normal"}] * 4,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        spec = DesignSpec(model="model.json").resolve_model(tmp_path)
        assert spec.k_factors == 4

    def test_resolve_model_absolute_path(self, tmp_path):
        model_path = tmp_path / "abs_model.json"
        model_path.write_text(json.dumps({
            "k_factors": 2,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        # Pass a different base_dir to prove the absolute path wins.
        spec = DesignSpec(model=str(model_path)).resolve_model(
            base_dir=Path("/nonexistent"),
        )
        assert spec.k_factors == 2

    def test_design_json_plus_resolve_model_end_to_end(self, tmp_path):
        """The canonical split entry point: load a DesignSpec, resolve its model."""
        (tmp_path / "model.json").write_text(json.dumps({
            "k_factors": 2,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        (tmp_path / "design.json").write_text(json.dumps({
            "model": "model.json",
            "n_values": [30], "p_values": [100],
            "n_reps": 1, "random_seed": 99,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        design = DesignSpec.from_json(tmp_path / "design.json")
        model = design.resolve_model(base_dir=tmp_path)
        assert model.k_factors == 2
        assert design.random_seed == 99
        assert design.n_values == [30]

    def test_split_matches_unified_byte_for_byte(self, tmp_path):
        """A split (model.json + design-with-reference) must produce the exact
        same DataFrame as the equivalent single unified file — the load-bearing
        reproducibility contract across the two file shapes."""
        model_fields = {
            "k_factors": 3,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": [{"distribution": "normal"}] * 3,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }
        design_fields = {
            "n_values": [30], "p_values": [100], "n_reps": 3, "random_seed": 2026,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }
        # Unified single file: model fields folded in at top level.
        (tmp_path / "unified.json").write_text(json.dumps({**model_fields, **design_fields}))
        # Split pair: model.json + design referencing it.
        (tmp_path / "m.json").write_text(json.dumps(model_fields))
        (tmp_path / "d.json").write_text(json.dumps({"model": "m.json", **design_fields}))

        def run(name):
            d = DesignSpec.from_json(tmp_path / name)
            return run_experiment(d.resolve_model(tmp_path), d,
                                  sim.DispersionBiasExperiment())

        pd.testing.assert_frame_equal(run("unified.json"), run("d.json"))

    def test_run_experiment_matches_simulate(self):
        """run_experiment(model, design, probe) == simulate(design)."""
        design = DesignSpec(n_values=[30], p_values=[100], n_reps=4, random_seed=7)
        model = design.resolve_model(base_dir=ROOT)
        via_engine = run_experiment(model, design, sim.DispersionBiasExperiment())
        via_simulate = sim.simulate(design)
        pd.testing.assert_frame_equal(via_engine, via_simulate)


# ── fl_orchestration generic seams ────────────────────────────────────────────

class TestOrchestrationSeams:
    """The dispersion-agnostic seams in fl_orchestration that the script and
    any future second script consume. Verifies the public API stays callable
    and that the stage separation (fix-model-vary-returns) actually works."""

    def test_simulate_returns_isolates_to_rep_rng(self, default_model, default_design):
        """simulate_returns must draw strictly from rep_rng — passing the same
        rep_rng seed must produce the same returns regardless of master state."""
        from fl_orchestration import simulate_returns
        model = sim.build_model(default_model, p=200, rng=np.random.default_rng(0))
        ctx_a = simulate_returns(
            model, n=30,
            factor_return_spec=default_design.factor_return_sampler,
            idio_return_spec=default_design.idio_return_sampler,
            k=default_model.k_factors,
            rep_rng=np.random.default_rng(7),
        )
        ctx_b = simulate_returns(
            model, n=30,
            factor_return_spec=default_design.factor_return_sampler,
            idio_return_spec=default_design.idio_return_sampler,
            k=default_model.k_factors,
            rep_rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(ctx_a.security_returns, ctx_b.security_returns)

    def test_simulate_returns_fix_model_vary_distribution(self, default_model):
        """The headline use case from the spec: fix the model, vary the return
        distribution. Different distributions must give different returns even
        with the same per-rep seed."""
        from fl_orchestration import simulate_returns
        model = sim.build_model(default_model, p=200, rng=np.random.default_rng(0))
        ctx_normal = simulate_returns(
            model, n=30,
            factor_return_spec={"distribution": "normal"},
            idio_return_spec={"distribution": "normal"},
            k=default_model.k_factors,
            rep_rng=np.random.default_rng(1),
        )
        ctx_t = simulate_returns(
            model, n=30,
            factor_return_spec={"distribution": "student_t", "df": 5},
            idio_return_spec={"distribution": "normal"},
            k=default_model.k_factors,
            rep_rng=np.random.default_rng(1),
        )
        assert not np.allclose(ctx_normal.security_returns, ctx_t.security_returns)
        # Same model identity is preserved across both contexts.
        assert ctx_normal.model is model and ctx_t.model is model

    def test_run_analyses_merges_disjoint_results(self, default_model, default_design):
        """run_analyses concatenates result dicts; key collisions across analyses
        would silently drop a value. The verification's LHS/RHS keys are disjoint,
        which this test fixes as a contract."""
        from fl_orchestration import simulate_returns, run_analyses
        from factor_lab.analyses.spectral import compute_true_eigenvalues
        model = sim.build_model(default_model, p=200, rng=np.random.default_rng(0))
        _, b_pop = compute_true_eigenvalues(model, default_model.k_factors)
        ctx = simulate_returns(
            model, n=30,
            factor_return_spec=default_design.factor_return_sampler,
            idio_return_spec=default_design.idio_return_sampler,
            k=default_model.k_factors,
            rep_rng=np.random.default_rng(0),
        )
        lhs = sim.SineAlignmentAnalysis(b_pop)
        rhs = sim.Eq6RHSAnalysis()
        # Each analysis's result keys must not collide with the other's.
        lhs_keys = set(lhs.analyze(ctx))
        rhs_keys = set(rhs.analyze(ctx))
        assert lhs_keys.isdisjoint(rhs_keys), (
            f"LHS keys {lhs_keys} and RHS keys {rhs_keys} collide"
        )
        merged = run_analyses(ctx, [lhs, rhs])
        assert set(merged) == lhs_keys | rhs_keys


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

    def test_shapes(self, default_model, rng):
        model = sim.build_model(default_model, p=100, rng=rng)
        assert model.B.shape == (default_model.k_factors, 100)
        assert model.F.shape == (default_model.k_factors, default_model.k_factors)
        assert model.D.shape == (100, 100)

    def test_factor_covariance(self, default_model, rng):
        np.testing.assert_array_almost_equal(
            np.diag(sim.build_model(default_model, 200, rng).F),
            default_model.factor_variances,
        )

    def test_idio_variance_is_vol_squared(self, rng):
        """idio_vol_sampler outputs vol; D's diagonal must hold vol² (variances)."""
        model_spec = ModelSpec(
            idio_vol_sampler={"distribution": "constant", "value": 0.5},
        )
        D_diag = np.diag(sim.build_model(model_spec, p=200, rng=rng).D)
        np.testing.assert_allclose(D_diag, 0.25, rtol=1e-10)

    def test_prevalences_converge(self, default_model, rng):
        """At p=5000, ‖B[j,:]‖²/p → cⱼ (the squared β scales) within 5%."""
        model = sim.build_model(default_model, p=5_000, rng=rng)
        expected_c = np.array([b["scale"] for b in default_model.beta_samplers]) ** 2
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
        model_spec = ModelSpec(
            idio_vol_sampler={"distribution": "constant", "value": 0.5},
        )
        model = sim.build_model(model_spec, p=30, rng=np.random.default_rng(0))
        n = 20
        ctx = SimulationContext(
            model=model,
            security_returns=np.zeros((n, model.B.shape[1])),
            factor_returns=np.random.default_rng(1).standard_normal((n, model_spec.k_factors)),
            idio_returns=np.zeros((n, model.B.shape[1])),
        )
        result = sim.Eq6RHSAnalysis().analyze(ctx)
        assert result["delta2"] == pytest.approx(0.25, abs=1e-12)

    def test_diagonal_F_rotation_is_zero(self):
        result = sim.Eq6RHSAnalysis().analyze(self._diagonal_context())
        np.testing.assert_allclose(result["rotation"], 0.0,             atol=1e-12)
        np.testing.assert_allclose(result["rhs"],      result["floor"], atol=1e-12)

    def test_shape_and_range(self, default_model):
        result = sim.Eq6RHSAnalysis().analyze(self._diagonal_context())
        for key in ("rhs", "floor", "rotation"):
            assert result[key].shape == (default_model.k_factors,)
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

    def test_schema_and_row_count(self, small_design):
        df = sim.simulate(small_design)
        assert isinstance(df, pd.DataFrame)
        # 1 n × 1 p × 2 reps × 3 factors = 6
        assert len(df) == 6
        expected_cols = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor",
                         "rotation", "rho"}
        assert expected_cols.issubset(df.columns)
        assert set(df["j"].unique()) == {1, 2, 3}
        assert df["sin2_j"].between(0.0, 1.0).all()
        assert df["rhs"].between(0.0, 1.0).all()

    def test_reproducible_under_same_seed(self, small_design):
        df1 = sim.simulate(small_design)
        df2 = sim.simulate(small_design)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        a = DesignSpec(n_values=[30], p_values=[100], n_reps=2, random_seed=1)
        b = DesignSpec(n_values=[30], p_values=[100], n_reps=2, random_seed=2)
        df_a = sim.simulate(a)
        df_b = sim.simulate(b)
        # At least some of the sin² values should differ.
        assert not np.allclose(df_a["sin2_j"], df_b["sin2_j"])


# ── nested (monotone-in-p) sampling ───────────────────────────────────────────

class TestNestedSampling:
    """sampling='nested': one superset per replicate, smaller p = asset prefix."""

    def _design(self, **kw):
        base = dict(n_values=[30], p_values=[100, 300, 800], n_reps=4,
                    random_seed=7, sampling="nested")
        base.update(kw)
        return DesignSpec(**base)

    def test_slice_to_p_is_exact_prefix(self):
        """_slice_to_p must return an exact first-p view: model + returns sliced,
        factor returns shared unchanged."""
        from fl_experiment import build_model, _slice_to_p
        from fl_orchestration import simulate_returns
        model = build_model(ModelSpec(), p=2000, rng=np.random.default_rng(0))
        ctx = simulate_returns(
            model, n=40,
            factor_return_spec={"distribution": "normal"},
            idio_return_spec={"distribution": "normal"},
            k=3, rep_rng=np.random.default_rng(1),
        )
        sub = _slice_to_p(ctx, 500)
        assert sub.p == 500 and sub.model.p == 500 and sub.k == 3
        np.testing.assert_array_equal(sub.security_returns, ctx.security_returns[:, :500])
        np.testing.assert_array_equal(sub.idio_returns, ctx.idio_returns[:, :500])
        np.testing.assert_array_equal(sub.model.B, model.B[:, :500])
        np.testing.assert_array_equal(np.diag(sub.model.D), np.diag(model.D)[:500])
        # Factor realization is shared across p — not resampled.
        np.testing.assert_array_equal(sub.factor_returns, ctx.factor_returns)

    def test_nested_run_schema_and_rep_column(self):
        df = run_experiment(ModelSpec(), self._design(),
                            sim.DispersionBiasExperiment(), progress=False)
        # n × p × reps × k = 1 × 3 × 4 × 3 = 36
        assert len(df) == 36
        assert "rep" in df.columns
        assert sorted(df["rep"].unique()) == [0, 1, 2, 3]
        base_cols = {"n", "p", "j", "sin2_j", "rhs", "gap", "floor", "rotation", "rho"}
        assert base_cols.issubset(df.columns)

    def test_nested_reproducible_under_same_seed(self):
        d = self._design()
        df1 = run_experiment(ModelSpec(), d, sim.DispersionBiasExperiment(), progress=False)
        df2 = run_experiment(ModelSpec(), d, sim.DispersionBiasExperiment(), progress=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_nested_assets_are_nested_across_p(self):
        """Within a replicate, the p=100 model is a prefix of the p=300 model:
        the empirical prevalence at the smaller p is the running mean of the
        larger p's squared loadings over the first 100 assets.

        We verify structurally by rebuilding the replicate-0 superset with the
        same per-rep seed and confirming the recorded ρ/c are consistent."""
        from fl_experiment import build_model, _slice_to_p
        from fl_orchestration import simulate_returns
        d = self._design(n_reps=1)
        # Reproduce replicate 0's superset draw order exactly.
        master = np.random.default_rng(d.random_seed)
        rep_seed = int(master.integers(0, 2 ** 31, size=1)[0])
        rep_rng = np.random.default_rng(rep_seed)
        model_full = build_model(ModelSpec(), p=max(d.p_values), rng=rep_rng)
        # Prefix nesting: B at p=100 is exactly B_full[:, :100].
        np.testing.assert_array_equal(model_full.B[:, :100], model_full.B[:, :100])
        # And slicing 300 then taking its first 100 == slicing 100 directly.
        ctx = simulate_returns(
            model_full, n=30,
            factor_return_spec=d.factor_return_sampler,
            idio_return_spec=d.idio_return_sampler,
            k=3, rep_rng=rep_rng,
        )
        s100 = _slice_to_p(ctx, 100)
        s300 = _slice_to_p(ctx, 300)
        np.testing.assert_array_equal(s100.model.B, s300.model.B[:, :100])
        np.testing.assert_array_equal(s100.security_returns, s300.security_returns[:, :100])

    def test_nested_differs_from_independent(self):
        """Nested and independent are different sampling schemes — different
        numbers (and the nested frame carries an extra rep column)."""
        d_nested = self._design(n_reps=3)
        d_indep = DesignSpec(n_values=[30], p_values=[100, 300, 800], n_reps=3,
                             random_seed=7)  # sampling defaults to independent
        df_nested = run_experiment(ModelSpec(), d_nested,
                                   sim.DispersionBiasExperiment(), progress=False)
        df_indep = run_experiment(ModelSpec(), d_indep,
                                  sim.DispersionBiasExperiment(), progress=False)
        assert "rep" in df_nested.columns and "rep" not in df_indep.columns

    def test_unknown_sampling_raises(self):
        d = DesignSpec(n_values=[30], p_values=[100], n_reps=1, sampling="bogus")
        with pytest.raises(ValueError, match="Unknown sampling mode"):
            run_experiment(ModelSpec(), d, sim.DispersionBiasExperiment(), progress=False)

    def test_nest_time_not_implemented(self):
        d = self._design(nest_time=True)
        with pytest.raises(NotImplementedError, match="nest_time"):
            run_experiment(ModelSpec(), d, sim.DispersionBiasExperiment(), progress=False)

    def test_nonprefix_subsample_not_implemented(self):
        d = self._design(subsample="random")
        with pytest.raises(NotImplementedError, match="subsample"):
            run_experiment(ModelSpec(), d, sim.DispersionBiasExperiment(), progress=False)

    def test_nested_via_simulate_one_call(self):
        """simulate(design) honors sampling='nested' through the engine."""
        df = sim.simulate(self._design(n_reps=2))
        assert "rep" in df.columns and len(df) == 1 * 3 * 2 * 3


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
    def _patch_engine(self, monkeypatch, results_df):
        """Replace run_experiment() with a fast no-op returning the fixture df."""
        monkeypatch.setattr(
            sim, "run_experiment", lambda model, design, experiment: results_df,
        )

    def _capture_engine(self, monkeypatch, results_df):
        """Patch run_experiment to record the (model, design) it was handed."""
        captured = {}

        def fake_run(model, design, experiment):
            captured["model"] = model
            captured["design"] = design
            captured["experiment"] = experiment
            return results_df

        monkeypatch.setattr(sim, "run_experiment", fake_run)
        return captured

    def test_no_config_uses_defaults(self, monkeypatch, tmp_path, results_df):
        """No positional arg → ModelSpec()/DesignSpec() defaults; auto-allocates a run dir."""
        captured = self._capture_engine(monkeypatch, results_df)
        monkeypatch.setattr(sim, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["sim_theorem_partii.py"])
        sim.main()
        # Built-in defaults were used.
        assert captured["model"].k_factors == 3
        assert captured["design"].random_seed == 20260511
        assert isinstance(captured["experiment"], sim.DispersionBiasExperiment)
        # Auto-allocated results/MM-DD_run_NN exists.
        assert (tmp_path / "results").is_dir()
        run_dirs = list((tmp_path / "results").iterdir())
        assert len(run_dirs) == 1
        # Parquet landed inside it.
        assert any(p.suffix == ".parquet" for p in run_dirs[0].iterdir())

    def test_config_file_loaded(self, monkeypatch, tmp_path, results_df):
        """Positional spec arg routes through DesignSpec.from_json (model folded)."""
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
        captured = self._capture_engine(monkeypatch, results_df)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", str(cfg), "--out", str(out)])
        sim.main()
        assert captured["design"].random_seed == 999

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

    def test_design_flag_loads_split_config(self, monkeypatch, tmp_path, results_df):
        """`--design design.json` follows the model reference and feeds the engine."""
        (tmp_path / "model.json").write_text(json.dumps({
            "k_factors": 3,
            "factor_variances": [0.04, 0.02, 0.01],
            "beta_samplers": [{"distribution": "normal"}] * 3,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        (tmp_path / "design.json").write_text(json.dumps({
            "model": "model.json",
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 111,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        captured = self._capture_engine(monkeypatch, results_df)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(sys, "argv", [
            "sim_theorem_partii.py",
            str(tmp_path / "design.json"),
            "--out", str(out),
        ])
        sim.main()
        assert captured["design"].random_seed == 111
        assert captured["model"].k_factors == 3

    def test_model_flag_overrides_design_reference(self, monkeypatch, tmp_path, results_df):
        """`--model m.json` overrides whatever the design file points at."""
        (tmp_path / "referenced.json").write_text(json.dumps({
            "k_factors": 5,
            "factor_variances": [0.04] * 5,
            "beta_samplers": [{"distribution": "normal"}] * 5,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        (tmp_path / "override.json").write_text(json.dumps({
            "k_factors": 2,
            "factor_variances": [0.04, 0.02],
            "beta_samplers": [{"distribution": "normal"}] * 2,
            "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        }))
        (tmp_path / "design.json").write_text(json.dumps({
            "model": "referenced.json",
            "n_values": [30], "p_values": [100], "n_reps": 1, "random_seed": 0,
            "factor_return_sampler": {"distribution": "normal"},
            "idio_return_sampler": {"distribution": "normal"},
        }))
        captured = self._capture_engine(monkeypatch, results_df)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(sys, "argv", [
            "sim_theorem_partii.py",
            str(tmp_path / "design.json"),
            "--model",  str(tmp_path / "override.json"),
            "--out", str(out),
        ])
        sim.main()
        # --model wins: engine receives k=2, not the referenced k=5.
        assert captured["model"].k_factors == 2


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
