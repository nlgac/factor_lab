"""
tests/test_pipeline_e2e.py
==========================
End-to-end pipeline tests: run the REAL pipeline unmocked through the public
entry points (run_experiment / simulate / the CLI main), asserting both on the
outputs (schema, values, files on disk, reproducibility, the RMSE-falls-with-p
trend) AND on the loguru stage logs the pipeline emits.

Unlike the unit tests in test_sim_theorem_partii.py, nothing here is mocked —
these exercise config → model → returns → analysis → DataFrame → parquet/figures
through the same calls the script and notebook make.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_theorem_partii as sim
from fl_experiment import ModelSpec, DesignSpec, run_experiment


# ── loguru capture ────────────────────────────────────────────────────────────

@pytest.fixture
def captured_logs():
    """Capture every loguru record (TRACE and above) emitted during the test.

    Adds a temporary sink that appends each record dict to a list, then removes
    it on teardown so it cannot leak into other tests.
    """
    records: list[dict] = []
    sink_id = logger.add(lambda m: records.append(m.record), level="TRACE")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _messages(records, level=None):
    return [r["message"] for r in records
            if level is None or r["level"].name == level]


def _level_of(records, needle):
    """Return the level name of the first captured message containing ``needle``."""
    for r in records:
        if needle in r["message"]:
            return r["level"].name
    return None


# A tiny config: fast, but enough cells/reps to exercise the whole pipeline.
def _small_design(**kw):
    base = dict(n_values=[30], p_values=[100, 300], n_reps=5, random_seed=2026)
    base.update(kw)
    return DesignSpec(**base)


def _design_json(tmp_path, **overrides):
    cfg = {
        "k_factors": 3,
        "n_values": [30],
        "p_values": [100, 300],
        "n_reps": 5,
        "random_seed": 2026,
        "factor_vols": [0.04, 0.02, 0.01],
        "beta_samplers": [{"distribution": "normal"}] * 3,
        "idio_vol_sampler": {"distribution": "constant", "value": 1.0},
        "factor_return_sampler": {"distribution": "normal"},
        "idio_return_sampler": {"distribution": "normal"},
    }
    cfg.update(overrides)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(cfg))
    return path


# ── run_experiment / simulate end to end ──────────────────────────────────────

class TestEngineE2E:

    def test_run_experiment_full_pipeline(self):
        df = run_experiment(ModelSpec(), _small_design(),
                            sim.DispersionBiasExperiment(), progress=False)
        # 1 n × 2 p × 5 reps × 3 j = 30
        assert len(df) == 30
        assert set(df.columns) == {"n", "p", "j", "sin2_j", "rhs", "gap",
                                   "floor", "rotation", "rho"}
        assert set(df["j"].unique()) == {1, 2, 3}
        assert set(df["p"].unique()) == {100, 300}
        assert df["sin2_j"].between(0.0, 1.0).all()
        assert df["rhs"].between(0.0, 1.0).all()
        # gap is exactly sin2_j - rhs
        np.testing.assert_allclose(df["gap"], df["sin2_j"] - df["rhs"], atol=1e-12)

    def test_simulate_one_call_matches_run_experiment(self):
        design = _small_design()
        via_simulate = sim.simulate(design)
        via_engine = run_experiment(design.resolve_model(ROOT), design,
                                    sim.DispersionBiasExperiment(), progress=False)
        pd.testing.assert_frame_equal(via_simulate, via_engine)

    def test_reproducible_across_full_pipeline(self):
        d = _small_design()
        pd.testing.assert_frame_equal(sim.simulate(d), sim.simulate(d))

    def test_rmse_falls_with_p(self):
        """The headline confirmation: pooled RMSE of (sin² − RHS) for the
        strongest factor shrinks as p grows. Run a sweep with real signal and
        assert the largest-p RMSE is below the smallest-p RMSE."""
        design = DesignSpec(n_values=[60], p_values=[200, 4000],
                            n_reps=25, random_seed=2026)
        df = run_experiment(ModelSpec(), design,
                            sim.DispersionBiasExperiment(), progress=False)
        rmse = (df[df["j"] == 1].groupby("p")["gap"]
                .apply(lambda g: float(np.sqrt((g ** 2).mean()))))
        assert rmse.loc[4000] < rmse.loc[200], rmse.to_dict()


# ── stage logging ─────────────────────────────────────────────────────────────

class TestPipelineLogging:

    def test_stage_logs_emitted(self, captured_logs):
        run_experiment(ModelSpec(), _small_design(),
                       sim.DispersionBiasExperiment(), progress=False)
        msgs = "\n".join(_messages(captured_logs))
        for needle in ("Running independent sweep", "Starting n = 30",
                       "built model", "sampled returns", "ran 2 analyses",
                       "cell n=30", "Sweep complete"):
            assert needle in msgs, f"missing stage log: {needle!r}"

    def test_stage_log_levels(self, captured_logs):
        """Each stage logs at its intended level — INFO for the sweep envelope,
        DEBUG for per-cell structure, TRACE for per-rep work."""
        run_experiment(ModelSpec(), _small_design(),
                       sim.DispersionBiasExperiment(), progress=False)
        assert _level_of(captured_logs, "Running independent sweep") == "INFO"
        assert _level_of(captured_logs, "Sweep complete") == "INFO"
        assert _level_of(captured_logs, "built model") == "DEBUG"
        assert _level_of(captured_logs, "cell n=30") == "DEBUG"
        assert _level_of(captured_logs, "sampled returns") == "TRACE"
        assert _level_of(captured_logs, "ran 2 analyses") == "TRACE"

    def test_sweep_complete_reports_row_count(self, captured_logs):
        df = run_experiment(ModelSpec(), _small_design(),
                            sim.DispersionBiasExperiment(), progress=False)
        complete = [m for m in _messages(captured_logs) if "Sweep complete" in m]
        assert complete and str(len(df)) in complete[-1]

    def test_nested_logs_superset_build(self, captured_logs):
        run_experiment(ModelSpec(), _small_design(n_reps=3, sampling="nested"),
                       sim.DispersionBiasExperiment(), progress=False)
        msgs = "\n".join(_messages(captured_logs))
        assert "Running nested sweep" in msgs
        assert "built superset model at p_max=300" in msgs


# ── CLI main() on disk ─────────────────────────────────────────────────────────

class TestCLIPipeline:

    def test_main_writes_and_reloads_parquet(self, tmp_path, monkeypatch, captured_logs):
        cfg = _design_json(tmp_path)
        out = tmp_path / "result.parquet"
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", str(cfg), "--out", str(out)])
        sim.main()
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 30
        assert {"n", "p", "j", "sin2_j", "rhs", "gap"}.issubset(df.columns)
        # Pipeline announced the save and the sweep completion.
        msgs = "\n".join(_messages(captured_logs))
        assert "Saved 30 rows" in msgs and "Sweep complete" in msgs

    def test_main_auto_allocates_run_dir(self, tmp_path, monkeypatch, captured_logs):
        cfg = _design_json(tmp_path)
        monkeypatch.setattr(sim, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["sim_theorem_partii.py", str(cfg)])
        sim.main()
        run_dirs = list((tmp_path / "results").iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "sim_thmptii.parquet").exists()
        assert "Auto-allocated run directory" in "\n".join(_messages(captured_logs))

    def test_main_plot_save_writes_figures(self, tmp_path, monkeypatch):
        cfg = _design_json(tmp_path)
        out = tmp_path / "result.parquet"
        monkeypatch.setattr(sys, "argv", ["sim_theorem_partii.py", str(cfg),
                                          "--plot-save", "--out", str(out)])
        sim.main()
        assert out.exists()
        for fig in ("fig_theorem1_convergence_v2.png",
                    "fig_theorem1_scatter_v2.png",
                    "fig_theorem1_components_v2.png"):
            assert (tmp_path / fig).exists(), f"missing figure {fig}"

    def test_main_nested_parquet_has_rep_column(self, tmp_path, monkeypatch):
        cfg = _design_json(tmp_path, sampling="nested", n_reps=3)
        out = tmp_path / "nested.parquet"
        monkeypatch.setattr(sys, "argv",
                            ["sim_theorem_partii.py", str(cfg), "--out", str(out)])
        sim.main()
        df = pd.read_parquet(out)
        assert "rep" in df.columns
        assert sorted(df["rep"].unique()) == [0, 1, 2]
