"""
tests/test_experiment_framework.py
==================================
The generic Experiment scaffolding: BaseExperiment (default setup + fail-fast
hook validation), the name registry (register_experiment / get_experiment), and
a second probe (Corollary 4 / subspace distance) that reuses the engine and an
existing analysis through a *different* record schema — the seam's stress test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl_experiment_setup import (
    BaseExperiment, register_experiment, get_experiment, registered_experiments,
    ModelSpec, DesignSpec,
)
from fl_experiment_runner import run_experiment
import sim_theorem_partii as sim
import sim_corollary4 as cor4


# ── BaseExperiment ────────────────────────────────────────────────────────────

class TestBaseExperiment:

    def test_default_setup_is_noop(self):
        class OK(BaseExperiment):
            def cell_setup(self, model, n, p): return []
            def record(self, n, p, merged): return []
        assert OK().setup() is None   # inherited no-op

    def test_missing_record_raises_at_definition(self):
        with pytest.raises(TypeError, match=r"must define record\(\)"):
            class Bad(BaseExperiment):
                def cell_setup(self, model, n, p): return []

    def test_missing_cell_setup_raises_at_definition(self):
        with pytest.raises(TypeError, match=r"must define cell_setup\(\)"):
            class Bad(BaseExperiment):
                def record(self, n, p, merged): return []


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_shipped_probes_registered(self):
        assert get_experiment("dispersion_bias") is sim.DispersionBiasExperiment
        assert get_experiment("subspace_distance") is cor4.SubspaceDistanceExperiment
        assert {"dispersion_bias", "subspace_distance"}.issubset(registered_experiments())

    def test_experiment_name_attribute_set(self):
        assert sim.DispersionBiasExperiment.experiment_name == "dispersion_bias"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="no experiment registered"):
            get_experiment("does_not_exist")

    def test_register_and_lookup_roundtrip(self):
        @register_experiment("_tmp_probe_roundtrip")
        class Tmp(BaseExperiment):
            def cell_setup(self, model, n, p): return []
            def record(self, n, p, merged): return []
        assert get_experiment("_tmp_probe_roundtrip") is Tmp
        assert Tmp.experiment_name == "_tmp_probe_roundtrip"

    def test_duplicate_name_raises(self):
        @register_experiment("_tmp_probe_dup")
        class A(BaseExperiment):
            def cell_setup(self, model, n, p): return []
            def record(self, n, p, merged): return []
        with pytest.raises(ValueError, match="already registered"):
            @register_experiment("_tmp_probe_dup")
            class B(BaseExperiment):
                def cell_setup(self, model, n, p): return []
                def record(self, n, p, merged): return []

    def test_register_rejects_class_missing_hooks(self):
        with pytest.raises(TypeError, match=r"must define record\(\)"):
            @register_experiment("_tmp_probe_bad")
            class Bad:   # not a BaseExperiment, missing record
                def cell_setup(self, model, n, p): return []

    def test_register_rejects_non_class(self):
        with pytest.raises(TypeError, match="decorates a class"):
            register_experiment("_tmp_fn")(lambda: None)


# ── Second probe: Corollary 4 (subspace distance) ─────────────────────────────

class TestSubspaceDistanceProbe:
    """The seam's stress test: a new theorem via one Experiment, reusing
    Eq6RHSAnalysis, with a one-row-per-rep schema instead of k per-factor rows."""

    def _design(self, **kw):
        base = dict(n_values=[60], p_values=[200, 5000], n_reps=40, random_seed=1)
        base.update(kw)
        return DesignSpec(**base)

    def test_runs_through_same_engine_scalar_schema(self):
        df = run_experiment(ModelSpec(), self._design(),
                            cor4.SubspaceDistanceExperiment(), progress=False)
        # one row per (n, p, rep) — NOT per factor
        assert len(df) == 1 * 2 * 40
        assert set(df.columns) == {"n", "p", "d_gr2_obs", "d_gr2_pred", "gap"}
        assert "j" not in df.columns

    def test_reuses_eq6rhs_floors_for_prediction(self):
        """d_gr2_pred must equal Σ_j floor_j from the reused Eq6RHSAnalysis."""
        from fl_experiment_setup import build_model, simulate_returns, run_analyses
        from factor_lab.analyses.spectral import compute_true_eigenvalues
        model = build_model(ModelSpec(), p=1000, rng=np.random.default_rng(0))
        _, b_pop = compute_true_eigenvalues(model, model.k)
        ctx = simulate_returns(
            model, n=60,
            factor_return_spec={"distribution": "normal"},
            idio_return_spec={"distribution": "normal"},
            k=3, rep_rng=np.random.default_rng(1),
        )
        merged = run_analyses(ctx, [cor4.SubspaceDistanceAnalysis(b_pop), sim.Eq6RHSAnalysis()])
        rows = cor4.SubspaceDistanceExperiment().record(60, 1000, merged)
        assert rows[0]["d_gr2_pred"] == pytest.approx(float(np.sum(merged["floor"])))

    def test_reproducible(self):
        d = self._design()
        df1 = run_experiment(ModelSpec(), d, cor4.SubspaceDistanceExperiment(), progress=False)
        df2 = run_experiment(ModelSpec(), d, cor4.SubspaceDistanceExperiment(), progress=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_subspace_distance_converges_with_p(self):
        """Corollary 4: pooled RMSE of (d_Gr² − Σ floors) falls as p grows."""
        df = run_experiment(ModelSpec(), self._design(),
                            cor4.SubspaceDistanceExperiment(), progress=False)
        rmse = df.groupby("p")["gap"].apply(lambda g: float(np.sqrt((g ** 2).mean())))
        assert rmse.loc[5000] < rmse.loc[200]

    def test_observed_distance_in_range(self):
        df = run_experiment(ModelSpec(), self._design(n_reps=10),
                            cor4.SubspaceDistanceExperiment(), progress=False)
        # d_Gr² for a k=3 subspace lies in [0, k].
        assert df["d_gr2_obs"].between(0.0, 3.0 + 1e-9).all()

    def test_print_summary_output(self, capsys):
        df = run_experiment(ModelSpec(), self._design(n_reps=8),
                            cor4.SubspaceDistanceExperiment(), progress=False)
        cor4.print_summary(df)
        out = capsys.readouterr().out
        assert "RMSE of" in out and "d_Gr" in out
