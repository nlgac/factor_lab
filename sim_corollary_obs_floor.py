"""
sim_corollary_obs_floor.py
==========================
A *third* theorem probe on the same engine as ``sim_theorem_partii.py`` and
``sim_corollary4.py`` — verifying the **observable lower bound** (the
"observable floor" corollary) of *Multifactor Dispersion Bias with Per-Column
Prevalence*.

The claim
---------
Equation (6) Part (ii) gives the analytic floor δ²/(nρⱼ+δ²) as the irreducible,
out-of-subspace part of the alignment defect sin²∠(hⱼ, b̄ⱼ). That floor depends
on ρⱼ — an eigenvalue of D̂ = C^{1/2}(FᵀF/n)C^{1/2}, built from the *latent*
loadings/Gram and factor returns — so it is not directly observable.

The corollary says it can be recovered anyway, from the sample spectrum alone.
With s_{p,1} ≥ … ≥ s_{p,n} the singular values of Y/√n (so s²_{p,j} are the dual
sample eigenvalues), define the average non-spike eigenvalue and per-spike ratio

    ℓ²_p := (1/(n-k)) Σ_{j>k} s²_{p,j},     ψ²_{p,j} := 1 − ℓ²_p / s²_{p,j}.

Then, conditional on F and a.s. as p → ∞,

    ℓ²_p / s²_{p,j}  ⟶  δ² / (n ρⱼ + δ²)   =   the analytic floor,

so ℓ²_p / s²_{p,j} is an *observable* estimator of the otherwise-latent floor,
and a lower bound on the squared cosine ⟨hⱼ, b̄ⱼ⟩² in the limit.

Scale invariance
----------------
Both ℓ²_p and s²_{p,j} carry the same 1/n from the Y/√n convention, so the ratio
ℓ²_p / s²_{p,j} is computed directly from the **raw** dual-Gram spectrum
g = eig(YᵀY) — the 1/n cancels. No covariance scaling is needed.

Composition
-----------
This probe stands alone: unlike ``sim_corollary4.py`` (which reuses
``Eq6RHSAnalysis``), the analytic floor here is **inlined** so the file has no
dependency on the Part (ii) probe. One analysis, ``ObservableFloorAnalysis``,
emits all three per-factor series.

Record schema: **k per-factor rows per replication**,
``(n, p, j, sin2_j, floor_obs, floor_true, gap)`` where
``gap = floor_obs − floor_true`` is the quantity whose RMSE → 0.

Usage
-----
    python sim_corollary_obs_floor.py            # small smoke sweep + RMSE table
    from fl_experiment_setup import get_experiment
    ObservableFloorExperiment = get_experiment("observable_floor")

Notebook idiom
--------------
    from fl_experiment_setup import ModelSpec, DesignSpec
    from fl_experiment_runner import run_experiment
    from sim_corollary_obs_floor import ObservableFloorExperiment
    df = run_experiment(ModelSpec(), DesignSpec(), ObservableFloorExperiment())
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.analysis import SimulationContext
from factor_lab.analyses.spectral import compute_true_eigenvalues
from factor_lab.analyses import compute_sine_alignment

from fl_experiment_setup import (
    ModelSpec, DesignSpec, BaseExperiment, register_experiment,
)
from fl_experiment_runner import run_experiment

__all__ = [
    "ObservableFloorAnalysis",
    "ObservableFloorExperiment",
    "simulate",
    "print_summary",
    "main",
]


# ── Observed LHS + observable proxy + analytic floor ─────────────────────────


class ObservableFloorAnalysis:
    """Per-factor observable-floor quantities for one replication.

    Computes, for each factor j ∈ {1, …, k}:

      sin2_j      observed alignment defect  sin²∠(hⱼ, b̄ⱼ)            [LHS]
      floor_obs   observable proxy           ℓ²_p / s²_{p,j}          [corollary]
      floor_true  analytic floor             δ² / (n ρⱼ + δ²)         [Theorem 1]

    ``hⱼ`` is the j-th top left singular vector of Y (computed via the n×n Gram
    trick — uncentered, matching the theorem). ``b̄ⱼ`` (rows of ``b_pop``, the
    population loading directions, eigenvectors of Σ₀ = BΣ_F Bᵀ/p) is injected at
    construction so ARPACK runs once per (n, p) cell.

    The proxy uses the raw dual-Gram spectrum g = eig(YᵀY): with the spikes the
    top k and the non-spikes the remaining n − k,

        floor_obs_j = mean(g_{k+1}, …, g_n) / g_j   =   ℓ²_p / s²_{p,j},

    the common 1/n of the Y/√n convention cancelling between numerator and
    denominator.

    The analytic floor is **inlined** (no import of the Part (ii) RHS): ρⱼ are the
    eigenvalues of D̂ = C^{1/2}(FᵀF/n)C^{1/2}, with empirical prevalences
    cⱼ = ‖B[j,:]‖²/p and δ² the mean idiosyncratic variance from ``model.D``.

    Example:
        _, b_pop = compute_true_eigenvalues(model, model.k)
        analysis = ObservableFloorAnalysis(b_pop)
        result = analysis.analyze(context)
        # {"sin2_j": (k,), "floor_obs": (k,), "floor_true": (k,)}
    """

    def __init__(self, b_pop: np.ndarray):
        self.b_pop = b_pop   # (k, p), rows are population loading directions b̄ⱼ

    def analyze(self, context: SimulationContext) -> dict:
        k, n = context.k, context.T
        Y = context.security_returns.T            # (p, n)

        # n×n dual Gram (uncentered); descending spectrum g₁ ≥ … ≥ g_n ≥ 0.
        G = Y.T @ Y
        vals, vecs = np.linalg.eigh(G)
        order = np.argsort(vals)[::-1]
        g = np.maximum(vals[order], 0.0)
        V_top = vecs[:, order[:k]]

        # Observed LHS: hⱼ = top-k left singular frame of Y via the Gram trick.
        s = np.sqrt(g[:k])
        H = (Y @ V_top) / np.where(s > 1e-14, s, 1.0)     # (p, k)
        sin2, _ = compute_sine_alignment(self.b_pop, H.T)

        # Observable proxy ℓ²_p / s²_{p,j}; the 1/n cancels, so use raw g.
        ell2 = g[k:n].mean()                              # average non-spike eig
        floor_obs = ell2 / g[:k]                          # length-k

        # Analytic floor δ²/(nρⱼ+δ²), inlined (mirrors Eq6RHSAnalysis lines 206–214).
        F = context.factor_returns.T                      # (k, n)
        c_half = np.sqrt((context.model.B ** 2).mean(axis=1))   # √cⱼ
        D_hat = (c_half[:, None] * (F @ F.T / n)) * c_half[None, :]
        rhos = np.sort(np.linalg.eigvalsh(D_hat))[::-1]   # descending ρⱼ
        delta2 = float(np.diag(context.model.D).mean())
        floor_true = delta2 / (n * rhos + delta2)

        return {"sin2_j": sin2, "floor_obs": floor_obs, "floor_true": floor_true}


# ── The probe ─────────────────────────────────────────────────────────────────


def _rep_records(k: int, n: int, p: int, res: dict) -> list[dict]:
    """Flatten one replication's result into k per-factor rows."""
    return [
        {"n": n, "p": p, "j": j + 1,
         "sin2_j":     float(res["sin2_j"][j]),
         "floor_obs":  float(res["floor_obs"][j]),
         "floor_true": float(res["floor_true"][j]),
         "gap":        float(res["floor_obs"][j] - res["floor_true"][j])}
        for j in range(k)
    ]


@register_experiment("observable_floor")
class ObservableFloorExperiment(BaseExperiment):
    """Observable-floor probe: proxy ℓ²ₚ/s²ₚⱼ vs analytic floor δ²/(nρⱼ+δ²).

    ``cell_setup`` computes the population directions once per cell (ARPACK,
    RNG-free) and returns the single self-contained analysis. ``record`` emits k
    per-factor rows per replication. No engine change; a new question is a new
    Experiment.

    Notebook idiom::

        from fl_experiment_setup import ModelSpec, DesignSpec
        from fl_experiment_runner import run_experiment
        from sim_corollary_obs_floor import ObservableFloorExperiment
        df = run_experiment(ModelSpec(), DesignSpec(), ObservableFloorExperiment())
    """

    def cell_setup(self, model, n: int, p: int):
        _, b_pop = compute_true_eigenvalues(model, model.k)
        return [ObservableFloorAnalysis(b_pop)]

    def record(self, n: int, p: int, merged: dict) -> list[dict]:
        k = len(merged["sin2_j"])
        return _rep_records(k, n, p, merged)


# ── Single-call driver ────────────────────────────────────────────────────────


def simulate(design: DesignSpec, *, base_dir: Path = ROOT) -> pd.DataFrame:
    """Run the observable-floor verification from a :class:`DesignSpec`.

    Resolves the design's model (inline, referenced, or defaults) and hands both
    to the generic engine with the observable-floor probe.
    """
    model = design.resolve_model(base_dir)
    return run_experiment(model, design, ObservableFloorExperiment())


# ── Summary + entry point ─────────────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    """RMSE of the gap (proxy − analytic floor) by (n, p, j); should → 0 in p."""
    tbl = (
        df.groupby(["n", "p", "j"])["gap"]
        .apply(lambda s: float(np.sqrt((s ** 2).mean())))
        .rename("RMSE")
        .reset_index()
        .pivot(index=["n", "p"], columns="j", values="RMSE")
    )
    tbl.columns = [f"j={c}" for c in tbl.columns]
    print("\nRMSE of (ℓ²ₚ/s²ₚⱼ − δ²/(nρⱼ+δ²))  [smaller is better; → 0 as p grows]\n")
    print(tbl.to_string(float_format="{:.5f}".format))
    print()


def main() -> None:
    logger.info("Observable floor (Corollary) — small smoke sweep")
    design = DesignSpec(
        n_values=[60], p_values=[200, 500, 1000, 2000, 5000],
        n_reps=50, random_seed=20260511,
    )
    df = run_experiment(ModelSpec(), design, ObservableFloorExperiment())
    print("rows:", len(df), " columns:", list(df.columns))
    print_summary(df)


if __name__ == "__main__":
    main()
