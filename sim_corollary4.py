"""
sim_corollary4.py
=================
A *second* theorem probe, built to exercise the same engine as
``sim_theorem_partii.py`` — and to show that a new question plugs in by writing
one ``Experiment``, reusing existing analyses, with no engine change.

Checks **Corollary 4** (Grassmannian subspace distance) of *Multifactor
Dispersion Bias with Per-Column Prevalence*: as p → ∞ (conditional on F, a.s.),

    d_Gr²(col(H), B)  →  Σ_j  δ² / (n ρ_j + δ²)

i.e. the squared subspace distance between the sample loading subspace col(H) and
the population loading subspace col(B̄) converges to the **sum of the per-factor
floors** — the in-subspace rotation cancels in the Grassmannian metric.

Composition in action
---------------------
- LHS (observed): a new ``SubspaceDistanceAnalysis`` — the squared principal-angle
  distance between col(H) and col(B̄).
- RHS (predicted): the *existing* ``Eq6RHSAnalysis`` from the dispersion probe,
  summed over factors (``Σ floor_j``). Reused verbatim — no copy.

The record schema differs from the dispersion probe: **one scalar row per
replication** ``(n, p, d_gr2_obs, d_gr2_pred, gap)`` instead of k per-factor rows.
That difference is the point — it pressure-tests the Experiment seam.

Usage
-----
    python sim_corollary4.py                 # small smoke sweep + RMSE table
    from fl_experiment_setup import get_experiment
    SubspaceDistanceExperiment = get_experiment("subspace_distance")
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
from fl_experiment_setup import (
    ModelSpec, DesignSpec, BaseExperiment, register_experiment,
)
from fl_experiment_runner import run_experiment
from sim_theorem_partii import Eq6RHSAnalysis

__all__ = ["SubspaceDistanceAnalysis", "SubspaceDistanceExperiment", "main"]


# ── Observed LHS: subspace distance ───────────────────────────────────────────


class SubspaceDistanceAnalysis:
    """Observed LHS of Corollary 4 for one replication.

    Squared Grassmannian (projection) distance between the sample loading
    subspace col(H) and the population subspace col(B̄):

        d² = Σ_l sin²θ_l = k − ‖Hᵀ B̄‖_F²,

    where θ_l are the principal angles. H is the top-k left singular frame of Y
    (via the n×n Gram trick); B̄ rows are the population loading directions
    (eigenvectors of Σ₀), injected at construction.
    """

    def __init__(self, b_pop: np.ndarray):
        self.b_pop = b_pop   # (k, p), orthonormal rows

    def analyze(self, context: SimulationContext) -> dict:
        k = context.k
        Y = context.security_returns.T   # (p, n)
        G = Y.T @ Y
        vals, vecs = np.linalg.eigh(G)
        idx = np.argsort(vals)[::-1][:k]
        s = np.sqrt(np.maximum(vals[idx], 0.0))
        H = (Y @ vecs[:, idx]) / np.where(s > 1e-14, s, 1.0)   # (p, k), orthonormal cols
        # Principal-angle cosines between col(H) and col(B̄).
        cos = np.linalg.svd(H.T @ self.b_pop.T, compute_uv=False)
        cos = np.clip(cos, 0.0, 1.0)
        return {"d_gr2_obs": float(np.sum(1.0 - cos ** 2))}


# ── The probe ─────────────────────────────────────────────────────────────────


@register_experiment("subspace_distance")
class SubspaceDistanceExperiment(BaseExperiment):
    """Corollary 4 probe: observed subspace distance vs. predicted Σ of floors.

    ``cell_setup`` computes the population directions once per cell (ARPACK) and
    returns the new LHS analysis paired with the *reused* ``Eq6RHSAnalysis``.
    ``record`` emits one scalar row per replication, summing Eq6RHS's per-factor
    floors for the prediction.
    """

    def cell_setup(self, model, n: int, p: int):
        _, b_pop = compute_true_eigenvalues(model, model.k)
        return [SubspaceDistanceAnalysis(b_pop), Eq6RHSAnalysis()]

    def record(self, n: int, p: int, merged: dict) -> list[dict]:
        # RHS = Σ_j δ²/(nρ_j+δ²) — exactly the sum of Eq6RHS's per-factor floors.
        pred = float(np.sum(merged["floor"]))
        obs = float(merged["d_gr2_obs"])
        return [{"n": n, "p": p, "d_gr2_obs": obs, "d_gr2_pred": pred,
                 "gap": obs - pred}]


# ── Summary + entry point ─────────────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    """RMSE of (d_Gr²_obs − d_Gr²_pred) by (n, p); should fall as p grows."""
    tbl = (
        df.groupby(["n", "p"])["gap"]
        .apply(lambda g: float(np.sqrt((g ** 2).mean())))
        .rename("RMSE").reset_index().pivot(index="n", columns="p", values="RMSE")
    )
    print("\nRMSE of (d_Gr² − Σ floors)  [smaller is better; should → 0 as p grows]\n")
    print(tbl.to_string(float_format="{:.5f}".format))
    print()


def main() -> None:
    logger.info("Corollary 4 (subspace distance) — small smoke sweep")
    design = DesignSpec(
        n_values=[60], p_values=[200, 500, 1000, 2000, 5000],
        n_reps=50, random_seed=20260511,
    )
    df = run_experiment(ModelSpec(), design, SubspaceDistanceExperiment())
    print("rows:", len(df), " columns:", list(df.columns))
    print_summary(df)


if __name__ == "__main__":
    main()
