#!/usr/bin/env python3
"""
standalone_dispersion_9panel.py  — DRIVER (no theorem math here)
================================================================
Large-p, heavy-tailed dispersion-bias experiment + 9-panel figure.

This script contains **no theorem math**. Every quantity in Eq. (17) — the measured
``sin^2 angle(h_j, b_j)`` and the ``floor + weight*rotation`` decomposition — is computed by the
*unmodified* ``sim_theorem_partii.DispersionBiasExperiment``, driven by
``fl_experiment_runner.run_experiment``. This file only: (1) defines the model + sweep, (2) calls
the experiment, (3) draws the figure from the result frame.

To run it, drop this file together with the three modules shipped in the same tarball —

    fl_experiment_setup.py    fl_experiment_runner.py    sim_theorem_partii.py

— next to a base ``factor_lab`` package (i.e. import ``factor_lab`` must work), then:

    python standalone_dispersion_9panel.py                       # defaults
    python standalone_dispersion_9panel.py --reps 100 --pmax 100000
    python standalone_dispersion_9panel.py --out fig.png

Requires: a base ``factor_lab`` install exposing FactorModelBuilder, FlexibleReturnsSimulator,
create_sampler, SimulationContext, FactorModelData, and
factor_lab.analyses.{compute_true_eigenvalues, compute_sine_alignment, register_manifold_distance}
(plus numpy, pandas, matplotlib, loguru; tqdm optional).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; delete this line for an interactive window
import matplotlib.pyplot as plt

from loguru import logger
logger.remove()                # keep the console clean (the tqdm bar is enough)

# ── The experiment + the theorem live entirely in these (unmodified) modules ──
from fl_experiment_setup import ModelSpec, DesignSpec
from fl_experiment_runner import run_experiment
from sim_theorem_partii import DispersionBiasExperiment      # <- all Eq.(17) math is in here

# ── Configuration (model + heavy-tailed return process) ───────────────────────
MODEL = ModelSpec(
    k_factors=3,
    factor_vols=[0.16, 0.08, 0.06],                          # sigma_j
    beta_samplers=[
        {"distribution": "normal", "loc": 1.0, "scale": 1.0},   # factor 1: market-like, c_1 = 2
        {"distribution": "normal", "loc": 0.0, "scale": 1.0},   # factor 2: zero-mean
        {"distribution": "normal", "loc": 0.0, "scale": 1.0},   # factor 3: zero-mean
    ],
    idio_vol_sampler={"distribution": "constant", "value": 0.4},   # delta = 0.4
)

# Heavy tails: Student-t factor + idiosyncratic returns. df > 4 keeps a finite 4th moment
# (the theorem's assumption); df <= 4 stresses it and slows convergence.
HEAVY_TAIL = dict(
    factor_return_sampler={"distribution": "student_t", "df": 6, "loc": 0.0, "scale": 1.0},
    idio_return_sampler={"distribution": "student_t", "df": 5, "loc": 0.0, "scale": 1.0},
)

P_VALUES = [200, 500, 1000, 2000, 5000, 10000, 20000]


# ── 9-panel figure (plotting only — no theorem math) ──────────────────────────
def plot_9panel(df: pd.DataFrame, out_path: str):
    """Draw the 3x3 figure from a DispersionBiasExperiment result frame
    (columns n, p, j, sin2_j, rhs, floor): sin^2 decomposition | paired gap | out-of-subspace share."""
    NAVY, GRAY = "#1f3864", "#555555"
    COLORS = ["tab:blue", "tab:orange", "tab:green"]
    SIN2 = ([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.0"])

    g = df.assign(gap=df["sin2_j"] - df["rhs"])              # residual, for the plot only
    agg = g.groupby(["p", "j"]).agg(
        s2_meas=("sin2_j", "mean"), s2_meas_se=("sin2_j", "sem"),
        s2_theory=("rhs", "mean"), s2_theory_se=("rhs", "sem"),
        s2_oos=("floor", "mean"),
        gap=("gap", "mean"), gap_se=("gap", "sem"),
    ).reset_index()
    P = sorted(df["p"].unique())
    cats = [f"{p:,}" for p in P]
    reps = int(df.groupby(["p", "j"]).size().iloc[0])

    fig, axes = plt.subplots(3, 3, figsize=(13.3, 9.6), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1, 0.7, 0.7]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.17, hspace=0.13, wspace=0.08)
    cap = dict(fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=5)
    for j in (1, 2, 3):
        a = agg[agg["j"] == j].set_index("p").loc[P]
        x = np.arange(len(P))
        meas, oos, theory = (a["s2_meas"].to_numpy(), a["s2_oos"].to_numpy(),
                             a["s2_theory"].to_numpy())
        ax = axes[0, j - 1]
        ax.bar(x, oos, 0.7, color="#4878a8",
               label=r"out-of-subspace: $\delta^2/(n\lambda_{n,j}+\delta^2)$")
        ax.bar(x, theory - oos, 0.7, bottom=oos, color="#f28e2b", label="in-subspace")
        ax.plot(x, meas, "o-", color="black", label=r"measured $\angle(h,\bar b)$", zorder=5)
        ax.errorbar(x, meas, yerr=1.96 * a["s2_meas_se"].to_numpy(), **cap)
        ax.errorbar(x, theory, yerr=1.96 * a["s2_theory_se"].to_numpy(), **cap)
        ax.set_xticks(x, cats, fontsize=8)
        ax.set_title(f"factor {j}", color=NAVY)
        axg = axes[1, j - 1]
        axg.axhline(0, color="0.6", lw=0.8, ls="--")
        axg.errorbar(x, a["gap"].to_numpy(), yerr=1.96 * a["gap_se"].to_numpy(),
                     fmt="o-", color=COLORS[j - 1], ms=4, lw=1.2, capsize=2)
        axg.set_xticks(x, cats, fontsize=8)
        axs = axes[2, j - 1]
        axs.bar(x, oos / theory, 0.7, color=COLORS[j - 1], alpha=0.55)
        axs.set_xticks(x, cats, fontsize=8)
        axs.set_xlabel("p (assets)")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_yticks(*SIN2)
    axes[0, 0].set_ylabel(r"average $\sin^2$")
    axes[1, 0].set_ylabel("gap = mean(meas − theory)\n[sin², paired]")
    axes[2, 0].set_ylabel("out-of-subspace share")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(True, color="0.85", lw=0.5)
        ax.label_outer()
    fig.suptitle(f"Heavy-tailed (Student-t) dispersion bias — growing p, fixed n = {df['n'].iloc[0]}",
                 color=NAVY, y=0.985)
    fig.text(0.5, 0.025,
             "Top — sin²∠(h,b̄) = out-of-subspace + in-subspace (additive); black line = measured; "
             f"caps/strip = 95% CI (R = {reps}).   Middle — paired gap → 0 as p grows (the theorem).   "
             f"Returns: Student-t (factor df={HEAVY_TAIL['factor_return_sampler']['df']}, "
             f"idio df={HEAVY_TAIL['idio_return_sampler']['df']}).",
             ha="center", va="top", fontsize=8, color=GRAY, wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Heavy-tailed dispersion-bias experiment + 9-panel figure "
                    "(driver; all theorem math is in sim_theorem_partii.py).")
    ap.add_argument("--n", type=int, default=63, help="sample length (periods)")
    ap.add_argument("--reps", type=int, default=100, help="replications per p")
    ap.add_argument("--pmax", type=int, default=None,
                    help="extend the default p-grid up to this value (e.g. 100000)")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--out", type=str, default="dispersion_9panel.png")
    args = ap.parse_args()

    p_values = list(P_VALUES)
    if args.pmax and args.pmax > p_values[-1]:
        p_values.append(args.pmax)

    design = DesignSpec(
        n_values=[args.n], p_values=p_values, n_reps=args.reps,
        random_seed=args.seed, sampling="nested", **HEAVY_TAIL,
    )
    print(f"sweep: p={p_values}, n={args.n}, reps={args.reps}, seed={args.seed}, "
          f"Student-t df=({HEAVY_TAIL['factor_return_sampler']['df']},"
          f"{HEAVY_TAIL['idio_return_sampler']['df']})")

    # ALL Eq.(17) math happens inside this call (sim_theorem_partii, unmodified):
    df = run_experiment(MODEL, design, DispersionBiasExperiment(), progress=True)
    print(f"rows: {len(df)}")
    plot_9panel(df, args.out)


if __name__ == "__main__":
    main()
