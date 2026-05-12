"""
sim_theorem1_eq20_v2.py
=======================
Numerical verification of Theorem 1, Equation (20) from:

    "Multifactor Dispersion Bias under a Per-Column Prevalence Condition" (v7)
    §3.3, diagonal-Gram theorem.

The claim: for k=3 factors with G∞ = diag(τⱼ) (diagonal Gram), conditional
on X and almost surely as p → ∞,

    sin²∠(hⱼ, bⱼ)  →  δ²/(nρⱼ+δ²)  +  nρⱼ/(nρⱼ+δ²) · (1 − (ŵⱼ)ⱼ²)
                        ──────────────    ──────────────────────────────
                           floor               rotation

where ρⱼ and ŵⱼ are the j-th eigenvalue/vector of D̂ = C^{1/2}(XX^T/n)C^{1/2}.

Changes from v1
---------------
- `SineAlignmentAnalysis` replaces `Eq20LHSAnalysis`.  The LHS column is now
  named "sin2_j" throughout (DataFrame, plots, RMSE table).
- `compute_sine_alignment` (from factor_lab.analyses) is registered with
  `register_manifold_distance` so it also appears in `ManifoldDistanceAnalysis`
  results.  Note: that path uses centered PCA, which differs from the uncentered
  Gram trick used here; treat the registered value as a diagnostic only.

Setup
-----
- Loadings: B[j,:] has i.i.d. N(0, τⱼ) entries, independent across j.
  Empirical prevalence ‖B[j,:]‖²/p → τⱼ; diagonal Gram G∞ = diag(τⱼ).
- Factor returns: columns of X are i.i.d. N(0, F), F = diag(σⱼ²).
- Noise: idiosyncratic entries are i.i.d. N(0, δ²).
- Population loading directions bⱼ: top-k eigenvectors of Σ = B'FB + D.
- The model is drawn once per (n, p) cell and held fixed; X and Z are
  redrawn each replication (simulating the conditional-on-X regime).

Outputs
-------
- sim_theorem1_results_v2.csv       — raw per-rep records
- fig_theorem1_convergence_v2.png   — gap sin²∠−RHS vs p, for each n and factor
- fig_theorem1_scatter_v2.png       — sin²∠ vs RHS scatter at p=P_VALUES[-2]
- fig_theorem1_components_v2.png    — floor and rotation convergence separately
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from factor_lab.model_builder import FactorModelBuilder
from factor_lab.flexible_simulator import ReturnsSimulator
from factor_lab.distributions import create_sampler
from factor_lab.analysis import SimulationContext
from factor_lab.analyses.spectral import compute_true_eigenvalues
from factor_lab.analyses import compute_sine_alignment, register_manifold_distance

# ── Experiment parameters ─────────────────────────────────────────────────────

K = 3

# Factor return variances. Assumption 3 requires c₁σ₁² > c₂σ₂² > c₃σ₃².
# With TAU2 = [1.0, 0.8, 0.6] the effective spikes are d_j = τⱼ·σⱼ²:
#   d₁ = 0.040 > d₂ = 0.016 > d₃ = 0.006  ✓
SIGMA2 = np.array([0.04, 0.02, 0.01])

# Per-factor loading entry variances → prevalences cⱼ ≈ τⱼ (by LLN)
TAU2 = np.array([1.00, 0.80, 0.60])

# Idiosyncratic noise variance
DELTA2 = 1.0

N_VALUES = [30, 60, 120]
P_VALUES = [200, 500, 1000, 2000, 5000, 10_000]
N_REPS   = 300
SEED     = 20260511

# n-value used for the floor/rotation component plot (middle of N_VALUES)
N_SHOW = N_VALUES[1]

# ── SimulationAnalysis implementations ───────────────────────────────────────


class SineAlignmentAnalysis:
    """
    Observed LHS of equation (20): sin²∠(hⱼ, bⱼ) for each factor j.

    hⱼ: j-th top left singular vector of Y  (estimated loading direction,
        computed via the n×n Gram trick — uncentered, matching the theorem).
    bⱼ: j-th population loading direction, injected at construction.

    Population directions are passed at construction so that ARPACK runs once
    per model (once per (n, p) cell), not once per replication.

    Example:
        _, b_pop = compute_true_eigenvalues(model, K)
        analysis = SineAlignmentAnalysis(b_pop)
        result = analysis.analyze(context)
        # {"sin2_j": array shape (K,), "dist_sine": float}
    """

    def __init__(self, b_pop: np.ndarray):
        self.b_pop = b_pop   # (k, p), rows are population directions bⱼ

    def analyze(self, context: SimulationContext) -> dict:
        k = context.k
        Y = context.security_returns.T   # (p, n)
        # Top-k left SVs of Y via the n×n Gram Y^T Y (uncentered).
        # Cost O(p·n²) vs O(p²·n) for the full SVD.
        G = Y.T @ Y
        vals, vecs = np.linalg.eigh(G)
        idx = np.argsort(vals)[::-1][:k]
        s = np.sqrt(np.maximum(vals[idx], 0.0))
        H = (Y @ vecs[:, idx]) / np.where(s > 1e-14, s, 1.0)   # (p, k)
        # H.T has shape (k, p); b_pop has shape (k, p)
        sin2, dist = compute_sine_alignment(self.b_pop, H.T)
        return {"sin2_j": sin2, "dist_sine": dist}


class Eq20RHSAnalysis:
    """
    Predicted RHS of equation (20): floor + weight × rotation for each factor j.

    Uses factor returns X from the context and empirical prevalences cⱼ = ‖B[j,:]‖²/p
    from the model loadings.

    Example:
        analysis = Eq20RHSAnalysis(delta2=1.0)
        result = analysis.analyze(context)
        # keys: "rhs", "floor", "rotation", "rhos"
    """

    def __init__(self, delta2: float):
        self.delta2 = delta2

    def analyze(self, context: SimulationContext) -> dict:
        k, n = context.k, context.T
        X = context.factor_returns.T                     # (k, n)
        c_half = np.sqrt((context.model.B ** 2).mean(axis=1))   # √cⱼ
        D_hat = (c_half[:, None] * (X @ X.T / n)) * c_half[None, :]
        vals, vecs = np.linalg.eigh(D_hat)
        idx = np.argsort(vals)[::-1]
        rhos = vals[idx]
        W = vecs[:, idx]
        floor = self.delta2 / (n * rhos + self.delta2)
        weight = n * rhos / (n * rhos + self.delta2)
        # (ŵⱼ)ⱼ is the j-th component of the j-th eigenvector; squaring removes sign.
        rotation = 1.0 - np.diag(W) ** 2
        return {"rhs": floor + weight * rotation, "floor": floor,
                "rotation": rotation, "rhos": rhos}


# ── Model construction ────────────────────────────────────────────────────────


def build_model(p: int, rng: np.random.Generator):
    """Build a k-factor model for the diagonal-Gram experiment.

    Loading entries for factor j are i.i.d. N(0, τⱼ), so by LLN
    ‖B[j,:]‖²/p → τⱼ and off-diagonal Gram entries → 0, giving G∞ = diag(τⱼ).
    Idiosyncratic volatility is uniform at δ for all assets.

    Example:
        model = build_model(p=1000, rng=np.random.default_rng(0))
        # model.B.shape == (3, 1000), model.F == diag(SIGMA2), model.D == DELTA2·I
    """
    return FactorModelBuilder(rng=rng).build(
        p=p,
        k=K,
        beta_samplers=[
            create_sampler("normal", rng, loc=0, scale=np.sqrt(tau))
            for tau in TAU2
        ],
        idio_vol_sampler=create_sampler("constant", rng, value=np.sqrt(DELTA2)),
        factor_variances=SIGMA2.tolist(),
    )


# ── Simulation helpers ────────────────────────────────────────────────────────


def _rep_records(n: int, p: int, lhs_res: dict, rhs_res: dict) -> list[dict]:
    """Flatten one replication's analysis results into K per-factor records."""
    gap = lhs_res["sin2_j"] - rhs_res["rhs"]
    return [
        {"n": n, "p": p, "j": j + 1,
         "sin2_j":   float(lhs_res["sin2_j"][j]),
         "rhs":      float(rhs_res["rhs"][j]),
         "gap":      float(gap[j]),
         "floor":    float(rhs_res["floor"][j]),
         "rotation": float(rhs_res["rotation"][j]),
         "rho":      float(rhs_res["rhos"][j])}
        for j in range(K)
    ]


# ── Main simulation ───────────────────────────────────────────────────────────


def simulate() -> pd.DataFrame:
    # Register sine distance once per process so ManifoldDistanceAnalysis
    # picks it up when used alongside this simulation.  The registration is
    # guarded to be idempotent; the centering difference vs. uncentered Y
    # used here means this value is diagnostic only.
    from factor_lab.analyses.manifold import _EXTRA_DISTANCES
    if 'dist_sine' not in _EXTRA_DISTANCES:
        register_manifold_distance(
            'dist_sine',
            lambda bt, be: compute_sine_alignment(bt, be)[1],
        )

    rng_master = np.random.default_rng(SEED)
    simulator = ReturnsSimulator()   # stateless; all draws go through samplers
    rhs_analysis = Eq20RHSAnalysis(DELTA2)
    records: list[dict] = []

    for n in N_VALUES:
        logger.info("Starting n = {}", n)
        for p in tqdm(P_VALUES, desc=f"n={n}", unit="p"):

            # Build model once per (n, p) cell — fresh β each time.
            model = build_model(p, rng_master)

            # Population directions computed once here; ARPACK skips the rep loop.
            _, b_pop = compute_true_eigenvalues(model, K)
            lhs_analysis = SineAlignmentAnalysis(b_pop)

            logger.debug("n={}, p={}: c={:.4f}, {:.4f}, {:.4f}",
                         n, p, *(model.B ** 2).mean(axis=1))

            # Independent seed per rep; master rng advances only here.
            rep_seeds = rng_master.integers(0, 2 ** 31, size=N_REPS)
            for _ in range(N_REPS):
                rep_rng = np.random.default_rng(int(rep_seeds[_]))
                normal = create_sampler("normal", rep_rng)
                # factor and idio draws share rep_rng via the same sampler —
                # sequential draws are independent, replication is reproducible.
                sim_out = simulator.simulate(
                    model=model, n_periods=n,
                    factor_return_samplers=normal,
                    idio_return_sampler=normal,
                )
                context = SimulationContext(
                    model=model,
                    security_returns=sim_out["security_returns"],
                    factor_returns=sim_out["factor_returns"],
                    idio_returns=sim_out["idio_returns"],
                )
                records.extend(
                    _rep_records(n, p, lhs_analysis.analyze(context),
                                 rhs_analysis.analyze(context))
                )

    return pd.DataFrame(records)


# ── Plotting helpers ──────────────────────────────────────────────────────────

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green (j=1,2,3)
# Line style keyed by n-value; derived from N_VALUES so they stay in sync.
_N_LINE = dict(zip(N_VALUES, ["-", "--", "-."]))


def _agg_by_p(grp) -> tuple:
    """Return (p_vals, medians, q25s, q75s) for a SeriesGroupBy grouped by p."""
    p_vals = sorted(grp.groups)
    return (
        p_vals,
        [grp.get_group(p).median()       for p in p_vals],
        [grp.get_group(p).quantile(0.25) for p in p_vals],
        [grp.get_group(p).quantile(0.75) for p in p_vals],
    )


def _save_fig(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved {}", out_path.name)


# ── Plotting functions ────────────────────────────────────────────────────────


def plot_convergence(df: pd.DataFrame, out_path: Path) -> None:
    """Gap sin²∠ − RHS vs p, median ± IQR, for each n and factor."""
    fig, axes = plt.subplots(1, K, figsize=(14, 4), sharey=False)
    for ax, j in zip(axes, range(1, K + 1)):
        sub = df[df["j"] == j]
        for n in N_VALUES:
            p_vals, med, q25, q75 = _agg_by_p(sub[sub["n"] == n].groupby("p")["gap"])
            ax.plot(p_vals, med, linestyle=_N_LINE[n], color="k",
                    linewidth=1.5, label=f"n={n}")
            ax.fill_between(p_vals, q25, q75, alpha=0.15, color="k")
        ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(f"Factor j={j}", fontsize=11)
        ax.set_xlabel("p")
        if j == 1:
            ax.set_ylabel(r"$\sin^2\angle$ − RHS  (gap)")
        ax.legend(fontsize=8)
    fig.suptitle(
        r"Convergence of gap  $\sin^2\angle(h_j, b_j)$ − RHS  to zero as $p \to \infty$"
        "\n"
        r"Equation (20), Theorem 1  ($G_\infty = \mathrm{diag}(\tau_j)$, "
        f"$n_{{\\mathrm{{rep}}}}={N_REPS}$)",
        fontsize=11,
    )
    _save_fig(fig, out_path)


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """sin²∠ vs RHS scatter at the second-largest p value."""
    p_scatter = sorted(df["p"].unique())[-2]
    sub = df[df["p"] == p_scatter]
    fig, axes = plt.subplots(1, K, figsize=(13, 4))
    for ax, j in zip(axes, range(1, K + 1)):
        d = sub[sub["j"] == j]
        for n in N_VALUES:
            dn = d[d["n"] == n]
            ax.scatter(dn["rhs"], dn["sin2_j"], s=6, alpha=0.4, label=f"n={n}")
        lo = min(d["rhs"].min(), d["sin2_j"].min()) - 0.01
        hi = max(d["rhs"].max(), d["sin2_j"].max()) + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="45°")
        ax.set_xlabel("RHS (predicted)", fontsize=9)
        ax.set_ylabel(r"$\sin^2\angle(h_j, b_j)$  (observed)", fontsize=9)
        ax.set_title(f"Factor j={j}", fontsize=11)
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle(
        r"$\sin^2\angle(h_j, b_j)$ vs RHS of Equation (20)"
        f" at p={p_scatter:,}\n"
        "Each point is one (X, Z) replication",
        fontsize=11,
    )
    _save_fig(fig, out_path)


def plot_components(df: pd.DataFrame, out_path: Path) -> None:
    """Floor and rotation terms vs p for each factor at n=N_SHOW.

    Row 0: predicted floor vs observed sin²∠ — shows the floor is p-stable.
    Row 1: rotation term 1 − (ŵⱼ)ⱼ² — shows the rotation vanishes as p grows.
    """
    sub = df[df["n"] == N_SHOW]
    fig, axes = plt.subplots(2, K, figsize=(13, 7), sharex=True)
    for j_idx, j in enumerate(range(1, K + 1)):
        d = sub[sub["j"] == j]
        p_vals, floor_med, _, _ = _agg_by_p(d.groupby("p")["floor"])
        _,      sin2_med,  _, _ = _agg_by_p(d.groupby("p")["sin2_j"])
        _,      rot_med, rot_q25, rot_q75 = _agg_by_p(d.groupby("p")["rotation"])
        color = _COLORS[j_idx]

        ax0 = axes[0, j_idx]
        ax0.plot(p_vals, floor_med, color=color, linewidth=2, label="predicted floor")
        ax0.plot(p_vals, sin2_med, color="k", linestyle=":", linewidth=1.2,
                 label=r"observed $\sin^2\angle$")
        ax0.set_xscale("log")
        ax0.set_title(f"Factor j={j}", fontsize=11)
        if j_idx == 0:
            ax0.set_ylabel(r"Floor  (predicted vs $\sin^2\angle$)", fontsize=9)
        ax0.legend(fontsize=8)

        ax1 = axes[1, j_idx]
        ax1.plot(p_vals, rot_med, color=color, linewidth=2,
                 label=r"rotation  $1 - (\hat{w}_j)_j^2$")
        ax1.fill_between(p_vals, rot_q25, rot_q75, alpha=0.25, color=color)
        ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax1.set_xlabel("p")
        ax1.set_xscale("log")
        if j_idx == 0:
            ax1.set_ylabel("Rotation  (median ± IQR)", fontsize=9)
        ax1.legend(fontsize=8)

    fig.suptitle(
        f"Floor and rotation components of Equation (20),  n={N_SHOW}\n"
        "Rotation → 0 as p → ∞; floor is p-stable",
        fontsize=11,
    )
    _save_fig(fig, out_path)


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact RMSE table: RMSE of (sin²∠ − RHS) by (n, p, j)."""
    tbl = (
        df.groupby(["n", "p", "j"])["gap"]
        .apply(lambda g: np.sqrt((g ** 2).mean()))
        .rename("RMSE")
        .reset_index()
        .pivot(index=["n", "p"], columns="j", values="RMSE")
    )
    tbl.columns = [f"j={c}" for c in tbl.columns]
    print("\nRMSE of (sin²∠ − RHS)  [smaller is better; should → 0 as p grows]\n")
    print(tbl.to_string(float_format="{:.5f}".format))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    logger.info(
        "Simulation: k={}, n={}, p={}, reps={}, seed={}",
        K, N_VALUES, P_VALUES, N_REPS, SEED,
    )
    logger.info("σ²={}, τ²={}, δ²={}, spikes={}",
                SIGMA2.tolist(), TAU2.tolist(), DELTA2, (TAU2 * SIGMA2).tolist())

    df = simulate()

    csv_path = ROOT / "sim_theorem1_results_v2.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved {} rows to {}", len(df), csv_path.name)

    print_summary(df)

    plot_convergence(df, ROOT / "fig_theorem1_convergence_v2.png")
    plot_scatter(df,     ROOT / "fig_theorem1_scatter_v2.png")
    plot_components(df,  ROOT / "fig_theorem1_components_v2.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
