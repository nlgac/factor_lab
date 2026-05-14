"""
fl_graphics.py
==============
Graphics for the Theorem 1 / Equation (20) simulation study.

Consumes a pandas DataFrame — produced by sim_theorem_partii.simulate() or
loaded from a .parquet or .csv file — with columns:

    n, p, j, sin2_j, rhs, gap, floor, rotation, rho

Can be run standalone:

    python fl_graphics.py                          # loads default parquet path
    python fl_graphics.py sim_thmptii.parquet

Or called programmatically after a live simulation:

    from fl_graphics import plot_all
    plot_all(df, out_dir=Path("."), n_show=60)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

ROOT = Path(__file__).resolve().parent

_DEFAULT_DATA_PATH = ROOT / "sim_thmptii.parquet"


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_results(path: Path | str) -> pd.DataFrame:
    """Load simulation results from a .parquet or .csv file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _save_fig(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_path.name)


# ── Public plot functions ─────────────────────────────────────────────────────


def plot_convergence(df: pd.DataFrame, out_path: Path) -> None:
    """Gap sin²∠ − RHS vs p, median ± IQR, for each n and factor."""
    sns.set_theme(style="whitegrid", context="paper")

    g = sns.relplot(
        data=df, x="p", y="gap", hue="n", col="j",
        kind="line", estimator=np.median, errorbar=("pi", 50),
        facet_kws={"sharey": False}, height=3.5, aspect=1.2,
        palette="tab10",
    )
    g.set(xscale="log")
    g.map(plt.axhline, y=0, color="red", linestyle=":", linewidth=0.8)
    g.set_axis_labels("p", r"$\sin^2\angle$ − RHS (gap)")
    g.set_titles(col_template="Factor j={col_name}")
    g.fig.suptitle(
        r"Convergence of gap $\sin^2\angle(h_j, b_j)$ − RHS to zero as $p \to \infty$"
        "\n"
        r"Equation (20), Theorem 1 ($G_\infty = \mathrm{diag}(\tau_j)$)",
        y=1.08,
    )
    _save_fig(g.fig, out_path)


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """sin²∠ vs RHS scatter at the second-largest p value."""
    p_scatter = sorted(df["p"].unique())[-2]
    sub = df[df["p"] == p_scatter]

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.relplot(
        data=sub, x="rhs", y="sin2_j", hue="n", col="j",
        kind="scatter", alpha=0.4, s=15,
        height=3.5, aspect=1.1, palette="tab10",
    )

    def _draw_45(**kwargs):
        ax = plt.gca()
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", linewidth=0.8)

    g.map(_draw_45)
    g.set_axis_labels("RHS (predicted)", r"$\sin^2\angle(h_j, b_j)$ (observed)")
    g.set_titles(col_template="Factor j={col_name}")
    g.fig.suptitle(
        r"$\sin^2\angle(h_j, b_j)$ vs RHS of Equation (20)"
        f" at p={p_scatter:,}\n"
        "Each point is one (X, Z) replication",
        y=1.08,
    )
    _save_fig(g.fig, out_path)


def plot_components(
    df: pd.DataFrame,
    out_path: Path,
    n_show: int,
    top_margin: float = 0.03,
) -> None:
    """Floor and rotation terms vs p for each factor at n=n_show.

    Boxplots on a categorical p-axis (equally spaced).
    Row 0 (Floor / Observed): predicted floor vs observed sin²∠.
    Row 1 (Rotation): rotation term 1 − (ŵⱼ)ⱼ² — p-stable, set by F and C alone.

    top_margin: fractional headroom added above the top-row data maximum
                to prevent whisker clipping (default 0.03 = 3%).
    """
    sub = df[df["n"] == n_show].copy()
    df_melt = sub.melt(
        id_vars=["n", "p", "j"],
        value_vars=["floor", "sin2_j", "rotation"],
        var_name="metric", value_name="value",
    )
    df_melt["row_group"] = df_melt["metric"].apply(
        lambda x: "Rotation" if x == "rotation" else "Floor / Observed"
    )

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.catplot(
        data=df_melt, x="p", y="value", hue="metric",
        col="j", row="row_group",
        kind="box", showfliers=False,
        sharey="row", height=3.0, aspect=1.2,
        palette="Set2",
    )
    g.set_axis_labels("Ambient dimension (p)", "Value")
    g.set_titles(row_template="{row_name}", col_template="Factor j={col_name}")

    y_top = df_melt[df_melt["row_group"] == "Floor / Observed"]["value"].max()
    for ax in g.axes[0, :]:
        ax.set_ylim(top=y_top * (1 + top_margin))

    for ax in g.axes[1, :]:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    g.fig.suptitle(
        f"Floor and rotation components of Equation (20), n={n_show}\n"
        "Both terms are p-stable; the formula becomes exact as p → ∞",
        y=1.05,
    )
    _save_fig(g.fig, out_path)


def plot_all(
    df: pd.DataFrame,
    out_dir: Path,
    n_show: int | None = None,
) -> None:
    """Generate all three plots and save to out_dir.

    Example:
        plot_all(df, Path("."), n_show=60)
        plot_all(df, Path("."))   # infers n_show as median of df["n"].unique()
    """
    if n_show is None:
        n_values = sorted(df["n"].unique())
        n_show = n_values[len(n_values) // 2]

    plot_convergence(df, out_dir / "fig_theorem1_convergence_v2.png")
    plot_scatter(df,     out_dir / "fig_theorem1_scatter_v2.png")
    plot_components(df,  out_dir / "fig_theorem1_components_v2.png", n_show=n_show)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DATA_PATH
    logger.info("Loading results from {}", data_path)
    df = load_results(data_path)
    logger.info("Loaded {} rows", len(df))
    plot_all(df, data_path.parent)
    logger.info("Done.")


if __name__ == "__main__":
    main()
