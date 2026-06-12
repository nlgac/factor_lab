"""
fl_graphics.py
==============
Dispersion-bias figures for the Theorem 1 / Equation (20) simulation study.

These are the *study-specific* figures (gap-convergence, LHS-vs-RHS scatter,
floor/rotation components). The generic save/IO/dispatch plumbing lives in
``fl_visualization``; this module supplies the dispersion-bias content and
registers each figure into that harness, so a future script can render them via:

    from fl_visualization import render_figures
    render_figures(df, out_dir, names=["theorem1_convergence", ...], n_show=60)

or keep using the convenience wrappers preserved here.

Consumes a pandas DataFrame — produced by sim_theorem_partii.simulate() or loaded
from a .parquet/.csv file — with columns:

    n, p, j, sin2_j, rhs, gap, floor, rotation, rho

Standalone:

    python fl_graphics.py                          # loads default parquet path
    python fl_graphics.py sim_thmptii.parquet

Programmatic:

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

from fl_visualization import (
    register_figure,
    render_figures,
    load_results,
    set_theme,
    save_fig as _save_fig,
)

ROOT = Path(__file__).resolve().parent

_DEFAULT_DATA_PATH = ROOT / "sim_thmptii.parquet"

# Study-specific figure names and their output filenames, in render order.
THEOREM_FIGURES = ["theorem1_convergence", "theorem1_scatter", "theorem1_components"]

__all__ = [
    "load_results",
    "plot_convergence",
    "plot_scatter",
    "plot_components",
    "plot_all",
    "THEOREM_FIGURES",
    "main",
]


def _infer_n_show(df: pd.DataFrame) -> int:
    """Median of the available n values (the original plot_all default)."""
    n_values = sorted(df["n"].unique())
    return n_values[len(n_values) // 2]


# ── Public plot functions ─────────────────────────────────────────────────────


def plot_convergence(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Gap sin²∠ − RHS vs p, median ± IQR, for each n and factor."""
    set_theme()

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


def plot_scatter(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """sin²∠ vs RHS scatter at the second-largest p value."""
    p_scatter = sorted(df["p"].unique())[-2]
    sub = df[df["p"] == p_scatter]

    set_theme()

    g = sns.relplot(
        data=sub, x="rhs", y="sin2_j", hue="n", col="j",
        kind="scatter", alpha=0.4, s=15,
        height=3.5, aspect=1.1, palette="tab10",
    )

    def _draw_45(**kw):
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
    n_show: int | None = None,
    top_margin: float = 0.03,
    **kwargs,
) -> None:
    """Floor and rotation terms vs p for each factor at n=n_show.

    Boxplots on a categorical p-axis (equally spaced).
    Row 0 (Floor / Observed): predicted floor vs observed sin²∠.
    Row 1 (Rotation): rotation term 1 − (ŵⱼ)ⱼ² — p-stable, set by F and C alone.

    n_show: which n to slice; inferred as the median n when None.
    top_margin: fractional headroom above the top-row data maximum to prevent
                whisker clipping (default 0.03 = 3%).
    """
    if n_show is None:
        n_show = _infer_n_show(df)

    sub = df[df["n"] == n_show].copy()
    df_melt = sub.melt(
        id_vars=["n", "p", "j"],
        value_vars=["floor", "sin2_j", "rotation"],
        var_name="metric", value_name="value",
    )
    df_melt["row_group"] = df_melt["metric"].apply(
        lambda x: "Rotation" if x == "rotation" else "Floor / Observed"
    )

    set_theme()

    g = sns.catplot(
        data=df_melt, x="p", y="value", hue="metric",
        col="j", row="row_group",
        kind="box", showfliers=False,
        sharey="row", height=3.0, aspect=1.2,
        palette="Set2",
    )
    g.set_axis_labels("Ambient dimension (p)", "Value")
    g.set_titles(row_template="{row_name}", col_template="Factor j={col_name}")

    # Render the legend entries as mathtext ($...$); matplotlib's built-in
    # mathtext needs no TeX install. Relabel the existing legend artists so
    # seaborn's hue→color mapping stays intact.
    TERM_LABELS = {
        "floor":    r"$\delta^2 / (n\rho_j + \delta^2)$  (floor)",
        "sin2_j":   r"$\sin^2\angle(h_j, \bar b_j)$  (observed)",
        "rotation": r"$1 - \hat w_{jj}^2$  (rotation)",
    }
    handles, labels = [], []
    if g.legend is not None:
        labels = [TERM_LABELS.get(t.get_text(), t.get_text()) for t in g.legend.texts]
        handles = [h for h in g.legend.legend_handles if h is not None]
        g.legend.remove()  # drop seaborn's default outside legend

    y_top = df_melt[df_melt["row_group"] == "Floor / Observed"]["value"].max()
    for ax in g.axes[0, :]:
        ax.set_ylim(top=y_top * (1 + top_margin))

    for ax in g.axes[1, :]:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    # Reserve interior margins (bottom for the legend, top for the suptitle)
    # rather than letting a tight bbox widen the canvas — the figure keeps its
    # native width and the legend sits in a horizontal band *under* the panels,
    # within their horizontal bounds. Saved here rather than via _save_fig,
    # whose no-arg tight_layout + bbox_inches="tight" would undo the reservation.
    g.figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    if handles:
        g.figure.legend(handles, labels, title="term", fontsize=8, ncol=len(handles),
                        loc="lower center", bbox_to_anchor=(0.5, 0.0))
    g.figure.suptitle(
        f"Floor and rotation components of Equation (20), n={n_show}\n"
        "Both terms are p-stable; the formula becomes exact as p → ∞",
        y=0.985,
    )
    g.figure.savefig(out_path, dpi=150)
    plt.close(g.figure)
    logger.info("Saved {}", Path(out_path).name)


# ── Registry wiring ───────────────────────────────────────────────────────────

register_figure("theorem1_convergence", plot_convergence, "fig_theorem1_convergence_v2.png")
register_figure("theorem1_scatter",     plot_scatter,     "fig_theorem1_scatter_v2.png")
register_figure("theorem1_components",  plot_components,  "fig_theorem1_components_v2.png")


def plot_all(
    df: pd.DataFrame,
    out_dir: Path,
    n_show: int | None = None,
) -> None:
    """Generate all three theorem figures and save to out_dir.

    Thin wrapper over the harness: dispatches the registered THEOREM_FIGURES with
    a resolved n_show. Output filenames are unchanged.

    Example:
        plot_all(df, Path("."), n_show=60)
        plot_all(df, Path("."))   # infers n_show as median of df["n"].unique()
    """
    if n_show is None:
        n_show = _infer_n_show(df)
    render_figures(df, out_dir, names=THEOREM_FIGURES, n_show=n_show)


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
