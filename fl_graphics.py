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
    # Corollary (observable floor)
    "plot_obs_floor_convergence",
    "plot_obs_floor_gap",
    "plot_all_obs_floor",
    "OBS_FLOOR_FIGURES",
    "OBS_FLOOR_CONVERGENCE_PER_FACTOR",
    "OBS_FLOOR_GAP_PER_FACTOR",
    # Corollary 4 (subspace distance)
    "plot_corollary4_convergence",
    "plot_corollary4_gap",
    "plot_all_corollary4",
    "COROLLARY4_FIGURES",
    # Eq.(17) decomposition (9-panel)
    "nine_panel",
    "nine_panel_decomposition",
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


# ════════════════════════════════════════════════════════════════════════════
# Corollary (observable floor): ℓ²ₚ/s²ₚⱼ → δ²/(nρⱼ+δ²)
# ════════════════════════════════════════════════════════════════════════════
#
# Consumes the dataframe from sim_corollary_obs_floor.ObservableFloorExperiment:
#     columns: n, p, j, sin2_j, floor_obs, floor_true, gap
#
# Two registered figures:
#   obs_floor_convergence — per-factor: proxy ℓ²ₚ/s²ₚⱼ converging onto the flat
#                           analytic floor, with the observed sin²∠ as the upper
#                           envelope the floor lower-bounds.
#   obs_floor_gap         — per-factor gap RMSE on log–log, the convergence rate.

OBS_FLOOR_FIGURES = ["obs_floor_convergence", "obs_floor_gap"]

_OBS_FLOOR_LABELS = {
    "floor_obs":  r"$\ell^2_p / s^2_{p,j}$  (observable)",
    "floor_true": r"$\delta^2/(n\rho_j+\delta^2)$  (analytic floor)",
    "sin2_j":     r"$\sin^2\angle(h_j,\bar b_j)$  (observed)",
}
_OBS_FLOOR_PALETTE = {
    "floor_obs":  "#c0392b",   # the star: observable proxy
    "floor_true": "#2c3e50",   # the target it approaches
    "sin2_j":     "#7f8c8d",   # the upper envelope (floor + rotation)
}
_OBS_FLOOR_DASHES = {
    "floor_obs":  "",          # solid
    "floor_true": (4, 2),      # dashed reference
    "sin2_j":     (1, 1),      # dotted envelope
}


def plot_obs_floor_convergence(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Per-factor convergence of the observable proxy onto the analytic floor.

    One panel per factor j (independent y-scales — the floors differ by an order
    of magnitude across factors). Three series vs p on a log axis:

      • floor_obs  ℓ²ₚ/s²ₚⱼ        — observable proxy (solid red), converging;
      • floor_true δ²/(nρⱼ+δ²)     — analytic floor (dashed navy), p-stable;
      • sin2_j     sin²∠(hⱼ,b̄ⱼ)   — observed defect (dotted grey), the upper
                                     envelope the floor lower-bounds.

    Lines are medians over replications; bands are the 25–75% range.
    """
    set_theme()
    long = df.melt(
        id_vars=["n", "p", "j"],
        value_vars=["floor_obs", "floor_true", "sin2_j"],
        var_name="metric", value_name="value",
    )
    g = sns.relplot(
        data=long, x="p", y="value",
        hue="metric", style="metric", col="j",
        kind="line", estimator=np.median, errorbar=("pi", 50),
        palette=_OBS_FLOOR_PALETTE, dashes=_OBS_FLOOR_DASHES,
        height=3.5, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log")
    g.set_axis_labels("p (assets, log scale)", r"$\sin^2\angle$ (alignment defect)")
    g.set_titles(col_template="Factor j={col_name}")
    # Relabel with mathtext and move the legend OUT of the panels (it otherwise
    # lands inside the last facet and overlaps the data). Rebuild it as a single
    # key anchored to the right of all three panels.
    if g.legend is not None:
        handles = [h for h in g.legend.legend_handles if h is not None]
        labels = [_OBS_FLOOR_LABELS.get(t.get_text(), t.get_text())
                  for t in g.legend.texts]
        g.legend.remove()
        g.figure.legend(
            handles, labels, title="", fontsize=8,
            loc="center left", bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
    g.figure.suptitle(
        r"Observable lower bound: $\ell^2_p / s^2_{p,j} \to \delta^2/(n\rho_j+\delta^2)$"
        r" as $p \to \infty$"
        "\n"
        "the analytic floor, recovered from the sample spectrum alone",
        y=1.08,
    )
    _save_fig(g.figure, out_path)


def plot_obs_floor_gap(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Per-factor gap RMSE |proxy − analytic floor| vs p on log–log axes."""
    set_theme()
    rmse = (
        df.groupby(["p", "j"])["gap"]
        .apply(lambda s: float(np.sqrt((s ** 2).mean())))
        .rename("rmse").reset_index()
    )
    g = sns.relplot(
        data=rmse, x="p", y="rmse", col="j",
        kind="line", marker="o", color="#c0392b",
        height=3.5, aspect=1.15, facet_kws={"sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("p (assets, log scale)", "gap RMSE (log scale)")
    g.set_titles(col_template="Factor j={col_name}")
    g.figure.suptitle(
        r"Convergence rate: RMSE$(\ell^2_p/s^2_{p,j} - \delta^2/(n\rho_j+\delta^2)) \to 0$",
        y=1.05,
    )
    _save_fig(g.figure, out_path)


register_figure("obs_floor_convergence", plot_obs_floor_convergence, "fig_obs_floor_convergence.png")
register_figure("obs_floor_gap",         plot_obs_floor_gap,         "fig_obs_floor_gap.png")


def plot_all_obs_floor(df: pd.DataFrame, out_dir: Path) -> None:
    """Render both observable-floor figures into out_dir."""
    render_figures(df, out_dir, names=OBS_FLOOR_FIGURES)


# ── Per-factor (single-panel) variants ───────────────────────────────────────
#
# Same content as the 3-panel combined figures above, but one standalone,
# individually-titled figure per factor — so the combined view and the three
# singles can both appear in the deck. Bands are kept (median + 25–75% IQR);
# drop them later by removing errorbar=... if Lisa prefers clean lines.

OBS_FLOOR_CONVERGENCE_PER_FACTOR = [f"obs_floor_convergence_f{j}" for j in (1, 2, 3)]
OBS_FLOOR_GAP_PER_FACTOR = [f"obs_floor_gap_f{j}" for j in (1, 2, 3)]


def _plot_obs_floor_convergence_single(df: pd.DataFrame, out_path: Path, j: int) -> None:
    """Single-factor convergence panel: proxy vs analytic floor vs observed."""
    set_theme()
    sub = df[df["j"] == j]
    long = sub.melt(
        id_vars=["n", "p", "j"],
        value_vars=["floor_obs", "floor_true", "sin2_j"],
        var_name="metric", value_name="value",
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.lineplot(
        data=long, x="p", y="value", hue="metric", style="metric",
        estimator=np.median, errorbar=("pi", 50),
        palette=_OBS_FLOOR_PALETTE, dashes=_OBS_FLOOR_DASHES, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("p (assets, log scale)")
    ax.set_ylabel(r"$\sin^2\angle$ (alignment defect)")
    ax.set_title(f"Observable floor — Factor j={j}")
    # Relabel with mathtext and place the key outside, right of the panel.
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, _OBS_FLOOR_LABELS[l]) for h, l in zip(handles, labels)
            if l in _OBS_FLOOR_LABELS]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if keep:
        ax.legend([h for h, _ in keep], [l for _, l in keep],
                  title="", fontsize=8, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save_fig(fig, out_path)


def _plot_obs_floor_gap_single(df: pd.DataFrame, out_path: Path, j: int) -> None:
    """Single-factor gap RMSE panel on log–log axes."""
    set_theme()
    sub = df[df["j"] == j]
    rmse = sub.groupby("p")["gap"].apply(lambda s: float(np.sqrt((s ** 2).mean())))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rmse.index.values, rmse.values, "o-", color="#c0392b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (assets, log scale)")
    ax.set_ylabel("gap RMSE (log scale)")
    ax.set_title(f"Convergence rate — Factor j={j}")
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, out_path)


def _make_single(fn, j):
    """Bind a per-factor plotter to a fixed j as a (df, out_path, **kwargs) renderer."""
    def _renderer(df, out_path, **kwargs):
        fn(df, out_path, j)
    _renderer.__name__ = f"{fn.__name__}_f{j}"
    return _renderer


for _j in (1, 2, 3):
    register_figure(
        f"obs_floor_convergence_f{_j}",
        _make_single(_plot_obs_floor_convergence_single, _j),
        f"fig_obs_floor_convergence_f{_j}.png",
    )
    register_figure(
        f"obs_floor_gap_f{_j}",
        _make_single(_plot_obs_floor_gap_single, _j),
        f"fig_obs_floor_gap_f{_j}.png",
    )


# ════════════════════════════════════════════════════════════════════════════
# Corollary 4: Grassmannian subspace distance  d_Gr² → Σⱼ δ²/(nρⱼ+δ²)
# ════════════════════════════════════════════════════════════════════════════
#
# Consumes the dataframe from sim_corollary4.SubspaceDistanceExperiment:
#     columns: n, p, d_gr2_obs, d_gr2_pred, gap   (one scalar row per replication)
#
# Two registered figures:
#   corollary4_convergence — observed d_Gr² vs predicted Σ floors across p, each
#                            a median with a 25–75% band; the curves pull together.
#   corollary4_gap         — gap RMSE on log–log, trending to zero.

COROLLARY4_FIGURES = ["corollary4_convergence", "corollary4_gap"]


def _median_band(df: pd.DataFrame, value: str, by: str = "p",
                 lo: float = 0.25, hi: float = 0.75):
    """Return (median, lower-quantile, upper-quantile) Series grouped by ``by``."""
    grp = df.groupby(by)[value]
    return grp.median(), grp.quantile(lo), grp.quantile(hi)


def plot_corollary4_convergence(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Observed subspace distance vs predicted Σ floors across p (single panel).

    Median d_Gr²(col H, col B̄) (observed) and median Σⱼ δ²/(nρⱼ+δ²) (predicted)
    vs p on a log axis, each with a 25–75% inter-replication band. The two pull
    together as p → ∞ — the Corollary 4 statement, the in-subspace rotation
    having cancelled in the Grassmannian metric.
    """
    set_theme()
    obs_m, obs_lo, obs_hi = _median_band(df, "d_gr2_obs")
    pred_m, pred_lo, pred_hi = _median_band(df, "d_gr2_pred")
    p = obs_m.index.values

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(p, obs_m.values, "o-", color="#c0392b",
            label=r"observed  $d_{\mathrm{Gr}}^2(\mathrm{col}\,H,\ \mathrm{col}\,\bar B)$")
    ax.fill_between(p, obs_lo.values, obs_hi.values, color="#c0392b", alpha=0.15)
    ax.plot(p, pred_m.values, "s--", color="#2c3e50",
            label=r"predicted  $\sum_j \delta^2/(n\rho_j+\delta^2)$")
    ax.fill_between(p, pred_lo.values, pred_hi.values, color="#2c3e50", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("p (assets, log scale)")
    ax.set_ylabel(r"squared subspace distance  $d_{\mathrm{Gr}}^2$")
    ax.set_title(r"Corollary 4: $d_{\mathrm{Gr}}^2 \to \sum_j \delta^2/(n\rho_j+\delta^2)$ as $p \to \infty$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)


def plot_corollary4_gap(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Gap RMSE (d_Gr² − Σ floors) vs p on log–log axes."""
    set_theme()
    rmse = df.groupby("p")["gap"].apply(lambda s: float(np.sqrt((s ** 2).mean())))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(rmse.index.values, rmse.values, "o-", color="#2980b9")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (assets, log scale)")
    ax.set_ylabel("gap RMSE (log scale)")
    ax.set_title(r"Gap RMSE $\to 0$:  $\mathrm{RMSE}(d_{\mathrm{Gr}}^2 - \sum_j \mathrm{floor}_j)$")
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, out_path)


register_figure("corollary4_convergence", plot_corollary4_convergence, "fig_corollary4_convergence.png")
register_figure("corollary4_gap",         plot_corollary4_gap,         "fig_corollary4_gap.png")


def plot_all_corollary4(df: pd.DataFrame, out_dir: Path) -> None:
    """Render both Corollary 4 figures into out_dir."""
    render_figures(df, out_dir, names=COROLLARY4_FIGURES)



# ── Eq.(17) decomposition: 9-panel figure (sin² | gap | out-of-subspace share) ──

_NP_COLORS = ["tab:blue", "tab:orange", "tab:green"]
_NP_NAVY, _NP_GRAY = "#1f3864", "#555555"
_NP_SIN2_TICKS = ([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.0"])
_NP_LBL_MEAS = r"measured $\angle(h, \bar b)$"
_NP_LBL_OOS = r"out-of-subspace: $\delta^2/(n\lambda_{n,j}+\delta^2)$"
_NP_LBL_INSUB = "in-subspace"


def _np_summarize(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per (key, factor j): component means + SEMs of measured/theory and the paired gap."""
    g = df.assign(
        s2_meas=df["sin2_j"], s2_theory=df["rhs"], s2_oos=df["floor"],
        gap=df["sin2_j"] - df["rhs"],
    )
    return g.groupby([key, "j"]).agg(
        s2_meas=("s2_meas", "mean"), s2_meas_se=("s2_meas", "sem"),
        s2_theory=("s2_theory", "mean"), s2_theory_se=("s2_theory", "sem"),
        s2_oos=("s2_oos", "mean"),
        gap=("gap", "mean"), gap_se=("gap", "sem"),
    ).reset_index()


def _np_caption(reps: int) -> str:
    return (
        "Top — sin²∠(h, b̄) = out-of-subspace + in-subspace (additive); black line = measured; "
        f"caps = 95% CI (R = {reps}, SE = sd/√R).   "
        "Middle — gap = mean(measured − theory) sin² (paired): 0 ⇒ theory matches, "
        ">0 ⇒ measured exceeds prediction.   "
        "Bottom — out-of-subspace share = oos/(oos+in-sub) of the predicted total."
    )


def nine_panel(avg, key, order, cats, xlabel, suptitle, reps=None):
    """Render the 3×3 Eq.(17) decomposition figure from a pre-summarized frame.

    Rows: sin² (out-of-subspace + in-subspace stack, measured line) | gap | out-of-subspace
    share. Columns: factors 1–3. ``avg`` is an :func:`_np_summarize` frame keyed by ``key``
    ('p' or 'n'); ``order`` the key values in plot order; ``cats`` their tick labels; ``reps``
    drives the 95% CI caption (None hides it). Returns ``(fig, axes)``.
    """
    fig, axes = plt.subplots(3, 3, figsize=(13.3, 9.6), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1, 0.7, 0.7]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.17, hspace=0.13, wspace=0.08)
    cap = dict(fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=5)
    for j in (1, 2, 3):
        a = avg[avg["j"] == j].set_index(key).loc[order]
        meas, oos, theory = (a["s2_meas"].to_numpy(), a["s2_oos"].to_numpy(),
                             a["s2_theory"].to_numpy())
        x = np.arange(len(cats))
        ax = axes[0, j - 1]
        ax.bar(x, oos, 0.7, color="#4878a8", label=_NP_LBL_OOS)
        ax.bar(x, theory - oos, 0.7, bottom=oos, color="#f28e2b", label=_NP_LBL_INSUB)
        ax.plot(x, meas, "o-", color="black", label=_NP_LBL_MEAS, zorder=5)
        ax.errorbar(x, meas, yerr=1.96 * a["s2_meas_se"].to_numpy(), **cap)
        ax.errorbar(x, theory, yerr=1.96 * a["s2_theory_se"].to_numpy(), **cap)
        ax.set_xticks(x, cats, fontsize=8)
        ax.set_title(f"factor {j}", color=_NP_NAVY)
        axg = axes[1, j - 1]
        axg.axhline(0, color="0.6", lw=0.8, ls="--")
        axg.errorbar(x, a["gap"].to_numpy(), yerr=1.96 * a["gap_se"].to_numpy(),
                     fmt="o-", color=_NP_COLORS[j - 1], ms=4, lw=1.2, capsize=2)
        axg.set_xticks(x, cats, fontsize=8)
        axs = axes[2, j - 1]
        axs.bar(x, oos / theory, 0.7, color=_NP_COLORS[j - 1], alpha=0.55,
                label="out-of-subspace share")
        axs.set_xticks(x, cats, fontsize=8)
        axs.set_xlabel(xlabel)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_yticks(*_NP_SIN2_TICKS)
    axes[0, 0].set_ylabel(r"average $\sin^2$")
    axes[1, 0].set_ylabel("gap = mean(meas − theory)\n[sin², paired]")
    axes[2, 0].set_ylabel("out-of-subspace share")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(True, color="0.85", lw=0.5)
        ax.label_outer()
    fig.suptitle(suptitle, color=_NP_NAVY, y=0.985)
    if reps is not None:
        fig.text(0.5, 0.025, _np_caption(reps), ha="center", va="top",
                 fontsize=8, color=_NP_GRAY, wrap=True)
    return fig, axes


def _validate_decomp_df(df):
    """Raise ``ValueError`` if ``df`` is not a plottable DispersionBiasExperiment sweep."""
    required = {"n", "p", "j", "sin2_j", "rhs", "floor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "nine_panel_decomposition: result frame is missing required column(s) "
            f"{sorted(missing)} — expected a DispersionBiasExperiment sweep with columns "
            f"{sorted(required)}.")
    factors = {int(j) for j in df["j"].unique()}
    if factors != {1, 2, 3}:
        raise ValueError(
            "nine_panel_decomposition: this figure is built for 3 factors (j = 1, 2, 3); "
            f"got j = {sorted(factors)}.")
    n_uniq, p_uniq = df["n"].nunique(), df["p"].nunique()
    if not ((p_uniq > 1 and n_uniq == 1) or (n_uniq > 1 and p_uniq == 1)):
        raise ValueError(
            "nine_panel_decomposition: needs exactly one swept axis — several p at a single n "
            f"(growing-p) or several n at a single p (growing-n); got {p_uniq} distinct p and "
            f"{n_uniq} distinct n.")


def nine_panel_decomposition(df, *, key=None, suptitle=None, reps=None, out_path=None):
    """Plot the 9-panel Eq.(17) decomposition from an experiment result frame.

    **Pure plotter — it does not run anything.** ``df`` must be the per-rep output of a
    ``DispersionBiasExperiment`` sweep (columns ``n, p, j, sin2_j, rhs, floor``), with factors
    ``j = 1, 2, 3`` and exactly one swept axis (several ``p`` at a single ``n``, or several ``n``
    at a single ``p``). The swept axis, x labels, title and replicate count (for the CI caption)
    are inferred from ``df``; override with ``key`` / ``suptitle`` / ``reps``. Raises
    ``ValueError`` if ``df`` is incompatible. Saves to ``out_path`` if given. Returns
    ``(fig, axes)``::

        from fl_experiment_runner import run_experiment
        from sim_theorem_partii import DispersionBiasExperiment
        df = run_experiment(model, design, DispersionBiasExperiment())
        fig, _ = nine_panel_decomposition(df)
    """
    _validate_decomp_df(df)
    if key is None:
        key = "p" if df["p"].nunique() > 1 else "n"
    order = sorted(df[key].unique())
    cats = [f"{v:,}" for v in order] if key == "p" else [str(v) for v in order]
    xlabel = "p (assets)" if key == "p" else "n (periods)"
    if suptitle is None:
        suptitle = (f"Growing p, fixed n = {int(df['n'].iloc[0])}" if key == "p"
                    else f"Fixed p = {int(df['p'].iloc[0]):,}, growing n")
    if reps is None:
        reps = (int(df["rep"].nunique()) if "rep" in df.columns
                else int(df.groupby(["n", "p", "j"]).size().iloc[0]))
    fig, axes = nine_panel(_np_summarize(df, key), key, order, cats, xlabel, suptitle, reps=reps)
    if out_path is not None:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    return fig, axes


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
