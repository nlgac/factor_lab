"""
fl_plot.py
==========
Composable panel kit for the dispersion-bias figures.

Purely additive: nothing here imports from or modifies ``fl_graphics`` /
``fl_visualization``; existing figures and notebooks are untouched. New
notebooks compose figures from small panel *marks* instead of copying whole
figure functions between cells.

Layers (dependency direction is strictly downward):

    Theme       — style constants (colors, fills, caption font); one default
    sweep_axis  — the categorical swept-axis convention (p commas, n plain)
    data        — frame → arrays: rep_dists / summarize / derived columns
    marks       — draw one visual element onto an ax; return a caption fragment
    grid        — compose rows of marks × factor columns into a finished figure

Two-tier marks: the ``draw_*`` functions are the true primitives (arrays in,
artists out — no DataFrames), and the capitalized mark classes (``BoxDist``,
``ViolinDist``, ``Band``, ``MeanCI``, ``BandStack``, ``QuantileBand``,
``RefOverlay``) bind column names so ``grid`` can drive them from a tidy
per-replicate frame. Use the classes for composition, the functions for
one-off custom axes work.

Notebook idiom::

    from fl_plot import grid, Row, BandStack, BoxDist, ViolinDist, MeanCI, Band, RefOverlay

    fig, axes = grid(
        df_p, "p",
        rows=[
            Row(BandStack(), ylabel=r"average $\\sin^2$", ylim=(0, 1), height=1.0),
            Row(BoxDist("gap"), ylabel="gap = meas − theory\\n[per rep]"),
        ],
        suptitle="Growing p, fixed n=63",
        out_path=OUT_DIR / "my_figure",       # writes .png (and .pdf if asked)
    )

The tidy-frame contract matches the existing sweeps: columns ``n, p, j`` plus
whatever value columns the marks reference; one row per (replicate, factor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = [
    "Theme", "THEME", "sweep_axis", "SweepAxis",
    "rep_dists", "summarize",
    "draw_box_dist", "draw_violin_dist", "draw_band", "draw_mean_ci",
    "draw_band_stack", "draw_quantile_band", "draw_ref_overlay",
    "draw_disk_kde", "draw_disk_frame",
    "BoxDist", "ViolinDist", "Band", "MeanCI", "BandStack", "QuantileBand",
    "RefOverlay", "DiskDensity", "Row", "grid", "save",
]


# ── Theme ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    """Style constants. The default matches the existing fl_graphics look, so
    composed figures sit next to legacy ones without a visible seam."""

    # Indexed by factor identity (j − 1), so factor 4 is red in every figure it
    # appears in, regardless of which columns or overlays a figure shows.
    factor_colors: tuple = ("tab:blue", "tab:orange", "tab:green",
                            "tab:red", "tab:purple", "tab:brown")
    factor_cmaps: tuple = ("Blues", "Oranges", "Greens",
                           "Reds", "Purples", "YlOrBr")   # sequential twins of factor_colors
    # Also indexed by factor identity. Empty string = no hatch (the default
    # everywhere); set e.g. ("", "//", "xx") in a mono/grayscale theme so factor
    # identity survives without color.
    factor_hatches: tuple = ("",) * 6
    navy: str = "#1f3864"            # titles
    gray: str = "#555555"            # captions
    oos_fill: str = "#4878a8"        # out-of-subspace band
    oos_line: str = "#2c4a6e"
    insub_fill: str = "#f28e2b"      # in-subspace band
    insub_line: str = "#b5651d"
    overlay: str = "#b5651d"         # reference-overlay series
    grid_color: str = "0.85"
    zero_line_color: str = "0.6"
    caption_fontsize: float = 8.0
    tick_fontsize: float = 8.0
    sin2_ticks: tuple = ((0, 0.25, 0.5, 0.75, 1.0), ("0", "0.25", "0.50", "0.75", "1.0"))
    ci_caps: dict = field(default_factory=lambda: dict(
        fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=6))

    def factor_color(self, idx: int) -> str:
        return self.factor_colors[idx % len(self.factor_colors)]

    def factor_cmap(self, idx: int) -> str:
        return self.factor_cmaps[idx % len(self.factor_cmaps)]

    def factor_hatch(self, idx: int) -> str:
        return self.factor_hatches[idx % len(self.factor_hatches)]


THEME = Theme()


# ── Swept axis ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SweepAxis:
    """The categorical swept-axis convention, computed once per figure.

    ``order`` — swept values in plot order; ``cats`` — tick labels (p gets
    thousands commas, n stays plain); ``x`` — integer positions; ``xlabel``.
    """

    key: str
    order: tuple
    cats: tuple
    xlabel: str

    @property
    def x(self) -> np.ndarray:
        return np.arange(len(self.order))


def sweep_axis(df: pd.DataFrame, key: str) -> SweepAxis:
    """Build the :class:`SweepAxis` for ``key`` ('p' or 'n') from a tidy frame."""
    order = tuple(sorted(df[key].unique()))
    if key == "p":
        cats, xlabel = tuple(f"{v:,}" for v in order), "p (assets)"
    else:
        cats, xlabel = tuple(str(v) for v in order), "n (periods)"
    return SweepAxis(key=key, order=order, cats=cats, xlabel=xlabel)


# ── Data: frame → arrays ──────────────────────────────────────────────────────


def rep_dists(sub: pd.DataFrame, ax_spec: SweepAxis, col: str) -> list:
    """Per-replicate arrays of ``col`` at each swept value, in plot order.
    ``sub`` is the single-factor slice of the tidy frame."""
    return [sub.loc[sub[ax_spec.key] == v, col].to_numpy() for v in ax_spec.order]


def summarize(sub: pd.DataFrame, ax_spec: SweepAxis, col: str) -> pd.DataFrame:
    """mean/sem/median/q1/q3 of ``col`` per swept value (single-factor slice),
    reindexed to plot order."""
    g = (sub.groupby(ax_spec.key)[col]
            .agg(mean="mean", sem="sem", median="median",
                 q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)))
    return g.reindex(list(ax_spec.order))


def n_reps(df: pd.DataFrame) -> int:
    """Replicate count per (n, p, j) cell — used in captions."""
    return int(df.groupby(["n", "p", "j"]).size().iloc[0])


# ── Draw primitives (arrays in, artists out) ──────────────────────────────────


def draw_zero_line(ax, theme: Theme = THEME) -> None:
    ax.axhline(0, color=theme.zero_line_color, lw=0.8, ls="--", zorder=1)


def draw_box_dist(ax, ax_spec: SweepAxis, dists, color, *, width=0.55,
                  hatch="", theme: Theme = THEME) -> None:
    """Per-replicate boxplots at each swept value: IQR box, 1.5·IQR whiskers,
    black median, fliers hidden. ``hatch`` fills the boxes with a pattern
    (drawn in the edge color) on top of the translucent face. Hatched boxes
    bake the translucency into the facecolor instead of patch alpha — patch
    alpha also fades the hatch strokes, which the PDF backend renders as
    near-invisible."""
    if hatch:
        boxprops = dict(facecolor=mcolors.to_rgba(color, 0.55),
                        edgecolor=color, hatch=hatch)
    else:
        boxprops = dict(facecolor=color, alpha=0.55, edgecolor=color)
    ax.boxplot(
        dists, positions=ax_spec.x, widths=width, patch_artist=True,
        showfliers=False, zorder=3,
        medianprops=dict(color="black", lw=1.1),
        boxprops=boxprops,
        whiskerprops=dict(color=color), capprops=dict(color=color),
    )
    ax.set_xlim(ax_spec.x[0] - 0.5, ax_spec.x[-1] + 0.5)
    ax.set_xticks(ax_spec.x, ax_spec.cats, fontsize=theme.tick_fontsize)


def draw_violin_dist(ax, ax_spec: SweepAxis, dists, color, *, clip=(0.005, 0.995),
                     width=0.8, theme: Theme = THEME) -> None:
    """Per-replicate violins: KDE body over the inner ``clip`` quantile range
    (heavy tails would otherwise flatten every body to a spike), inner summary
    on the untrimmed sample — 1.5·IQR line, thick IQR bar, white median dot."""
    bodies, positions = [], []
    for arr, xi in zip(dists, ax_spec.x):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        inside = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
        lo, hi = (inside.min(), inside.max()) if inside.size else (med, med)
        body = a
        if clip is not None and a.size > 1:
            c_lo, c_hi = np.quantile(a, clip)
            body = a[(a >= c_lo) & (a <= c_hi)]
        if body.size >= 2 and np.ptp(body) > 0:
            bodies.append(body)
            positions.append(xi)
        else:
            ax.plot([xi - width / 2, xi + width / 2], [med, med],
                    color=color, lw=1.6, zorder=3)
        ax.vlines(xi, lo, hi, color="0.25", lw=0.9, zorder=5)
        ax.vlines(xi, q1, q3, color="0.15", lw=4.0, zorder=6)
        ax.plot([xi], [med], marker="o", ms=3.0, mfc="white", mec="0.15",
                mew=0.7, zorder=7)
    if bodies:
        parts = ax.violinplot(bodies, positions=positions, widths=width,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.45)
            b.set_linewidth(0.8)
            b.set_zorder(3)
    ax.set_xlim(ax_spec.x[0] - 0.5, ax_spec.x[-1] + 0.5)
    ax.set_xticks(ax_spec.x, ax_spec.cats, fontsize=theme.tick_fontsize)


def draw_band(ax, ax_spec: SweepAxis, level, *, fill, line, base=None,
              theme: Theme = THEME) -> None:
    """One shaded band from ``base`` (default 0) up to ``level`` with its
    topline — the single-component theory band."""
    x = ax_spec.x
    lo = np.zeros_like(np.asarray(level, dtype=float)) if base is None else np.asarray(base)
    ax.fill_between(x, lo, level, color=fill, alpha=0.45, zorder=1)
    ax.plot(x, level, "-", color=line, lw=1.2, zorder=3)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x, ax_spec.cats, fontsize=theme.tick_fontsize)


def draw_mean_ci(ax, ax_spec: SweepAxis, mean, sem, *, color="black",
                 theme: Theme = THEME) -> None:
    """Measured mean as dots+line with 95% CI caps."""
    x = ax_spec.x
    ax.plot(x, mean, "o-", color=color, lw=1.2, zorder=5)
    ax.errorbar(x, mean, yerr=1.96 * np.asarray(sem), **theme.ci_caps)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x, ax_spec.cats, fontsize=theme.tick_fontsize)


def draw_band_stack(ax, ax_spec: SweepAxis, lower, upper, measured, measured_sem,
                    *, theme: Theme = THEME) -> None:
    """The two-component additive stack: 0→lower (out-of-subspace fill),
    lower→upper (in-subspace fill), toplines, and the measured mean on top."""
    x = ax_spec.x
    ax.fill_between(x, 0, lower, color=theme.oos_fill, alpha=0.45)
    ax.fill_between(x, lower, upper, color=theme.insub_fill, alpha=0.45)
    ax.plot(x, lower, "-", color=theme.oos_line, lw=1.2)
    ax.plot(x, upper, "-", color=theme.insub_line, lw=1.2)
    draw_mean_ci(ax, ax_spec, measured, measured_sem, theme=theme)


def draw_quantile_band(ax, ax_spec: SweepAxis, q1, med, q3, *, fill, line,
                       theme: Theme = THEME) -> None:
    """Q1–Q3 band with quartile toplines and a heavy median line — the
    distribution-of-a-level view (no theory reference implied)."""
    x = ax_spec.x
    ax.fill_between(x, q1, q3, color=fill, alpha=0.45, zorder=1)
    ax.plot(x, q1, "-", color=line, lw=0.8, zorder=3)
    ax.plot(x, q3, "-", color=line, lw=0.8, zorder=3)
    ax.plot(x, med, "-", color=line, lw=2.0, zorder=4)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x, ax_spec.cats, fontsize=theme.tick_fontsize)


def draw_ref_overlay(ax, ax_spec: SweepAxis, q1, med, q3, *, color=None,
                     theme: Theme = THEME) -> None:
    """Overlay a second quantity's median (dashed, dotted markers) and Q1–Q3
    (translucent band) over an existing panel, on the same y-axis."""
    color = color or theme.overlay
    x = ax_spec.x
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0, zorder=8)
    ax.plot(x, med, "--", color=color, lw=1.8, zorder=9)
    ax.plot(x, med, "o", color=color, ms=3.5, zorder=9)


def draw_disk_frame(ax, *, radius=1.0, pad=0.15, cross=True, labels=None) -> None:
    """Unit-disk frame for direction-projection panels: boundary circle,
    optional axis cross, equal aspect, spines/ticks off.

    ``labels`` — optional (x_label, y_label) drawn at the positive axis ends.
    """
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color="#B4B2A9", lw=1.3, zorder=2)
    a = radius * (1 + pad)
    if cross:
        ax.plot([-a, a], [0, 0], color="0.55", ls=":", lw=1.0, zorder=1)
        ax.plot([0, 0], [-a, a], color="0.55", ls=":", lw=1.0, zorder=1)
    if labels is not None:
        ax.text(a, 0.03 * radius, labels[0], color="0.3", fontsize=9,
                ha="right", va="bottom")
        ax.text(0.03 * radius, a, labels[1], color="0.3", fontsize=9,
                ha="left", va="top")
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_disk_kde(ax, xy, cmap, *, gridsize=220, bw_method=None, alpha=0.85,
                  floor_frac=0.02, radius=1.0):
    """Smooth KDE heatmap of 2-D points confined to the disk of ``radius``.

    Adapted from the standalone ``radial_graphs/disk_heatmap.py`` prototype.
    ``xy`` is (N, 2) — e.g. per-replicate projections of unit vectors onto a
    2-D reference plane, which live in the unit disk by construction. The KDE
    is evaluated on our own grid and masked outside the disk (library 2-D
    density plots assume rectangular support and paint meaningless mass at
    r > 1); cells below ``floor_frac`` of the layer's own max are masked too,
    so overlaid layers read as islands of color rather than full-disk washes.
    Returns the QuadMesh (for a colorbar).
    """
    from scipy.stats import gaussian_kde

    xy = np.asarray(xy, dtype=float)
    kde = gaussian_kde(xy.T, bw_method=bw_method)
    g = np.linspace(-radius, radius, gridsize)
    X, Y = np.meshgrid(g, g)
    dens = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    mask = (X ** 2 + Y ** 2 > radius ** 2) | (dens < floor_frac * dens.max())
    dens = np.ma.array(dens, mask=mask)
    return ax.pcolormesh(X, Y, dens, cmap=cmap, shading="auto", alpha=alpha, zorder=3)


# ── Marks: column-bound wrappers the composer can drive ───────────────────────
#
# A mark exposes:
#   draw(ax, ax_spec, sub, j_idx, theme)  — sub is the single-factor tidy slice
#   caption                              — fragment for the assembled caption
#   legend_handles(theme)                — optional, shown on the first panel


class BoxDist:
    """Per-replicate boxplots of ``col``; dashed zero reference by default."""

    caption = "box = IQR, whiskers 1.5·IQR, line = median, fliers hidden"

    def __init__(self, col: str, *, zero_line: bool = True, width: float = 0.55):
        self.col, self.zero_line, self.width = col, zero_line, width

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        if self.zero_line:
            draw_zero_line(ax, theme)
        draw_box_dist(ax, ax_spec, rep_dists(sub, ax_spec, self.col),
                      theme.factor_color(j_idx), width=self.width,
                      hatch=theme.factor_hatch(j_idx), theme=theme)

    def legend_handles(self, theme):
        return []


class ViolinDist:
    """Per-replicate violins of ``col``; see :func:`draw_violin_dist`."""

    caption = ("violin = KDE over the inner 99% of replicates, thick bar = IQR, "
               "thin bar = 1.5·IQR range, white dot = median")

    def __init__(self, col: str, *, zero_line: bool = True, clip=(0.005, 0.995),
                 width: float = 0.8):
        self.col, self.zero_line, self.clip, self.width = col, zero_line, clip, width

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        if self.zero_line:
            draw_zero_line(ax, theme)
        draw_violin_dist(ax, ax_spec, rep_dists(sub, ax_spec, self.col),
                         theme.factor_color(j_idx), clip=self.clip,
                         width=self.width, theme=theme)

    def legend_handles(self, theme):
        return []


class Band:
    """Single band from 0 up to mean(``col``) — a theory limit by default; pass
    ``caption`` to describe a band of something else (e.g. an observable estimate)."""

    caption = "shaded band = theory limit (mean over replicates)"

    def __init__(self, col: str, *, label: str | None = None,
                 fill: str | None = None, line: str | None = None,
                 caption: str | None = None):
        self.col, self.label, self.fill, self.line = col, label, fill, line
        if caption is not None:
            self.caption = caption

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        a = summarize(sub, ax_spec, self.col)
        draw_band(ax, ax_spec, a["mean"].to_numpy(),
                  fill=self.fill or theme.oos_fill,
                  line=self.line or theme.oos_line, theme=theme)

    def legend_handles(self, theme):
        return [Patch(facecolor=self.fill or theme.oos_fill, alpha=0.45,
                      edgecolor=self.line or theme.oos_line,
                      label=self.label or self.col)]


class MeanCI:
    """Measured mean of ``col`` with 95% CI caps (black dots + line by default;
    pass ``caption`` alongside a non-black ``color`` so the footer stays honest)."""

    caption = "black dots = mean with 95% CI caps"

    def __init__(self, col: str, *, label: str | None = None, color: str = "black",
                 caption: str | None = None):
        self.col, self.label, self.color = col, label, color
        if caption is not None:
            self.caption = caption

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        a = summarize(sub, ax_spec, self.col)
        draw_mean_ci(ax, ax_spec, a["mean"].to_numpy(), a["sem"].to_numpy(),
                     color=self.color, theme=theme)

    def legend_handles(self, theme):
        return [plt.Line2D([], [], marker="o", color=self.color, lw=1.2,
                           label=self.label or self.col)]


class BandStack:
    """The additive out-of-subspace + in-subspace stack with the measured line.

    Defaults to the standard tidy-frame columns (``floor``, ``rhs``, ``sin2_j``)."""

    caption = ("shaded bands = out-of-subspace + in-subspace (additive); "
               "black line = measured mean with 95% CI caps")

    def __init__(self, lower: str = "floor", upper: str = "rhs",
                 measured: str = "sin2_j", *, labels=None):
        self.lower, self.upper, self.measured = lower, upper, measured
        self.labels = labels or (r"out-of-subspace", r"in-subspace", r"measured")

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        lo = summarize(sub, ax_spec, self.lower)["mean"].to_numpy()
        up = summarize(sub, ax_spec, self.upper)["mean"].to_numpy()
        m = summarize(sub, ax_spec, self.measured)
        draw_band_stack(ax, ax_spec, lo, up, m["mean"].to_numpy(),
                        m["sem"].to_numpy(), theme=theme)

    def legend_handles(self, theme):
        return [
            Patch(facecolor=theme.oos_fill, alpha=0.45, label=self.labels[0]),
            Patch(facecolor=theme.insub_fill, alpha=0.45, label=self.labels[1]),
            plt.Line2D([], [], marker="o", color="black", lw=1.2, label=self.labels[2]),
        ]


class QuantileBand:
    """Q1–Q3 band + median line of ``col``'s per-replicate distribution —
    the honest top row for a level with no theory reference."""

    caption = "shaded band = Q1–Q3 across replicates, dark line = median"

    def __init__(self, col: str, *, label: str | None = None,
                 fill: str | None = None, line: str | None = None):
        self.col, self.label, self.fill, self.line = col, label, fill, line

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        a = summarize(sub, ax_spec, self.col)
        draw_quantile_band(ax, ax_spec, a["q1"].to_numpy(), a["median"].to_numpy(),
                           a["q3"].to_numpy(), fill=self.fill or theme.oos_fill,
                           line=self.line or theme.oos_line, theme=theme)

    def legend_handles(self, theme):
        return [
            Patch(facecolor=self.fill or theme.oos_fill, alpha=0.45,
                  label=(self.label or self.col) + " Q1–Q3"),
            plt.Line2D([], [], color=self.line or theme.oos_line, lw=2.0,
                       label=(self.label or self.col) + " median"),
        ]


class RefOverlay:
    """Overlay ``col``'s median + Q1–Q3 as a reference series on every panel
    of the row (dashed line + translucent band, distinct color)."""

    def __init__(self, col: str, *, label: str | None = None, color: str | None = None):
        self.col, self.label, self.color = col, label, color

    @property
    def caption(self):
        return (f"overlaid {self.label or self.col}: dashed line = median, "
                "translucent band = Q1–Q3")

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        a = summarize(sub, ax_spec, self.col)
        draw_ref_overlay(ax, ax_spec, a["q1"].to_numpy(), a["median"].to_numpy(),
                         a["q3"].to_numpy(), color=self.color, theme=theme)

    def legend_handles(self, theme):
        c = self.color or theme.overlay
        lbl = self.label or self.col
        return [
            plt.Line2D([], [], color=c, lw=1.8, ls="--", marker="o", ms=3.5,
                       label=f"{lbl} median"),
            Patch(facecolor=c, alpha=0.18, label=f"{lbl} Q1–Q3"),
        ]


class DiskDensity:
    """KDE density of per-replicate 2-D direction projections on the unit disk.

    ``xcol`` / ``ycol`` — columns holding the projection coordinates (points
    must satisfy x² + y² ≤ 1, e.g. coordinates of a unit vector in a 2-D
    reference plane). ``at`` — optional ``{column: value}`` filter applied to
    the factor slice, so a stack of Rows can walk the sweep
    (``Row(DiskDensity(..., at={"n": 20}), ylabel="n = 20")``, …).

    Unlike the sweep marks this one ignores the categorical x-axis entirely:
    it draws the disk frame (equal aspect, axes off) and the density cloud in
    the factor column's sequential colormap. Compose disk rows with other
    disk rows — mixing them with swept-axis rows in one ``grid`` would share
    x limits between a disk and a categorical axis.

    ``factor`` — pin the mark to one factor (overlay use: several ``DiskDensity``
    marks with different ``factor=`` in one Row share a panel, each cloud in its
    own factor's colormap — coloring follows factor *identity*, not layout).
    ``mean_arrow`` — draw an arrow from the disk center to the cloud's mean
    (the circular resultant: length 1 = perfectly concentrated on the boundary,
    0 = no net in-plane direction), annotated with its length.
    ``ref_point`` — optional (x, y, label) marker (e.g. the population
    direction); ``scatter`` — draw up to that many raw points over the cloud.
    """

    def __init__(self, xcol: str, ycol: str, *, at: dict | None = None,
                 factor: int | None = None, mean_arrow: bool = False,
                 median_arrow: bool = False, kde: bool = True,
                 cmap: str | None = None, label: str | None = None, labels=None,
                 ref_point=None, scatter: int = 0, bw_method=None,
                 floor_frac: float = 0.02, alpha: float = 0.85, seed: int = 0):
        self.xcol, self.ycol, self.at = xcol, ycol, at
        self.factor, self.mean_arrow, self.label = factor, mean_arrow, label
        self.median_arrow = median_arrow
        self.kde = kde                      # False -> raw dots only (with scatter=N)
        self.cmap, self.labels, self.ref_point = cmap, labels, ref_point
        self.scatter, self.bw_method = scatter, bw_method
        self.floor_frac, self.alpha, self.seed = floor_frac, alpha, seed

    @property
    def caption(self):
        if not self.kde:
            base = ("disk = per-replicate projections as dots "
                    "(boundary = unit norm)")
        else:
            base = ("disk = KDE of per-replicate direction projections "
                    "(boundary = unit norm; low-density cells masked)")
        if self.mean_arrow:
            base += ("; arrow = mean of the projected directions, annotated with "
                     "its length (1 = concentrated, 0 = no net direction)")
        if self.median_arrow:
            base += "; dashed arrow = coordinate-wise median"
        return base

    def draw(self, ax, ax_spec, sub, j_idx, theme):
        if self.at:
            for c, v in self.at.items():
                sub = sub[sub[c] == v]
        xy = sub[[self.xcol, self.ycol]].to_numpy()
        draw_disk_frame(ax, labels=self.labels)
        if self.kde and len(xy) >= 3:
            draw_disk_kde(ax, xy, self.cmap or theme.factor_cmap(j_idx),
                          bw_method=self.bw_method, floor_frac=self.floor_frac,
                          alpha=self.alpha)
        if self.scatter and len(xy):
            take = min(self.scatter, len(xy))
            idx = np.random.default_rng(self.seed).choice(len(xy), take, replace=False)
            dot_color = "0.2" if self.kde else theme.factor_color(j_idx)
            ax.scatter(xy[idx, 0], xy[idx, 1], s=5, color=dot_color,
                       alpha=0.35 if self.kde else 0.25, lw=0, zorder=4)
        if self.median_arrow and len(xy):
            qx, qy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
            col = theme.factor_color(j_idx)
            ax.annotate("", xy=(qx, qy), xytext=(0, 0), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3,
                                        linestyle="--", shrinkA=0, shrinkB=0))
        if self.mean_arrow and len(xy):
            mx, my = float(xy[:, 0].mean()), float(xy[:, 1].mean())
            length = float(np.hypot(mx, my))
            col = theme.factor_color(j_idx)
            ax.annotate("", xy=(mx, my), xytext=(0, 0), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                        shrinkA=0, shrinkB=0))
            # Label just past the arrowhead (offset outward; fallback offset at 0).
            ux, uy = (mx / length, my / length) if length > 1e-9 else (0.7, 0.7)
            import matplotlib.patheffects as _pe
            ax.text(mx + 0.09 * ux, my + 0.09 * uy, f"{length:.2f}", color=col,
                    fontsize=8, ha="center", va="center", zorder=8,
                    path_effects=[_pe.withStroke(linewidth=1.6, foreground="black")])
        if self.ref_point is not None:
            x, y, lbl = self.ref_point
            ax.scatter([x], [y], color="#E24B4A", edgecolor="#A32D2D", s=70, zorder=5)
            ax.text(x + 0.05, y + 0.07, lbl, color="#791F1F", fontsize=9)

    def legend_handles(self, theme):
        if self.label is None:
            return []
        idx = (self.factor - 1) if self.factor else 0
        col = plt.get_cmap(self.cmap or theme.factor_cmap(idx))(0.65)
        return [Patch(facecolor=col, alpha=self.alpha, label=self.label)]


# ── Composition ───────────────────────────────────────────────────────────────


@dataclass
class Row:
    """One figure row: one or more marks drawn onto the same axes, an axis
    label, and optional per-row geometry/limits.

    ``marks`` may be a single mark or a sequence (drawn in order — e.g. a
    ``Band`` under a ``MeanCI``, or a ``BoxDist`` plus a ``RefOverlay``); the
    same marks are drawn into every column. When each column needs *different*
    marks (e.g. one coordinate-plane disk per column), pass ``cells`` instead:
    a list with one mark-list per column, matched to the grid's columns in
    order. Exactly one of ``marks`` / ``cells`` should be given.
    """

    marks: object = None
    ylabel: str = ""
    height: float = 0.8
    ylim: tuple | None = None
    yticks: tuple | None = None
    cells: list | None = None

    def mark_list(self):
        return list(self.marks) if isinstance(self.marks, (list, tuple)) else [self.marks]

    def marks_for_col(self, ci: int):
        if self.cells is not None:
            m = self.cells[ci]
            return list(m) if isinstance(m, (list, tuple)) else [m]
        return self.mark_list()

    def all_marks(self):
        if self.cells is not None:
            return [m for cell in self.cells
                    for m in (cell if isinstance(cell, (list, tuple)) else [cell])]
        return self.mark_list()


def grid(df: pd.DataFrame, key: str, rows, *, suptitle=None, caption_extra=None,
         caption: bool = True, out_path=None, formats=("png",), dpi=150,
         sharey="row", factors=None, col_titles=None, figsize=None,
         legend_row: int | None = 0, theme: Theme = THEME):
    """Compose ``rows`` of marks × factor columns into a finished figure.

    - columns are the factors (``df["j"]`` values, sorted) with navy titles;
      ``col_titles`` overrides the default "factor j" labels (e.g. plane names
      when the columns aren't factor slices);
    - colors key off factor *identity* (j − 1), so a figure showing factors
      [2, 3] keeps their canonical colors; a mark with a ``factor`` attribute
      pins itself to that factor's slice and color regardless of its column
      (overlay use);
    - x is the categorical swept axis for ``key`` ('p' or 'n');
    - ``sharey``: "row" (default), True (one scale everywhere), or False;
    - the bottom caption is assembled from each mark's fragment + R;
    - ``legend_row``: row whose marks contribute the legend on its first
      panel (None disables);
    - saves ``out_path`` with each extension in ``formats`` when given.

    Returns ``(fig, axes)``.
    """
    ax_spec = sweep_axis(df, key)
    factors = factors or sorted(df["j"].unique())
    rows = list(rows)
    nrows, ncols = len(rows), len(factors)
    reps = n_reps(df)

    if figsize is None:
        figsize = (13.3, sum(1.0 if r.height >= 1 else 0.8 for r in rows) * 3.4 + 0.8)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex="col", sharey=sharey,
        gridspec_kw={"height_ratios": [r.height for r in rows]}, squeeze=False)
    # Margins reserved in inches (suptitle + factor titles above; xlabel + caption
    # below), so spacing holds across 1-row and 5-row figures alike.
    height_in = figsize[1]
    top_in = 0.85 if suptitle else 0.45
    fig.subplots_adjust(left=0.09, right=0.97, top=1 - top_in / height_in,
                        bottom=1.05 / height_in, hspace=0.13, wspace=0.08)

    for ci, j in enumerate(factors):
        sub = df[df["j"] == j]
        for ri, row in enumerate(rows):
            ax = axes[ri, ci]
            for mark in row.marks_for_col(ci):
                jm = getattr(mark, "factor", None)
                if jm is None:
                    mark.draw(ax, ax_spec, sub, int(j) - 1, theme)
                else:                      # pinned mark: its own slice + color
                    mark.draw(ax, ax_spec, df[df["j"] == jm], int(jm) - 1, theme)
            if ri == 0:
                title = col_titles[ci] if col_titles else f"factor {j}"
                ax.set_title(title, color=theme.navy)
            if ri == nrows - 1:
                ax.set_xlabel(ax_spec.xlabel)
    for ri, row in enumerate(rows):
        if row.ylim is not None:
            axes[ri, 0].set_ylim(*row.ylim)
        if row.yticks is not None:
            axes[ri, 0].set_yticks(*row.yticks)
        if axes[ri, 0].axison:
            axes[ri, 0].set_ylabel(row.ylabel)
        elif row.ylabel:
            # Axis-off rows (e.g. DiskDensity) swallow set_ylabel — draw the row
            # label as rotated text just left of the panel instead.
            axes[ri, 0].text(-0.06, 0.5, row.ylabel, transform=axes[ri, 0].transAxes,
                             rotation=90, ha="right", va="center", color=theme.navy)
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(True, color=theme.grid_color, lw=0.5)
        ax.label_outer()

    if legend_row is not None:
        handles = []
        for mark in rows[legend_row].all_marks():
            handles.extend(mark.legend_handles(theme))
        uniq = {}
        for h in handles:                      # dedupe by label (cells rows repeat marks)
            uniq.setdefault(h.get_label(), h)
        if uniq:
            axes[legend_row, 0].legend(handles=list(uniq.values()), fontsize=7.5,
                                       loc="upper right")

    if suptitle:
        fig.suptitle(suptitle, color=theme.navy, y=1 - 0.18 / height_in)

    if caption:
        fragments, seen = [], set()
        for row in rows:
            for mark in row.all_marks():
                frag = getattr(mark, "caption", "")
                if frag and frag not in seen:
                    seen.add(frag)
                    fragments.append(frag)
        text = ".   ".join(f[0].upper() + f[1:] for f in fragments)
        text += f".   Per-replicate marks use R = {reps}."
        if caption_extra:
            text += f"   {caption_extra}"
        fig.text(0.5, 0.015, text, ha="center", va="bottom",
                 fontsize=theme.caption_fontsize, color=theme.gray, wrap=True)

    if out_path is not None:
        save(fig, out_path, formats=formats, dpi=dpi)
    return fig, axes


def save(fig, path_stem, *, formats=("png",), dpi=150) -> list:
    """Save ``fig`` as ``path_stem`` + each extension; returns written paths."""
    stem = Path(path_stem)
    if stem.suffix:                       # tolerate a full filename
        stem = stem.with_suffix("")
    written = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        written.append(p)
    return written
