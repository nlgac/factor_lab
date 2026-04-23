"""
factor_sims_plots.py - Faceted catplot visualisations for factor_sims output
=============================================================================

Produces a single figure with two rows (grassmann, stiefel-canonical) and
one column per target radius, showing box plots of sample-target distances
across ambient dimensions p. A dashed reference line marks the nominal target
radius in each column.

Usage
-----
    from factor_sims import build_spec, run_simulation
    from factor_sims_plots import plot_results

    results = run_simulation(build_spec('toy'))
    plot_results(results, output_dir='factor_sims_output/figures')

Or from the command line to re-plot saved CSVs:

    python factor_sims_plots.py distances_all.csv --output figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


# ---------------------------------------------------------------------------
# Plot constants — change here to restyle globally
# ---------------------------------------------------------------------------

# Default distance types for plots.
# truth-target is excluded: it equals radius by construction, so its box plot
# would be a flat line already shown by the dashed reference line.
# sample-truth is excluded by default; include via --sample-truth / sample_truth=True.
_PLOT_DISTANCE_TYPES_DEFAULT: tuple[str, ...] = ("sample-target",)
_PLOT_DISTANCE_TYPES_WITH_TRUTH: tuple[str, ...] = ("sample-target", "sample-truth")

_PLOT_STYLE: dict = dict(style="whitegrid", context="paper")

_CATPLOT_KW: dict = dict(
    kind="box",
    x="p",
    y="distance",
    hue="distance_type",
    col="radius_label",
    row="metric",
    sharey="row",
    sharex = True,
    height=3.0,
    aspect=1.1,
    linewidth=0.8,
    showfliers=False,
)

_REFLINE_STYLE: dict = dict(ls="--", lw=1.2, color="black", alpha=0.7)

_SUPTITLE_FONTSIZE: int = 14
_SAVE_DPI: int = 220


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_results(results, output_dir: str | Path = "factor_sims_output/figures",
                 sample_truth: bool = False) -> None:
    """
    Produce and save the distance figure from a SimResults object.

    Parameters
    ----------
    results : SimResults
        Output of factor_sims.run_simulation().
    output_dir : str or Path
        Directory for saved figures. Created if it does not exist.
    sample_truth : bool
        When True, include sample-truth distances as an additional hue level.
        Pass True when run_simulation was called with sample_truth=True.

    Example
    -------
        results = run_simulation(spec, sample_truth=True)
        plot_results(results, 'output/figures', sample_truth=True)
        # writes output/figures/distances.png
        #   rows: grassmann, stiefel-canonical
        #   cols: r=0.1, r=0.5, r=1.0
        #   hue: sample-target, sample-truth
    """
    plot_dataframe(results.long_df, output_dir, sample_truth=sample_truth)


def plot_dataframe(df: pd.DataFrame, output_dir: str | Path = "figures", sample_truth: bool = False) -> None:
    """
    Produce and save a single figure with one row per metric and one column
    per target radius.

    Accepts the raw CSV output of factor_sims directly, so you can re-plot
    without re-running the simulation:

        df = pd.read_csv('distances_all.csv')
        plot_dataframe(df, 'figures/')

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns:
        dimension, p, n, radius, rep, metric, distance_type, distance,
        radius_label, n_label.
    output_dir : str or Path
        Directory for saved figures. Created if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise radius_label to one decimal place so r=0.10 and r=0.1 merge.
    # factor_sims uses f"r={r:.2f}" but the old simulation used f"r={r:.1f}".
    df = df.copy()
    df["radius_label"] = df["radius"].map(lambda r: f"r={r:.1f}")

    metrics = sorted(df["metric"].unique())
    logger.info("Plotting {} metric(s) as rows: {}", len(metrics), metrics)

    save_path = output_dir / "distances.png"
    distance_types = _PLOT_DISTANCE_TYPES_WITH_TRUTH if sample_truth else _PLOT_DISTANCE_TYPES_DEFAULT
    logger.info("Rendering combined figure -> {} (hue: {})", save_path, distance_types)
    _plot_all_metrics(df, save_path, distance_types=distance_types)
    logger.info("Saved {}", save_path)


# ---------------------------------------------------------------------------
# Private helpers — each has one responsibility
# ---------------------------------------------------------------------------


def _plot_all_metrics(df: pd.DataFrame, save_path: Path,
                      distance_types: tuple[str, ...] = _PLOT_DISTANCE_TYPES_DEFAULT) -> None:
    """Produce and save the combined figure: rows=metrics, cols=radii."""
    plot_df = _filter_plot_data(df, distance_types=distance_types)
    if plot_df.empty:
        logger.warning("No plottable data found, skipping.")
        return

    col_order, row_order = _derive_facet_orders(plot_df)
    radius_map = _build_radius_map(plot_df)

    sns.set_theme(**_PLOT_STYLE)
    g = _build_catplot(plot_df, col_order, row_order, distance_types=distance_types)
    _annotate_axes(g, col_order, radius_map)
    _set_figure_titles(g, col_order, row_order)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(g.fig)


def _filter_plot_data(df: pd.DataFrame, metric: str | None = None,
                      distance_types: tuple[str, ...] = _PLOT_DISTANCE_TYPES_DEFAULT) -> pd.DataFrame:
    """Subset to rows for the given distance types (all metrics if metric is None)."""
    mask = df["distance_type"].isin(distance_types)
    if metric is not None:
        mask &= df["metric"] == metric
    return df[mask].copy()


def _derive_facet_orders(plot_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sorted column (radius) and row (metric) label orders for the FacetGrid.

    Radius order is numerical to prevent lexicographic accidents.
    Metric order is alphabetical (grassmann before stiefel-canonical).
    """
    col_order = [f"r={r:.1f}" for r in sorted(plot_df["radius"].unique())]
    row_order = sorted(plot_df["metric"].unique())
    return col_order, row_order


def _build_radius_map(plot_df: pd.DataFrame) -> dict[str, float]:
    """Map each radius label (e.g. 'r=0.3') to its float value for reference lines."""
    return {f"r={r:.1f}": r for r in sorted(plot_df["radius"].unique())}


def _build_catplot(
    plot_df: pd.DataFrame,
    col_order: list[str],
    row_order: list[str],
    distance_types: tuple[str, ...] = _PLOT_DISTANCE_TYPES_DEFAULT,
) -> sns.FacetGrid:
    """Construct the seaborn FacetGrid."""
    return sns.catplot(
        data=plot_df,
        **{
            **_CATPLOT_KW,
            "col_order": col_order,
            "row_order": row_order,
            "hue_order": list(distance_types),
        },
    )


def _annotate_axes(
    g: sns.FacetGrid,
    col_order: list[str],
    radius_map: dict[str, float],
) -> None:
    """Add reference lines and axis labels to every subplot."""
    for axes_row in g.axes:
        for label, ax in zip(col_order, axes_row):
            ax.axhline(radius_map[label], **_REFLINE_STYLE)
            ax.set_xlabel("Ambient dimension (p)")
            ax.set_ylabel("Distance")


def _set_figure_titles(
    g: sns.FacetGrid,
    col_order: list[str],
    row_order: list[str],
) -> None:
    """Set panel titles, figure suptitle, and strip the legend title."""
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.suptitle(
        "Sample-Target Distance vs (p, radius, metric)",
        fontsize=_SUPTITLE_FONTSIZE,
        y=1.02,
    )
    g.fig.subplots_adjust(top=0.85)
    if g._legend:
        g._legend.set_title("")


# ---------------------------------------------------------------------------
# CLI entry point — re-plot from saved CSV without re-running simulation
# ---------------------------------------------------------------------------


def main() -> None:
    """Re-plot from a saved distances_all.csv file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-plot factor_sims distance output from a saved CSV.",
        epilog="""
examples:
  python factor_sims_plots.py distances_all.csv
  python factor_sims_plots.py distances_all.csv --output my_figures/
  python factor_sims_plots.py distances_all.csv --sample-truth
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_file", type=Path, help="Path to distances_all.csv")
    parser.add_argument("--output", type=Path, default=Path("figures"),
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--sample-truth", action="store_true",
                        help="Include sample-truth distances as a hue level "
                             "(only meaningful if the CSV contains them).")
    args = parser.parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(args.csv_file)

    logger.info("Loading {}", args.csv_file)
    df = pd.read_csv(args.csv_file)
    logger.info("Loaded {} rows", len(df))
    plot_dataframe(df, args.output, sample_truth=args.sample_truth)
    logger.info("Done.")


if __name__ == "__main__":
    main()
