"""
factor_sims_inspect.py - Histogram inspection for saved model and returns data
==============================================================================

Plots histograms of the arrays saved by factor_sims.py --save-model and
--save-returns.

Usage
-----
    # Model arrays (B and D)
    python factor_sims_inspect.py model model.npz

    # Returns arrays (factor and idiosyncratic)
    python factor_sims_inspect.py returns returns.npz

    # Save figures instead of displaying
    python factor_sims_inspect.py model model.npz --output figures/
    python factor_sims_inspect.py returns returns.npz --output figures/

From Python
-----------
    from factor_sims_inspect import plot_model_histograms, plot_returns_histograms
    import numpy as np

    m = np.load('model.npz')
    plot_model_histograms(m['B'], m['F'], m['D'])

    r = np.load('returns.npz')
    plot_returns_histograms(r['factor_returns'], r['idio_returns'])
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Model histograms  (B and D)
# ---------------------------------------------------------------------------


def plot_model_histograms(
    B: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    bins: int = 60,
    output_path: Path | None = None,
) -> None:
    """
    Histogram grid for the population model arrays B and D.

    B (k × p): one subplot per factor row — distribution of loadings across
    all p assets for that factor.

    D (p × p diagonal): one subplot — distribution of idiosyncratic variances
    across all assets. D is diagonal so only the diagonal is plotted.

    F (k × k diagonal): printed as a table in the figure title; too few
    values to warrant a histogram.

    Parameters
    ----------
    B : ndarray, shape (k, p)
    F : ndarray, shape (k, k)
    D : ndarray, shape (p, p)
    bins : int
        Number of histogram bins. Default 60.
    output_path : Path or None
        If given, save the figure here instead of displaying it.
    """
    k, p = B.shape
    d_diag = np.diag(D)                       # (p,) idio variances
    f_diag = np.diag(F)                        # (k,) factor variances

    # k rows for B factors + 1 row for D diagonal
    n_rows = k + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 2.5 * n_rows))
    fig.suptitle(
        f"Population model histograms  |  k={k}, p={p}\n"
        f"Factor variances F: {[f'{v:.4f}' for v in f_diag]}",
        fontsize=12, y=1.01,
    )

    for i in range(k):
        ax = axes[i]
        ax.hist(B[i], bins=bins, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_title(f"B row {i}  —  factor {i} loadings across {p:,} assets",
                     fontsize=10)
        ax.set_xlabel("Loading value")
        ax.set_ylabel("Count")
        _add_stats(ax, B[i])

    ax = axes[k]
    ax.hist(d_diag, bins=bins, color='darkorange', edgecolor='none', alpha=0.85)
    ax.set_title(f"D diagonal  —  idiosyncratic variances across {p:,} assets",
                 fontsize=10)
    ax.set_xlabel("Idio variance")
    ax.set_ylabel("Count")
    _add_stats(ax, d_diag)

    fig.tight_layout()
    _save_or_show(fig, output_path, "model_histograms.png")


# ---------------------------------------------------------------------------
# Returns histograms (factor returns and idio returns)
# ---------------------------------------------------------------------------


def plot_returns_histograms(
    factor_returns: np.ndarray,
    idio_returns: np.ndarray,
    bins: int = 60,
    output_path: Path | None = None,
) -> None:
    """
    Histogram grid for simulated factor and idiosyncratic returns.

    factor_returns (num_sim × num_obs × k): one subplot per factor — all
    draws across every simulation window pooled together.

    idio_returns (num_sim × num_obs × p): one subplot — all idio draws
    pooled across all assets and windows. A single subplot is sufficient
    because by construction each asset's idio draws are N(0, d_i) where
    d_i is its idio variance; pooling them shows the marginal distribution.

    Parameters
    ----------
    factor_returns : ndarray, shape (num_sim, num_obs, k)
    idio_returns   : ndarray, shape (num_sim, num_obs, p)
    bins : int
        Number of histogram bins. Default 60.
    output_path : Path or None
        If given, save the figure here instead of displaying it.
    """
    num_sim, num_obs, k = factor_returns.shape
    p = idio_returns.shape[2]
    total_draws = num_sim * num_obs

    # k factor rows + 1 idio row
    n_rows = k + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 2.5 * n_rows))
    fig.suptitle(
        f"Returns histograms  |  k={k}, p={p}, "
        f"{num_sim} sims × {num_obs} obs = {total_draws:,} draws per series",
        fontsize=12, y=1.01,
    )

    for i in range(k):
        ax = axes[i]
        # Pool all simulation windows for this factor: shape (total_draws,)
        draws = factor_returns[:, :, i].ravel()
        ax.hist(draws, bins=bins, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_title(f"Factor {i} returns  —  {total_draws:,} pooled draws", fontsize=10)
        ax.set_xlabel("Return")
        ax.set_ylabel("Count")
        _add_stats(ax, draws)

    ax = axes[k]
    # Pool all assets, all windows — shape (num_sim * num_obs * p,)
    # At large p this can be several hundred million values; sample if needed.
    all_idio = idio_returns.ravel()
    max_sample = 2_000_000
    if len(all_idio) > max_sample:
        logger.info("Idio returns: {:,} total values — sampling {:,} for histogram",
                    len(all_idio), max_sample)
        rng = np.random.default_rng(0)
        all_idio = rng.choice(all_idio, size=max_sample, replace=False)
    ax.hist(all_idio, bins=bins, color='darkorange', edgecolor='none', alpha=0.85)
    ax.set_title(
        f"Idio returns  —  pooled across {p:,} assets, {total_draws:,} obs/asset",
        fontsize=10,
    )
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    _add_stats(ax, all_idio)

    fig.tight_layout()
    _save_or_show(fig, output_path, "returns_histograms.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_stats(ax: plt.Axes, values: np.ndarray) -> None:
    """Add mean and std annotation to an axes."""
    mean, std = float(np.mean(values)), float(np.std(values))
    ax.axvline(mean, color='black', lw=1.2, ls='--', alpha=0.8)
    ax.text(0.97, 0.93, f"mean={mean:.4f}  std={std:.4f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))


def _save_or_show(fig: plt.Figure, output_path: Path | None, default_name: str) -> None:
    """Save to file or display interactively."""
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / default_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        logger.info("Saved {}", output_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot histograms from factor_sims .npz output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python factor_sims_inspect.py model model.npz
  python factor_sims_inspect.py returns returns.npz
  python factor_sims_inspect.py model model.npz --output figures/
  python factor_sims_inspect.py returns returns.npz --bins 80 --output figures/
""",
    )
    parser.add_argument(
        'kind',
        choices=['model', 'returns'],
        help="'model' for B/F/D histograms; 'returns' for factor/idio return histograms.",
    )
    parser.add_argument(
        'npz_file',
        type=Path,
        help="Path to the .npz file saved by factor_sims.py.",
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help="Output directory or file path. If omitted, displays interactively.",
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=60,
        help="Number of histogram bins (default: 60).",
    )
    args = parser.parse_args()

    if not args.npz_file.exists():
        raise FileNotFoundError(args.npz_file)

    logger.info("Loading {}", args.npz_file)
    data = np.load(args.npz_file)
    logger.info("Arrays: {}", list(data.keys()))

    if args.kind == 'model':
        plot_model_histograms(
            data['B'], data['F'], data['D'],
            bins=args.bins,
            output_path=args.output,
        )
    else:
        plot_returns_histograms(
            data['factor_returns'], data['idio_returns'],
            bins=args.bins,
            output_path=args.output,
        )


if __name__ == '__main__':
    main()
