
from seaborn.objects import Path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Used by both analyze_results and plot_distance_comparison.
METRICS = [
    ("grassmannian", "grassmann_sampling", "grassmann_perturb"),
    ("procrustes",   "procrustes_sampling", "procrustes_perturb"),
    ("chordal",      "chordal_sampling",    "chordal_perturb"),
]



def distance_histograms(out_dict: dict, subsample_sizes: 
    list, perturbation_epsilons: list,
    output_dir: Path = None) -> plt.Figure:
    """
    Produce one histogram-grid figure per distance metric.

    Layout:  rows = subsample sizes (p),  cols = perturbation epsilons (eps)

    Each panel overlays:
      blue   – sampling error distances (n_windows values, one per window)
      orange – perturbation distances at that eps  (n_windows * 20 values)

    Dashed vertical lines mark the means of each distribution.  A stats
    annotation in the top-right corner shows mean and variance.

    Parameters
    ----------
    out_dict   : dict       output of run_perturbation_study
    subsample_sizes : list of int
        Asset counts at which sampling error is measured.
    perturbation_epsilons : list of float
        Geodesic distances at which perturbation frames are generated.
    output_dir : Path or None
        If provided, each figure is saved as
        ``output_dir/distance_comparison_<metric>.png``.

    Returns
    -------
    The last matplotlib Figure created (one per metric).
    """
    sample_truth  = out_dict["sample_truth_distance_results"]
    sample_perturb = out_dict["sample_perturb_distance_results"]
    truth_perturb = out_dict["truth_perturb_distance_results"]

    n_p   = len(subsample_sizes)
    n_eps = len(perturbation_epsilons)

    for metric_name, samp_key, perturb_key in METRICS:
        fig, axes = plt.subplots(
            n_p, n_eps,
            figsize=(4 * n_eps, 3.5 * n_p),
            squeeze=False,
        )
        fig.suptitle(f"Sample vs Perturbation/Target and Truth vs. Perturbation/Target – {metric_name}",
                     fontsize=13, y=1.01)

        for row_i, p in enumerate(subsample_sizes):

            for col_j, eps in enumerate(perturbation_epsilons):
                s = np.array(sample_perturb[(eps,p)][perturb_key])  # (n_windows,)

                ax = axes[row_i][col_j]
                d  = np.array(truth_perturb[(eps, p)][perturb_key])
                
                #shared bins from Freedman-Diaconis rule applied to combined data for better visual comparison
                shared_bins = np.histogram_bin_edges(np.concatenate([s, d]), bins='fd')
                
                # Density-normalised so both distributions are visually comparable
                ax.hist(d, bins=shared_bins, density=True, alpha=0.6, color="darkorange",
                        label=f"perturb ε={eps}, truth -> perturb/target")
                ax.hist(s, bins=shared_bins, density=True, alpha=0.6, color="steelblue",
                        label="sampling error, truth -> sample")

                #overlay KDE curves for smoother comparison
                xs = np.linspace(shared_bins[0], shared_bins[-1], 300)
                ax.plot(xs, gaussian_kde(d)(xs), color='darkorange', lw=2)
                ax.plot(xs, gaussian_kde(s)(xs), color='steelblue', lw=2)
                
                # Mean lines
                ax.axvline(d.mean(), color="darkorange", linestyle="--", linewidth=1.2,label='truth -> perturb/target mean distance')
                ax.axvline(s.mean(), color="steelblue",  linestyle="--", linewidth=1.2, label='sample -> perturb/target mean distance')

                ax.set_title(f"p={p}, ε={eps}", fontsize=9)
                ax.set_xlabel("distance to truth")
                ax.set_ylabel("density")

                stats_txt = (
                    f"perturb μ={d.mean():.4f}\n"
                    f"sample  μ={s.mean():.4f}"
                )
                ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                        fontsize=7, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

                if row_i == 0 and col_j == 0:
                    ax.legend(fontsize=7)

        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / f"distance_comparison_{metric_name}.png",
                        dpi=150, bbox_inches="tight")
        plt.show()

    return fig