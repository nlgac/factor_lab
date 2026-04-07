from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import svd
from scipy.linalg import eigh


def orthonormalize(A: np.ndarray) -> np.ndarray:
    """Return a column-orthonormal basis spanning the columns of A."""
    Q, R = np.linalg.qr(A)
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1.0
    return Q * sign


def principal_angle_distance_from_overlap(M: np.ndarray) -> float:
    """
    Distance from the overlap matrix B^T U_hat.

    For k=1 this is the spherical geodesic distance on S^{p-1}.
    For k>=2 this is the Grassmann geodesic distance.
    """
    M = np.asarray(M)
    if M.shape == (1, 1):
        return float(np.arccos(np.clip(M[0, 0], -1.0, 1.0)))
    s = svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    return float(np.linalg.norm(theta))


def sample_residual_gram(
    m: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample Z^T Z with Z of shape (m, n), using a Bartlett draw when m >= n.
    """
    if m <= 0:
        return np.zeros((n, n))

    if m < n:
        Z = rng.normal(size=(m, n))
        return Z.T @ Z

    A = np.zeros((n, n))
    diag = np.sqrt(rng.chisquare(df=np.arange(m, m - n, -1)))
    A[np.arange(n), np.arange(n)] = diag
    tril_idx = np.tril_indices(n, k=-1)
    A[tril_idx] = rng.normal(size=len(tril_idx[0]))
    return A @ A.T


class CanonicalGeometry:
    """Canonical truth frame and canonical target-complement directions."""

    def __init__(self, p: int, k: int) -> None:
        if p < 2 * k:
            raise ValueError("This implementation requires p >= 2k.")
        self.p = p
        self.k = k
        self.U_true = np.eye(p, k)
        self.G = np.eye(p, 2 * k)[:, k : 2 * k]


def generate_sample_overlaps(
    p: int,
    n: int,
    lambdas: tuple[float, ...],
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the overlaps U_true^T U_hat and G^T U_hat without building the full p x n matrix.

    By rotational invariance, the true frame can be fixed to the first k coordinate axes.
    The remaining noise contribution enters through a Wishart term in the sample Gram matrix.
    """
    k = len(lambdas)
    if p < 2 * k:
        raise ValueError("p must satisfy p >= 2k.")
    if np.any(np.asarray(lambdas) <= sigma**2):
        raise ValueError("Each lambda must be larger than sigma^2.")

    signal_scales = np.sqrt(np.asarray(lambdas) - sigma**2)

    F = rng.normal(size=(k, n))
    E_u = rng.normal(size=(k, n))
    E_g = rng.normal(size=(k, n))
    residual_df = p - 2 * k

    A = signal_scales[:, None] * F + sigma * E_u
    B = sigma * E_g
    residual_gram = sigma**2 * sample_residual_gram(residual_df, n, rng)
    gram = A.T @ A + B.T @ B + residual_gram

    n_obs = gram.shape[0]
    evals, V = eigh(gram, subset_by_index=[n_obs - k, n_obs - 1])
    evals = np.clip(evals[::-1], 0.0, None)
    svals = np.sqrt(evals)
    V = V[:, ::-1]

    inv_s = np.where(svals > 0, 1.0 / svals, 1.0)
    S_u = (A @ V) * inv_s
    S_g = (B @ V) * inv_s

    if k == 1 and S_u[0, 0] < 0.0:
        S_u *= -1.0
        S_g *= -1.0

    return S_u, S_g


def target_overlap_with_sample(
    radius: float,
    S_u: np.ndarray,
    S_g: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return V^T U_hat for a target at the prescribed distance from the truth."""
    k = S_u.shape[0]

    if k == 1:
        return np.array([[np.cos(radius) * S_u[0, 0] + np.sin(radius) * S_g[0, 0]]])

    weights = np.abs(rng.normal(size=k))
    weights = weights / np.linalg.norm(weights)
    theta = radius * weights

    R = orthonormalize(rng.normal(size=(k, k)))
    cos_part = np.diag(np.cos(theta)) @ (R.T @ S_u)
    sin_part = np.diag(np.sin(theta)) @ S_g
    return cos_part + sin_part


def distance_name(k: int) -> str:
    return "sphere-geodesic" if k == 1 else "grassmann"


def run_simulation(
    ps: list[int],
    ns: list[int],
    radii: list[float],
    lambdas_by_dim: dict[int, tuple[float, ...]],
    n_reps: int = 12,
    n_targets_per_rep: int = 8,
    sigma: float = 1.0,
    seed: int = 20260403,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full simulation grid across dimensions, sample sizes, and radii."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for dim, lambdas in lambdas_by_dim.items():
        for n in ns:
            for radius in radii:
                for p in ps:
                    for rep in range(n_reps):
                        S_u, S_g = generate_sample_overlaps(
                            p=p,
                            n=n,
                            lambdas=lambdas,
                            sigma=sigma,
                            rng=rng,
                        )

                        records.append(
                            {
                                "dimension": dim,
                                "n": n,
                                "radius": radius,
                                "p": p,
                                "rep": rep,
                                "distance_type": "sample-truth",
                                "distance": principal_angle_distance_from_overlap(S_u),
                                "distance_metric": distance_name(dim),
                            }
                        )

                        for _ in range(n_targets_per_rep):
                            VtUh = target_overlap_with_sample(
                                radius=radius,
                                S_u=S_u,
                                S_g=S_g,
                                rng=rng,
                            )
                            records.append(
                                {
                                    "dimension": dim,
                                    "n": n,
                                    "radius": radius,
                                    "p": p,
                                    "rep": rep,
                                    "distance_type": "sample-target",
                                    "distance": principal_angle_distance_from_overlap(VtUh),
                                    "distance_metric": distance_name(dim),
                                }
                            )
                            records.append(
                                {
                                    "dimension": dim,
                                    "n": n,
                                    "radius": radius,
                                    "p": p,
                                    "rep": rep,
                                    "distance_type": "truth-target",
                                    "distance": radius,
                                    "distance_metric": distance_name(dim),
                                }
                            )

    long_df = pd.DataFrame.from_records(records)
    long_df["radius_label"] = long_df["radius"].map(lambda x: f"r = {x:.1f}")
    long_df["n_label"] = long_df["n"].map(lambda x: f"n = {x}")
    long_df["p_str"] = long_df["p"].astype(str)

    summary_df = (
        long_df.groupby(
            ["dimension", "n", "radius", "p", "distance_type", "distance_metric"],
            as_index=False,
        )["distance"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: np.quantile(x, 0.25),
            q75=lambda x: np.quantile(x, 0.75),
            min="min",
            max="max",
        )
        .sort_values(["dimension", "n", "radius", "p", "distance_type"])
        .reset_index(drop=True)
    )

    return long_df, summary_df


def make_dimension_facet_plot(
    long_df: pd.DataFrame,
    dim: int,
    save_path: Path,
) -> None:
    """Create one faceted plot for a fixed latent dimension."""
    plot_df = long_df[
        (long_df["dimension"] == dim)
        & (long_df["distance_type"].isin(["sample-target", "sample-truth"]))
    ].copy()

    radius_order = [f"r = {r:.1f}" for r in sorted(plot_df["radius"].unique())]
    n_order = [f"n = {n}" for n in sorted(plot_df["n"].unique())]
    p_order = [str(p) for p in sorted(plot_df["p"].unique())]

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.catplot(
        data=plot_df,
        kind="box",
        x="p_str",
        y="distance",
        hue="distance_type",
        col="radius_label",
        row="n_label",
        order=p_order,
        col_order=radius_order,
        row_order=n_order,
        hue_order=["sample-target", "sample-truth"],
        sharey=True,
        height=3.0,
        aspect=1.1,
        linewidth=0.8,
        showfliers=False,
        legend=True,
    )

    radius_map = {f"r = {r:.1f}": r for r in sorted(plot_df["radius"].unique())}
    for axes_row in g.axes:
        for radius_label, ax in zip(radius_order, axes_row):
            ax.axhline(
                radius_map[radius_label],
                linestyle="--",
                linewidth=1.2,
                color="black",
            )
            ax.set_xlabel("p")
            ax.set_ylabel("Distance")

    metric_label = "Sphere geodesic distance" if dim == 1 else "Grassmann distance"
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.subplots_adjust(top=0.90, wspace=0.10, hspace=0.18)
    g.fig.suptitle(
        f"Dimension {dim}: {metric_label} across radius and sample size",
        fontsize=14,
    )
    if g._legend is not None:
        g._legend.set_title("")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def write_readme(output_dir: Path, config: dict[str, object]) -> None:
    """Write a compact README for the output directory."""
    lines = [
        "Distance simulation outputs",
        "",
        "Configuration:",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Files:",
            "- long_results.csv: all simulated distances.",
            "- summary_results.csv: grouped summaries.",
            "- figures/dimension_1_faceted.png: 1-factor sphere-distance results.",
            "- figures/dimension_2_faceted.png: 2-factor Grassmann-distance results.",
            "- figures/dimension_3_faceted.png: 3-factor Grassmann-distance results.",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ps = [100, 500, 1000, 2000, 5000, 10000]
    ns = [63, 126]
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    lambdas_by_dim = {
        1: (9.0,),
        2: (9.0, 4.0),
        3: (9.0, 4.0, 2.0),
    }
    n_reps = 12
    n_targets_per_rep = 8
    sigma = 1.0
    seed = 20260403

    output_dir = Path("")
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    long_df, summary_df = run_simulation(
        ps=ps,
        ns=ns,
        radii=radii,
        lambdas_by_dim=lambdas_by_dim,
        n_reps=n_reps,
        n_targets_per_rep=n_targets_per_rep,
        sigma=sigma,
        seed=seed,
    )

    long_df.to_csv(output_dir / "long_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)

    for dim in sorted(lambdas_by_dim):
        make_dimension_facet_plot(
            long_df=long_df,
            dim=dim,
            save_path=figure_dir / f"dimension_{dim}_faceted.png",
        )

    write_readme(
        output_dir=output_dir,
        config={
            "ps": ps,
            "ns": ns,
            "radii": radii,
            "lambdas_by_dim": lambdas_by_dim,
            "sigma": sigma,
            "n_reps": n_reps,
            "n_targets_per_rep": n_targets_per_rep,
            "seed": seed,
        },
    )

    print("Saved updated outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()
