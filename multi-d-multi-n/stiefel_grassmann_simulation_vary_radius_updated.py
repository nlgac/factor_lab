from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.linalg import svd


"""
Simulation updates relative to the uploaded script:
- keep both n = 63 and n = 126;
- extend the experiment from k = 2 to k in {1, 2, 3};
- use the same simulation pipeline for all three dimensions;
- for k = 1, use the geodesic distance on the sphere S^{p-1};
- for k in {2, 3}, use the Grassmann geodesic distance based on principal angles;
- combine the five radius settings and the two n settings into one faceted figure for each dimension.
"""


FACTOR_LAMBDAS: dict[int, tuple[float, ...]] = {
    1: (9.0,),
    2: (9.0, 4.0),
    3: (9.0, 4.0, 2.25),
}


def orthonormalize(A: np.ndarray) -> np.ndarray:
    """Return a column-orthonormal basis spanning the columns of A."""
    Q, R = np.linalg.qr(A)
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1.0
    return Q * sign


def sphere_geodesic_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Geodesic distance on the unit sphere S^{p-1}."""
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    inner = np.clip(float(u @ v), -1.0, 1.0)
    return float(np.arccos(inner))


def grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Grassmann geodesic distance based on principal angles."""
    U = orthonormalize(U)
    V = orthonormalize(V)
    s = svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    return float(np.linalg.norm(theta))


def frame_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Unified distance for k = 1, 2, 3."""
    k = U.shape[1]
    if k == 1:
        return sphere_geodesic_distance(U[:, 0], V[:, 0])
    return grassmann_distance(U, V)


def generate_factor_sample(
    p: int,
    n: int,
    k: int,
    lambdas: tuple[float, ...],
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one sample from a k-factor Gaussian model and return:
    - U_true: the population k-frame;
    - U_hat: the estimated top-k sample frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(lambdas) != k:
        raise ValueError(f"Expected {k} eigenvalues, received {len(lambdas)}.")

    lambdas_arr = np.asarray(lambdas, dtype=float)
    if np.any(lambdas_arr <= sigma**2):
        raise ValueError("Each leading eigenvalue must be larger than sigma^2.")

    U_true = orthonormalize(rng.normal(size=(p, k)))
    signal_scales = np.sqrt(lambdas_arr - sigma**2)

    F = rng.normal(size=(k, n))
    E = rng.normal(size=(p, n))
    X = U_true @ (signal_scales[:, None] * F) + sigma * E
    X = X - X.mean(axis=1, keepdims=True)

    gram = X.T @ X
    evals, V = np.linalg.eigh(gram)
    idx = np.argsort(evals)[::-1][:k]
    evals = np.clip(evals[idx], 0.0, None)
    svals = np.sqrt(evals)
    V = V[:, idx]

    U_hat = X @ V
    U_hat = U_hat / np.where(svals > 0, svals, 1.0)
    U_hat = orthonormalize(U_hat)

    if k == 1 and float(U_hat[:, 0] @ U_true[:, 0]) < 0.0:
        U_hat = -U_hat

    return U_true, U_hat


def construct_target(
    U_true: np.ndarray,
    radius: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Construct a target point at geodesic distance `radius` from U_true.

    For k = 1 this is a point on S^{p-1}.
    For k in {2, 3} this is a point on Gr(p, k).
    """
    if rng is None:
        rng = np.random.default_rng()

    p, k = U_true.shape

    if k == 1:
        Z = rng.normal(size=(p, 1))
        Z = Z - U_true @ (U_true.T @ Z)
        G = orthonormalize(Z)
        V = U_true * np.cos(radius) + G * np.sin(radius)
        V = orthonormalize(V)
        if float(V[:, 0] @ U_true[:, 0]) < 0.0:
            V = -V
        return V

    direction = np.abs(rng.normal(size=k))
    direction = direction / np.linalg.norm(direction)
    theta = radius * direction

    R = orthonormalize(rng.normal(size=(k, k)))
    U_rot = U_true @ R

    Z = rng.normal(size=(p, k))
    Z = Z - U_true @ (U_true.T @ Z)
    G = orthonormalize(Z)

    V = U_rot @ np.diag(np.cos(theta)) + G @ np.diag(np.sin(theta))
    return orthonormalize(V)


def run_simulation(
    ps: list[int],
    radii: list[float],
    ns: list[int],
    dims: list[int],
    n_reps: int = 20,
    n_targets_per_rep: int = 5,
    factor_lambdas: dict[int, tuple[float, ...]] | None = None,
    sigma: float = 1.0,
    seed: int = 20260404,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full simulation for all dimensions, radii, and sample sizes."""
    if factor_lambdas is None:
        factor_lambdas = FACTOR_LAMBDAS

    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for dim in dims:
        lambdas = factor_lambdas[dim]

        for n in ns:
            for radius in radii:
                for p in ps:
                    for rep in range(n_reps):
                        U_true, U_hat = generate_factor_sample(
                            p=p,
                            n=n,
                            k=dim,
                            lambdas=lambdas,
                            sigma=sigma,
                            rng=rng,
                        )

                        d_st = frame_distance(U_hat, U_true)
                        records.append(
                            {
                                "dim": dim,
                                "n": n,
                                "radius": radius,
                                "p": p,
                                "rep": rep,
                                "distance_type": "sample-truth",
                                "distance": d_st,
                            }
                        )

                        for _ in range(n_targets_per_rep):
                            V = construct_target(U_true, radius=radius, rng=rng)
                            d_sv = frame_distance(U_hat, V)
                            d_tv = frame_distance(U_true, V)

                            records.append(
                                {
                                    "dim": dim,
                                    "n": n,
                                    "radius": radius,
                                    "p": p,
                                    "rep": rep,
                                    "distance_type": "sample-target",
                                    "distance": d_sv,
                                }
                            )
                            records.append(
                                {
                                    "dim": dim,
                                    "n": n,
                                    "radius": radius,
                                    "p": p,
                                    "rep": rep,
                                    "distance_type": "truth-target",
                                    "distance": d_tv,
                                }
                            )

    long_df = pd.DataFrame.from_records(records)

    summary_df = (
        long_df.groupby(["dim", "n", "radius", "p", "distance_type"], as_index=False)["distance"]
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
        .sort_values(["dim", "n", "radius", "p", "distance_type"])
        .reset_index(drop=True)
    )

    return long_df, summary_df


def make_facet_plot_for_dimension(
    long_df: pd.DataFrame,
    dim: int,
    ps: list[int],
    radii: list[float],
    ns: list[int],
    save_path: Path,
) -> None:
    """Create one faceted figure for a fixed dimension."""
    plot_df = long_df[
        (long_df["dim"] == dim)
        & (long_df["distance_type"].isin(["sample-target", "sample-truth"]))
    ].copy()

    plot_df["p_str"] = pd.Categorical(
        plot_df["p"].astype(str),
        categories=[str(p) for p in ps],
        ordered=True,
    )
    plot_df["radius_label"] = pd.Categorical(
        plot_df["radius"].map(lambda x: f"radius = {x:g}"),
        categories=[f"radius = {x:g}" for x in radii],
        ordered=True,
    )
    plot_df["n_label"] = pd.Categorical(
        plot_df["n"].map(lambda x: f"n = {x}"),
        categories=[f"n = {x}" for x in ns],
        ordered=True,
    )

    palette = {
        "sample-target": "#4C72B0",
        "sample-truth": "#DD8452",
    }

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=plot_df,
        kind="box",
        x="p_str",
        y="distance",
        hue="distance_type",
        row="n_label",
        col="radius_label",
        order=[str(p) for p in ps],
        row_order=[f"n = {x}" for x in ns],
        col_order=[f"radius = {x:g}" for x in radii],
        hue_order=["sample-target", "sample-truth"],
        palette=palette,
        height=3.5,
        aspect=1.0,
        linewidth=0.9,
        fliersize=1.0,
        sharey=True,
        legend=False,
    )

    for row_idx, n in enumerate(ns):
        for col_idx, radius in enumerate(radii):
            ax = g.axes[row_idx, col_idx]
            ax.axhline(radius, color="black", linestyle="--", linewidth=1.6)
            ax.tick_params(axis="x", rotation=45)
            if row_idx == len(ns) - 1:
                ax.set_xlabel("p")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Distance")
            else:
                ax.set_ylabel("")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.figure.subplots_adjust(top=0.88, bottom=0.08, wspace=0.12, hspace=0.22)
    g.figure.suptitle(f"{dim}D case")

    legend_handles = [
        Patch(facecolor=palette["sample-target"], edgecolor="black", label="sample-target"),
        Patch(facecolor=palette["sample-truth"], edgecolor="black", label="sample-truth"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="truth-target"),
    ]
    g.figure.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True)

    g.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(g.figure)


def write_readme(output_dir: Path, config: dict[str, object]) -> None:
    """Write a short README for the saved results."""
    lines = [
        "Stiefel/Grassmann distance simulation results",
        "",
        "Configuration:",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Contents:",
            "- long_results.csv: all simulated distances",
            "- summary_results.csv: grouped summary statistics",
            "- figures/dim_1_facet.png: 1D faceted figure",
            "- figures/dim_2_facet.png: 2D faceted figure",
            "- figures/dim_3_facet.png: 3D faceted figure",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ps = [100, 500, 1000, 2000, 5000, 10000]
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    ns = [63, 126]
    dims = [1, 2, 3]
    n_reps = 20
    n_targets_per_rep = 5
    sigma = 1.0
    seed = 20260404

    output_dir = Path("stiefel_dimension_n_radius_results")
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    long_df, summary_df = run_simulation(
        ps=ps,
        radii=radii,
        ns=ns,
        dims=dims,
        n_reps=n_reps,
        n_targets_per_rep=n_targets_per_rep,
        factor_lambdas=FACTOR_LAMBDAS,
        sigma=sigma,
        seed=seed,
    )

    long_df.to_csv(output_dir / "long_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)

    for dim in dims:
        make_facet_plot_for_dimension(
            long_df=long_df,
            dim=dim,
            ps=ps,
            radii=radii,
            ns=ns,
            save_path=figures_dir / f"dim_{dim}_facet.png",
        )

    write_readme(
        output_dir=output_dir,
        config={
            "ps": ps,
            "radii": radii,
            "ns": ns,
            "dims": dims,
            "factor_lambdas": FACTOR_LAMBDAS,
            "sigma": sigma,
            "n_reps": n_reps,
            "n_targets_per_rep": n_targets_per_rep,
            "seed": seed,
        },
    )

    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
