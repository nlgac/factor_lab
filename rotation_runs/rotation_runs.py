"""
rotation_runs.py — stability of the rotation term across unseeded runs.

Repeats the Part-(ii) experiment with ``random_seed=None`` (fresh OS entropy
each run, nested sampling, n=60) and compares the rotation term
1 − (ŵⱼ)ⱼ² across runs: within a run the boxplots are p-invariant, so any
spread across runs is pure sampling noise from the finite replication count.

Writes to nb_outputs/:

    rotation_runs.parquet        — combined per-replication rows (+ run column)
    rotation_runs_boxplots.png   — rotation boxplots grouped by factor,
                                   x = p, hue = run

With ``--dense``, sweeps every integer p in [200, 2000] instead (~50s per
run x reps/40): the per-run median traced over the dense grid shows the
nested-sampling random-walk wander that the coarse grid reads as "flat".
Writes rotation_runs_dense.parquet, rotation_dense_box.png (boxplots at
p = 200, 500, 1000, 2000 only), and rotation_dense_line.png.

Usage:

    python rotation_runs/rotation_runs.py                  # 5 runs x 40 reps
    python rotation_runs/rotation_runs.py 10 100           # n_runs n_reps
    python rotation_runs/rotation_runs.py --dense          # dense p-grid variant
    python rotation_runs/rotation_runs.py --dense 3 40
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (we live in a subdir)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import seaborn as sns
from loguru import logger

from fl_experiment_setup import ModelSpec, DesignSpec
from fl_experiment_runner import run_experiment
from sim_theorem_partii import DispersionBiasExperiment

OUT_DIR = ROOT / "nb_outputs"

P_VALUES = [200, 500, 1000, 2000, 5000, 10000, 20000]
P_DENSE = list(range(200, 2001))            # every integer p
P_DENSE_BOX = [200, 500, 1000, 2000]        # boxplot slice of the dense grid
N_PERIODS = 60


def simulate_runs(
    n_runs: int = 5, n_reps: int = 40, p_values: list[int] = P_VALUES
) -> pd.DataFrame:
    """n_runs unseeded sweeps, concatenated with a 1-based ``run`` column."""
    model = ModelSpec()  # canonical diagonal-Gram model
    runs = []
    for r in range(1, n_runs + 1):
        design = DesignSpec(
            n_values=[N_PERIODS], p_values=p_values, n_reps=n_reps,
            random_seed=None, sampling="nested",
        )
        logger.info("run {}/{}", r, n_runs)
        runs.append(
            run_experiment(model, design, DispersionBiasExperiment()).assign(run=r)
        )
    return pd.concat(runs, ignore_index=True)


def plot_rotation_boxplots(big: pd.DataFrame) -> sns.FacetGrid:
    """Rotation boxplots grouped by factor: x = p, hue = run."""
    n_runs = big["run"].nunique()
    n_reps = big.groupby(["run", "p", "j"]).size().iloc[0]
    sns.set_theme(style="whitegrid", context="paper")
    g = sns.catplot(
        data=big, x="p", y="rotation", hue="run", col="j",
        kind="box", showfliers=False,
        height=3.2, aspect=1.3, palette="tab10",
    )
    g.set_titles(col_template="Factor j={col_name}")
    g.set_axis_labels("Ambient dimension (p)", "rotation term")
    g.figure.suptitle(
        f"Rotation term across {n_runs} unseeded runs "
        f"(nested sampling, n={N_PERIODS}, {n_reps} reps)", y=1.06,
    )
    return g


def plot_rotation_medians(big: pd.DataFrame) -> sns.FacetGrid:
    """Per-run median rotation traced over p: x = p, one line per run."""
    med = big.groupby(["run", "p", "j"])["rotation"].median().reset_index()
    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=med, x="p", y="rotation", hue="run", col="j",
        kind="line", linewidth=1.0, height=3.2, aspect=1.3,
        palette="tab10",
    )
    g.set_titles(col_template="Factor j={col_name}")
    g.set_axis_labels("Ambient dimension (p)", "median rotation")
    g.figure.suptitle(
        f"Median rotation per run, traced over p in "
        f"[{big.p.min()}, {big.p.max()}]", y=1.06,
    )
    return g


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    args = [a for a in sys.argv[1:] if a != "--dense"]
    dense = "--dense" in sys.argv[1:]
    n_runs = int(args[0]) if len(args) > 0 else 5
    n_reps = int(args[1]) if len(args) > 1 else 40

    OUT_DIR.mkdir(exist_ok=True)
    if dense:
        big = simulate_runs(n_runs, n_reps, p_values=P_DENSE)
        big.to_parquet(OUT_DIR / "rotation_runs_dense.parquet")
        g = plot_rotation_boxplots(big[big.p.isin(P_DENSE_BOX)])
        g.figure.savefig(OUT_DIR / "rotation_dense_box.png", dpi=140, bbox_inches="tight")
        g = plot_rotation_medians(big)
        g.figure.savefig(OUT_DIR / "rotation_dense_line.png", dpi=140, bbox_inches="tight")
        logger.info("Saved rotation_runs_dense.parquet, rotation_dense_box.png, "
                    "rotation_dense_line.png to {}", OUT_DIR)
    else:
        big = simulate_runs(n_runs, n_reps)
        big.to_parquet(OUT_DIR / "rotation_runs.parquet")
        g = plot_rotation_boxplots(big)
        g.figure.savefig(OUT_DIR / "rotation_runs_boxplots.png", dpi=150, bbox_inches="tight")
        logger.info("Saved rotation_runs.parquet and rotation_runs_boxplots.png to {}", OUT_DIR)


if __name__ == "__main__":
    main()
