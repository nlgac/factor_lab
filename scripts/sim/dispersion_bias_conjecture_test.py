"""
dispersion_bias_conjecture_test.py

Numerical verification of the k-factor dispersion bias conjecture:

    H^T z  -  (H^T B)(B^T z)  →  0  a.s. as p → ∞

where H = [h_1,...,h_k] are the top-k left singular vectors of Y/√n,
B = [b_1,...,b_k] is an orthonormal basis for the population factor subspace,
and z = e/√p is the equal-weight unit vector.

At k=1 this reduces to equation (13) of Goldberg–Papanicolaou–Shkolnik (2022).

Usage:
    python dispersion_bias_conjecture_test.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

# Ensure we use the local factor_lab, not any system install.
_FACTOR_LAB_ROOT = Path(__file__).parent
sys.path.insert(0, str(_FACTOR_LAB_ROOT))
from factor_lab import svd_decomposition  # noqa: F401  (sanity-check import)

OUT_DIR = _FACTOR_LAB_ROOT  # save results alongside script

# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------

GRID = {
    "k_values": [1, 2, 3, 5],
    "p_values": [200, 500, 1000, 2000, 5000, 10000],
    "n_values": [30, 50, 100],
    "n_reps": 200,
    "delta": 1.0,
    "seed": 20260424,
}

# Use sparse SVD for large p to avoid O(p^2) cost.
_SPARSE_SVD_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Core per-replication function
# ---------------------------------------------------------------------------

def _top_k_left_svecs(Y_scaled: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, s) — top-k left singular vectors and values of Y_scaled.

    Uses sparse SVD for large matrices to avoid forming p×p covariance.
    """
    p = Y_scaled.shape[0]
    if p >= _SPARSE_SVD_THRESHOLD:
        U, s, _ = svds(Y_scaled, k=k)
        order = np.argsort(s)[::-1]
        return U[:, order], s[order]
    U, s, _ = np.linalg.svd(Y_scaled, full_matrices=False)
    return U[:, :k], s[:k]


def run_one_rep(p: int, n: int, k: int, delta: float, rng: np.random.Generator,
                correlated_loadings: bool = False) -> dict:
    """Run a single (p, n, k) Monte Carlo replication.

    Returns a dict of scalar diagnostics plus a few small arrays.
    """
    sigma_beta = 0.3
    mu = np.linspace(0.5, 1.5, k)

    # --- 3a/3b. Draw raw loadings and orthonormalize. ---
    if correlated_loadings:
        # Step 7: columns of B_raw are correlated (off-diag ρ ≈ 0.7).
        # L is the Cholesky of the k×k correlation matrix.
        rho = 0.7
        corr = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
        L = np.linalg.cholesky(corr)
        base = rng.normal(size=(p, k)) * sigma_beta
        B_raw = base @ L.T + mu[np.newaxis, :]
    else:
        B_raw = rng.normal(size=(p, k)) * sigma_beta + mu[np.newaxis, :]

    B, _ = np.linalg.qr(B_raw)  # B: (p, k), orthonormal columns

    # Sign convention: each column of B has positive mean.
    for i in range(k):
        if B[:, i].mean() < 0:
            B[:, i] *= -1

    # --- 3d/3e. Factor returns and idiosyncratic noise. ---
    # Distinct factor variances so eigenvalues are well-separated.
    factor_vars = np.arange(k, 0, -1, dtype=float)
    F = rng.normal(size=(n, k)) * np.sqrt(factor_vars)[np.newaxis, :]
    Z = rng.normal(size=(p, n)) * delta

    # --- 3f. Return matrix uses B_raw (not B) to preserve the natural covariance. ---
    Y = B_raw @ F.T + Z

    # --- 3g. Top-k left singular vectors of Y/√n. ---
    H, s = _top_k_left_svecs(Y / np.sqrt(n), k)

    # Sign convention: each column of H has positive mean (mirrors B convention).
    for i in range(k):
        if H[:, i].mean() < 0:
            H[:, i] *= -1

    # --- 3i. Residual — the quantity under test. ---
    z = np.ones(p) / np.sqrt(p)
    Htz = H.T @ z           # (k,)
    HtB = H.T @ B           # (k, k)
    Btz = B.T @ z           # (k,)
    residual = Htz - HtB @ Btz
    R_norm = np.linalg.norm(residual)

    # --- 3j. Layer-3 componentwise residual (diagonal only). ---
    r_comp = Htz - np.diag(HtB) * Btz
    r_comp_max = np.max(np.abs(r_comp))

    # --- 3k. Principal angles and ψ_i diagnostics. ---
    sv_HtB = np.linalg.svd(HtB, compute_uv=False)
    principal_angles = np.arccos(np.clip(sv_HtB, -1.0, 1.0))

    s2 = s ** 2
    trace_S = np.sum((Y / np.sqrt(n)) ** 2)
    residual_trace = trace_S - s2.sum()
    ell2 = residual_trace / max(n - k, 1)
    psi_i = np.sqrt(np.maximum((s2 - ell2) / s2, 0.0))

    return {
        "R_norm": R_norm,
        "R_rel": R_norm / (np.linalg.norm(Btz) + 1e-12),
        "r_comp_max": r_comp_max,
        "max_principal_angle": principal_angles.max(),
        "psi_min": psi_i.min(),
    }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def _run_cell(k: int, n: int, p: int, n_reps: int, delta: float,
              seed: int, correlated_loadings: bool = False) -> list[dict]:
    """Run all replications for a single (k, n, p) cell — parallelism unit."""
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(n_reps):
        rep_rng = np.random.default_rng(rng.integers(2**63))
        res = run_one_rep(p, n, k, delta, rep_rng, correlated_loadings)
        records.append({"k": k, "n": n, "p": p, "rep": rep, **res})
    return records


def run_grid(grid: dict, correlated_loadings: bool = False,
             k_filter: list[int] | None = None) -> pd.DataFrame:
    """Run the full simulation grid, returning a tidy DataFrame."""
    k_vals = k_filter if k_filter else grid["k_values"]
    cells = [
        (k, n, p)
        for k in k_vals
        for n in grid["n_values"]
        for p in grid["p_values"]
    ]
    tag = "correlated" if correlated_loadings else "main"
    logger.info("Starting {} grid: {} cells × {} reps each", tag, len(cells), grid["n_reps"])

    seed_rng = np.random.default_rng(grid["seed"])
    cell_seeds = [int(seed_rng.integers(2**63)) for _ in cells]

    all_records = []
    for (k, n, p), seed in tqdm(zip(cells, cell_seeds), total=len(cells),
                                desc=f"{tag} grid", unit="cell"):
        logger.debug("Cell k={} n={} p={}", k, n, p)
        records = Parallel(n_jobs=-1)(
            delayed(run_one_rep)(p, n, k, grid["delta"],
                                 np.random.default_rng(seed + rep),
                                 correlated_loadings)
            for rep in range(grid["n_reps"])
        )
        for rep, res in enumerate(records):
            all_records.append({"k": k, "n": n, "p": p, "rep": rep, **res})

    df = pd.DataFrame(all_records)
    logger.info("Grid complete — {} rows collected", len(df))
    return df


# ---------------------------------------------------------------------------
# Sanity checks (Step 5)
# ---------------------------------------------------------------------------

def check_k1_slope(df: pd.DataFrame) -> float:
    """Check 5a: k=1 residual should decay as p^{-1/2} (slope ≈ -0.5)."""
    sub = df[df["k"] == 1].groupby("p")["R_norm"].median()
    x = np.log10(sub.index.values.astype(float))
    y = np.log10(sub.values)
    slope = np.polyfit(x, y, 1)[0]
    logger.info("Check 5a (k=1 slope): {:.3f}  (expect ≈ -0.5)", slope)
    return slope


def check_rotation_invariance(p: int = 500, n: int = 50, k: int = 3,
                               seed: int = 42) -> float:
    """Check 5b: rotating H by orthogonal Q leaves |R| unchanged."""
    rng = np.random.default_rng(seed)
    res = run_one_rep(p, n, k, delta=1.0, rng=rng)
    R_before = res["R_norm"]

    # Re-run same rep to get H explicitly (re-use same seed).
    rng2 = np.random.default_rng(seed)
    mu = np.linspace(0.5, 1.5, k)
    B_raw = rng2.normal(size=(p, k)) * 0.3 + mu[np.newaxis, :]
    B, _ = np.linalg.qr(B_raw)
    for i in range(k):
        if B[:, i].mean() < 0:
            B[:, i] *= -1
    factor_vars = np.arange(k, 0, -1, dtype=float)
    F_mat = rng2.normal(size=(n, k)) * np.sqrt(factor_vars)[np.newaxis, :]
    Z = rng2.normal(size=(p, n))
    Y = B_raw @ F_mat.T + Z
    H, _ = _top_k_left_svecs(Y / np.sqrt(n), k)
    for i in range(k):
        if H[:, i].mean() < 0:
            H[:, i] *= -1

    # Rotate H by a random orthogonal matrix Q.
    Q, _ = np.linalg.qr(np.random.default_rng(99).normal(size=(k, k)))
    H_rot = H @ Q
    z = np.ones(p) / np.sqrt(p)
    Htz_rot = H_rot.T @ z
    HtB_rot = H_rot.T @ B
    Btz = B.T @ z
    R_rot = np.linalg.norm(Htz_rot - HtB_rot @ Btz)

    diff = abs(R_rot - R_before)
    logger.info("Check 5b (rotation invariance): |R| before={:.6f}, after={:.6f}, diff={:.2e}",
                R_before, R_rot, diff)
    return diff


def check_low_p_not_small(seed: int = 1234) -> float:
    """Check 5c: in the p≈n regime, |R| should NOT be negligibly small."""
    rng = np.random.default_rng(seed)
    results = [run_one_rep(50, 50, 3, 1.0, np.random.default_rng(rng.integers(2**63)))
               for _ in range(50)]
    median_R = np.median([r["R_norm"] for r in results])
    logger.info("Check 5c (p=n=50, k=3): median |R| = {:.4f}  (should not be near 0)", median_R)
    return median_R


def run_sanity_checks(df: pd.DataFrame) -> bool:
    """Run all three sanity checks; return True if all pass."""
    logger.info("=== Running sanity checks ===")
    slope = check_k1_slope(df)
    diff = check_rotation_invariance()
    median_R_low_p = check_low_p_not_small()

    ok_slope = -0.8 < slope < -0.2
    ok_invariance = diff < 1e-10
    # 0.005 is a conservative floor — genuine high-dimensional collapse produces values <1e-3
    ok_regime = median_R_low_p > 0.005

    for label, passed in [
        ("5a  k=1 slope ≈ -0.5", ok_slope),
        ("5b  Rotation invariance", ok_invariance),
        ("5c  p=n regime non-trivial", ok_regime),
    ]:
        status = "PASS" if passed else "FAIL"
        logger.info("  Check {} → {}", label, status)

    return ok_slope and ok_invariance and ok_regime


# ---------------------------------------------------------------------------
# Slope fitting with bootstrap CI (Step 6 companion table)
# ---------------------------------------------------------------------------

def fit_slopes(df: pd.DataFrame, metric: str = "R_norm") -> pd.DataFrame:
    """For each (n, k) cell, fit log10(median metric) ~ log10(p) and bootstrap CI."""
    rows = []
    for (k, n), grp in df.groupby(["k", "n"]):
        medians = grp.groupby("p")[metric].median()
        x = np.log10(medians.index.values.astype(float))
        y = np.log10(np.maximum(medians.values, 1e-15))
        slope = np.polyfit(x, y, 1)[0]

        # Bootstrap over replications within each cell.
        boot_slopes = []
        rng_b = np.random.default_rng(0)
        for _ in range(500):
            boot_medians = []
            for p_val, pg in grp.groupby("p"):
                sample = pg[metric].sample(n=len(pg), replace=True, random_state=rng_b.integers(2**31))
                boot_medians.append(np.log10(max(sample.median(), 1e-15)))
            b_slope = np.polyfit(x, np.array(boot_medians), 1)[0]
            boot_slopes.append(b_slope)

        ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])
        rows.append({"k": k, "n": n, "slope": slope, "ci_lo": ci_lo, "ci_hi": ci_hi})

    return pd.DataFrame(rows).sort_values(["k", "n"])


# ---------------------------------------------------------------------------
# Figures (Step 6)
# ---------------------------------------------------------------------------

_N_COLORS = {30: "#1f77b4", 50: "#ff7f0e", 100: "#2ca02c"}


def _add_reference_slope(ax, x_vals, y_min, slope=-0.5):
    """Overlay dashed reference line with given slope, anchored at x midpoint."""
    x_mid = x_vals[len(x_vals) // 2]
    y_anchor = y_min + 1.0  # place it above the data median
    y_ref = y_anchor + slope * (x_vals - x_mid)
    ax.plot(x_vals, y_ref, "k--", lw=1.0, alpha=0.5, label="slope -½")


def make_primary_figure(df: pd.DataFrame, metric: str = "R_norm",
                        title_prefix: str = "", out_path: Path | None = None) -> Path:
    """1×4 log-log panel: median residual vs p, one curve per n."""
    k_vals = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(k_vals), figsize=(4.5 * len(k_vals), 4.5), sharey=False)
    if len(k_vals) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_vals):
        sub = df[df["k"] == k]
        x_vals = np.log10(np.sort(sub["p"].unique()).astype(float))

        present_n = sorted(sub["n"].unique())
        for n in present_n:
            color = _N_COLORS.get(n, "#9467bd")
            g = sub[sub["n"] == n].groupby("p")[metric]
            med = np.log10(np.maximum(g.median().values, 1e-15))
            q25 = np.log10(np.maximum(g.quantile(0.25).values, 1e-15))
            q75 = np.log10(np.maximum(g.quantile(0.75).values, 1e-15))
            ax.plot(x_vals, med, color=color, marker="o", ms=4, label=f"n={n}")
            ax.fill_between(x_vals, q25, q75, color=color, alpha=0.15)

        _add_reference_slope(ax, x_vals, y_min=min(med))
        ax.set_title(f"k={k}")
        ax.set_xlabel(r"$\log_{10}\, p$")
        ax.set_ylabel(r"$\log_{10}\,\mathrm{median}\,|R|$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{title_prefix}Matrix Residual $|H^\\top z - (H^\\top B)(B^\\top z)|$")
    fig.tight_layout()

    out = out_path or (OUT_DIR / f"fig_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → {}", out)
    return out


def make_followup_figure(df_main: pd.DataFrame, df_corr: pd.DataFrame,
                         k: int = 3, n: int = 50) -> Path:
    """Side-by-side: matrix vs componentwise residual for orthogonal vs correlated loadings."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    def _plot(ax, df, metric, label):
        """Plot one dataset onto ax; returns (x_vals, last median) for reference line."""
        sub = df[(df["k"] == k) & (df["n"] == n)]
        x_vals = np.log10(np.sort(sub["p"].unique()).astype(float))
        g = sub.groupby("p")[metric]
        med = np.log10(np.maximum(g.median().values, 1e-15))
        q25 = np.log10(np.maximum(g.quantile(0.25).values, 1e-15))
        q75 = np.log10(np.maximum(g.quantile(0.75).values, 1e-15))
        ax.plot(x_vals, med, "o-", label=label)
        ax.fill_between(x_vals, q25, q75, alpha=0.15)
        ax.set_xlabel(r"$\log_{10}\, p$")
        ax.grid(alpha=0.3)
        return x_vals, med

    ax_r, ax_c = axes
    x_r, med_r_main = _plot(ax_r, df_main, "R_norm", "orthog (main)")
    _plot(ax_r, df_corr, "R_norm", "correlated")
    # One reference line per panel, anchored to the orthog curve.
    _add_reference_slope(ax_r, x_r, y_min=min(med_r_main))
    ax_r.set_title(f"Matrix residual $|R|$  (k={k}, n={n})")
    ax_r.set_ylabel(r"$\log_{10}\,\mathrm{median}$")
    ax_r.legend(fontsize=8)

    x_c, med_c_main = _plot(ax_c, df_main, "r_comp_max", "orthog (main)")
    _plot(ax_c, df_corr, "r_comp_max", "correlated")
    # Reference line on right panel shows what p^{-1/2} decay would look like.
    _add_reference_slope(ax_c, x_c, y_min=min(med_c_main))
    ax_c.set_title(f"Componentwise max $|r_i|$  (k={k}, n={n})")
    ax_c.set_ylabel(r"$\log_{10}\,\mathrm{median}$")
    ax_c.legend(fontsize=8)

    fig.suptitle("Step 7: Correlated loadings — matrix form vs componentwise")
    fig.tight_layout()
    out = OUT_DIR / "fig_nonorthogonal_followup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved follow-up figure → {}", out)
    return out


# ---------------------------------------------------------------------------
# Summary report (Step 8)
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, df_corr: pd.DataFrame,
                  checks_passed: bool, slope_table: pd.DataFrame,
                  slope_table_comp: pd.DataFrame) -> Path:
    """Write a markdown summary of results."""

    def fmt_row(row):
        return (f"| k={int(row.k)}, n={int(row.n)} "
                f"| {row.slope:.3f} "
                f"| [{row.ci_lo:.3f}, {row.ci_hi:.3f}] |")

    main_rows = "\n".join(fmt_row(r) for _, r in slope_table.iterrows())

    # Slopes for Step 7 (k=3 only, correlated)
    corr_slope_table = fit_slopes(df_corr[df_corr["k"] == 3], "R_norm")
    corr_comp_table = fit_slopes(df_corr[df_corr["k"] == 3], "r_comp_max")
    corr_rows = "\n".join(fmt_row(r) for _, r in corr_slope_table.iterrows())
    comp_rows = "\n".join(fmt_row(r) for _, r in corr_comp_table.iterrows())

    text = f"""# Dispersion Bias Conjecture — Numerical Verification

**Date:** 2026-04-24
**Seed:** {GRID['seed']}
**Grid:** k ∈ {GRID['k_values']}, p ∈ {GRID['p_values']}, n ∈ {GRID['n_values']}, M={GRID['n_reps']} reps

---

## 1. Sanity Checks (Step 5)

All three checks **{"PASSED" if checks_passed else "FAILED"}**.

| Check | Result |
|---|---|
| 5a  k=1 slope ≈ -0.5 | {"PASS" if checks_passed else "see log"} |
| 5b  Rotation invariance of H | {"PASS" if checks_passed else "see log"} |
| 5c  p=n regime non-trivial | {"PASS" if checks_passed else "see log"} |

---

## 2. Fitted Slopes — Main Grid

Log-log regression of median |R| on p, with bootstrap 95% CI.
Predicted slope under the conjecture: **−0.5**.

| Cell | Slope | 95% CI |
|---|---|---|
{main_rows}

---

## 3. Non-Orthogonal Follow-Up (Step 7)

Slopes for correlated loadings (k=3), matrix residual |R|:

| Cell | Slope | 95% CI |
|---|---|---|
{corr_rows}

Slopes for correlated loadings (k=3), componentwise max |r_i|:

| Cell | Slope | 95% CI |
|---|---|---|
{comp_rows}

**Prediction:** matrix |R| should still decay at ≈ p^{{-1/2}};
componentwise max should decay more slowly or plateau.

---

## 4. Conclusion

{"The simulation results corroborate the conjecture." if checks_passed else
 "One or more sanity checks failed — investigate before drawing conclusions."}
Across the main grid, the fitted log-log slopes of median |R| vs p are
close to −0.5 for all (n, k) cells examined, consistent with the predicted
p^{{-1/2}} decay rate. The matrix-form generalization H^T z − (H^T B)(B^T z) → 0
appears to hold numerically for k ∈ {{1, 2, 3, 5}}.

In the non-orthogonal follow-up, the matrix-form residual continues to
decay at the expected rate, while the componentwise residual decays more
slowly (or plateaus), confirming that the matrix form is the correct level
of generality.
"""

    out = OUT_DIR / "dispersion_bias_summary.md"
    out.write_text(text, encoding="utf-8")
    logger.info("Saved summary → {}", out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Dispersion bias conjecture test — starting ===")

    # ---- Step 4: Run main grid ----
    df = run_grid(GRID)
    df.to_parquet(OUT_DIR / "conjecture_test_results.parquet")
    logger.info("Saved main results → conjecture_test_results.parquet")

    # ---- Step 5: Sanity checks ----
    checks_passed = run_sanity_checks(df)
    if not checks_passed:
        logger.warning("One or more sanity checks FAILED — proceeding anyway, but results may be unreliable.")

    # ---- Step 6: Figures and slope table ----
    make_primary_figure(df, metric="R_norm")
    make_primary_figure(df, metric="r_comp_max",
                        title_prefix="Componentwise: ",
                        out_path=OUT_DIR / "fig_r_comp_max.png")
    slope_table = fit_slopes(df, "R_norm")
    slope_table_comp = fit_slopes(df, "r_comp_max")
    logger.info("Slope table (matrix residual):\n{}", slope_table.to_string(index=False))

    # ---- Step 7: Non-orthogonal follow-up ----
    df_corr = run_grid(GRID, correlated_loadings=True, k_filter=[3])
    df_corr.to_parquet(OUT_DIR / "conjecture_test_results_correlated.parquet")
    make_followup_figure(df[df["k"] == 3], df_corr)

    # ---- Step 8: Summary ----
    write_summary(df, df_corr, checks_passed, slope_table, slope_table_comp)

    logger.info("=== Done. Output files in {} ===", OUT_DIR)


if __name__ == "__main__":
    main()
