#!/usr/bin/env python3
"""
bias_correction_demo.py
=======================
Illustrates the dispersion bias and the James-Stein correction for k = 2 factors.

Model
-----
    Y = B F' + Z,   Y in R^{p x n},  B in R^{p x 2},  F in R^{n x 2},  Z in R^{p x n}

Loadings (two-block, mutually orthogonal per Assumption 2.5'):
    beta_1 = (3, ..., 3,  1, ..., 1)     first half = 3, second half = 1
    beta_2 = (-1, ..., -1, 3, ..., 3)   first half = -1, second half = 3

    Verify orthogonality:  p/2 * 3*(-1) + p/2 * 1*3 = 0.  OK.
    mu_1 = 2,  mu_2 = 1  (both positive, satisfying Assumption 2.2')
    alpha_1 = alpha_2 = sqrt(5)   (loading scale = sqrt(mean^2 + variance))
    c_1 = mu_1/alpha_1 = 2/sqrt(5),   c_2 = mu_2/alpha_2 = 1/sqrt(5)
    |Pi_B z|^2 -> c_1^2 + c_2^2 = 4/5 + 1/5 = 1.0   (z lies in B)

Factor returns F: iid N(0, sigma_j^2) across time and factors.
Noise:           Z_it ~ iid N(0, delta^2).

Chosen parameters (n=60, delta=1, sigma_1=0.10, sigma_2=0.05):
    alpha_j^2 * n * sigma_j^2:  factor 1 -> 5*60*0.01 = 3.0,  factor 2 -> 5*60*0.0025 = 0.75
    psi_inf,1 = sqrt(3 / 4)    ~ 0.866   (moderately strong factor)
    psi_inf,2 = sqrt(0.75/1.75) ~ 0.655  (weaker factor)
    Theoretical bias = (1-0.75)*0.8 + (1-3/7)*0.2 = 0.200 + 0.114 = 0.314

James-Stein estimator
---------------------
    psi_hat_i  = sqrt(max(0, 1 - delta2_hat * p / S_i^2))
    where S_i     = i-th singular value of Y   (converges to sqrt(p*(alpha_i^2|X_i|^2+delta^2)))
          delta2_hat = ||( I - H H') Y ||_F^2 / ((p-k)*n)

    JS corrected squared projection:
    |Pi^JS z|^2 = sum_i  (h_i' z)^2 / psi_hat_i^2   ->  |Pi_B z|^2 = 1

Simulation
----------
For p in [50, 100, 200, 500, 1000, 2000, 5000], M=400 Monte Carlo draws each.
Report: mean +/- std of sample and JS-corrected projections, plus estimated psi_i.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Parameters ────────────────────────────────────────────────────────────────
RNG_SEED = 42
N_OBS    = 60            # n  (time periods per window)
K        = 2             # number of factors
DELTA    = 1.0           # idiosyncratic noise std
SIGMA    = [0.10, 0.05]  # factor-return standard deviations
P_VALUES = [50, 100, 200, 500, 1000, 2000, 5000]
N_MC     = 400           # Monte Carlo repetitions per p-slice

# ── Analytical constants ──────────────────────────────────────────────────────
# Loading scale: alpha_j = sqrt((a^2 + b^2)/2) for block values (a, b)
# Factor 1: a=3, b=1  ->  alpha_1 = sqrt(5)
# Factor 2: a=-1, b=3 ->  alpha_2 = sqrt(5)
ALPHA    = [np.sqrt(5), np.sqrt(5)]
MU_INF   = [2.0, 1.0]                                  # cross-sectional means
C        = [m / a for m, a in zip(MU_INF, ALPHA)]      # eq-wt loading coefficients

def psi_asymptotic(sigma_j, alpha_j, n, delta):
    """Theoretical limit psi_inf,j = alpha_j*|X_j| / sqrt(alpha_j^2*|X_j|^2 + delta^2),
    with |X_j|^2 replaced by its expectation n*sigma_j^2."""
    signal = alpha_j**2 * n * sigma_j**2
    return np.sqrt(signal / (signal + delta**2))

PSI_THEORY  = [psi_asymptotic(SIGMA[j], ALPHA[j], N_OBS, DELTA) for j in range(K)]
POP_PROJ    = sum(c**2 for c in C)          # |Pi_B z|^2 -> 1.0
BIAS_THEORY = sum((1 - PSI_THEORY[j]**2) * C[j]**2 for j in range(K))

# ── Loading matrix ────────────────────────────────────────────────────────────
def make_B(p):
    """Build the (p, 2) loading matrix with two orthogonal block vectors."""
    half = p // 2
    b1 = np.concatenate([np.full(half, 3.0),  np.full(p - half, 1.0)])
    b2 = np.concatenate([np.full(half, -1.0), np.full(p - half, 3.0)])
    return np.column_stack([b1, b2])  # (p, 2)

# ── One Monte Carlo draw ──────────────────────────────────────────────────────
def one_draw(p, rng):
    """
    Simulate one (Y, B, F, Z) draw and return three scalars:
      sample_proj  = |Pi_H z|^2         (naive, biased)
      js_proj      = |Pi^JS z|^2        (corrected)
      psi_hat      = array of k estimated shrinkage factors
    """
    B  = make_B(p)                                      # (p, k)
    z  = np.ones(p) / np.sqrt(p)                        # equal-weight portfolio

    # Draw factor returns and noise
    F  = rng.standard_normal((N_OBS, K)) * SIGMA        # (n, k)
    Z  = rng.standard_normal((p, N_OBS)) * DELTA        # (p, n)
    Y  = B @ F.T + Z                                    # (p, n)

    # Top-k SVD of Y
    U, sv, _ = np.linalg.svd(Y, full_matrices=False)
    H  = U[:, :K]                                       # (p, k) left singular vectors
    S  = sv[:K]                                         # top-k singular values of Y

    # Sample squared projection
    coords      = H.T @ z                               # (k,)  h_i' z
    sample_proj = float(np.sum(coords**2))

    # Noise-variance estimate from factor-model residuals
    Y_fit      = H @ (H.T @ Y)
    delta2_hat = np.sum((Y - Y_fit)**2) / ((p - K) * N_OBS)

    # Shrinkage-factor estimates: psi_i = sqrt(max(0, 1 - delta2_hat * p / S_i^2))
    # Derivation: S_i^2/p -> alpha_i^2|X_i|^2 + delta^2 (from Lemma A.2' Part 2)
    psi_hat = np.sqrt(np.maximum(0.0, 1.0 - delta2_hat * p / S**2))

    # JS-corrected squared projection
    js_proj = float(np.sum(coords**2 / np.maximum(psi_hat**2, 1e-8)))

    return sample_proj, js_proj, psi_hat

# ── Simulation loop ───────────────────────────────────────────────────────────
def run_simulation():
    rng  = np.random.default_rng(RNG_SEED)
    rows = []
    for p in P_VALUES:
        sp_list, jp_list, ph_list = [], [], []
        for _ in range(N_MC):
            sp, jp, ph = one_draw(p, rng)
            sp_list.append(sp)
            jp_list.append(jp)
            ph_list.append(ph)

        sp = np.array(sp_list)
        jp = np.array(jp_list)
        ph = np.array(ph_list)   # (N_MC, k)

        rows.append({
            'p':           p,
            'sample_mean': np.mean(sp),
            'sample_std':  np.std(sp),
            'js_mean':     np.mean(jp),
            'js_std':      np.std(jp),
            'psi1_hat':    np.mean(ph[:, 0]),
            'psi2_hat':    np.mean(ph[:, 1]),
            'bias_sample': POP_PROJ - np.mean(sp),
            'bias_js':     POP_PROJ - np.mean(jp),
        })
    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 72)
    print("  k=2 Dispersion Bias and James-Stein Correction: Illustration")
    print("=" * 72)
    print()
    print("  Model:  Y = B F' + Z,  p x n,  k=2 factors")
    print(f"  n (observations):      {N_OBS}")
    print(f"  delta (idio noise std): {DELTA}")
    print(f"  sigma_1, sigma_2:       {SIGMA[0]}, {SIGMA[1]}")
    print()
    print("  Loading vectors (two-block, orthogonal):")
    print("    beta_1 = [3,...,3, 1,...,1]   mu_1=2  alpha_1=sqrt(5)  c_1=2/sqrt(5)")
    print("    beta_2 = [-1,...,-1,3,...,3]  mu_2=1  alpha_2=sqrt(5)  c_2=1/sqrt(5)")
    print()
    print("  Theoretical limits (p -> inf, n=60 fixed):")
    print(f"    |Pi_B z|^2              = {POP_PROJ:.4f}   (true factor exposure of z)")
    print(f"    psi_inf,1               = {PSI_THEORY[0]:.4f}   (shrinkage, factor 1)")
    print(f"    psi_inf,2               = {PSI_THEORY[1]:.4f}   (shrinkage, factor 2)")
    print(f"    Asymptotic bias         = {BIAS_THEORY:.4f}   (= |Pi_B z|^2 - lim E|Pi_H z|^2)")
    print(f"    lim E|Pi_H z|^2         = {POP_PROJ - BIAS_THEORY:.4f}")
    print(f"    lim E|Pi^JS z|^2        = {POP_PROJ:.4f}   (corrected -> true value)")
    print()

    print(f"  Monte Carlo: M={N_MC} draws per p-slice")
    print()

    df = run_simulation()

    # Print table
    col_w = 72
    hdr = (f"{'p':>5}  {'|Pi_H z|^2':^16}  {'|Pi^JS z|^2':^16}  "
           f"{'psi1_hat':>8}  {'psi2_hat':>8}  {'bias(raw)':>9}  {'bias(JS)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(
            f"{int(r.p):>5}  "
            f"{r.sample_mean:.4f} +/- {r.sample_std:.4f}  "
            f"{r.js_mean:.4f} +/- {r.js_std:.4f}  "
            f"{r.psi1_hat:.4f}    "
            f"{r.psi2_hat:.4f}    "
            f"{r.bias_sample:>9.4f}  "
            f"{r.bias_js:>9.5f}"
        )

    print()
    print(f"  Theory:  bias -> {BIAS_THEORY:.4f},  |Pi_H|^2 -> {POP_PROJ - BIAS_THEORY:.4f},  "
          f"|Pi^JS|^2 -> {POP_PROJ:.4f}")
    print(f"           psi_1 -> {PSI_THEORY[0]:.4f},  psi_2 -> {PSI_THEORY[1]:.4f}")
    print()

    out = Path('/sessions/ecstatic-serene-bell/mnt/factor_lab/bias_correction_demo_results.csv')
    df.to_csv(out, index=False)
    print(f"  Results saved to {out}")

    return df

if __name__ == '__main__':
    main()
