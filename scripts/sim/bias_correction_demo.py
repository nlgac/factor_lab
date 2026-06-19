#!/usr/bin/env python3
"""
bias_correction_demo.py
=======================
Companion to `dispersion_bias_correction_v2.md`.  Illustrates the dispersion
bias and the James-Stein correction implied by Theorem 3.1' (cleaned-proof
form).  The simple diagonal correction implemented here applies under
Assumptions 2.5' (orthogonal loadings) and 2.6' (orthogonal factor returns);
extensions are discussed in v2 sec 5 and in `KT_extension_to_nonorthogonal_factors.md`.

Model
-----
    Y = B F' + Z,   Y in R^{p x n},  B in R^{p x 2},  F in R^{n x 2},  Z in R^{p x n}

Loadings (two-block, mutually orthogonal per Assumption 2.5'):
    beta_1 = (3, ..., 3,  1, ..., 1)   first half = 3,  second half = 1
    beta_2 = (-1,...,-1,  3, ..., 3)   first half = -1, second half = 3

    Verify orthogonality:  p/2 * 3*(-1) + p/2 * 1*3 = 0.  OK.
    mu_1 = 2,  mu_2 = 1                      (Assumption 2.2')
    alpha_1 = alpha_2 = sqrt(5)              (loading scale = sqrt(mean^2 + variance))
    c_1 = mu_1/alpha_1 = 2/sqrt(5),  c_2 = mu_2/alpha_2 = 1/sqrt(5)
    |Pi_B z|^2 -> c_1^2 + c_2^2 = 4/5 + 1/5 = 1.0   (z lies in B asymptotically)

Factor returns F: iid N(0, sigma_j^2) across time and factors.
Noise:           Z_it ~ iid N(0, delta^2).

Chosen parameters (n=60, delta=1, sigma_1=0.10, sigma_2=0.05):
    alpha_j^2 * n * sigma_j^2:  factor 1 -> 5*60*0.01 = 3.00,  factor 2 -> 5*60*0.0025 = 0.75
    psi_inf,1 = sqrt(3 / 4)       ~ 0.866   (moderately strong factor)
    psi_inf,2 = sqrt(0.75 / 1.75) ~ 0.655   (weaker factor)
    Theoretical bias = (1-0.75)*0.8 + (1-3/7)*0.2 = 0.200 + 0.114 = 0.314

James-Stein estimator (v2 sec 3-4)
----------------------------------
Notation: sigma_i = sigma_i(Y) = i-th singular value of Y (NOT of Y/sqrt(n)).
By Lemma A.2' Part 2,  sigma_i^2 / p  ->  alpha_i^2 |X_i|^2 + delta^2  (under 2.6').

    psi_hat_i^2 = max( tau,  1 - p * delta2_hat / sigma_i^2 )
    delta2_hat  = ||(I - H H') Y ||_F^2  /  ((p-k) n)

    JS-corrected vector estimator (per-coordinate inversion form):
        Pi^JS z  =  sum_i  (h_i' z / psi_hat_i)  h_i
    JS-corrected squared projection:
        |Pi^JS z|^2  =  sum_i  (h_i' z)^2 / psi_hat_i^2   ->  |Pi_B z|^2 = 1   a.s.

The floor `tau` > 0 prevents 1/psi^2 from blowing up when sigma_i^2 barely
clears p*delta2_hat (weak-factor regime).  tau in [1e-2, 1e-1] is typical.

Simulation
----------
For p in [50, 100, 200, 500, 1000, 2000, 5000], M=400 Monte Carlo draws each.
Reports (i) mean +/- std of sample and JS-corrected projections, (ii) full MSE
decomposition (bias-squared, variance, MSE, ratio).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Parameters
RNG_SEED = 42
N_OBS    = 60                            # n  (time periods per window)
K        = 2                             # number of factors
DELTA    = 1.0                           # idiosyncratic noise std
SIGMA    = [0.10, 0.05]                  # factor-return standard deviations
P_VALUES = [50, 100, 200, 500, 1000, 2000, 5000]
N_MC     = 400                           # Monte Carlo repetitions per p-slice
TAU      = 1e-2                          # floor on psi_hat^2 (caps inflation 1/psi^2 at 1/TAU)

# Analytical constants
# Loading scale: alpha_j = sqrt((a^2 + b^2)/2) for block values (a, b)
# Factor 1: a=3, b=1   ->  alpha_1 = sqrt(5)
# Factor 2: a=-1, b=3  ->  alpha_2 = sqrt(5)
ALPHA  = [np.sqrt(5), np.sqrt(5)]
MU_INF = [2.0, 1.0]                                 # cross-sectional means
C      = [m / a for m, a in zip(MU_INF, ALPHA)]     # eq-wt loading coefficients

def psi_asymptotic(sigma_j, alpha_j, n, delta):
    """Theoretical limit psi_inf,j = alpha_j |X_j| / sqrt(alpha_j^2 |X_j|^2 + delta^2),
    with |X_j|^2 replaced by its expectation n * sigma_j^2."""
    signal = alpha_j**2 * n * sigma_j**2
    return np.sqrt(signal / (signal + delta**2))

PSI_THEORY  = [psi_asymptotic(SIGMA[j], ALPHA[j], N_OBS, DELTA) for j in range(K)]
POP_PROJ    = sum(c**2 for c in C)                  # |Pi_B z|^2 -> 1.0
BIAS_THEORY = sum((1 - PSI_THEORY[j]**2) * C[j]**2 for j in range(K))

# Loading matrix
def make_B(p):
    """Build the (p, 2) loading matrix with two orthogonal block vectors."""
    half = p // 2
    b1 = np.concatenate([np.full(half, 3.0),  np.full(p - half, 1.0)])
    b2 = np.concatenate([np.full(half, -1.0), np.full(p - half, 3.0)])
    return np.column_stack([b1, b2])                # (p, 2)

# Single Monte Carlo draw
def one_draw(p, rng, tau=TAU):
    """
    Simulate one (Y, B, F, Z) draw and return:
      sample_proj  = |Pi_H z|^2          (naive, biased)
      js_proj      = |Pi^JS z|^2         (corrected, squared norm of per-coord-inversion form)
      psi_hat      = (k,) array of estimated shrinkage factors
    """
    B = make_B(p)                                   # (p, k)
    z = np.ones(p) / np.sqrt(p)                     # equal-weight portfolio

    # Draw factor returns and idiosyncratic noise
    F = rng.standard_normal((N_OBS, K)) * SIGMA     # (n, k)
    Z = rng.standard_normal((p, N_OBS)) * DELTA     # (p, n)
    Y = B @ F.T + Z                                 # (p, n)

    # Top-k SVD of Y
    U, sv, _ = np.linalg.svd(Y, full_matrices=False)
    H     = U[:, :K]                                # (p, k) left singular vectors
    sigma = sv[:K]                                  # top-k singular values of Y

    # Sample squared projection
    coords      = H.T @ z                           # (k,)  h_i' z
    sample_proj = float(np.sum(coords**2))

    # Noise-variance estimate from factor-model residuals
    Y_fit      = H @ (H.T @ Y)
    delta2_hat = np.sum((Y - Y_fit)**2) / ((p - K) * N_OBS)

    # Shrinkage-factor estimates with floor tau (v2 sec 4.1):
    #   psi_hat_i^2 = max(tau,  1 - p * delta2_hat / sigma_i^2)
    # Lemma A.2' Part 2: sigma_i^2 / p -> alpha_i^2 |X_i|^2 + delta^2 under 2.6'.
    psi2_hat = np.maximum(tau, 1.0 - delta2_hat * p / sigma**2)
    psi_hat  = np.sqrt(psi2_hat)

    # JS-corrected squared projection (per-coordinate-inversion form, then squared norm)
    js_proj = float(np.sum(coords**2 / psi2_hat))

    return sample_proj, js_proj, psi_hat

# Simulation loop with MSE decomposition
def run_simulation(tau=TAU):
    rng  = np.random.default_rng(RNG_SEED)
    rows = []
    for p in P_VALUES:
        sp_list, jp_list, ph_list = [], [], []
        for _ in range(N_MC):
            sp, jp, ph = one_draw(p, rng, tau)
            sp_list.append(sp)
            jp_list.append(jp)
            ph_list.append(ph)

        sp = np.array(sp_list)
        jp = np.array(jp_list)
        ph = np.array(ph_list)                      # (N_MC, k)

        # Bias and MSE decomposition (v2 sec 7)
        bias_sample  = POP_PROJ - sp.mean()
        bias_js      = POP_PROJ - jp.mean()
        bias2_sample = bias_sample ** 2
        bias2_js     = bias_js ** 2
        var_sample   = sp.var()
        var_js       = jp.var()
        mse_sample   = bias2_sample + var_sample
        mse_js       = bias2_js + var_js
        mse_ratio    = mse_sample / mse_js if mse_js > 0 else float('inf')

        rows.append({
            'p':             p,
            'sample_mean':   sp.mean(),
            'sample_std':    sp.std(),
            'js_mean':       jp.mean(),
            'js_std':        jp.std(),
            'psi1_hat':      ph[:, 0].mean(),
            'psi2_hat':      ph[:, 1].mean(),
            'bias_sample':   bias_sample,
            'bias_js':       bias_js,
            'bias2_sample':  bias2_sample,
            'var_sample':    var_sample,
            'mse_sample':    mse_sample,
            'bias2_js':      bias2_js,
            'var_js':        var_js,
            'mse_js':        mse_js,
            'mse_ratio':     mse_ratio,
        })
    return pd.DataFrame(rows)

# k = 1 worked example (v2 sec 2)
def k1_example_demo(rng=None, p=2000):
    """
    Single-factor example from v2 sec 2: uniform loadings beta = e (so alpha = 1
    and b = z = e/sqrt(p), giving c = 1).  Population |Pi_B z|^2 = 1.
    Asymptotically |Pi_H z|^2 -> psi_inf^2 = |X|^2 / (|X|^2 + delta^2).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED + 1)

    n     = N_OBS
    delta = DELTA
    sigma = 0.10                                    # factor return std
    z     = np.ones(p) / np.sqrt(p)
    beta  = np.ones(p)                              # uniform loadings, alpha = 1
    X     = rng.standard_normal(n) * sigma          # one realised factor-return path
    Z     = rng.standard_normal((p, n)) * delta
    Y     = np.outer(beta, X) + Z

    # Top-1 SVD
    U, sv, _ = np.linalg.svd(Y, full_matrices=False)
    h     = U[:, 0]
    sig1  = sv[0]

    # Sample / JS projections
    hz          = float(h @ z)
    sample_proj = hz ** 2
    Y_fit       = np.outer(h, h @ Y)
    delta2_hat  = np.sum((Y - Y_fit)**2) / ((p - 1) * n)
    psi2_hat    = max(TAU, 1.0 - p * delta2_hat / sig1**2)
    psi_hat     = np.sqrt(psi2_hat)
    js_proj     = sample_proj / psi2_hat

    psi_theory      = np.sqrt((np.sum(X**2)) / (np.sum(X**2) + delta**2))
    psi_theory_E    = np.sqrt((n * sigma**2) / (n * sigma**2 + delta**2))

    print()
    print("  k=1 worked example (v2 sec 2)")
    print("  -----------------------------")
    print(f"    p = {p}, n = {n}, sigma = {sigma}, delta = {delta}, alpha = 1")
    print(f"    Population:        |Pi_B z|^2 = 1.0000   (z is in B)")
    print(f"    Theory (E[|X|^2]): psi_inf    = {psi_theory_E:.4f}    -> lim |Pi_H z|^2 = {psi_theory_E**2:.4f}")
    print(f"    Theory (this |X|): psi_inf    = {psi_theory:.4f}    -> lim |Pi_H z|^2 = {psi_theory**2:.4f}")
    print(f"    Sample:            |Pi_H z|^2 = {sample_proj:.4f}    psi_hat        = {psi_hat:.4f}")
    print(f"    JS-corrected:      |Pi^JS z|^2 = {js_proj:.4f}")
    print()

# Pretty-printed tables
def print_projection_table(df):
    """Section 6.3 of v2: convergence of sample / JS projections vs p."""
    hdr = (f"{'p':>5}  {'|Pi_H z|^2':^16}  {'|Pi^JS z|^2':^16}  "
           f"{'psi1_hat':>8}  {'psi2_hat':>8}  {'bias(raw)':>9}  {'bias(JS)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(
            f"{int(r.p):>5}  "
            f"{r.sample_mean:.4f} +/- {r.sample_std:.4f}  "
            f"{r.js_mean:.4f} +/- {r.js_std:.4f}  "
            f"{r.psi1_hat:>8.4f}  "
            f"{r.psi2_hat:>8.4f}  "
            f"{r.bias_sample:>9.4f}  "
            f"{r.bias_js:>9.5f}"
        )
    print(f"{'inf':>5}  {POP_PROJ - BIAS_THEORY:>16.4f}  {POP_PROJ:>16.4f}  "
          f"{PSI_THEORY[0]:>8.4f}  {PSI_THEORY[1]:>8.4f}  "
          f"{BIAS_THEORY:>9.4f}  {0.0:>9.5f}    (theory)")

def print_mse_table(df):
    """Section 7.1 of v2: MSE decomposition (bias^2 + variance)."""
    hdr = (f"{'p':>5}  {'bias2(raw)':>10}  {'var(raw)':>10}  {'MSE(raw)':>10}  "
           f"{'bias2(JS)':>10}  {'var(JS)':>10}  {'MSE(JS)':>10}  {'ratio':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(
            f"{int(r.p):>5}  "
            f"{r.bias2_sample:>10.4f}  "
            f"{r.var_sample:>10.4f}  "
            f"{r.mse_sample:>10.4f}  "
            f"{r.bias2_js:>10.4f}  "
            f"{r.var_js:>10.4f}  "
            f"{r.mse_js:>10.4f}  "
            f"{r.mse_ratio:>6.1f}x"
        )

# Main
def main():
    print()
    print("=" * 78)
    print("  k=2 Dispersion Bias and James-Stein Correction (Theorem 3.1', cleaned form)")
    print("=" * 78)
    print()
    print(f"  Model:  Y = B F' + Z,  p x n,  k = {K} factors")
    print(f"  n (observations):       {N_OBS}")
    print(f"  delta (idio noise std): {DELTA}")
    print(f"  sigma_1, sigma_2:       {SIGMA[0]}, {SIGMA[1]}")
    print(f"  Stability floor tau:    {TAU}    (caps 1/psi^2 inflation at 1/tau = {1/TAU:.0f})")
    print()
    print("  Loading vectors (two-block, orthogonal):")
    print("    beta_1 = [3,...,3, 1,...,1]   mu_1=2  alpha_1=sqrt(5)  c_1=2/sqrt(5)")
    print("    beta_2 = [-1,...,-1,3,...,3]  mu_2=1  alpha_2=sqrt(5)  c_2=1/sqrt(5)")
    print()
    print("  Asymptotic limits (p -> inf, n=60 fixed):")
    print(f"    |Pi_B z|^2              = {POP_PROJ:.4f}    (true factor exposure of z)")
    print(f"    psi_inf,1               = {PSI_THEORY[0]:.4f}    (shrinkage, factor 1)")
    print(f"    psi_inf,2               = {PSI_THEORY[1]:.4f}    (shrinkage, factor 2)")
    print(f"    Asymptotic bias         = {BIAS_THEORY:.4f}    (= |Pi_B z|^2 - lim E|Pi_H z|^2)")
    print(f"    lim E|Pi_H z|^2         = {POP_PROJ - BIAS_THEORY:.4f}")
    print(f"    lim E|Pi^JS z|^2        = {POP_PROJ:.4f}    (corrected -> true value)")
    print()

    # k = 1 sanity demonstration first
    k1_example_demo()

    print(f"  Monte Carlo: M = {N_MC} draws per p-slice")
    print()

    df = run_simulation()

    # Section 6.3 table
    print("  Convergence of sample / JS projections (v2 sec 6.3):")
    print()
    print_projection_table(df)
    print()

    # Section 7.1 MSE decomposition table
    print("  MSE decomposition: bias^2 + variance, raw vs JS (v2 sec 7.1):")
    print()
    print_mse_table(df)
    print()

    # Save full data
    out = Path(__file__).parent / 'bias_correction_demo_results.csv'
    df.to_csv(out, index=False)
    print(f"  Full results saved to {out}")
    print()

    return df

if __name__ == '__main__':
    main()
