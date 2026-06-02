"""
rotation_check.py
=================
Verifies the claim: "When n is small and k > 1, the finite-sample eigenvector
misalignment of M-hat can be substantial" -- even when G_B and Sigma_F are both
diagonal (so non-diagonal Gram structure is NOT required).

Uses factor_lab (FactorModelBuilder + FlexibleReturnsSimulator) with Gaussian
defaults to build the population model and draw returns.

Setting
-------
k-factor model  Y = B F + Z,  p -> inf, n and k fixed.

  G_B     = diag(c)          -- loading Gram limit  (diagonal)
  Sigma_F = diag(sigma2)     -- population factor covariance  (diagonal)
  M       = G_B^{1/2} Sigma_F G_B^{1/2}  -- population signal matrix (diagonal)
  M-hat   = G_B^{1/2} (F F' / n) G_B^{1/2}  -- sample signal matrix

Because M is diagonal, its eigenvectors are w_j = e_j (standard basis).
Any rotation sin^2 angle(w-hat_j, e_j) is purely finite-n sampling noise.

Two outputs
-----------
1. Mean rotation by n  (averaged over n_reps draws of F via factor_lab).
2. Single-draw full decomposition:
      sin^2 angle(h_j, b_j)  ->  floor  +  weight * rotation  [Theorem 1]
   at n=10, p=500, delta^2=1.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in dir() else os.getcwd())
import factor_sims as fs

# -- Parameters ----------------------------------------------------------------
k        = 3
p        = 500
sigma2   = np.array([0.10, 0.08, 0.06])   # factor return variances
delta2   = 1.0                              # idiosyncratic variance (common)
n_reps   = 5_000
SEED     = 42

# Prevalences c_j = lim ||beta_j||^2 / p.  With beta_j ~ N(0, c_j) iid,
# ||beta_j||^2 / p -> c_j by the LLN.  G_B = diag(c) -- DIAGONAL.
c = np.array([1.0, 0.9, 0.8])

# -- Population quantities -----------------------------------------------------
C12     = np.diag(np.sqrt(c))
Sigma_F = np.diag(sigma2)
M       = C12 @ Sigma_F @ C12    # diagonal; eigenvalues lambda_j = c_j * sigma2_j
lam_pop = np.diag(M)
# w_j = e_j  (M diagonal with distinct eigenvalues -> standard basis eigenvectors)


def make_model(p, rng):
    """Build population model via factor_lab with Gaussian loadings."""
    builder = fs.FactorModelBuilder(rng=rng)
    beta_samplers = [
        fs.create_sampler("normal", rng, loc=0.0, scale=float(np.sqrt(c[j])))
        for j in range(k)
    ]
    idio_vol_sampler = fs.create_sampler("constant", rng, value=float(np.sqrt(delta2)))
    return builder.build(
        p=p,
        k=k,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=sigma2.tolist(),
    )


def draw_F_and_Mhat(model, n, rng):
    """
    Draw n factor returns via FlexibleReturnsSimulator and return
    (F_draw [k x n], M-hat [k x k]).
    """
    simulator = fs.FlexibleReturnsSimulator(rng=rng)
    # factor_lab scales each sampler by sqrt(diag(F)) internally (F = diag(sigma^2)), so the samplers here should be standardized (scale=1.0).
    # Passing scale=sqrt(sigma2) here would double-scale the factor returns (variance being sigma2**2) and shrink the measured eigenvector rotation.
    factor_return_samplers = [
        fs.create_sampler("normal", rng, loc=0.0, scale=1.0)
        for j in range(k)
    ]
    # D = diag(delta^2) via the builder, so the idio sampler here should be standardized too.
    idio_return_sampler = fs.create_sampler("normal", rng, loc=0.0, scale=1.0)
    result = simulator.simulate(
        model=model,
        n_periods=n,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler,
    )
    F_draw = result["factor_returns"].T          # -> (k, n)
    Mhat   = C12 @ (F_draw @ F_draw.T / n) @ C12
    return F_draw, Mhat


def eigenvector_rotations(Mhat):
    """Return sin^2 angle(w-hat_j, e_j) for j=0..k-1 (descending eigenvalue order)."""
    _, vecs = np.linalg.eigh(Mhat)    # ascending order
    vecs = vecs[:, ::-1]               # -> descending
    return np.array([1.0 - (vecs[:, j] @ np.eye(k)[j])**2 for j in range(k)])


# -- Table 1: mean rotation vs n -----------------------------------------------
print("Parameters")
print(f"  k={k},  p={p},  c={c},  sigma^2={sigma2},  delta^2={delta2}")
print("  G_B = diag(c) [DIAGONAL],  Sigma_F = diag(sigma^2) [DIAGONAL]")
print("  w_j = e_j  (rotation = 0 only as n -> inf)")
print()

rng_main = np.random.default_rng(SEED)
model = make_model(p, rng_main)

print(f"{'n':>5}  {'E[sin2(w1,e1)]':>16}  {'E[sin2(w2,e2)]':>16}  {'E[sin2(w3,e3)]':>16}")
print("-" * 60)

for n in [5, 10, 20, 50, 200]:
    sins = np.zeros((n_reps, k))
    rng_n = np.random.default_rng(SEED + n)
    for r in range(n_reps):
        _, Mhat = draw_F_and_Mhat(model, n, rng_n)
        sins[r] = eigenvector_rotations(Mhat)
    means = sins.mean(axis=0)
    print(f"{n:>5}  {means[0]:>16.4f}  {means[1]:>16.4f}  {means[2]:>16.4f}")

# -- Table 2: single-draw full decomposition at n=10 --------------------------
print()
print(f"Single-draw decomposition  (n=10, p={p}, delta^2={delta2}, seed={SEED})")
print("  sin^2 angle(h_j, b_j)  ->  floor  +  weight * rotation  [Theorem 1]")
print()

rng_demo = np.random.default_rng(SEED)
model_demo = make_model(p, rng_demo)
_, Mhat_demo = draw_F_and_Mhat(model_demo, n=10, rng=rng_demo)
lam_hat, what_vecs = np.linalg.eigh(Mhat_demo)
lam_hat   = lam_hat[::-1]
what_vecs = what_vecs[:, ::-1]

hdr = (f"  {'j':>2}  {'lam_j':>8}  {'lam_hat':>8}  {'SNR_hat':>8}  "
       f"{'floor':>7}  {'sin2rot':>8}  {'wt*rot':>7}  {'RHS':>7}")
print(hdr)
print("  " + "-" * 65)
n_demo = 10
for j in range(k):
    lh      = lam_hat[j]
    floor_j = delta2 / (n_demo * lh + delta2)
    sin2rot = 1.0 - (what_vecs[:, j] @ np.eye(k)[j])**2
    wt      = n_demo * lh / (n_demo * lh + delta2)
    snr     = n_demo * lh / delta2
    rhs     = floor_j + wt * sin2rot
    print(f"  {j+1:>2}  {lam_pop[j]:>8.4f}  {lh:>8.4f}  {snr:>8.3f}  "
          f"{floor_j:>7.4f}  {sin2rot:>8.4f}  {wt*sin2rot:>7.4f}  {rhs:>7.4f}")

print()
print("Conclusion: substantial rotation at small n even with fully diagonal")
print("G_B and Sigma_F.  Non-diagonal Gram structure is NOT required.")
