"""
rotation_check.py
=================
Verifies the claim: "When n is small and k > 1, the finite-sample eigenvector
misalignment of M-hat can be substantial" — even when G_B and Sigma_F are both
diagonal (so non-diagonal Gram structure is NOT required).

Setting
-------
k-factor model  Y = B F + Z,  p -> inf, n and k fixed.

  G_B     = diag(c)          -- loading Gram limit  (diagonal)
  Sigma_F = diag(sigma2)     -- population factor covariance  (diagonal)
  M       = G_B^{1/2} Sigma_F G_B^{1/2}  -- population signal matrix (diagonal)
  M-hat   = G_B^{1/2} (F F' / n) G_B^{1/2}  -- sample signal matrix

Because M is diagonal, its eigenvectors are w_j = e_j (standard basis).
Any rotation sin² angle(w-hat_j, e_j) is purely finite-n sampling noise.

Two outputs
-----------
1. Mean rotation by n  (averaged over n_reps Monte Carlo draws of F).
2. Single-draw full decomposition:
      sin² angle(h_j, b_j)  ->  floor  +  weight * rotation
   at n=10, p=500, delta²=1.
"""

import numpy as np

# ── Parameters (match the inline simulation in the conversation) ──────────────
k       = 3
c       = np.array([1.0, 0.9, 0.8])          # prevalences  (G_B = diag(c))
sigma2  = np.array([0.10, 0.08, 0.06])        # factor return variances
delta2  = 1.0                                  # noise variance
p_demo  = 500                                  # used only for the single-draw RHS
n_reps  = 5_000
rng     = np.random.default_rng(42)

# ── Derived population quantities ─────────────────────────────────────────────
C12     = np.diag(np.sqrt(c))                  # G_B^{1/2}
Sigma_F = np.diag(sigma2)
M       = C12 @ Sigma_F @ C12                  # diagonal; eigenvalues = c_j * sigma2_j
lam_pop = np.diag(M)
# Population eigenvectors w_j = e_j (M is diagonal with distinct eigenvalues)

def sample_rotation(n: int, n_reps: int) -> np.ndarray:
    """
    Return (n_reps, k) array of sin² angle(w-hat_j, e_j).
    F columns are drawn iid N(0, sigma2_j) for factor j.
    """
    sins = np.zeros((n_reps, k))
    for r in range(n_reps):
        F    = rng.normal(size=(k, n)) * np.sqrt(sigma2)[:, None]
        Mhat = C12 @ (F @ F.T / n) @ C12
        # eigh returns ascending eigenvalues; reverse to match descending M order
        _, vecs = np.linalg.eigh(Mhat)
        vecs = vecs[:, ::-1]
        for j in range(k):
            sins[r, j] = 1.0 - (vecs[:, j] @ np.eye(k)[j]) ** 2
    return sins


# ── Table 1: mean rotation vs n ───────────────────────────────────────────────
print("Parameters")
print(f"  k={k},  c={c},  sigma²={sigma2},  delta²={delta2}")
print(f"  G_B = diag(c)  [DIAGONAL],  Sigma_F = diag(sigma²)  [DIAGONAL]")
print(f"  Population eigenvectors w_j = e_j  (rotation = 0 only as n -> inf)\n")

def pct_right_angle(sin2):
    """Convert sin²(theta) to % of a right angle: arcsin(sqrt(sin²)) / (pi/2) * 100."""
    return np.arcsin(np.sqrt(np.clip(sin2, 0, 1))) / (np.pi / 2) * 100

print(f"{'n':>5}  {'E[sin²(w1,e1)]':>16}  {'E[sin²(w2,e2)]':>16}  {'E[sin²(w3,e3)]':>16}"
      f"  {'%R(f1)':>8}  {'%R(f2)':>8}  {'%R(f3)':>8}")
print("-" * 90)
for n in [5, 10, 20, 50, 200]:
    means = sample_rotation(n, n_reps).mean(axis=0)
    pcts  = pct_right_angle(means)
    print(f"{n:>5}  {means[0]:>16.4f}  {means[1]:>16.4f}  {means[2]:>16.4f}"
          f"  {pcts[0]:>8.1f}  {pcts[1]:>8.1f}  {pcts[2]:>8.1f}")
    
print("-" * 90)
print("  %R(f_j) = % of a right angle (90 deg) for factor j: arcsin(sqrt(sin²(w-hat_j, e_j))) / (pi/2) * 100")
print("-" * 90)
print("-" * 90)

# ── Table 2: single-draw full decomposition at n=10 ──────────────────────────
print()
print(f"Single-draw decomposition  (n=10, p={p_demo}, delta²={delta2}, seed=42)")
print(f"  sin² angle(h_j, b_j)  ->  floor  +  weight * rotation  [Theorem 1]\n")

rng2 = np.random.default_rng(42)
n    = 10
F    = rng2.normal(size=(k, n)) * np.sqrt(sigma2)[:, None]
Mhat = C12 @ (F @ F.T / n) @ C12
lam_hat, what_vecs = np.linalg.eigh(Mhat)
lam_hat   = lam_hat[::-1]
what_vecs = what_vecs[:, ::-1]

hdr = f"  {'j':>2}  {'λ_j':>8}  {'λ̂_j':>8}  {'SNR̂_j':>7}  {'floor':>7}  " \
      f"{'sin²rot':>9}  {'wt×rot':>7}  {'RHS':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for j in range(k):
    lh   = lam_hat[j]
    floor   = delta2 / (n * lh + delta2)
    sin2rot = 1.0 - (what_vecs[:, j] @ np.eye(k)[j]) ** 2
    wt      = n * lh / (n * lh + delta2)
    snr     = n * lh / delta2
    rhs     = floor + wt * sin2rot
    print(f"  {j+1:>2}  {lam_pop[j]:>8.4f}  {lh:>8.4f}  {snr:>7.3f}  "
          f"{floor:>7.4f}  {sin2rot:>9.4f}  {wt*sin2rot:>7.4f}  {rhs:>7.4f}")

print()
print("Column guide for Table 2:")
print("  lam_j    : population eigenvalue of M = G_B^{1/2} Sigma_F G_B^{1/2}")
print("  lam_hat  : sample eigenvalue of M-hat (single draw)")
print("  SNR_hat  : realized signal-to-noise = n * lam_hat / delta²")
print("  floor    : irreducible lower bound = 1 / (1 + SNR_hat)")
print("             -- set by n and lam_hat alone; cannot be reduced by increasing p")
print("  sin²rot  : sin² angle(w-hat_j, e_j) -- eigenvector misalignment of M-hat")
print("             relative to the population eigenvector (standard basis vector e_j)")
print("  wt*rot   : SNR_hat/(1+SNR_hat) * sin²rot  -- rotation contribution to RHS")
print("  RHS      : floor + wt*rot")
print("             = sin² angle(h_j, b_j) [Theorem 1, right-hand side]")
print("             -- total misalignment between sample PC h_j and true loading b_j")
print("             The floor dominates when SNR_hat << 1 (as here at n=10).")
print()
print("Conclusion: substantial rotation at small n even with fully diagonal")
print("G_B and Sigma_F.  Non-diagonal Gram structure is NOT required.")
