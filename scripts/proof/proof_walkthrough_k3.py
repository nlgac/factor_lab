"""
proof_walkthrough_k3.py
=======================
A single, self-contained numerical walkthrough of the proof of Theorem Part (ii),
equation (5), for a concrete k=3, p=500, n=60 example.

The script mirrors the seven steps of Appendix B.3 exactly, printing the key
quantity at each step so you can see the proof moving.  No factor_lab imports
are needed — only numpy.

Parameters (diagonal-Gram case, G_inf = I_k)
---------------------------------------------
  k = 3, p = 500, n = 60
  prevalences:    c = [1.0, 0.8, 0.6]        (Assumption 1)
  factor variances: sigma2 = [0.04, 0.02, 0.01]
  noise variance: delta2 = 1.0

  Effective spikes  d_j = c_j * sigma2_j:
    d_1 = 0.040 > d_2 = 0.016 > d_3 = 0.006  (Assumption 3 satisfied)

Because G_inf = I_k the simplification Q=I, Lambda_G=I applies throughout,
so  Gamma_B = C,  M_hat = D_hat,  M = D (diagonal),  w_j = e_j.
"""

import numpy as np

rng = np.random.default_rng(20260522)

# ── Parameters ────────────────────────────────────────────────────────────────

k      = 3
p      = 500
n      = 60
c      = np.array([1.0, 0.8, 0.6])          # prevalences
sigma2 = np.array([0.04, 0.02, 0.01])        # factor return variances
delta2 = 1.0                                 # noise variance

C_half = np.diag(np.sqrt(c))                 # C^{1/2}
Gamma_B = np.diag(c)                         # C^{1/2} G_inf C^{1/2} = C (since G_inf=I)

sep = "─" * 70

# ── Draw the model ────────────────────────────────────────────────────────────

print(sep)
print("MODEL DRAW")
print(sep)

# Loading matrix B: each column j has i.i.d. N(0, c_j) entries → prevalence c_j
B = rng.normal(0, 1, size=(p, k)) * np.sqrt(c)           # (p, k)
empirical_c = (B**2).mean(axis=0)
print(f"  Empirical prevalences  c_j = {empirical_c}  (target: {c})")

# Factor returns F: (n, k), column j ~ N(0, sigma2_j)
F = rng.normal(0, 1, size=(n, k)) * np.sqrt(sigma2)       # (n, k)

# Noise Z: (p, n), i.i.d. N(0, delta2)
Z = rng.normal(0, np.sqrt(delta2), size=(p, n))

# Data matrix Y = B F^T + Z,  shape (p, n)
Y = B @ F.T + Z

# ── Population objects ────────────────────────────────────────────────────────

print(sep)
print("POPULATION OBJECTS")
print(sep)

# M = D = C^{1/2} Sigma_F C^{1/2} (diagonal since G_inf=I, Sigma_F diagonal)
# In the limit n→∞, F^T F/n → Sigma_F = diag(sigma2).
# We use the asymptotic D here (true population), not the finite-n D_hat.
Sigma_F = np.diag(sigma2)
D = C_half @ Sigma_F @ C_half                 # = diag(c_j * sigma2_j)
d = np.diag(D)
print(f"  Population spikes  d_j = c_j*sigma2_j = {d}")
print(f"  (Assumption 3: {d[0]:.4f} > {d[1]:.4f} > {d[2]:.4f}  ✓)" if
      d[0] > d[1] > d[2] else "  WARNING: Assumption 3 not satisfied")

# Population loading directions b̄_j = eigenvectors of Sigma_0 = B Sigma_F B^T / p
# Since G_inf=I:  w_j = e_j,  a_j^inf = C^{-1/2} e_j = e_j/sqrt(c_j)
# b̄_j = B a_j^inf / sqrt(a_j^inf^T Gamma_B a_j^inf)
#      = B e_j / (sqrt(c_j) * sqrt(c_j)) ... normalised below
a_inf = np.diag(1.0 / np.sqrt(c))            # columns are a_j^inf = e_j/sqrt(c_j)
b_bar_raw = B @ a_inf                         # (p, k), columns = B e_j / sqrt(c_j)
# Gamma_B-normalise: ||b̄_j||=1 requires (a_j^inf)^T Gamma_B a_j^inf = 1,
# which holds since w_j=e_j is unit and the formula gives ||b̄_j||²→1 a.s.
norms = np.linalg.norm(b_bar_raw, axis=0)
b_bar = b_bar_raw / norms                     # (p, k)  unit vectors
print(f"\n  ||b̄_j|| = {np.linalg.norm(b_bar, axis=0)}  (should be 1)")

# ── Step B.3.2: Expand W^(p) = Y^T Y / (np) ─────────────────────────────────

print(sep)
print("B.3.2  SMALL GRAM MATRIX  W^(p) = Y^T Y / (np)")
print(sep)

W_p = Y.T @ Y / (n * p)                      # (n, n)

# Theoretical limit W_inf = F Gamma_B F^T / n + (delta2/n) I_n
W_inf = F @ Gamma_B @ F.T / n + (delta2 / n) * np.eye(n)

diff_op = np.linalg.norm(W_p - W_inf, ord=2)
print(f"  ||W^(p) - W_inf||_op = {diff_op:.6f}  (should be small)")

# ── Step B.3.3: Eigenstructure of W_inf  (Lemma 7) ───────────────────────────

print(sep)
print("B.3.3  EIGENSTRUCTURE OF W_inf  (Lemma 7)")
print(sep)

# D_hat = C^{1/2} (F^T F / n) C^{1/2}
D_hat = C_half @ (F.T @ F / n) @ C_half       # (k, k)
rho, W_hat = np.linalg.eigh(D_hat)
# eigh returns ascending; reverse for descending
idx = np.argsort(rho)[::-1]
rho = rho[idx]                                # rho_1 > rho_2 > rho_3
W_hat = W_hat[:, idx]                         # columns are ŵ_j

print(f"  D_hat eigenvalues  rho_j = {rho}")
print(f"  (population limits d_j  = {d})")
print(f"  SNR_j = n*rho_j/delta2  = {n * rho / delta2}")

# Top-k eigenvalues of W_inf should be tau_j = rho_j + delta2/n
tau = rho + delta2 / n
evals_Winf = np.sort(np.linalg.eigvalsh(W_inf))[::-1]
print(f"\n  Lemma 7 predicts tau_j = rho_j + delta2/n = {tau}")
print(f"  Actual top-k evals of W_inf             = {evals_Winf[:k]}")
print(f"  Match: {np.allclose(tau, evals_Winf[:k], atol=1e-10)}")

# Top-k eigenvectors v_j of W_inf: formula (10) with G_inf=I => v_j = F^# ŵ_j / sqrt(n rho_j)
F_sharp = F @ C_half                          # F C^{1/2}, shape (n, k)  [= (F^#)^T in paper notation]
V_lemma7 = F_sharp @ W_hat / np.sqrt(n * rho) # (n, k), columns are v_j

# Compare to eigenvectors of W_inf directly
_, evecs_Winf = np.linalg.eigh(W_inf)
evecs_Winf = evecs_Winf[:, np.argsort(np.linalg.eigvalsh(W_inf))[::-1]]
for j in range(k):
    cos_angle = abs(V_lemma7[:, j] @ evecs_Winf[:, j])
    print(f"  |cos∠(v_{j+1}, evec_{j+1} of W_inf)| = {cos_angle:.10f}  (should be 1)")

# ── Step B.3.4: Spectral convergence  χ_{p,j} → v_j ─────────────────────────

print(sep)
print("B.3.4  SPECTRAL CONVERGENCE: chi_{p,j} -> v_j")
print(sep)

# Compute chi_{p,j}: eigenvectors of W^(p)
evals_Wp, evecs_Wp = np.linalg.eigh(W_p)
idx_Wp = np.argsort(evals_Wp)[::-1]
evals_Wp = evals_Wp[idx_Wp]
evecs_Wp = evecs_Wp[:, idx_Wp]                # chi_{p,j} in columns

print(f"  s²_{'{p,j}'}/p (evals of W^(p), top k) = {evals_Wp[:k]}")
print(f"  tau_j (evals of W_inf)                 = {tau}")

for j in range(k):
    cos_angle = abs(evecs_Wp[:, j] @ V_lemma7[:, j])
    print(f"  |cos∠(chi_{{p,{j+1}}}, v_{j+1})| = {cos_angle:.8f}  (should be near 1 at p=500)")

# ── Step B.3.5: Gamma_B-coordinate framework ─────────────────────────────────

print(sep)
print("B.3.5  GAMMA_B COORDINATES")
print(sep)

# g_j^inf = sqrt(n*rho_j / (n*rho_j + delta2)) * C^{-1/2} Q Lambda_G^{-1/2} ŵ_j
# With G_inf=I: Q=I, Lambda_G=I, so g_j^inf = sqrt(n*rho_j/(n*rho_j+delta2)) * C^{-1/2} ŵ_j
C_inv_half = np.diag(1.0 / np.sqrt(c))
scale = np.sqrt(n * rho / (n * rho + delta2))  # shape (k,)

g_inf = C_inv_half @ W_hat * scale             # (k, k), column j is g_j^inf

# a_j^inf = C^{-1/2} e_j (already computed above as columns of a_inf)
print("  g_j^inf  (columns, in R^k):")
for j in range(k):
    print(f"    g_{j+1}^inf = {g_inf[:, j]}")

print("\n  a_j^inf = e_j/sqrt(c_j)  (columns of a_inf):")
for j in range(k):
    print(f"    a_{j+1}^inf = {a_inf[:, j]}")

# ── Step B.3.6: Floor and in-subspace angle ───────────────────────────────────

print(sep)
print("B.3.6  FLOOR AND IN-SUBSPACE ANGLE  (Gamma_B inner products)")
print(sep)

# ||Pi_B h_j||² → (g_j^inf)^T Gamma_B g_j^inf = n*rho_j/(n*rho_j+delta2)
insubspace_norm2 = np.array([g_inf[:, j] @ Gamma_B @ g_inf[:, j] for j in range(k)])
floor_predicted   = delta2 / (n * rho + delta2)
print(f"  (g_j^inf)^T Gamma_B g_j^inf  = {insubspace_norm2}  (= n*rho/(n*rho+delta2))")
print(f"  n*rho_j/(n*rho_j+delta2)     = {n*rho/(n*rho+delta2)}")
print(f"  floor = 1 - above            = {floor_predicted}")

# <Pi_B h_j, b̄_j> → (g_j^inf)^T Gamma_B a_j^inf = sqrt(n*rho_j/(n*rho_j+delta2)) * ŵ_j^T w_j
# With G_inf=I: w_j=e_j, so ŵ_j^T w_j = (ŵ_j)_j = diagonal of W_hat
w_hat_dot_w = np.diag(W_hat)                  # ŵ_j^T e_j for j=1,2,3
inner_predicted = scale * w_hat_dot_w
inner_direct    = np.array([g_inf[:, j] @ Gamma_B @ a_inf[:, j] for j in range(k)])
print(f"\n  <Pi_B h_j, b̄_j> → scale * ŵ_j^T w_j = {inner_predicted}")
print(f"  Direct Gamma_B inner product          = {inner_direct}")
print(f"  Match: {np.allclose(inner_predicted, inner_direct, atol=1e-12)}")

# sin²∠(ŵ_j, w_j) = 1 - (ŵ_j^T e_j)²
rotation = 1.0 - w_hat_dot_w**2
print(f"\n  sin²∠(ŵ_j, w_j) = 1 - (ŵ_j^T e_j)² = {rotation}")

# ── Step B.3.7: Assembly ──────────────────────────────────────────────────────

print(sep)
print("B.3.7  ASSEMBLY — THEOREM PART (ii), EQUATION (5)")
print(sep)

weight = n * rho / (n * rho + delta2)
rhs_predicted = floor_predicted + weight * rotation
print(f"  floor                    = {floor_predicted}")
print(f"  weight = n*rho/(n*rho+d²)= {weight}")
print(f"  rotation = sin²∠(ŵ,w)   = {rotation}")
print(f"  RHS = floor + weight*rot = {rhs_predicted}")

# ── Observed LHS: sin²∠(h_j, b̄_j) ───────────────────────────────────────────

print(sep)
print("OBSERVED LHS  sin²∠(h_j, b̄_j)  vs  predicted RHS")
print(sep)

# Compute h_j via the n×n Gram trick
G_nn = Y.T @ Y                                # (n, n)
evals_G, evecs_G = np.linalg.eigh(G_nn)
idx_G = np.argsort(evals_G)[::-1]
s_vals = np.sqrt(np.maximum(evals_G[idx_G[:k]], 0.0))
H = (Y @ evecs_G[:, idx_G[:k]]) / s_vals      # (p, k), columns are h_j

sin2_obs = np.array([1.0 - (H[:, j] @ b_bar[:, j])**2 for j in range(k)])

print(f"  {'Factor':>8}  {'LHS observed':>16}  {'RHS predicted':>16}  {'gap':>10}")
for j in range(k):
    gap = sin2_obs[j] - rhs_predicted[j]
    print(f"  j={j+1:>5}    {sin2_obs[j]:>16.6f}  {rhs_predicted[j]:>16.6f}  {gap:>+10.6f}")

print(f"\n  (Gap → 0 as p → ∞; at p={p} some residual is expected)")

# ── Corollary 4: Grassmannian distance ───────────────────────────────────────

print(sep)
print("COROLLARY 4  Grassmannian distance  d²_Gr = sum_j delta²/(n*rho_j+delta²)")
print(sep)

d_Gr_predicted = floor_predicted.sum()
# Observed: ||H^T Pi_B^perp H||_F² = sum_j ||h_j^perp||²
Pi_B = B @ np.linalg.solve(B.T @ B, B.T)     # (p,p) — expensive; only for verification
h_perp_norms2 = np.array([1.0 - np.linalg.norm(Pi_B @ H[:, j])**2 for j in range(k)])
d_Gr_obs = h_perp_norms2.sum()

print(f"  Predicted d²_Gr = sum floor_j   = {d_Gr_predicted:.6f}")
print(f"  Observed  d²_Gr = sum ||h_j^perp||² = {d_Gr_obs:.6f}")
print(f"  Gap                              = {d_Gr_obs - d_Gr_predicted:+.6f}")

print(sep)
print("DONE")
print(sep)
