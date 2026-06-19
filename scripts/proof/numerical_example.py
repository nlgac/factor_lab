"""Numerical example for proof_summary_5ideas.tex (Theorem 1, k=2, n=8).

Picks B so that G_B^(p) = B^T B / p is visibly anisotropic (non-diagonal),
so the "G_B metric vs Euclidean metric" contrast in the final figure is real.
Prints every quantity that appears in the 5ideas write-up so the worked
example and the figure can both be built from these exact numbers.
"""
import numpy as np

np.random.seed(0)
p, n, k = 2000, 8, 2
delta = 0.5  # delta^2 = 0.25

# --- Loadings B: two correlated columns so G_B^(p) is anisotropic -------
z1 = np.random.randn(p)
z2 = np.random.randn(p)
col1 = 1.5 * z1
col2 = 1.0 * (0.6 * z1 + np.sqrt(1 - 0.6**2) * z2)
B = np.stack([col1, col2], axis=1)  # p x k

G_Bp = B.T @ B / p
print("G_B^(p) =\n", G_Bp)

# symmetric square root of G_Bp (2x2, SPD)
evals, evecs = np.linalg.eigh(G_Bp)
G_half = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
G_half_inv = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T

# --- Population factor covariance and one realization of F --------------
Sigma_f = np.diag([4.0, 1.0])  # factor 1 var 4, factor 2 var 1
F = np.linalg.cholesky(Sigma_f) @ np.random.randn(k, n)
FFn = F @ F.T / n
print("\nFF^T/n =\n", FFn)

# --- Noise and data -------------------------------------------------------
Z = delta * np.random.randn(p, n)
Y = B @ F + Z


def top_k(mat, k):
    w, v = np.linalg.eigh(mat)
    idx = np.argsort(w)[::-1][:k]
    return w[idx], v[:, idx]


# S_n^(p) = Y Y^T/(np)  ->  h_j, theta_{n,j}^(p)
theta, H = top_k(Y @ Y.T / (n * p), k)
print("\ntheta_{n,j}^(p) =", theta)

# W_n^(p) = Y^T Y/(np) -> chi_{n,j}^(p), same theta
thetaW, Xp = top_k(Y.T @ Y / (n * p), k)
print("theta check (dual side) =", thetaW)

# b_j: eigenvectors of Sigma_0^(p)/p = B Sigma_f B^T / p
lambda_j_p, b = top_k(B @ Sigma_f @ B.T / p, k)
print("\nlambda_j^(p) =", lambda_j_p)

# M^(p) = G_half Sigma_f G_half -> w_j^(p)
lambda_j_check, w_p = top_k(G_half @ Sigma_f @ G_half, k)
print("lambda_j^(p) check (via M^(p)) =", lambda_j_check)

# M_n = G_half (FF^T/n) G_half -> w_{n,j}, lambda_{n,j}
lambda_nj, w_n = top_k(G_half @ FFn @ G_half, k)
print("\nlambda_{n,j} =", lambda_nj)

# kappa_{n,j}
kappa = np.sqrt(n * lambda_nj / (n * lambda_nj + delta**2))
print("kappa_{n,j} =", kappa)
print("floor 1-kappa^2 =", 1 - kappa**2)

# --- Fix sign conventions: <h_j,b_j> >= 0, <w_n,j, w_j> >= 0 -------------
for j in range(k):
    if H[:, j] @ b[:, j] < 0:
        H[:, j] *= -1
    if w_n[:, j] @ w_p[:, j] < 0:
        w_n[:, j] *= -1

# --- Projector onto B = col(B) -------------------------------------------
Q, _ = np.linalg.qr(B)
PiB = Q @ Q.T

print("\n--- per-factor angles ---")
for j in range(k):
    hj, bj = H[:, j], b[:, j]
    Pihj = PiB @ hj
    sin2_out = 1 - np.linalg.norm(Pihj) ** 2  # ||Pi_B^perp h_j||^2
    in_sub_dir = Pihj / np.linalg.norm(Pihj)
    cos_in = np.dot(in_sub_dir, bj)
    sin2_in = 1 - cos_in**2
    sin2_total = 1 - np.dot(hj, bj) ** 2

    wn, wj = w_n[:, j], w_p[:, j]
    cos_w = np.dot(wn, wj)
    sin2_w = 1 - cos_w**2

    # Euclidean angle vs G_B angle between w_{n,j} and w_j in R^k
    gnorm_n = np.sqrt(wn @ G_Bp @ wn)
    gnorm_j = np.sqrt(wj @ G_Bp @ wj)
    cos_w_GB = (wn @ G_Bp @ wj) / (gnorm_n * gnorm_j)

    print(f"\nj={j+1}")
    print(f"  ||Pi_B^perp h_j||^2        = {sin2_out:.4f}")
    print(f"  predicted floor            = {1-kappa[j]**2:.4f}")
    print(f"  ||Pi_B h_j||^2             = {np.linalg.norm(Pihj)**2:.4f}  (kappa^2={kappa[j]**2:.4f})")
    print(f"  sin^2 angle(h_j,b_j)       = {sin2_total:.4f}")
    print(f"  sin^2 angle(in-sub,b_j)    = {sin2_in:.4f}")
    print(f"  sin^2 angle(w_nj,w_j) [Euclidean] = {sin2_w:.4f}")
    print(f"  cos angle(w_nj,w_j) [G_B metric]  = {cos_w_GB:.4f}  -> sin^2 = {1-cos_w_GB**2:.4f}")
    rhs = (1 - kappa[j] ** 2) + kappa[j] ** 2 * sin2_w
    print(f"  RHS = floor + kappa^2*sin^2_Eucl(w) = {rhs:.4f}  vs LHS sin^2(h_j,b_j)={sin2_total:.4f}")

print("\nw_n (cols) =\n", w_n)
print("w_p (cols) =\n", w_p)
print("R^T R_n =\n", w_p.T @ w_n)

print("\n\n=== Phi^(p)-coordinates: a_j^(p) and g_{n,j}^(p) ===")
for j in range(k):
    wj, wnj = w_p[:, j], w_n[:, j]
    aj = G_half_inv @ wj
    gnj = G_half_inv @ wnj
    # normalize aj to unit Euclidean norm of a_j? Actually a_j s.t. Phi(a_j)=b_j, ||b_j||=1
    # ||Phi(a_j)||^2 = a_j^T G_Bp a_j = w_j^T w_j = 1  (since w_j Euclidean unit)
    # so a_j already has G_Bp-norm 1. Similarly g_{n,j} has G_Bp-norm 1 (up to o(1)).
    GBnorm_a = np.sqrt(aj @ G_Bp @ aj)
    GBnorm_g = np.sqrt(gnj @ G_Bp @ gnj)
    eucl_norm_a = np.linalg.norm(aj)
    eucl_norm_g = np.linalg.norm(gnj)

    # G_B-metric angle between a_j and g_{n,j}
    cos_GB = (aj @ G_Bp @ gnj) / (GBnorm_a * GBnorm_g)
    sin2_GB = 1 - cos_GB**2

    # Euclidean angle between a_j and g_{n,j} (raw R^k coordinates)
    cos_E = (aj @ gnj) / (eucl_norm_a * eucl_norm_g)
    sin2_E = 1 - cos_E**2

    print(f"\nj={j+1}")
    print(f"  a_j^(p)            = {aj},  G_B-norm={GBnorm_a:.4f}, Eucl-norm={eucl_norm_a:.4f}")
    print(f"  g_{{n,{j+1}}}^(p)        = {gnj},  G_B-norm={GBnorm_g:.4f}, Eucl-norm={eucl_norm_g:.4f}")
    print(f"  G_B-metric angle:   sin^2 = {sin2_GB:.4f}")
    print(f"  Euclidean angle:    sin^2 = {sin2_E:.4f}")

print("\n\n=== Panel-3 (R^p, ambient) 2D coordinates for j=1 ===")
j = 0
hj, bj = H[:, j], b[:, j]
Pihj = PiB @ hj
in_dir = Pihj / np.linalg.norm(Pihj)
# 2D coords in orthonormal basis Q of B
b1_2d = Q.T @ bj
h1_2d = Q.T @ in_dir
print("b_1 (2D in B-basis)        =", b1_2d, " norm=", np.linalg.norm(b1_2d))
print("Pi_B h_1/||.|| (2D)        =", h1_2d, " norm=", np.linalg.norm(h1_2d))
cosv = b1_2d @ h1_2d
print("cos =", cosv, " sin^2 =", 1-cosv**2, " angle(deg) =", np.degrees(np.arccos(np.clip(cosv,-1,1))))

print("\n=== Panel 1/2 vectors for j=1, angles in degrees ===")
aj = G_half_inv @ w_p[:, 0]
gnj = G_half_inv @ w_n[:, 0]
print("a_1^(p)   =", aj)
print("g_{n,1}^(p) =", gnj)
GBnorm_a = np.sqrt(aj @ G_Bp @ aj); GBnorm_g = np.sqrt(gnj @ G_Bp @ gnj)
cos_GB = (aj @ G_Bp @ gnj)/(GBnorm_a*GBnorm_g)
print("G_B angle (deg) =", np.degrees(np.arccos(np.clip(cos_GB,-1,1))))
cos_E = (aj@gnj)/(np.linalg.norm(aj)*np.linalg.norm(gnj))
print("Euclidean angle (deg) =", np.degrees(np.arccos(np.clip(cos_E,-1,1))))

print("\nG_Bp matrix =\n", G_Bp)
print("eigvals/vecs of G_Bp:", np.linalg.eigh(G_Bp))
