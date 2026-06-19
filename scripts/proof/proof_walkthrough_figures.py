"""
proof_walkthrough_figures.py
============================
Generates all figures for proof_walkthrough_k3.md.
Saves PNGs to factor_lab/walkthrough_figs/.

Figures produced
----------------
  fig_w01_model_setup.png        — bar chart of model parameters (c, sigma2, d, SNR)
  fig_w02_Wp_convergence.png     — operator norm ||W(p)-W_inf|| vs p
  fig_w03_eigenspectrum.png      — eigenvalues of W_inf: signal spikes vs noise floor
  fig_w04_eigvec_alignment.png   — |cos angle(chi_pj, v_j)| vs p for each factor
  fig_w05_floor_rotation.png     — stacked bar: floor + rotation contribution per factor
  fig_w06_lhs_vs_rhs.png         — scatter sin²∠(h_j,b̄_j) observed vs predicted, vs p
  fig_w07_angle_decomp.png       — angle decomposition diagram (phi, theta) per factor
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent / "walkthrough_figs"
OUT.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
PURPLE = "#7F77DD"
GRAY   = "#888780"
LIGHT  = "#F1EFE8"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E0DED6",
    "grid.linewidth":    0.5,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "medium",
    "axes.labelsize":    11,
})

SEED = 20260522
rng  = np.random.default_rng(SEED)

# ── Fixed model parameters ────────────────────────────────────────────────────
k      = 3
n      = 60
delta2 = 1.0
c      = np.array([1.0, 0.8, 0.6])
sigma2 = np.array([0.04, 0.02, 0.01])
C_half = np.diag(np.sqrt(c))
Gamma_B = np.diag(c)

def draw_model(p, seed_offset=0):
    local = np.random.default_rng(SEED + seed_offset)
    B = local.normal(0, 1, (p, k)) * np.sqrt(c)
    F = local.normal(0, 1, (n, k)) * np.sqrt(sigma2)
    Z = local.normal(0, np.sqrt(delta2), (p, n))
    Y = B @ F.T + Z
    return B, F, Z, Y

# ── fig_w01: model setup ──────────────────────────────────────────────────────
def fig_model_setup():
    d     = c * sigma2
    snr   = n * d / delta2
    labels = ["Factor 1", "Factor 2", "Factor 3"]
    x = np.arange(k)
    w = 0.18

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    fig.suptitle("Example model parameters  (k=3, n=60, p=500, δ²=1)", fontweight="medium")

    data = [
        (c,      "Prevalence  cⱼ",              BLUE),
        (sigma2, "Factor variance  σⱼ²",         TEAL),
        (d,      "Spike  dⱼ = cⱼσⱼ²",           CORAL),
        (snr,    "SNR  = n·dⱼ / δ²",            AMBER),
    ]
    for ax, (vals, title, color) in zip(axes, data):
        bars = ax.bar(x, vals, color=color, width=0.5, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([f"j={j+1}" for j in range(k)])
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vals)*1.25)

    plt.tight_layout()
    path = OUT / "fig_w01_model_setup.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w02: ||W(p) - W_inf|| vs p ───────────────────────────────────────────
def fig_Wp_convergence():
    P_vals = [100, 200, 500, 1000, 2000, 5000, 10000]
    norms  = []
    for p in P_vals:
        B, F, Z, Y = draw_model(p)
        W_p   = Y.T @ Y / (n * p)
        W_inf = F @ Gamma_B @ F.T / n + (delta2 / n) * np.eye(n)
        norms.append(np.linalg.norm(W_p - W_inf, ord=2))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.loglog(P_vals, norms, "o-", color=BLUE, lw=1.8, ms=6, zorder=3)
    # Reference 1/sqrt(p) line
    ref = norms[0] * np.sqrt(P_vals[0]) / np.sqrt(np.array(P_vals))
    ax.loglog(P_vals, ref, "--", color=GRAY, lw=1.2, label=r"$\propto p^{-1/2}$")
    ax.set_xlabel("p  (number of assets, log scale)")
    ax.set_ylabel(r"$\|W^{(p)} - W_\infty\|_\mathrm{op}$  (log scale)")
    ax.set_title(r"B.3.2 — $W^{(p)}$ converges to $W_\infty$ in operator norm")
    ax.legend(frameon=False)
    ax.axvline(500, color=CORAL, lw=1, ls=":", label="p=500 (example)")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = OUT / "fig_w02_Wp_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w03: eigenspectrum of W_inf ──────────────────────────────────────────
def fig_eigenspectrum():
    p = 500
    B, F, Z, Y = draw_model(p)
    D_hat = C_half @ (F.T @ F / n) @ C_half
    rho = np.sort(np.linalg.eigvalsh(D_hat))[::-1]
    tau = rho + delta2 / n
    noise_floor = delta2 / n

    # Full spectrum of W_inf
    W_inf = F @ Gamma_B @ F.T / n + noise_floor * np.eye(n)
    evals_full = np.sort(np.linalg.eigvalsh(W_inf))[::-1]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(1, n + 1)
    ax.scatter(xs[k:], evals_full[k:], s=14, color=GRAY, zorder=3, label="Noise: $\\delta^2/n$")
    ax.scatter(xs[:k], evals_full[:k], s=50, color=BLUE, zorder=4, label="Signal spikes: $\\tau_j = \\rho_j + \\delta^2/n$")
    ax.axhline(noise_floor, color=GRAY, lw=1.2, ls="--", label=f"Noise floor $\\delta^2/n = {noise_floor:.4f}$")
    for j in range(k):
        ax.annotate(f"$\\tau_{j+1}={tau[j]:.4f}$\n$(\\rho_{j+1}={rho[j]:.4f})$",
                    xy=(j+1, tau[j]), xytext=(j+4, tau[j] + 0.004),
                    fontsize=8.5, color=BLUE,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue of $W_\\infty$")
    ax.set_title("B.3.3 — Lemma 7: eigenspectrum of $W_\\infty$ (signal + noise)")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    path = OUT / "fig_w03_eigenspectrum.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w04: eigenvector alignment |cos| vs p ─────────────────────────────────
def fig_eigvec_alignment():
    P_vals = [100, 200, 500, 1000, 2000, 5000, 10000]
    colors = [BLUE, TEAL, CORAL]
    cos_by_j = [[] for _ in range(k)]

    # Draw F once and hold it fixed — this is the conditional-on-F theorem.
    # Only B and Z vary with p; the target v_j is computed from this fixed F.
    F_fixed = np.random.default_rng(SEED + 99).normal(0, 1, (n, k)) * np.sqrt(sigma2)

    # Pre-compute the fixed target v_j from F_fixed.
    D_hat_fixed = C_half @ (F_fixed.T @ F_fixed / n) @ C_half
    rho_fixed, W_hat_fixed = np.linalg.eigh(D_hat_fixed)
    idx_fixed = np.argsort(rho_fixed)[::-1]
    rho_fixed  = rho_fixed[idx_fixed]
    W_hat_fixed = W_hat_fixed[:, idx_fixed]
    F_sharp_fixed = F_fixed @ C_half
    V_theory = F_sharp_fixed @ W_hat_fixed / np.sqrt(n * rho_fixed)   # (n, k), fixed target

    for ip, p in enumerate(P_vals):
        local = np.random.default_rng(SEED + 200 + ip)
        B = local.normal(0, 1, (p, k)) * np.sqrt(c)
        Z = local.normal(0, np.sqrt(delta2), (p, n))
        Y = B @ F_fixed.T + Z
        W_p = Y.T @ Y / (n * p)

        evals_Wp, evecs_Wp = np.linalg.eigh(W_p)
        chi = evecs_Wp[:, np.argsort(evals_Wp)[::-1]]

        for j in range(k):
            cos_by_j[j].append(abs(chi[:, j] @ V_theory[:, j]))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for j in range(k):
        ax.semilogx(P_vals, cos_by_j[j], "o-", color=colors[j],
                    lw=1.8, ms=6, label=f"j={j+1}", zorder=3)
    ax.axvline(500, color=GRAY, lw=1, ls=":", label="p=500 (example)")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("p  (log scale)")
    ax.set_ylabel(r"$|\cos\angle(\chi_{p,j},\, v_j)|$")
    ax.set_title(r"B.3.4 — Spectral convergence $\chi_{p,j} \to v_j$")
    ax.legend(frameon=False)
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    path = OUT / "fig_w04_eigvec_alignment.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w05: floor + rotation stacked bar ─────────────────────────────────────
def fig_floor_rotation():
    p = 500
    B, F, Z, Y = draw_model(p)
    D_hat = C_half @ (F.T @ F / n) @ C_half
    rho_eig, W_hat = np.linalg.eigh(D_hat)
    idx = np.argsort(rho_eig)[::-1]
    rho_eig = rho_eig[idx]; W_hat = W_hat[:, idx]

    floor  = delta2 / (n * rho_eig + delta2)
    weight = n * rho_eig / (n * rho_eig + delta2)
    rot    = 1.0 - np.diag(W_hat)**2
    rhs    = floor + weight * rot

    labels = ["Factor 1\n(SNR=2.74)", "Factor 2\n(SNR=0.94)", "Factor 3\n(SNR=0.41)"]
    x = np.arange(k)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x, floor,         color=CORAL,  width=0.5, label="Floor  $\\delta^2/(n\\rho_j+\\delta^2)$",  zorder=3)
    b2 = ax.bar(x, weight * rot,  color=PURPLE, width=0.5, bottom=floor,
                label="Rotation  $\\frac{n\\rho_j}{n\\rho_j+\\delta^2}\\sin^2\\angle(\\hat w_j,w_j)$", zorder=3)

    # Annotate total RHS
    for i, (f, wr, r) in enumerate(zip(floor, weight*rot, rhs)):
        ax.text(i, r + 0.012, f"RHS={r:.3f}", ha="center", va="bottom", fontsize=9, color="#333")
        ax.text(i, f/2,       f"{f:.3f}",     ha="center", va="center", fontsize=8.5, color="white", fontweight="medium")
        ax.text(i, f + wr/2,  f"{wr:.3f}",    ha="center", va="center", fontsize=8.5, color="white", fontweight="medium")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\sin^2\angle(h_j,\bar b_j)$  contribution")
    ax.set_title("B.3.7 — Assembly: floor + weighted rotation per factor")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    path = OUT / "fig_w05_floor_rotation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w06: observed LHS vs predicted RHS, vs p ─────────────────────────────
def fig_lhs_vs_rhs():
    P_vals = [200, 500, 1000, 2000, 5000]
    colors = [BLUE, TEAL, CORAL]
    lhs_by_j = [[] for _ in range(k)]
    rhs_by_j = [[] for _ in range(k)]

    for p in P_vals:
        B, F, Z, Y = draw_model(p)

        # Population b_bar
        C_inv_half = np.diag(1.0 / np.sqrt(c))
        a_inf = C_inv_half  # columns are e_j/sqrt(c_j)
        b_bar_raw = B @ a_inf
        b_bar = b_bar_raw / np.linalg.norm(b_bar_raw, axis=0)

        # h_j via n×n Gram trick
        G_nn = Y.T @ Y
        evals_G, evecs_G = np.linalg.eigh(G_nn)
        idx_G = np.argsort(evals_G)[::-1]
        s_vals = np.sqrt(np.maximum(evals_G[idx_G[:k]], 0.0))
        H = (Y @ evecs_G[:, idx_G[:k]]) / s_vals

        # RHS
        D_hat = C_half @ (F.T @ F / n) @ C_half
        rho_eig, W_hat = np.linalg.eigh(D_hat)
        idx_r = np.argsort(rho_eig)[::-1]
        rho_eig = rho_eig[idx_r]; W_hat = W_hat[:, idx_r]
        floor  = delta2 / (n * rho_eig + delta2)
        weight = n * rho_eig / (n * rho_eig + delta2)
        rot    = 1.0 - np.diag(W_hat)**2
        rhs    = floor + weight * rot

        for j in range(k):
            lhs_by_j[j].append(1.0 - (H[:, j] @ b_bar[:, j])**2)
            rhs_by_j[j].append(rhs[j])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
    for j, ax in enumerate(axes):
        ax.plot(P_vals, lhs_by_j[j], "o-", color=colors[j], lw=1.8, ms=6, label="LHS observed", zorder=3)
        ax.plot(P_vals, rhs_by_j[j], "s--", color=GRAY, lw=1.4, ms=5, label="RHS predicted", zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"Factor j={j+1}  (SNR≈{n*c[j]*sigma2[j]/delta2:.2f})")
        ax.set_xlabel("p  (log scale)")
        ax.set_ylabel(r"$\sin^2\angle(h_j,\bar b_j)$")
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0, 1)
    fig.suptitle("LHS observed vs RHS predicted — convergence as p → ∞", fontweight="medium")
    plt.tight_layout()
    path = OUT / "fig_w06_lhs_vs_rhs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# ── fig_w07: angle decomposition diagram ──────────────────────────────────────
def fig_angle_decomp():
    p = 500
    B, F, Z, Y = draw_model(p)

    # Population b_bar
    C_inv_half = np.diag(1.0 / np.sqrt(c))
    b_bar_raw = B @ C_inv_half
    b_bar = b_bar_raw / np.linalg.norm(b_bar_raw, axis=0)

    # h_j
    G_nn = Y.T @ Y
    evals_G, evecs_G = np.linalg.eigh(G_nn)
    idx_G = np.argsort(evals_G)[::-1]
    s_vals = np.sqrt(np.maximum(evals_G[idx_G[:k]], 0.0))
    H = (Y @ evecs_G[:, idx_G[:k]]) / s_vals

    # Projection onto B subspace
    Pi_B_light = B @ np.linalg.solve(B.T @ B, B.T)

    sin2_total = np.array([1.0 - (H[:, j] @ b_bar[:, j])**2 for j in range(k)])
    h_par      = np.array([Pi_B_light @ H[:, j] for j in range(k)])
    h_perp_n2  = 1.0 - np.array([np.linalg.norm(h_par[j])**2 for j in range(k)])
    h_par_norm = np.array([np.linalg.norm(h_par[j]) for j in range(k)])
    insubspace_sin2 = np.array([
        1.0 - (h_par[j] @ b_bar[:, j] / h_par_norm[j])**2
        for j in range(k)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors_j = [BLUE, TEAL, CORAL]
    snr_vals = [2.74, 0.94, 0.41]

    for j, ax in enumerate(axes):
        total_angle = np.degrees(np.arcsin(np.sqrt(np.clip(sin2_total[j], 0, 1))))
        perp_angle  = np.degrees(np.arcsin(np.sqrt(np.clip(h_perp_n2[j], 0, 1))))
        insubsp_ang = np.degrees(np.arcsin(np.sqrt(np.clip(insubspace_sin2[j], 0, 1))))

        # Draw unit circle arc
        theta_arr = np.linspace(0, np.pi/2, 200)
        ax.plot(np.cos(theta_arr), np.sin(theta_arr), color=LIGHT, lw=1)

        # b_bar direction (target) — always at 90° for clarity
        b_angle = np.pi/2
        ax.annotate("", xy=(np.cos(b_angle)*0.95, np.sin(b_angle)*0.95), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))
        ax.text(np.cos(b_angle)*1.0, np.sin(b_angle)*1.05, r"$\bar{b}_j$", ha="center", fontsize=10, color=GRAY)

        # h_j direction
        h_angle = b_angle - np.radians(total_angle)
        ax.annotate("", xy=(np.cos(h_angle)*0.95, np.sin(h_angle)*0.95), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colors_j[j], lw=2))
        ax.text(np.cos(h_angle)*1.05, np.sin(h_angle)*1.0, r"$h_j$", ha="center", fontsize=10, color=colors_j[j])

        # Horizontal line = subspace boundary
        ax.axhline(np.sin(np.radians(perp_angle)), color=AMBER, lw=1.2, ls="--", alpha=0.7)
        ax.text(0.05, np.sin(np.radians(perp_angle)) + 0.03,
                f"||h⊥||={np.sqrt(h_perp_n2[j]):.2f}\n(> floor)", fontsize=8, color=AMBER)

        # Arc for total angle
        arc_theta = np.linspace(h_angle, b_angle, 60)
        ax.plot(np.cos(arc_theta)*0.55, np.sin(arc_theta)*0.55, color=colors_j[j], lw=1.5)
        mid = (h_angle + b_angle)/2
        ax.text(np.cos(mid)*0.62, np.sin(mid)*0.62,
                f"{total_angle:.1f}°", fontsize=8.5, color=colors_j[j], ha="center")

        ax.set_xlim(-0.1, 1.2); ax.set_ylim(-0.1, 1.2)
        ax.set_aspect("equal"); ax.axis("off")
        floor_pred = delta2 / (n * c[j] * sigma2[j] + delta2)   # δ²/(nd_j+δ²), population floor
        ax.set_title(f"j={j+1}  SNR≈{snr_vals[j]}\nφ={total_angle:.1f}°  floor={floor_pred:.3f}", fontsize=10)

    fig.suptitle("B.3.1 — Angle decomposition: h_j relative to b̄_j and the signal subspace B", fontweight="medium")
    plt.tight_layout()
    path = OUT / "fig_w07_angle_decomp.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# -- fig_w08: gap vs p (all three factors) ------------------------------------
def fig_gap_vs_p():
    P_vals = [200, 500, 1000, 2000, 5000, 10000]
    colors = [BLUE, TEAL, CORAL]
    gaps = [[] for _ in range(k)]

    for p in P_vals:
        B, F, Z, Y = draw_model(p)
        C_inv_half = np.diag(1.0 / np.sqrt(c))
        b_bar_raw = B @ C_inv_half
        b_bar = b_bar_raw / np.linalg.norm(b_bar_raw, axis=0)
        G_nn = Y.T @ Y
        evals_G, evecs_G = np.linalg.eigh(G_nn)
        idx_G = np.argsort(evals_G)[::-1]
        s_vals = np.sqrt(np.maximum(evals_G[idx_G[:k]], 0.0))
        H = (Y @ evecs_G[:, idx_G[:k]]) / s_vals
        D_hat = C_half @ (F.T @ F / n) @ C_half
        rho_eig, W_hat = np.linalg.eigh(D_hat)
        idx_r = np.argsort(rho_eig)[::-1]
        rho_eig = rho_eig[idx_r]; W_hat = W_hat[:, idx_r]
        floor  = delta2 / (n * rho_eig + delta2)
        weight = n * rho_eig / (n * rho_eig + delta2)
        rot    = 1.0 - np.diag(W_hat)**2
        rhs    = floor + weight * rot
        for j in range(k):
            lhs = 1.0 - (H[:, j] @ b_bar[:, j])**2
            gaps[j].append(abs(lhs - rhs[j]))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for j in range(k):
        ax.loglog(P_vals, gaps[j], "o-", color=colors[j], lw=1.8, ms=6,
                  label=f"j={j+1}  (SNR={n*c[j]*sigma2[j]/delta2:.2f})", zorder=3)
    ref = gaps[0][0] * np.sqrt(P_vals[0]) / np.sqrt(np.array(P_vals))
    ax.loglog(P_vals, ref, "--", color=GRAY, lw=1.1, label=r"$\propto p^{-1/2}$")
    ax.axvline(500, color=GRAY, lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("p  (log scale)")
    ax.set_ylabel(r"|LHS $-$ RHS|  (log scale)")
    ax.set_title(r"Gap $|\sin^2\angle - \mathrm{RHS}|$ decays as $p \to \infty$")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    path = OUT / "fig_w08_gap_vs_p.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")

# -- Run all ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig_model_setup()
    fig_Wp_convergence()
    fig_eigenspectrum()
    fig_eigvec_alignment()
    fig_floor_rotation()
    fig_lhs_vs_rhs()
    fig_angle_decomp()
    fig_gap_vs_p()
    print("All figures saved to", OUT)
