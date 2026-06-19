"""Regenerate fig_isometry_pythagorean_split.pdf for proof_summary_5ideas.tex.

Three panels, one row.

  Panel 1: R^k, pre-symmetrization coordinates a_1^(p), g_{n,1}^(p), read with
           the (naive) Euclidean metric.  Both vectors live on the G_B^(p)-unit
           ellipse {x : x' G_B^(p) x = 1}; the ellipse is drawn explicitly.
           The naive Euclidean angle between them (~17 deg) differs from theta.
  Panel 2: R^k, the symmetrized coordinates w_1=(G_B^(p))^{1/2}a_1^(p) and
           w_{n,1}=(G_B^(p))^{1/2}g_{n,1}^(p) -- i.e. the actual w_j, w_{n,j}
           of the theorem. Plotted against the unit circle; angle ~9.8 deg,
           labeled as theta ~ 10 deg, matching panel 3.
  Panel 3: B subset R^p (ambient), b_1 and Pi_B h_1 / ||Pi_B h_1||, Euclidean
           metric. Angle ~9.9 deg, labeled as ~10 deg, matching panel 2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams["mathtext.fontset"] = "cm"

# G_B^(p) from the numerical example
G_B = np.array([[2.153, 0.851],
                [0.851, 0.959]])

# G_B^(p)-unit ellipse: {x : x' G x = 1}  =  {G^{-1/2} u : ||u||=1}
eigvals, Q = np.linalg.eigh(G_B)
G_inv_sqrt = Q @ np.diag(1.0 / np.sqrt(eigvals)) @ Q.T   # = (G_B^{1/2})^{-1}

def ellipse_points(n=400):
    t = np.linspace(0, 2 * np.pi, n)
    u = np.stack([np.cos(t), np.sin(t)])          # unit circle in R^2
    return G_inv_sqrt @ u                          # G_B-unit ellipse in R^2

# Panel 1: real a_1^(p), g_{n,1}^(p) from the numerical example;
# both satisfy x' G_B x = 1 (they are G_B-unit vectors by construction).
v_a1  = np.array([-0.653, -0.069])   # (Phi^(p))^{-1}(b_1)
v_gn1 = np.array([-0.735,  0.147])   # (Phi^(p))^{-1}(Pi_B h_1 / ||Pi_B h_1||)

# Panel 2: real w_1, w_{n,1} = (G_B^(p))^{1/2} a_1^(p), (G_B^(p))^{1/2} g_{n,1}^(p)
w1 = np.array([-0.95340047, -0.30170776])
wn1 = np.array([-0.99083178, -0.13510143])

# Panel 3: real b_1, Pi_B h_1 / ||.|| in the ambient B (2D chart)
b1_2d = np.array([0.99850103, -0.05473292])
h1_2d = np.array([0.99312273, 0.11707797])


def angle_arc(ax, v1, v2, radius, label, color="0.35", label_pos=None):
    a1deg = np.degrees(np.arctan2(v1[1], v1[0]))
    a2deg = np.degrees(np.arctan2(v2[1], v2[0]))
    lo, hi = sorted([a1deg, a2deg])
    if hi - lo > 180:
        lo, hi = hi - 360, lo
    arc = Arc((0, 0), 2 * radius, 2 * radius, angle=0,
              theta1=lo, theta2=hi, color=color, lw=1.3)
    ax.add_patch(arc)
    if label_pos is None:
        mid = np.radians((lo + hi) / 2)
        label_pos = (1.28 * radius * np.cos(mid), 1.28 * radius * np.sin(mid))
    ax.text(label_pos[0], label_pos[1], label, color=color, fontsize=11,
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def draw_vector(ax, v, color, label, label_offset=(0.06, 0.06)):
    ax.annotate("", xy=v, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                  shrinkA=0, shrinkB=0))
    ax.text(v[0] + label_offset[0], v[1] + label_offset[1], label,
            color=color, fontsize=12)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

t = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(t), np.sin(t)])

# ---------------------------------------------------------------- Panel 1: Euclidean reading of a_1^(p), g_{n,1}^(p) -- WRONG
ax = axes[0]
# unit circle (reference / wrong geometry)
ax.plot(circle[0], circle[1], color="#7570b3", lw=1.3, ls="--",
        label=r"$\{x:\,\|x\|=1\}$")
# G_B-unit ellipse (where a_1^(p) and g_{n,1}^(p) actually live)
ell = ellipse_points()
ax.plot(ell[0], ell[1], color="#e7298a", lw=2.0,
        label=r"$\{x:\,x^\top G_B^{(p)}x=1\}$")

draw_vector(ax, v_a1, "#1b9e77", r"$a_1^{(p)}$", label_offset=(-0.55, -0.20))
draw_vector(ax, v_gn1, "#d95f02", r"$g_{n,1}^{(p)}$", label_offset=(-0.60, 0.18))
# Euclidean angle between the real vectors (~17 deg)
angle_arc(ax, v_a1, v_gn1, 0.45, r"$\approx17^\circ$", color="#1b1b1b",
          label_pos=(-0.80, -0.35))

ax.set_title(r"$\mathbb{R}^k$ under the Euclidean metric", fontsize=13)
ax.text(0, -1.55, r"$a_1^{(p)},g_{n,1}^{(p)}$ live on the $G_B^{(p)}$-ellipse (pink);"
                   "\n" r"naive Euclidean angle $\approx17^\circ\ne\theta$",
        ha="center", va="top", fontsize=9.5, color="0.3")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

# ---------------------------------------------------------------- Panel 2: w_1, w_{n,1} (symmetrized coords) -- RIGHT, = theta
ax = axes[1]
ax.plot(circle[0], circle[1], color="#7570b3", lw=1.6,
        label=r"$\{x:\,\|x\|=1\}$")

draw_vector(ax, w1, "#1b9e77", r"$w_1$", label_offset=(-0.50, -0.10))
draw_vector(ax, wn1, "#d95f02", r"$w_{n,1}$", label_offset=(-0.55, 0.15))
angle_arc(ax, w1, wn1, 0.30, r"$\theta\approx10^\circ$", color="#1b1b1b",
          label_pos=(-0.60, -0.55))

ax.set_title(r"$\mathbb{R}^k$ under the $G_B^{(p)}$ metric", fontsize=13)
ax.text(0, -1.55, r"$w_1=(G_B^{(p)})^{1/2}a_1^{(p)}$, $w_{n,1}=(G_B^{(p)})^{1/2}g_{n,1}^{(p)}$:"
                   "\nangle $=\\theta$, matches panel 3",
        ha="center", va="top", fontsize=9.5, color="0.3")
ax.legend(loc="upper right", fontsize=8, frameon=False)

# ---------------------------------------------------------------- Panel 3: ambient B subset R^p, Euclidean (ground truth, theta)
ax = axes[2]
ax.plot(circle[0], circle[1], color="#7570b3", lw=1.6,
        label=r"$\{x:\,\|x\|=1\}$")

draw_vector(ax, b1_2d, "#1b9e77", r"$b_1$", label_offset=(0.06, -0.30))
draw_vector(ax, h1_2d, "#d95f02", r"$\Pi_Bh_1/\|\Pi_Bh_1\|$", label_offset=(-1.55, 0.20))
angle_arc(ax, b1_2d, h1_2d, 0.30, r"$\approx10^\circ\approx\theta$", color="#1b1b1b",
          label_pos=(0.75, -0.55))

ax.set_title(r"$\mathcal{B}\subset\mathbb{R}^p$ (ambient), Euclidean metric", fontsize=13)
ax.text(0, -1.55, "ground truth in the growing space;\n"
                   r"matches panel 2's $\theta$ via the isometry $\Phi^{(p)}$",
        ha="center", va="top", fontsize=9.5, color="0.3")
ax.legend(loc="upper right", fontsize=8, frameon=False)

# ---------------------------------------------------------------- shared styling
for ax in axes:
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle(r"Idea 5: the angle $\theta$ is metric-dependent in $\mathbb{R}^k$, "
              r"but isometric to the ambient angle in $\mathcal{B}\subset\mathbb{R}^p$",
              fontsize=12, y=1.04)
fig.tight_layout(rect=[0, 0.05, 1, 0.98])
import os
outdir = os.path.join(os.path.dirname(__file__), "../../figures")
fig.savefig(os.path.join(outdir, "fig_isometry_pythagorean_split.pdf"),
            bbox_inches="tight")
print("done")
