import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple
from scipy.linalg import expm, subspace_angles, qr, det
from scipy.sparse.linalg import eigsh
sqrt = np.sqrt

p_values = [100, 200, 500, 1000, 2000, 5000, 10000]
k = 3
ns = [63, 126]
radii = [0.1, 0.3, 0.5, 0.7, 0.9]
n_trials = 100
factor_variances = np.array([.16**2, 0.05**2, 0.05**2])
idio_variance = .1
n_perturbations = 100
seed = 42
#k = 3 factor model, 16% 5% and 5% volatility 



if len(factor_variances) > k:
    factor_variances = factor_variances[:k]

def orthonormalize(B: np.ndarray) -> np.ndarray:
    Q, _ = qr(B.T, mode='economic')
    return Q
#QR decomp 

def compute_grassmannian_distance(B_true, B_estimated):
    Q_true = orthonormalize(B_true)
    Q_estimated = orthonormalize(B_estimated)
    angles = subspace_angles(Q_true, Q_estimated)
    distance = float(np.linalg.norm(angles))
    return distance, angles
#takes two frames and then finds the Grassmann Distance

def generate_loadings(p, k, rng):
    B = np.zeros((k, p))
    B[0, :] = rng.normal(1.0, 1.0, p)
    for i in range(1, k):
        B[i, :] = rng.normal(0.0, 1.0, p)
    return B
#creates true factor matrix B

def simulate_returns(B, factor_variances, idio_variance, T, rng):
    k, p = B.shape
    f = rng.normal(size=(T, k)) * np.sqrt(factor_variances)
    e = rng.normal(size=(T, p)) * np.sqrt(idio_variance)
    return f @ B + e
#generates observations from the model 

def estimate_factors(returns, k):
    X = returns - returns.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k, :].T
#SVD on the returns and returns top k vectors 

def construct_epsilon_distance_perturbation(t, B, direction_rng):
    p, k = B.shape
    if k > p:
        B = B.T
        p, k = B.shape

    Z = direction_rng.standard_normal(B.shape)
    Delta = Z - B @ ((B.T @ Z + Z.T @ B) / 2)
    Delta = Delta / np.linalg.norm(Delta, 'fro')

    A = B.T @ Delta
    A = (A - A.T) / 2

    Delta_perp = Delta - B @ A
    Q_perp, R = np.linalg.qr(Delta_perp, mode='reduced')

    M = np.block([[A, -R.T],
                  [R, np.zeros((k, k))]])

    E = expm(t * M)
    M_t = E[:k, :k]
    N_t = E[k:, :k]

    return (B @ M_t + Q_perp @ N_t).T
#generaetes perturbations a geodesic away from the truth

def thm31_check(Q_est, Q_true, perturbation):
    k = Q_est.shape[1]
    if k > 1:
        h, b, z = Q_est, Q_true, perturbation
        lhs = np.abs(det(h.T @ z))
        rhs = np.abs(det(h.T @ b)) * np.abs(det(b.T @ z))
        return lhs, rhs
    h = Q_est[:, 0]
    b = Q_true[:, 0]
    z = perturbation
    lhs = abs(np.dot(h, z))
    rhs = abs(np.dot(h, b)) * abs(np.dot(b, z))
    return lhs, rhs
#Checks theorem 3.1, inner product / determinant method

#Pre-generate B and returns once at p_max
p_max = max(p_values)
rng_B = np.random.default_rng(seed)
B_max = generate_loadings(p_max, k, rng_B)

returns_collection = []
rng_sim = np.random.default_rng(p_max * seed)
for t in range(n_trials):
    returns_collection.append(
        simulate_returns(B_max, factor_variances, idio_variance, max(ns), rng_sim)
    )

#Main loop
records = []
thm31_records = []
perturbation_check = max(radii)

for p in p_values:
    B = B_max[:, :p]
    Q_true = orthonormalize(B)  #(p, k)

    for radius in radii:
        direction_rng = np.random.default_rng(seed + 2)
        perturbed_frames = []
        for _ in range(n_perturbations):
            pf = construct_epsilon_distance_perturbation(radius, Q_true, direction_rng)
            perturbed_frames.append(orthonormalize(pf))

        for n in ns:
            for t in range(n_trials):
                returns_used = returns_collection[t][:n, :p]
                Q_est = estimate_factors(returns_used, k)

                #sample-truth
                d_truth, _ = compute_grassmannian_distance(Q_true.T, Q_est.T)
                records.append({
                    'dimension': k,
                    'p': p,
                    'n': n,
                    'radius': radius,
                    'distance_type': 'sample-truth',
                    'distance': d_truth,
                    'metric': 'grassmann',
                })

                #sample-target
                for Q_perturb in perturbed_frames[:10]:
                    d_target, _ = compute_grassmannian_distance(Q_est.T, Q_perturb.T)
                    records.append({
                        'dimension': k,
                        'p': p,
                        'n': n,
                        'radius': radius,
                        'distance_type': 'sample-target',
                        'distance': d_target,
                        'metric': 'grassmann',
                    })

                #Theorem 3.1 check at largest radius only
                if radius == perturbation_check:
                    z = perturbed_frames[0]
                    lhs, rhs = thm31_check(Q_est, Q_true, z)
                    thm31_records.append({'p': p, 'n': n, 'lhs': lhs, 'rhs': rhs})

    print(f'p={p} done')

#Build DataFrames
long_df = pd.DataFrame(records)
long_df['radius_label'] = long_df['radius'].map(lambda x: f'r={x:.1f}')
long_df['n_label'] = long_df['n'].map(lambda x: f'n={x}')
thm31_df = pd.DataFrame(thm31_records)

print(long_df.head())
print(f'Total records: {len(long_df)}')

#Darwin plots 
col_order = [f'r={r:.1f}' for r in sorted(long_df['radius'].unique())]
row_order = [f'n={n}' for n in sorted(long_df['n'].unique())]
radius_map = {f'r={r:.1f}': r for r in sorted(long_df['radius'].unique())}

plot_df = long_df[long_df['distance_type'].isin(['sample-truth', 'sample-target'])].copy()

sns.set_theme(style='whitegrid', context='paper')
g = sns.catplot(
    data=plot_df,
    kind='box',
    x='p',
    y='distance',
    hue='distance_type',
    col='radius_label',
    row='n_label',
    col_order=col_order,
    row_order=row_order,
    hue_order=['sample-target', 'sample-truth'],
    sharey=True,
    height=3.0,
    aspect=1.1,
    linewidth=0.8,
    showfliers=False,
)
for axes_row in g.axes:
    for label, ax in zip(col_order, axes_row):
        ax.axhline(radius_map[label], ls='--', lw=1.2, color='black', alpha=0.7)
        ax.set_xlabel('Ambient dimension (p)')
        ax.set_ylabel('Distance')

g.set_titles(row_template='{row_name}', col_template='{col_name}')
g.fig.suptitle(f'k={k}: Grassmann distance vs (p, n, radius)', fontsize=14)
g.fig.subplots_adjust(top=0.90)
if g._legend:
    g._legend.set_title('')
plt.savefig('grassmann_distances.png', dpi=220, bbox_inches='tight')
plt.savefig('grassmann_distances.svg', bbox_inches='tight')
plt.show()

#Theorem 3.1 plot
thm_rows = []
for _, row in thm31_df.iterrows():
    thm_rows.append({'p': row['p'], 'Side': '|det(h\'z)| (LHS)', 'Value': row['lhs']})
    thm_rows.append({'p': row['p'], 'Side': '|det(h\'b)||det(b\'z)| (RHS)', 'Value': row['rhs']})
thm_plot_df = pd.DataFrame(thm_rows)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='p', y='Value', hue='Side', data=thm_plot_df, ax=ax,
            palette='Set1', medianprops=dict(color='black', linewidth=2))
ax.set_title(f'Theorem 3.1 Check: k={k}, perturbation ε={perturbation_check}')
ax.set_xlabel('Number of assets (p)')
ax.set_ylabel('Grassmann Kernel Measure')
ax.legend(title='')
plt.tight_layout()
plt.savefig('thm31_check.png', dpi=220, bbox_inches='tight')
plt.savefig('thm31_check.svg', bbox_inches='tight')
plt.show()
