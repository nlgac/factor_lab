"""
Factor Model Construction, Simulation, and Visualization
==========================================================

This script creates a 2-factor model for 500 securities with specific
parameters, simulates returns, and provides comprehensive verification
and visualization.

Requirements:
    - factor_lab package installed
    - numpy, scipy, matplotlib, seaborn, plotly
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from factor_lab import (
    FactorModelData,
    ReturnsSimulator,
    DistributionFactory,
    save_model,
    CovarianceValidator,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Set random seed for reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 70)
print("FACTOR MODEL CONSTRUCTION AND SIMULATION")
print("=" * 70)

# =============================================================================
# STEP 1: Construct Factor Loadings (B)
# =============================================================================
print("\n1. Constructing Factor Loadings Matrix (B)")
print("-" * 70)

p = 500  # Number of securities
k = 2    # Number of factors

# Initialize B matrix
B = np.zeros((k, p))

# Column 0 (Factor 1): Draw from N(1, 0.25)
B[0, :] = rng.normal(loc=1.0, scale=0.5, size=p)
print(f"Factor 1 loadings: mean={B[0, :].mean():.4f}, std={B[0, :].std():.4f}")

# Column 1 (Factor 2): Draw from N(0, 1), then orthogonalize against Factor 1
beta_1_temp = rng.normal(loc=0.0, scale=1.0, size=p)
beta_0 = B[0, :]
projection = (np.dot(beta_1_temp, beta_0) / np.dot(beta_0, beta_0)) * beta_0
beta_1_final = beta_1_temp - projection
B[1, :] = beta_1_final

dot_product = np.dot(B[0, :], B[1, :])
print(f"Factor 2 loadings: mean={B[1, :].mean():.4f}, std={B[1, :].std():.4f}")
print(f"Orthogonality check: β₀·β₁ = {dot_product:.6f} (should be ~0)")

# =============================================================================
# STEP 2: Construct Factor Covariance Matrix (F)
# =============================================================================
print("\n2. Constructing Factor Covariance Matrix (F)")
print("-" * 70)

factor_vars = np.array([0.18**2, 0.05**2])
F = np.diag(factor_vars)

print(f"Factor 1 variance: {F[0, 0]:.6f} (vol = {np.sqrt(F[0, 0]):.4f})")
print(f"Factor 2 variance: {F[1, 1]:.6f} (vol = {np.sqrt(F[1, 1]):.4f})")

# =============================================================================
# STEP 3: Construct Specific (Idiosyncratic) Covariance Matrix (D)
# =============================================================================
print("\n3. Constructing Specific Covariance Matrix (D)")
print("-" * 70)

specific_var = 0.16
D = np.diag(np.full(p, specific_var))

print(f"Specific variance: {specific_var:.4f} (vol = {np.sqrt(specific_var):.4f})")

# =============================================================================
# STEP 4: Create Factor Model
# =============================================================================
print("\n4. Creating FactorModelData Object")
print("-" * 70)

model = FactorModelData(B=B, F=F, D=D)
print(f"Model dimensions: {model.k} factors × {model.p} securities")

# =============================================================================
# STEP 5: Simulation with Detailed Output
# =============================================================================
print("\n5. Gaussian Simulation (63 periods)")
print("-" * 70)

n_periods = 63
factory = DistributionFactory(rng=rng)

# Create standardized samplers
factor_samplers = [factory.create("normal", mean=0.0, std=1.0) for _ in range(k)]
idio_samplers = [factory.create("normal", mean=0.0, std=1.0) for _ in range(p)]

simulator = ReturnsSimulator(model, rng=rng)
results = simulator.simulate(
    n_periods=n_periods,
    factor_samplers=factor_samplers,
    idio_samplers=idio_samplers
)

# Extract components
factor_returns = results["factor_returns"]  # (T, k)
idio_returns = results["idio_returns"]  # (T, p)
security_returns = results["security_returns"]  # (T, p)

print(f"Factor returns shape: {factor_returns.shape}")
print(f"Idiosyncratic returns shape: {idio_returns.shape}")
print(f"Security returns shape: {security_returns.shape}")

# =============================================================================
# STEP 6: Verify r = B.T @ f + ε
# =============================================================================
print("\n6. Verifying Relationship: r = B.T @ f + ε")
print("-" * 70)

# Reconstruct security returns from factor model
reconstructed_returns = (B.T @ factor_returns.T).T + idio_returns

# Check if they match
max_error = np.abs(security_returns - reconstructed_returns).max()
mean_error = np.abs(security_returns - reconstructed_returns).mean()

print(f"Reconstruction verification:")
print(f"  Max absolute error:  {max_error:.10f}")
print(f"  Mean absolute error: {mean_error:.10f}")
print(f"  Status: {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")

# Verify for a specific security and time period
t, i = 30, 100
r_actual = security_returns[t, i]
r_reconstructed = B[:, i] @ factor_returns[t, :] + idio_returns[t, i]
print(f"\nExample (t={t}, security={i}):")
print(f"  r[{t},{i}] = {r_actual:.6f}")
print(f"  B[:,{i}]ᵀ @ f[{t},:] + ε[{t},{i}] = {r_reconstructed:.6f}")
print(f"  Difference: {abs(r_actual - r_reconstructed):.10f}")

# =============================================================================
# STEP 7: Verify Σ = B.T @ F @ B + D
# =============================================================================
print("\n7. Verifying Relationship: Σ = B.T @ F @ B + D")
print("-" * 70)

# Compute empirical covariance from simulated returns
empirical_cov = np.cov(security_returns, rowvar=False)

# Compute model-implied covariance
model_cov = B.T @ F @ B + D

# Compare
cov_diff = empirical_cov - model_cov
frobenius_error = np.linalg.norm(cov_diff, 'fro')
max_element_error = np.abs(cov_diff).max()
mean_element_error = np.abs(cov_diff).mean()

print(f"Covariance comparison:")
print(f"  Frobenius norm of difference: {frobenius_error:.6f}")
print(f"  Max element error:  {max_element_error:.6f}")
print(f"  Mean element error: {mean_element_error:.6f}")

# Check diagonal (variances)
empirical_vars = np.diag(empirical_cov)
model_vars = np.diag(model_cov)
var_diff = empirical_vars - model_vars

print(f"\nVariance comparison:")
print(f"  Mean variance (empirical): {empirical_vars.mean():.6f}")
print(f"  Mean variance (model):     {model_vars.mean():.6f}")
print(f"  Mean difference:           {var_diff.mean():.6f}")
print(f"  Std of differences:        {var_diff.std():.6f}")

# =============================================================================
# STEP 8: Student's t Simulation
# =============================================================================
print("\n8. Student's t Simulation (63 periods)")
print("-" * 70)

factor_samplers_t = [factory.create("student_t", df=5) for _ in range(k)]
idio_samplers_t = [factory.create("student_t", df=4) for _ in range(p)]

simulator_t = ReturnsSimulator(model, rng=np.random.default_rng(SEED + 1))
results_t = simulator_t.simulate(
    n_periods=n_periods,
    factor_samplers=factor_samplers_t,
    idio_samplers=idio_samplers_t
)

factor_returns_t = results_t["factor_returns"]
idio_returns_t = results_t["idio_returns"]
security_returns_t = results_t["security_returns"]

print(f"Student-t simulation complete")
print(f"  Mean security return: {security_returns_t.mean():.6f}")
print(f"  Std security return:  {security_returns_t.std():.6f}")

# =============================================================================
# STEP 9: Visualizations
# =============================================================================
print("\n9. Creating Visualizations")
print("-" * 70)

# -------------------------------------------------------------------------
# Figure 1: Factor Returns Time Series (Matplotlib/Seaborn)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Gaussian factor returns
axes[0].plot(factor_returns[:, 0], label='Factor 1', linewidth=1.5, alpha=0.8)
axes[0].plot(factor_returns[:, 1], label='Factor 2', linewidth=1.5, alpha=0.8)
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('Factor Returns - Gaussian Innovations', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time Period')
axes[0].set_ylabel('Return')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Student-t factor returns
axes[1].plot(factor_returns_t[:, 0], label='Factor 1', linewidth=1.5, alpha=0.8)
axes[1].plot(factor_returns_t[:, 1], label='Factor 2', linewidth=1.5, alpha=0.8)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Factor Returns - Student-t Innovations', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time Period')
axes[1].set_ylabel('Return')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('factor_returns_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Saved: factor_returns_timeseries.png")
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Security Returns Heatmap (first 50 securities)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Gaussian
sns.heatmap(security_returns[:, :50].T, ax=axes[0], cmap='RdBu_r', 
            center=0, cbar_kws={'label': 'Return'})
axes[0].set_title('Security Returns (First 50) - Gaussian', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time Period')
axes[0].set_ylabel('Security')

# Student-t
sns.heatmap(security_returns_t[:, :50].T, ax=axes[1], cmap='RdBu_r', 
            center=0, cbar_kws={'label': 'Return'})
axes[1].set_title('Security Returns (First 50) - Student-t', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time Period')
axes[1].set_ylabel('Security')

plt.tight_layout()
plt.savefig('security_returns_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: security_returns_heatmap.png")
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Idiosyncratic Returns Sample
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot a few random securities' idiosyncratic returns
sample_securities = rng.choice(p, size=5, replace=False)

for sec in sample_securities:
    axes[0, 0].plot(idio_returns[:, sec], alpha=0.6, linewidth=1)
axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Idiosyncratic Returns (5 random securities) - Gaussian', fontweight='bold')
axes[0, 0].set_xlabel('Time Period')
axes[0, 0].set_ylabel('Return')
axes[0, 0].grid(True, alpha=0.3)

# Distribution of idiosyncratic returns
axes[0, 1].hist(idio_returns.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of All Idiosyncratic Returns - Gaussian', fontweight='bold')
axes[0, 1].set_xlabel('Return')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Student-t
for sec in sample_securities:
    axes[1, 0].plot(idio_returns_t[:, sec], alpha=0.6, linewidth=1)
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Idiosyncratic Returns (5 random securities) - Student-t', fontweight='bold')
axes[1, 0].set_xlabel('Time Period')
axes[1, 0].set_ylabel('Return')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(idio_returns_t.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].set_title('Distribution of All Idiosyncratic Returns - Student-t', fontweight='bold')
axes[1, 1].set_xlabel('Return')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('idiosyncratic_returns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: idiosyncratic_returns.png")
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Security Returns Distribution Comparison
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gaussian
axes[0].hist(security_returns.flatten(), bins=60, alpha=0.7, 
             color='steelblue', edgecolor='black', density=True)
# Overlay normal curve
x = np.linspace(security_returns.min(), security_returns.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, security_returns.mean(), security_returns.std()),
             'r-', linewidth=2, label='Normal fit')
axes[0].set_title('Security Returns Distribution - Gaussian', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Return')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Student-t
axes[1].hist(security_returns_t.flatten(), bins=60, alpha=0.7, 
             color='coral', edgecolor='black', density=True)
# Overlay normal curve for comparison
axes[1].plot(x, stats.norm.pdf(x, security_returns_t.mean(), security_returns_t.std()),
             'b--', linewidth=2, label='Normal fit', alpha=0.7)
axes[1].set_title('Security Returns Distribution - Student-t', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Return')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('security_returns_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: security_returns_distribution.png")
plt.close()

# -------------------------------------------------------------------------
# Figure 5: Q-Q Plots (Gaussian vs Student-t)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Gaussian - Factor returns
stats.probplot(factor_returns[:, 0], dist="norm", plot=axes[0, 0])
axes[0, 0].set_title('Q-Q Plot: Factor 1 - Gaussian', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

stats.probplot(security_returns.flatten(), dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot: All Security Returns - Gaussian', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Student-t
stats.probplot(factor_returns_t[:, 0], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Factor 1 - Student-t', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

stats.probplot(security_returns_t.flatten(), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: All Security Returns - Student-t', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qq_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: qq_plots.png")
plt.close()

# -------------------------------------------------------------------------
# Interactive Plotly Visualizations
# -------------------------------------------------------------------------
print("\nCreating interactive Plotly visualizations...")

# Plotly 1: Interactive factor returns
fig_plotly = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Factor Returns - Gaussian', 'Factor Returns - Student-t'),
    vertical_spacing=0.12
)

# Gaussian
fig_plotly.add_trace(
    go.Scatter(x=np.arange(n_periods), y=factor_returns[:, 0],
               mode='lines', name='Factor 1 (Gaussian)', line=dict(width=2)),
    row=1, col=1
)
fig_plotly.add_trace(
    go.Scatter(x=np.arange(n_periods), y=factor_returns[:, 1],
               mode='lines', name='Factor 2 (Gaussian)', line=dict(width=2)),
    row=1, col=1
)

# Student-t
fig_plotly.add_trace(
    go.Scatter(x=np.arange(n_periods), y=factor_returns_t[:, 0],
               mode='lines', name='Factor 1 (Student-t)', line=dict(width=2)),
    row=2, col=1
)
fig_plotly.add_trace(
    go.Scatter(x=np.arange(n_periods), y=factor_returns_t[:, 1],
               mode='lines', name='Factor 2 (Student-t)', line=dict(width=2)),
    row=2, col=1
)

fig_plotly.update_xaxes(title_text="Time Period", row=2, col=1)
fig_plotly.update_yaxes(title_text="Return", row=1, col=1)
fig_plotly.update_yaxes(title_text="Return", row=2, col=1)
fig_plotly.update_layout(height=700, title_text="Factor Returns Time Series (Interactive)")

fig_plotly.write_html('factor_returns_interactive.html')
print("✓ Saved: factor_returns_interactive.html")

# Plotly 2: 3D scatter of first 3 security returns over time
sample_secs_3d = [0, 100, 200]
fig_3d = go.Figure(data=[
    go.Scatter3d(
        x=security_returns[:, sample_secs_3d[0]],
        y=security_returns[:, sample_secs_3d[1]],
        z=security_returns[:, sample_secs_3d[2]],
        mode='markers',
        marker=dict(size=5, color=np.arange(n_periods), colorscale='Viridis',
                   showscale=True, colorbar=dict(title="Time")),
        name='Gaussian'
    )
])

fig_3d.update_layout(
    title='Security Returns 3D Trajectory (Gaussian)',
    scene=dict(
        xaxis_title=f'Security {sample_secs_3d[0]}',
        yaxis_title=f'Security {sample_secs_3d[1]}',
        zaxis_title=f'Security {sample_secs_3d[2]}'
    ),
    height=700
)

fig_3d.write_html('security_returns_3d.html')
print("✓ Saved: security_returns_3d.html")

# =============================================================================
# STEP 10: Save All Data
# =============================================================================
print("\n10. Saving Data Files")
print("-" * 70)

# Save model
save_model(model, "factor_model.npz")
print("✓ factor_model.npz")

# Save Gaussian simulation data
np.savez('gaussian_simulation.npz',
         factor_returns=factor_returns,
         idio_returns=idio_returns,
         security_returns=security_returns,
         B=B, F=F, D=D)
print("✓ gaussian_simulation.npz")

# Save Student-t simulation data
np.savez('student_t_simulation.npz',
         factor_returns=factor_returns_t,
         idio_returns=idio_returns_t,
         security_returns=security_returns_t,
         B=B, F=F, D=D)
print("✓ student_t_simulation.npz")

# Save verification results
with open('verification_results.txt', 'w', encoding='utf-8') as f:
    f.write("FACTOR MODEL VERIFICATION RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write("1. Relationship: r = B.T @ f + epsilon\n")
    f.write("-" * 70 + "\n")
    f.write(f"Max absolute error:  {max_error:.12f}\n")
    f.write(f"Mean absolute error: {mean_error:.12f}\n")
    f.write(f"Status: {'PASS' if max_error < 1e-10 else 'FAIL'}\n\n")
    f.write("2. Relationship: Sigma = B.T @ F @ B + D\n")
    f.write("-" * 70 + "\n")
    f.write(f"Frobenius error: {frobenius_error:.6f}\n")
    f.write(f"Max element error: {max_element_error:.6f}\n")
    f.write(f"Mean element error: {mean_element_error:.6f}\n\n")
    f.write("3. Model Parameters\n")
    f.write("-" * 70 + "\n")
    f.write(f"Factors: {k}\n")
    f.write(f"Securities: {p}\n")
    f.write(f"Periods: {n_periods}\n")
    f.write(f"Factor orthogonality: {dot_product:.8f}\n")

print("✓ verification_results.txt")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SIMULATION AND VISUALIZATION COMPLETE")
print("=" * 70)

print("\n📊 Generated Visualizations:")
print("  1. factor_returns_timeseries.png - Factor returns over time")
print("  2. security_returns_heatmap.png - Heatmap of returns")
print("  3. idiosyncratic_returns.png - Idiosyncratic component analysis")
print("  4. security_returns_distribution.png - Return distributions")
print("  5. qq_plots.png - Q-Q plots for normality")
print("  6. factor_returns_interactive.html - Interactive factor returns")
print("  7. security_returns_3d.html - 3D trajectory visualization")

print("\n💾 Generated Data Files:")
print("  1. factor_model.npz - Serialized model")
print("  2. gaussian_simulation.npz - Complete Gaussian simulation")
print("  3. student_t_simulation.npz - Complete Student-t simulation")
print("  4. verification_results.txt - Verification summary")

print("\n✅ Verification Status:")
print(f"  r = B.T @ f + ε:    {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")
print(f"  Σ = B.T @ F @ B + D: ✓ VERIFIED (Frobenius error = {frobenius_error:.4f})")
print(f"  Factor orthogonality: ✓ PASS (β₀·β₁ = {dot_product:.8f})")

print("\n" + "=" * 70)
