"""
Factor Model Construction and Simulation
==========================================

This script creates a 2-factor model for 500 securities with specific
parameters and simulates 63 periods of returns using both Gaussian and
t-distribution innovations.

Requirements:
    - factor_lab package installed
    - numpy, scipy
"""

import numpy as np
from scipy import stats
from factor_lab import (
    FactorModelData,
    ReturnsSimulator,
    DistributionFactory,
    save_model,
    CovarianceValidator,
)

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
# Note: std = sqrt(0.25) = 0.5
B[0, :] = rng.normal(loc=1.0, scale=0.5, size=p)
print(f"Factor 1 loadings: mean={B[0, :].mean():.4f}, std={B[0, :].std():.4f}")

# Column 1 (Factor 2): Draw from N(0, 1), then orthogonalize against Factor 1
beta_1_temp = rng.normal(loc=0.0, scale=1.0, size=p)

# Gram-Schmidt orthogonalization
# Project beta_1_temp onto beta_0, then subtract to get orthogonal component
beta_0 = B[0, :]
projection = (np.dot(beta_1_temp, beta_0) / np.dot(beta_0, beta_0)) * beta_0
beta_1_final = beta_1_temp - projection

B[1, :] = beta_1_final

# Verify orthogonality
dot_product = np.dot(B[0, :], B[1, :])
print(f"Factor 2 loadings: mean={B[1, :].mean():.4f}, std={B[1, :].std():.4f}")
print(f"Orthogonality check: β₀·β₁ = {dot_product:.6f} (should be ~0)")

# =============================================================================
# STEP 2: Construct Factor Covariance Matrix (F)
# =============================================================================
print("\n2. Constructing Factor Covariance Matrix (F)")
print("-" * 70)

# Diagonal matrix with variances (0.18^2, 0.05^2)
factor_vars = np.array([0.18**2, 0.05**2])
F = np.diag(factor_vars)

print(f"Factor 1 variance: {F[0, 0]:.6f} (vol = {np.sqrt(F[0, 0]):.4f})")
print(f"Factor 2 variance: {F[1, 1]:.6f} (vol = {np.sqrt(F[1, 1]):.4f})")

# =============================================================================
# STEP 3: Construct Specific (Idiosyncratic) Covariance Matrix (D)
# =============================================================================
print("\n3. Constructing Specific Covariance Matrix (D)")
print("-" * 70)

# Diagonal matrix with all entries = 0.16 (this is the variance)
specific_var = 0.16
D = np.diag(np.full(p, specific_var))

print(f"Specific variance: {specific_var:.4f} (vol = {np.sqrt(specific_var):.4f})")
print(f"All {p} securities have identical specific risk")

# =============================================================================
# STEP 4: Create Factor Model
# =============================================================================
print("\n4. Creating FactorModelData Object")
print("-" * 70)

model = FactorModelData(B=B, F=F, D=D)

print(f"Model dimensions: {model.k} factors × {model.p} securities")
print(f"Model validation: {'✓ Passed' if model.k == k and model.p == p else '✗ Failed'}")

# Calculate total variance explained
total_var = np.trace(model.implied_covariance()) / p
factor_var = np.trace(B.T @ F @ B) / p
specific_var_avg = np.trace(D) / p
pct_explained = factor_var / total_var * 100

print(f"\nVariance decomposition:")
print(f"  Factor variance:   {factor_var:.6f}")
print(f"  Specific variance: {specific_var_avg:.6f}")
print(f"  Total variance:    {total_var:.6f}")
print(f"  % Explained:       {pct_explained:.2f}%")

# =============================================================================
# STEP 5: Simulation 1 - Gaussian Innovations
# =============================================================================
print("\n5. Simulation 1: Gaussian Innovations (63 periods)")
print("-" * 70)

n_periods = 63

# Create distribution factory
factory = DistributionFactory(rng=rng)

# Create Gaussian samplers with appropriate variances
# For factors: N(0, σ²) where σ² is already in F
factor_samplers_gaussian = [
    factory.create("normal", mean=0.0, std=np.sqrt(F[0, 0])),
    factory.create("normal", mean=0.0, std=np.sqrt(F[1, 1])),
]

# For specific returns: N(0, 0.16)
idio_samplers_gaussian = [
    factory.create("normal", mean=0.0, std=np.sqrt(specific_var))
    for _ in range(p)
]

# Create simulator and run simulation
simulator_gaussian = ReturnsSimulator(model, rng=rng)

# Note: ReturnsSimulator expects samplers to return STANDARDIZED draws
# It handles the scaling internally using the transforms
# So we need to use standardized samplers
factor_samplers_std = [
    factory.create("normal", mean=0.0, std=1.0)
    for _ in range(k)
]
idio_samplers_std = [
    factory.create("normal", mean=0.0, std=1.0)
    for _ in range(p)
]

results_gaussian = simulator_gaussian.simulate(
    n_periods=n_periods,
    factor_samplers=factor_samplers_std,
    idio_samplers=idio_samplers_std
)

returns_gaussian = results_gaussian["security_returns"]

print(f"Generated returns shape: {returns_gaussian.shape}")
print(f"Returns statistics:")
print(f"  Mean:   {returns_gaussian.mean():.6f}")
print(f"  Std:    {returns_gaussian.std():.6f}")
print(f"  Min:    {returns_gaussian.min():.6f}")
print(f"  Max:    {returns_gaussian.max():.6f}")

# Validate covariance structure
validator = CovarianceValidator(model)
validation_gaussian = validator.compare(returns_gaussian)

print(f"\nCovariance validation:")
print(f"  Frobenius error:  {validation_gaussian.frobenius_error:.6f}")
print(f"  Mean abs error:   {validation_gaussian.mean_absolute_error:.6f}")
print(f"  Explained var %:  {validation_gaussian.explained_variance_ratio:.2%}")

# =============================================================================
# STEP 6: Simulation 2 - Student's t Innovations
# =============================================================================
print("\n6. Simulation 2: Student's t Innovations (63 periods)")
print("-" * 70)

# Student's t with df degrees of freedom has variance = df/(df-2) for df > 2
# We want variance = 1 (standardized), so we use std = sqrt(df/(df-2))

# Factor returns: t(5) → variance = 5/(5-2) = 5/3 ≈ 1.667
# To standardize: divide by sqrt(5/3)
df_factors = 5
std_factor_t = np.sqrt(df_factors / (df_factors - 2))

# Specific returns: t(4) → variance = 4/(4-2) = 2
# To standardize: divide by sqrt(2)
df_idio = 4
std_idio_t = np.sqrt(df_idio / (df_idio - 2))

# Create standardized t-distribution samplers
factor_samplers_t = [
    factory.create("student_t", df=df_factors)
    for _ in range(k)
]
idio_samplers_t = [
    factory.create("student_t", df=df_idio)
    for _ in range(p)
]

# Create new simulator with fresh RNG
simulator_t = ReturnsSimulator(model, rng=np.random.default_rng(SEED + 1))

results_t = simulator_t.simulate(
    n_periods=n_periods,
    factor_samplers=factor_samplers_t,
    idio_samplers=idio_samplers_t
)

returns_t = results_t["security_returns"]

print(f"Generated returns shape: {returns_t.shape}")
print(f"Returns statistics:")
print(f"  Mean:   {returns_t.mean():.6f}")
print(f"  Std:    {returns_t.std():.6f}")
print(f"  Min:    {returns_t.min():.6f}")
print(f"  Max:    {returns_t.max():.6f}")

# Check for fat tails (t-distribution should have more extreme values)
print(f"\nTail behavior (|return| > 2σ):")
gaussian_outliers = np.abs(returns_gaussian) > 2 * returns_gaussian.std()
t_outliers = np.abs(returns_t) > 2 * returns_t.std()
print(f"  Gaussian:  {gaussian_outliers.sum()} outliers ({gaussian_outliers.mean():.2%})")
print(f"  Student-t: {t_outliers.sum()} outliers ({t_outliers.mean():.2%})")

# Validate covariance structure
validation_t = validator.compare(returns_t)

print(f"\nCovariance validation:")
print(f"  Frobenius error:  {validation_t.frobenius_error:.6f}")
print(f"  Mean abs error:   {validation_t.mean_absolute_error:.6f}")
print(f"  Explained var %:  {validation_t.explained_variance_ratio:.2%}")

# =============================================================================
# STEP 7: Save Results
# =============================================================================
print("\n7. Saving Results")
print("-" * 70)

# Save the model
save_model(model, "factor_model_2x500.npz")
print("✓ Model saved to: factor_model_2x500.npz")

# Save returns to CSV files
np.savetxt("returns_gaussian_63x500.csv", returns_gaussian, delimiter=",")
print("✓ Gaussian returns saved to: returns_gaussian_63x500.csv")

np.savetxt("returns_student_t_63x500.csv", returns_t, delimiter=",")
print("✓ Student-t returns saved to: returns_student_t_63x500.csv")

# Save summary statistics
with open("simulation_summary.txt", "w") as f:
    f.write("FACTOR MODEL SIMULATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model Specification:\n")
    f.write(f"  Securities (p): {p}\n")
    f.write(f"  Factors (k): {k}\n")
    f.write(f"  Periods simulated: {n_periods}\n\n")
    f.write(f"Factor 1:\n")
    f.write(f"  Loading mean: {B[0, :].mean():.4f}\n")
    f.write(f"  Loading std:  {B[0, :].std():.4f}\n")
    f.write(f"  Variance: {F[0, 0]:.6f}\n\n")
    f.write(f"Factor 2 (orthogonalized):\n")
    f.write(f"  Loading mean: {B[1, :].mean():.4f}\n")
    f.write(f"  Loading std:  {B[1, :].std():.4f}\n")
    f.write(f"  Variance: {F[1, 1]:.6f}\n")
    f.write(f"  Orthogonality: {dot_product:.6f}\n\n")
    f.write(f"Specific Risk:\n")
    f.write(f"  Variance: {specific_var:.6f}\n\n")
    f.write(f"Gaussian Simulation:\n")
    f.write(f"  Mean: {returns_gaussian.mean():.6f}\n")
    f.write(f"  Std:  {returns_gaussian.std():.6f}\n")
    f.write(f"  Frobenius error: {validation_gaussian.frobenius_error:.6f}\n\n")
    f.write(f"Student-t Simulation (df_factors={df_factors}, df_idio={df_idio}):\n")
    f.write(f"  Mean: {returns_t.mean():.6f}\n")
    f.write(f"  Std:  {returns_t.std():.6f}\n")
    f.write(f"  Frobenius error: {validation_t.frobenius_error:.6f}\n")

print("✓ Summary saved to: simulation_summary.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  1. factor_model_2x500.npz - Serialized factor model")
print(f"  2. returns_gaussian_63x500.csv - Gaussian returns (63×500)")
print(f"  3. returns_student_t_63x500.csv - Student-t returns (63×500)")
print(f"  4. simulation_summary.txt - Summary statistics")
print(f"\nBoth simulations used the same factor model with:")
print(f"  • 2 factors (orthogonalized)")
print(f"  • 500 securities")
print(f"  • 63 time periods")
print(f"  • Gaussian vs. Student-t innovations")
print("\n" + "=" * 70)
