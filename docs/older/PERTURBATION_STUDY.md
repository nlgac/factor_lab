# Perturbation Study - Documentation

## Overview

The perturbation study compares two types of model differences:

1. **Sampling error**: Distance from true model to estimated model (finite sample)
2. **Rotational perturbation**: Distance from true model to slightly rotated model

**Key question:** How does sampling error compare to a small deterministic rotation?

---

## Scientific Motivation

### The Problem

Factor models have **rotational ambiguity**:
```
If B is a valid loading matrix, so is QB for any orthogonal Q
```

**Two natural questions:**

1. **How sensitive are distances to small rotations?**
   - Given true loadings B
   - Apply small rotation O ≈ I
   - Measure d(B, OB)

2. **How does sampling error compare?**
   - Given finite sample (T observations)
   - Estimate B̂ via SVD/PCA
   - Measure d(B, B̂)

**Insight:** If d(B, B̂) >> d(B, OB), then sampling error dominates rotational ambiguity!

### The Experiment

**Setup:**
- Create true model (B, F, D)
- Generate small rotation O ∈ O(p) with O ≈ I
- Create B_perturbed = O @ B
- Simulate 6,300 security returns
- Split into 100 subsets of 63 returns each

**For each subset:**
- Estimate B̂ from 63 observations
- Compute d_sample = d(B, B̂)
- Compute d_perturbed = d(B, O @ B)

**Analysis:**
- Scatter plot: d_sample vs. d_perturbed
- If points above y=x line: sampling error > perturbation
- If points below y=x line: perturbation > sampling error

---

## Usage

### Basic Usage

```bash
python perturbation_study.py perturbation_spec.json
```

### Specification File

Create `perturbation_spec.json`:

```json
{
  "p_assets": 500,           // Number of securities
  "k_factors": 3,            // Number of factors
  "n_total": 6300,           // Total observations (100 × 63)
  "n_subset": 63,            // Size of each subset
  "perturbation_size": 0.1,  // Rotation size (ε in exp(εA))
  "factor_variances": [0.0324, 0.01, 0.0025],  // Factor volatilities²
  "idio_variance": 0.01,     // Idiosyncratic variance
  "loading_mean": 0.0,       // Mean of loadings
  "loading_std": 1.0,        // Std of loadings
  "random_seed": 42          // For reproducibility
}
```

**Parameters explained:**

- **p_assets**: Number of stocks/securities (p)
- **k_factors**: Number of factors (k)
- **n_total**: Total simulated periods (must be divisible by n_subset)
- **n_subset**: Subset size for estimation (e.g., 63 = quarterly data for 5 years)
- **perturbation_size**: Controls rotation magnitude (ε)
  - 0.1 = small rotation (~10% from identity)
  - 0.01 = very small rotation (~1% from identity)
  - 1.0 = large rotation
- **factor_variances**: Diagonal of F (k values)
- **idio_variance**: All assets have same idiosyncratic variance
- **loading_mean/std**: Distribution of loadings B

---

## Output Files

### Directory Structure

```
perturbation_output/
├── perturbation_results.npz       # Complete results (main file)
├── scatter_stiefel.png            # Stiefel distance scatter plot
├── scatter_grassmannian.png       # Grassmannian distance scatter plot
├── histogram_full.png             # Full sample (6,300 obs) histogram
├── histograms_subsets.png         # 100 subset histograms (25×4 grid)
├── full_sample_stats.csv          # Pandas describe() for full sample
└── distances.csv                  # All distance measurements
```

### NPZ File Contents

**Model components:**
```python
data = np.load('perturbation_results.npz')

# True factor model
B_true = data['B_true']        # (k, p) - True loadings
F_true = data['F_true']        # (k, k) - True factor covariance
D_true = data['D_true']        # (p, p) - True idiosyncratic covariance

# Perturbation
O = data['O']                  # (p, p) - Small orthogonal matrix
B_perturbed = data['B_perturbed']  # (k, p) - Perturbed loadings
```

**Returns:**
```python
returns_full = data['returns_full']  # (6300, p) - All simulated returns
```

**Distances:**
```python
distances = data['distances']  # (100, 4) - Distance measurements

# Columns:
# 0: d_sample_stiefel     - Stiefel distance (B → B_sample)
# 1: d_pert_stiefel       - Stiefel distance (B → B_perturbed)
# 2: d_sample_grass       - Grassmannian distance (B → B_sample)
# 3: d_pert_grass         - Grassmannian distance (B → B_perturbed)
```

**Sample models (all 100):**
```python
B_samples = data['B_samples']  # (100, k, p) - Estimated loadings
F_samples = data['F_samples']  # (100, k, k) - Estimated factor covariances
D_samples = data['D_samples']  # (100, p, p) - Estimated idio covariances
```

**Metadata:**
```python
p_assets = data['p_assets']
k_factors = data['k_factors']
n_total = data['n_total']
n_subset = data['n_subset']
perturbation_size = data['perturbation_size']
```

---

## Interpreting Results

### Scatter Plots

**X-axis:** d_sample = distance from B to B̂ (sampling error)
**Y-axis:** d_perturbed = distance from B to O@B (rotation)

**Red dashed line:** y = x (equal distance)

**Interpretation:**

**Points above y=x (d_sample > d_perturbed):**
- Sampling error dominates
- Finite sample effects stronger than rotation
- High p/T regime (see APPENDIX_HIGH_DIMENSIONAL.md)

**Points below y=x (d_perturbed > d_sample):**
- Rotation dominates
- Good sample size
- Low p/T regime

**Points along y=x:**
- Sampling error ≈ perturbation
- Balanced regime

### Distance Metrics

**Stiefel distance (Procrustes):**
- Optimal alignment of frames
- Accounts for sign/permutation
- Smaller than Grassmannian (usually)

**Grassmannian distance:**
- Rotation-invariant subspace distance
- Fundamental geometric quantity
- Larger than Procrustes (usually)

**Both measure:** Angular separation between factor subspaces

### Histograms

**Full sample histogram:**
- Shows overall return distribution
- Check for normality
- Skewness/kurtosis reported

**100 subset histograms:**
- Visual check for stability
- Each subset has 63 observations
- Variability indicates sampling noise

---

## Example Analysis

### Scenario 1: Low p/T (Good Sample Size)

```json
{
  "p_assets": 100,
  "k_factors": 3,
  "n_total": 6300,
  "n_subset": 63,
  "perturbation_size": 0.1
}
```

**p/T = 100/63 ≈ 1.6** (moderate)

**Expected:**
- Most points near or below y=x
- Sampling error comparable to perturbation
- Good factor recovery

### Scenario 2: High p/T (Small Sample Size)

```json
{
  "p_assets": 500,
  "k_factors": 3,
  "n_total": 6300,
  "n_subset": 63,
  "perturbation_size": 0.1
}
```

**p/T = 500/63 ≈ 7.9** (high)

**Expected:**
- Most points above y=x
- Sampling error >> perturbation
- Finite sample effects dominate
- See bias correction (APPENDIX_HIGH_DIMENSIONAL.md)

### Scenario 3: Varying Perturbation Size

**Small perturbation (ε = 0.01):**
```json
{"perturbation_size": 0.01}
```
- Tiny rotation
- Points likely above y=x
- Sampling error >> perturbation

**Large perturbation (ε = 0.5):**
```json
{"perturbation_size": 0.5}
```
- Substantial rotation
- Points likely below y=x
- Perturbation >> sampling error

---

## Advanced Usage

### Loading Results for Further Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('perturbation_output/perturbation_results.npz')

# Extract distances
distances = data['distances']
d_sample_stiefel = distances[:, 0]
d_pert_stiefel = distances[:, 1]
d_sample_grass = distances[:, 2]
d_pert_grass = distances[:, 3]

# Compute statistics
print(f"Stiefel - Sample mean: {d_sample_stiefel.mean():.4f}")
print(f"Stiefel - Perturbation: {d_pert_stiefel[0]:.4f}")
print(f"Grassmannian - Sample mean: {d_sample_grass.mean():.4f}")
print(f"Grassmannian - Perturbation: {d_pert_grass[0]:.4f}")

# Ratio
ratio_stiefel = d_sample_stiefel.mean() / d_pert_stiefel[0]
ratio_grass = d_sample_grass.mean() / d_pert_grass[0]

print(f"\nRatio (sample/perturbation):")
print(f"  Stiefel: {ratio_stiefel:.2f}×")
print(f"  Grassmannian: {ratio_grass:.2f}×")

if ratio_grass > 1:
    print("\n➜ Sampling error dominates (high p/T regime)")
else:
    print("\n➜ Perturbation dominates (low p/T regime)")
```

### Comparing Multiple Scenarios

Run with different specifications:

```bash
# Scenario 1: Low p/T
python perturbation_study.py spec_low_pt.json
mv perturbation_output perturbation_low_pt

# Scenario 2: High p/T
python perturbation_study.py spec_high_pt.json
mv perturbation_output perturbation_high_pt

# Compare results
python compare_scenarios.py perturbation_low_pt perturbation_high_pt
```

---

## Theoretical Background

### Small Orthogonal Matrix Generation

**Method:** Matrix exponential of skew-symmetric matrix

```
O = exp(ε * A)

where A is skew-symmetric: A' = -A
```

**Properties:**
- O ∈ O(p) (orthogonal)
- det(O) = 1 (special orthogonal)
- ||O - I|| ≈ ε for small ε
- Smooth path from I to rotation

**Why skew-symmetric?**

For A skew-symmetric:
```
d/dt exp(tA) = A exp(tA)
```

This generates a **geodesic on SO(p)** starting at I with velocity A.

### Expected Sampling Error

From random matrix theory (see APPENDIX_HIGH_DIMENSIONAL.md):

```
E[d_G²(B, B̂)] ≈ k²p / (Tℓ²)
```

Where:
- k = number of factors
- p = number of assets
- T = sample size
- ℓ = signal strength

**For p/T large:** Sampling error grows!

### Perturbation Distance

For small rotation O ≈ I + εA:

```
d_G(B, OB) ≈ ε ||A||_F
```

**Comparison:**

If ε ||A||_F << k√(p/T):
➜ Sampling error dominates

If ε ||A||_F >> k√(p/T):
➜ Perturbation dominates

---

## Troubleshooting

### Issue: Script runs slowly

**Cause:** Large p or many subsets

**Solution:**
- Reduce p_assets
- Reduce n_total (but keep n_total/n_subset = 100)
- Use faster machine

### Issue: Distances all very similar

**Cause:** perturbation_size too close to sampling error

**Solution:**
- Try different perturbation_size values
- Run with multiple ε: [0.01, 0.05, 0.1, 0.5, 1.0]
- Compare across scenarios

### Issue: All points above y=x

**Cause:** High p/T regime (sampling error dominates)

**Solution:**
- This is expected! See APPENDIX_HIGH_DIMENSIONAL.md
- Reduce p or increase n_subset
- Apply bias corrections

### Issue: ImportError

**Cause:** Missing dependencies

**Solution:**
```bash
cd factor_lab_manifold_complete
pip install -r requirements.txt
```

---

## Example Workflow

### Complete Analysis Pipeline

```bash
# 1. Create specification
cat > my_spec.json << EOF
{
  "p_assets": 500,
  "k_factors": 3,
  "n_total": 6300,
  "n_subset": 63,
  "perturbation_size": 0.1,
  "factor_variances": [0.0324, 0.01, 0.0025],
  "idio_variance": 0.01,
  "random_seed": 42
}
EOF

# 2. Run study
python perturbation_study.py my_spec.json

# 3. View results
ls -lh perturbation_output/
open perturbation_output/scatter_grassmannian.png

# 4. Analyze in Python
python << EOF
import numpy as np
data = np.load('perturbation_output/perturbation_results.npz')
distances = data['distances']
print(f"Mean sampling error: {distances[:, 2].mean():.4f}")
print(f"Perturbation distance: {distances[0, 3]:.4f}")
print(f"Ratio: {distances[:, 2].mean() / distances[0, 3]:.2f}×")
EOF
```

---

## References

**Theoretical foundation:**
- See TECHNICAL_MANUAL.md §2 (Manifold Distances)
- See APPENDIX_HIGH_DIMENSIONAL.md §7 (Manifold Bias)

**Related analyses:**
- `build_and_simulate.py` - Standard factor analysis
- `demo.py` - Package demonstration

---

## Summary

The perturbation study provides:

✅ **Quantitative comparison** of sampling vs. rotation effects  
✅ **Visual diagnostics** via scatter plots  
✅ **Complete data** for further analysis  
✅ **Reproducible** results with random seed  
✅ **Flexible** specification via JSON  

**Use cases:**
- Understanding finite sample effects
- Calibrating perturbation size
- Validating estimation procedures
- Teaching random matrix theory
- Research on factor model stability

**Key insight:** In high-dimensional regimes (p/T > 5), sampling error typically dominates rotational ambiguity!
