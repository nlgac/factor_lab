# NPZ File Output Format

## Overview

When you run `build_and_simulate.py`, it creates comprehensive `.npz` files containing all simulation data and analysis results. These files match Gemini's original format and include everything you need for further analysis.

---

## Files Created

### Per Simulation
- `simulation_gaussian.npz` - Gaussian simulation results
- `simulation_student-t.npz` - Student-t simulation results
- etc. (one per simulation in config)

### Model
- `factor_model.npz` - Just the factor model (B, F, D)

---

## NPZ File Contents

Each `simulation_*.npz` file contains **15+ arrays** organized by category:

### 1. Raw Data
```python
'security_returns'    # (T, p) - Simulated returns
```

### 2. True Model
```python
'true_B'              # (k, p) - True factor loadings
'true_ortho_B'        # (k, p) - Orthonormalized version (Q from QR decomposition)
'true_F'              # (k, k) - True factor covariance
'true_D'              # (p, p) - True idiosyncratic covariance
'true_eigenvalues'    # (k,)   - Eigenvalues of Σ_true via LinearOperator
'true_eigenvectors'   # (k, p) - Eigenvectors of Σ_true
```

### 3. Sample Model (Estimated via SVD)
```python
'sample_B'            # (k, p) - Estimated factor loadings
'sample_F'            # (k, k) - Estimated factor covariance
'sample_D'            # (p, p) - Estimated idiosyncratic covariance
'sample_eigenvalues'  # (k,)   - Eigenvalues from PCA
'sample_eigenvectors' # (k, p) - Eigenvectors from PCA
```

### 4. Manifold Distances (Gemini's naming)
```python
'dist_grassmannian'       # float - Subspace distance (rotation-invariant)
'dist_stiefel_procrustes' # float - Optimal frame alignment
'dist_stiefel_chordal'    # float - Raw frame difference
'principal_angles'        # (k,)  - Principal angles between subspaces
```

### 5. Eigenvalue Metrics
```python
'eigenvalue_errors'              # (k,) - true - sample
'eigenvalue_relative_errors'     # (k,) - (true - sample) / true
'eigenvalue_rmse'                # float - Root mean squared error
'eigenvalue_max_error'           # float - Maximum absolute error
'eigenvalue_mean_relative_error' # float - Mean of absolute relative errors
```

### 6. Eigenvector Metrics
```python
'subspace_distance'        # float - L2 norm of principal angles
'procrustes_distance'      # float - After optimal rotation
'vector_correlations'      # (k,) - Per-vector canonical correlations
'mean_correlation'         # float - Mean of vector correlations
'min_correlation'          # float - Minimum correlation
'max_correlation'          # float - Maximum correlation
'grassmann_distance'       # float - Alternative subspace metric
'max_principal_angle'      # float - Largest principal angle
```

### 7. Optional (if computed)
```python
'optimal_rotation'         # (k, k) - Optimal rotation matrix
'aligned_eigenvectors'     # (k, p) - After Procrustes alignment
'aligned_frame'            # (p, k) - Aligned loading frame
```

---

## Example: Loading and Using NPZ Files

### Basic Loading
```python
import numpy as np

# Load file
data = np.load('simulation_gaussian.npz')

# Access arrays
true_B = data['true_B']
sample_B = data['sample_B']
returns = data['security_returns']

# Access scalars
grass_dist = float(data['dist_grassmannian'])
eigen_rmse = float(data['eigenvalue_rmse'])

print(f"Grassmannian distance: {grass_dist:.6f}")
print(f"Eigenvalue RMSE: {eigen_rmse:.6f}")
```

### Inspect File Contents
```python
# List all keys
print("Available arrays:", data.files)

# Check shapes
for key in data.files:
    val = data[key]
    if hasattr(val, 'shape'):
        print(f"{key:30s} {val.shape}")
    else:
        print(f"{key:30s} scalar: {float(val):.6f}")
```

### Using the Utility Script
```bash
# Inspect one file
python inspect_npz.py simulation_gaussian.npz

# Inspect all simulation files
python inspect_npz.py simulation_*.npz
```

---

## Example Output from inspect_npz.py

```
======================================================================
  File: simulation_gaussian.npz
======================================================================

📊 Contains 23 arrays:

  Raw Data:
    • security_returns             shape: (63, 500)       dtype: float64

  True Model:
    • true_B                       shape: (2, 500)        dtype: float64
    • true_D                       shape: (500, 500)      dtype: float64
    • true_F                       shape: (2, 2)          dtype: float64
    • true_eigenvalues             shape: (2,)            dtype: float64
    • true_eigenvectors            shape: (2, 500)        dtype: float64
    • true_ortho_B                 shape: (2, 500)        dtype: float64

  Sample Model:
    • sample_B                     shape: (2, 500)        dtype: float64
    • sample_D                     shape: (500, 500)      dtype: float64
    • sample_F                     shape: (2, 2)          dtype: float64
    • sample_eigenvalues           shape: (2,)            dtype: float64
    • sample_eigenvectors          shape: (2, 500)        dtype: float64

  Manifold Distances:
    • dist_grassmannian            scalar: 0.123456
    • dist_stiefel_chordal         scalar: 0.234567
    • dist_stiefel_procrustes      scalar: 0.012345
    • principal_angles             shape: (2,)            dtype: float64

  Eigenvalue Metrics:
    • eigenvalue_errors            shape: (2,)            dtype: float64
    • eigenvalue_max_error         scalar: 0.001234
    • eigenvalue_mean_relative_error scalar: 0.012345
    • eigenvalue_relative_errors   shape: (2,)            dtype: float64
    • eigenvalue_rmse              scalar: 0.001234

  Eigenvector Metrics:
    • grassmann_distance           scalar: 0.123456
    • max_correlation              scalar: 0.987654
    • max_principal_angle          scalar: 0.123456
    • mean_correlation             scalar: 0.912345
    • min_correlation              scalar: 0.876543
    • procrustes_distance          scalar: 0.012345
    • subspace_distance            scalar: 0.123456
    • vector_correlations          shape: (2,)            dtype: float64

======================================================================
  Summary
======================================================================

  Grassmannian Distance:  0.123456
  Procrustes Distance:    0.012345
  Chordal Distance:       0.234567

  Eigenvalue RMSE:        0.001234

  Mean Eigenvector Corr:  0.9123

  Model Dimensions:
    k (factors): 2
    p (assets):  500
    T (periods): 63
```

---

## Comparing to Gemini's Original Format

### ✅ Matches Gemini's Keys
```python
# Original Gemini keys (all included!)
[
    'security_returns',           # ✓
    'true_B',                     # ✓
    'true_ortho_B',              # ✓ Added!
    'true_F',                     # ✓
    'true_D',                     # ✓
    'true_eigenvalues',          # ✓ Added!
    'true_eigenvectors',         # ✓ Added!
    'sample_B',                   # ✓
    'sample_F',                   # ✓
    'sample_D',                   # ✓
    'sample_eigenvalues',        # ✓ Added!
    'sample_eigenvectors',       # ✓ Added!
    'dist_grassmannian',         # ✓
    'dist_stiefel_chordal',      # ✓ (our 'dist_chordal')
    'dist_stiefel_procrustes',   # ✓ (our 'dist_procrustes')
    'principal_angles'           # ✓
]
```

### ✅ Additional Metrics
We also include many additional useful metrics:
- Eigenvalue errors and RMSE
- Eigenvector correlations
- Procrustes rotation matrices
- More!

---

## Workflow Example

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Load simulation results
data = np.load('simulation_gaussian.npz')

# 2. Extract data
true_B = data['true_B']
true_ortho_B = data['true_ortho_B']
sample_B = data['sample_B']
returns = data['security_returns']

# 3. Compute additional metrics
loading_error = np.linalg.norm(true_B - sample_B, 'fro')
print(f"Loading Frobenius error: {loading_error:.4f}")

# 4. Visualize
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(true_B, aspect='auto', cmap='RdBu_r')
plt.title('True Loadings')
plt.colorbar()

plt.subplot(132)
plt.imshow(sample_B, aspect='auto', cmap='RdBu_r')
plt.title('Sample Loadings')
plt.colorbar()

plt.subplot(133)
plt.imshow(true_B - sample_B, aspect='auto', cmap='RdBu_r')
plt.title('Difference')
plt.colorbar()

plt.tight_layout()
plt.savefig('loading_comparison.png')

# 5. Check manifold distances
print(f"\nManifold Distances:")
print(f"  Grassmannian: {data['dist_grassmannian']:.6f}")
print(f"  Procrustes:   {data['dist_stiefel_procrustes']:.6f}")
print(f"  Chordal:      {data['dist_stiefel_chordal']:.6f}")

# 6. Check eigenvalue accuracy
print(f"\nEigenvalue Analysis:")
print(f"  RMSE: {data['eigenvalue_rmse']:.6f}")
for i, (true_ev, sample_ev) in enumerate(zip(data['true_eigenvalues'], 
                                              data['sample_eigenvalues'])):
    print(f"  λ_{i+1}: true={true_ev:.6f}, sample={sample_ev:.6f}")

# 7. Check eigenvector alignment
print(f"\nEigenvector Correlations:")
for i, corr in enumerate(data['vector_correlations']):
    print(f"  Vector {i+1}: {corr:.4f}")
print(f"  Mean: {data['mean_correlation']:.4f}")

data.close()
```

---

## File Sizes

Typical file sizes:
- `simulation_*.npz`: 5-50 MB (depending on p, k, T)
- `factor_model.npz`: <1 MB

The files are compressed with `np.savez_compressed`, so actual sizes are much smaller than the raw data.

---

## Best Practices

1. **Always close files** after loading:
   ```python
   data = np.load('file.npz')
   # ... use data ...
   data.close()
   ```

2. **Use context manager**:
   ```python
   with np.load('file.npz') as data:
       B = data['true_B']
       # ... use data ...
   # Automatically closed
   ```

3. **Check what's available**:
   ```python
   data = np.load('file.npz')
   print("Available:", sorted(data.files))
   ```

4. **Use inspect_npz.py** for quick inspection:
   ```bash
   python inspect_npz.py simulation_*.npz
   ```

---

## Summary

The enhanced `build_and_simulate.py` now saves **comprehensive NPZ files** with:
- ✅ All of Gemini's original keys
- ✅ Additional eigenvalue/eigenvector data
- ✅ Complete analysis results
- ✅ Organized, well-documented format
- ✅ Easy to load and use
- ✅ Utility script for inspection

**Run and you'll get everything you need!** 🎊
