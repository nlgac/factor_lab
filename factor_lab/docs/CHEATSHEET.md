# Factor Lab Manifold Complete - Quick Reference Cheatsheet

**Version**: 2.2.0 | **Python**: 3.8+ | **Tests**: 50/50 ✓

One-page reference for common tasks, patterns, and quick solutions.

---

## 🚀 Installation & Setup

```bash
# Extract package
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete

# Install dependencies
pip install -r requirements.txt

# Quick test
python demo.py

# Run tests
pytest tests/ -v  # Expect: 50 passed
```

---

## 📦 Essential Imports

```python
# Core functionality
from factor_lab import (
    FactorModelData,           # Factor model container
    svd_decomposition,         # Extract factors from returns
    ReturnsSimulator,          # Simulate returns
    DistributionFactory,       # Custom distributions
    save_model                 # Save/load models
)

# Analysis framework
from factor_lab.analysis import SimulationContext  # Context for analyses

# Built-in analyses
from factor_lab.analyses import Analyses  # Factory for analyses

# Visualization
from factor_lab.visualization import (
    create_manifold_dashboard,           # Static plots
    create_interactive_plotly_dashboard, # Interactive HTML
    print_verbose_results                # Console output
)
```

---

## ⚡ Quick Start Patterns

### Pattern 1: Complete Workflow (5 lines)

```python
model = FactorModelData(B=B, F=F, D=D)  # 1. Create model
results = ReturnsSimulator(model).simulate(n_periods=500)  # 2. Simulate
context = SimulationContext(model, **results)  # 3. Create context
analysis = Analyses.manifold_distances().analyze(context)  # 4. Analyze
print(f"Distance: {analysis['dist_grassmannian']:.4f}")  # 5. Report
```

### Pattern 2: Extract & Compare (3 lines)

```python
extracted = svd_decomposition(returns, k=3)  # Extract factors
context = SimulationContext(true_model, **sim_results)  # Create context
dist = Analyses.manifold_distances().analyze(context)['dist_grassmannian']
```

### Pattern 3: Full Analysis Suite

```python
# Run all analyses
manifold = Analyses.manifold_distances().analyze(context)
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)
eigvec = Analyses.eigenvector_comparison(k_components=5).analyze(context)

# Combine and visualize
all_results = {**manifold, **eigen, **eigvec}
create_manifold_dashboard(all_results, "output.png", context=context)
```

---

## 📊 Analysis Cheat Sheet

| Analysis | Command | Key Metrics |
|----------|---------|-------------|
| **Manifold** | `Analyses.manifold_distances()` | `dist_grassmannian`, `dist_procrustes` |
| **Eigenvalues** | `Analyses.eigenvalue_analysis(k_top=10)` | `eigenvalue_rmse`, `eigenvalue_errors` |
| **Eigenvectors** | `Analyses.eigenvector_comparison(k=5)` | `mean_correlation`, `subspace_distance` |
| **Custom** | `Analyses.custom(lambda ctx: {...})` | Your metrics |

---

## 🎯 Common Tasks

### Task: Build Factor Model

```python
k, p = 3, 100  # 3 factors, 100 assets
B = np.random.randn(k, p)
F = np.diag([0.09, 0.04, 0.01])  # Factor variances
D = np.diag(np.full(p, 0.01))    # Idio variances
model = FactorModelData(B=B, F=F, D=D)
```

### Task: Simulate Returns (Reproducible)

```python
simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = simulator.simulate(n_periods=500)

security_returns = results['security_returns']  # (T, p)
factor_returns = results['factor_returns']      # (T, k)
idio_returns = results['idio_returns']          # (T, p)
```

### Task: Extract Factors from Data

```python
# From returns matrix (T, p)
extracted_model = svd_decomposition(
    returns,
    k=3,         # Number of factors
    center=True  # Demean returns
)
```

### Task: Compare Factor Models

```python
# Setup context with ground truth
context = SimulationContext(
    model=true_model,  # Ground truth
    security_returns=simulated_returns,
    factor_returns=factor_returns,
    idio_returns=idio_returns
)

# Compare using manifold geometry
manifold = Analyses.manifold_distances().analyze(context)

print(f"Subspace match: {manifold['dist_grassmannian']:.4f}")
# < 0.1 = Excellent, < 0.3 = Good, ≥ 0.3 = Poor
```

### Task: Check Eigenvalue Accuracy

```python
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)

print(f"RMSE: {eigen['eigenvalue_rmse']:.4f}")
print(f"Errors: {eigen['eigenvalue_errors']}")
print(f"Relative: {eigen['eigenvalue_rel_errors']}")

# Plot spectrum
import matplotlib.pyplot as plt
plt.plot(eigen['true_eigenvalues'], 'o-', label='True')
plt.plot(eigen['sample_eigenvalues'], 's-', label='Sample')
plt.legend()
```

### Task: Eigenvector Comparison

```python
eigvec = Analyses.eigenvector_comparison(
    k_components=5,
    align_signs=True  # Handle sign ambiguity
).analyze(context)

print(f"Mean |correlation|: {eigvec['mean_correlation']:.4f}")
print(f"Per-component: {eigvec['aligned_correlations']}")
```

### Task: Custom Analysis

```python
# Quick lambda
custom = Analyses.custom(lambda ctx: {
    'total_var': float(np.trace(ctx.model.implied_covariance())),
    'frobenius_B': float(np.linalg.norm(ctx.model.B, 'fro'))
})
results = custom.analyze(context)

# Or full class for reusability
from factor_lab.analysis import SimulationAnalysis

class MyAnalysis(SimulationAnalysis):
    def analyze(self, context):
        return {'my_metric': compute_metric(context)}
```

### Task: Create Visualization

```python
# Static dashboard (publication quality)
create_manifold_dashboard(
    all_results,
    output_path="dashboard.png",
    figsize=(20, 14),
    context=context
)

# Interactive dashboard (for exploration)
create_interactive_plotly_dashboard(
    all_results,
    output_path="dashboard.html",
    context=context
)

# Console output (formatted)
print_verbose_results(all_results, context)
```

### Task: Monte Carlo Study

```python
grass_distances = []
for seed in range(100):
    simulator = ReturnsSimulator(model, rng=np.random.default_rng(seed))
    results = simulator.simulate(n_periods=500)
    context = SimulationContext(model, **results)
    manifold = Analyses.manifold_distances().analyze(context)
    grass_distances.append(manifold['dist_grassmannian'])

print(f"Mean: {np.mean(grass_distances):.4f}")
print(f"Std:  {np.std(grass_distances):.4f}")
```

### Task: Save/Load Model

```python
# Save
from factor_lab import save_model
save_model(model, 'my_model.npz')

# Load
data = np.load('my_model.npz')
loaded_model = FactorModelData(
    B=data['B'],
    F=data['F'],
    D=data['D']
)
```

---

## 🔬 Manifold Metrics Guide

### Grassmannian Distance

```python
dist_grassmannian = manifold['dist_grassmannian']

# Interpretation:
if dist_grassmannian < 0.1:
    print("✓ Excellent subspace recovery")
elif dist_grassmannian < 0.3:
    print("○ Good subspace recovery")
else:
    print("✗ Poor - need more data or better method")
```

**Properties**:
- ✓ Rotation invariant: d(U, RU) = 0
- ✓ Sign invariant: d(U, -U) = 0
- ✓ Permutation invariant
- Range: [0, π√k/2]

### Procrustes Distance

```python
dist_procrustes = manifold['dist_procrustes']
optimal_rotation = manifold['optimal_rotation']

# After optimal alignment
aligned_B = pca_B @ optimal_rotation
```

**Properties**:
- ✓ Finds optimal rotation
- ✓ Handles sign flips
- Lower bound: dist_procrustes ≤ dist_chordal

### Chordal Distance

```python
dist_chordal = manifold['dist_chordal']
```

**Properties**:
- ✗ NOT rotation invariant
- Useful for debugging
- Upper bound for Procrustes

---

## 📐 Mathematical Quick Reference

### Factor Model
```
r_t = B' f_t + ε_t
E[r_t] = 0
Cov(r) = Σ = B' F B + D
```

### Grassmannian Distance
```
d_G(U, V) = ||θ||_2
where θ are principal angles
```

### Procrustes Problem
```
min_R ||A - B R||_F
s.t. R' R = I

Solution: R_opt = U V' where U S V' = svd(B' A)
```

### Eigenvalue via LinearOperator
```
Instead of: Σ = B' F B + D  (O(p²) memory)
Use: matvec(v) = B'(F(Bv)) + Dv  (O(kp) memory)
```

---

## ⚡ Performance Tips

### Memory Optimization

```python
# ✓ GOOD: Uses O(kp) memory
eigen = Analyses.eigenvalue_analysis(k_top=10)

# ✗ BAD: Would use O(p²) memory
# Sigma = model.B.T @ model.F @ model.B + model.D  # Don't do this for large p!
```

**Memory Savings**:
- p=1,000, k=10: **100× reduction** (8 MB → 80 KB)
- p=10,000, k=10: **1000× reduction** (800 MB → 800 KB)

### Speed Optimization

```python
# Skip expensive eigenvector comparison if not needed
eigen = Analyses.eigenvalue_analysis(
    k_top=10,
    compare_eigenvectors=False  # Faster
)

# Use cached PCA decomposition
pca = context.pca_decomposition(k=3)  # Computed once
pca_again = context.pca_decomposition(k=3)  # From cache
```

### Recommended Problem Sizes

| Size | k | p | T | Time | Memory |
|------|---|-----|---|------|--------|
| Small | 2-3 | 20-100 | 100-500 | <1s | <10MB |
| Medium | 3-5 | 100-1000 | 500-2000 | ~5s | ~50MB |
| Large | 5-10 | 1000-5000 | 1000-5000 | ~60s | ~500MB |
| XLarge | 10-20 | 5000+ | 5000+ | ~10min | ~5GB |

---

## 🎨 Customization Recipes

### Recipe: Custom Color Palette

```python
import seaborn as sns
sns.set_palette("husl")  # Or "Set2", "colorblind", etc.
create_manifold_dashboard(results, "custom_colors.png")
```

### Recipe: High-Resolution Output

```python
create_manifold_dashboard(
    results,
    output_path="high_res.png",
    figsize=(24, 18),  # Larger figure
    context=context
)
# Then save with high DPI in your code
# plt.savefig(..., dpi=300)
```

### Recipe: Custom Metric

```python
def correlation_with_ewm(ctx):
    """Correlation with equal-weight market."""
    R = ctx.security_returns
    mkt = R.mean(axis=1)  # Equal-weight market
    cors = [np.corrcoef(R[:, i], mkt)[0, 1] for i in range(ctx.p)]
    return {
        'mean_beta': float(np.mean(cors)),
        'beta_dispersion': float(np.std(cors))
    }

custom = Analyses.custom(correlation_with_ewm)
results = custom.analyze(context)
```

---

## 🧪 Testing & Debugging

### Run Tests

```bash
# All tests (50 should pass)
pytest tests/ -v

# Specific suite
pytest tests/analysis/test_manifold.py -v

# With coverage
pytest tests/ --cov=factor_lab --cov-report=html

# Fast (no slow tests)
pytest tests/ -m "not slow"
```

### Debug Analysis

```python
# Check inputs
print(f"Model: k={context.k}, p={context.p}, T={context.T}")
print(f"B shape: {context.model.B.shape}")
print(f"Returns shape: {context.security_returns.shape}")

# Verify model structure
Sigma = context.model.implied_covariance()
print(f"Condition number: {np.linalg.cond(Sigma):.2e}")

# Check sample covariance
Sigma_sample = context.sample_covariance()
print(f"Sample rank: {np.linalg.matrix_rank(Sigma_sample)}")

# Verify PCA
pca = context.pca_decomposition(k=3)
print(f"PCA explained var: {np.trace(pca.B.T @ pca.F @ pca.B):.4f}")
```

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| ARPACK not converged | Too few iterations | Increase `maxiter` or loosen `tol` |
| Memory error | Problem too large | Reduce p or use smaller k_top |
| Poor recovery | T too small | Use T/p > 2, ideally T/p > 5 |
| Import error | Wrong directory | Add to path or `pip install -e .` |

---

## 📊 Interpretation Guidelines

### Grassmannian Distance

| Value | Interpretation | Confidence Level |
|-------|----------------|------------------|
| < 0.05 | Excellent | Use with confidence |
| 0.05-0.1 | Very Good | Proceed |
| 0.1-0.2 | Good | Acceptable |
| 0.2-0.3 | Moderate | Check assumptions |
| > 0.3 | Poor | Need more data |

### Eigenvalue RMSE

Depends on scale of eigenvalues. Check relative errors:

```python
rel_errors = eigen['eigenvalue_rel_errors']
max_rel_error = np.max(np.abs(rel_errors))

if max_rel_error < 0.05:
    print("✓ Excellent eigenvalue recovery")
elif max_rel_error < 0.10:
    print("○ Good eigenvalue recovery")
else:
    print("✗ Poor eigenvalue recovery")
```

### Eigenvector Correlation

```python
mean_corr = eigvec['mean_correlation']

if mean_corr > 0.95:
    print("✓ Excellent eigenvector recovery")
elif mean_corr > 0.85:
    print("○ Good eigenvector recovery")
else:
    print("✗ Poor eigenvector recovery")
```

---

## 🎯 One-Liners

```python
# Quick Grassmannian distance
Analyses.manifold_distances().analyze(context)['dist_grassmannian']

# Quick eigenvalue RMSE
Analyses.eigenvalue_analysis(k_top=5).analyze(context)['eigenvalue_rmse']

# Quick eigenvector correlation
Analyses.eigenvector_comparison(k=3).analyze(context)['mean_correlation']

# Complete analysis
{**Analyses.manifold_distances().analyze(context),
 **Analyses.eigenvalue_analysis(k_top=10).analyze(context),
 **Analyses.eigenvector_comparison(k=5).analyze(context)}

# Extract top 5 factors
svd_decomposition(returns, k=5)

# Simulate 1000 periods
ReturnsSimulator(model).simulate(n_periods=1000)['security_returns']
```

---

## 🔗 File Locations

### Inputs
```
model_spec.json          # Model configuration
perturbation_spec.json   # Perturbation configuration
your_returns.csv         # Your data
```

### Outputs
```
demo_output/                    # Demo outputs
    demo_dashboard.png          # Static visualization
    demo_interactive.html       # Interactive dashboard

output/                         # Build & simulate outputs
    *.png                       # Static dashboards
    *.html                      # Interactive dashboards
    *.npz                       # Saved results
```

---

## 💻 Command-Line Usage

```bash
# Run demo
python demo.py

# Run perturbation study
python perturbation_study.py perturbation_spec.json

# Run build & simulate (if available)
python build_and_simulate.py model_spec.json --seed 42

# Run tests
pytest tests/ -v

# Generate coverage report
pytest tests/ --cov=factor_lab --cov-report=html
open htmlcov/index.html  # View in browser
```

---

## 📚 Documentation Quick Links

| Topic | File | Section |
|-------|------|---------|
| **Overview** | `README.md` | Quick start, features |
| **API Reference** | `docs/API.md` | All functions, detailed |
| **This File** | `docs/CHEATSHEET.md` | Quick patterns |
| **Math Details** | `docs/TECHNICAL_MANUAL.md` | Theory, proofs |
| **High-Dim Theory** | `docs/APPENDIX_HIGH_DIMENSIONAL.md` | Asymptotics |
| **Code Examples** | `demo.py`, `examples/` | Working code |

---

## 🎓 Best Practices Checklist

### Before Running
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Understand your data (check T, p, k)
- [ ] Set random seed for reproducibility

### During Analysis
- [ ] Use T/p > 2 minimum (T/p > 5 better)
- [ ] Check multiple metrics (not just one)
- [ ] Visualize results (don't just look at numbers)
- [ ] Validate on multiple simulations

### After Analysis
- [ ] Save outputs with timestamps
- [ ] Document random seeds used
- [ ] Keep visualizations for reference
- [ ] Note package version used

---

## ⌨️ Interactive Python Session

```python
# Start interactive analysis
python -i demo.py

# Now you have: model, context, manifold, eigen, eigvec
>>> manifold['dist_grassmannian']
0.2134
>>> eigen['eigenvalue_rmse']
0.0234
>>> context.T, context.p, context.k
(500, 100, 3)

# Try your own analysis
>>> custom_result = Analyses.custom(lambda ctx: {
...     'my_metric': float(np.trace(ctx.model.F))
... }).analyze(context)
>>> custom_result
{'my_metric': 0.14}
```

---

## 🚦 Status Indicators

In verbose output:
- **✓** = Excellent (>90% quality)
- **○** = Good (70-90% quality)
- **✗** = Needs attention (<70% quality)

---

## 🆘 Emergency Debugging

```python
# Check everything
print("=" * 60)
print(f"Model: k={context.k}, p={context.p}, T={context.T}")
print(f"T/p ratio: {context.T/context.p:.2f}")
print(f"Returns range: [{context.security_returns.min():.2f}, "
      f"{context.security_returns.max():.2f}]")
print(f"Returns mean: {context.security_returns.mean():.4f}")
print(f"Returns std: {context.security_returns.std():.4f}")
print(f"Model total var: {np.trace(context.model.implied_covariance()):.4f}")
print(f"Sample total var: {np.trace(context.sample_covariance()):.4f}")
print("=" * 60)

# Test basic operations
try:
    pca = context.pca_decomposition(k=2)
    print("✓ PCA works")
except Exception as e:
    print(f"✗ PCA failed: {e}")

try:
    manifold = Analyses.manifold_distances().analyze(context)
    print("✓ Manifold analysis works")
except Exception as e:
    print(f"✗ Manifold failed: {e}")
```

---

## 📝 Pro Tips

1. **Always set random seed** for reproducibility:
   ```python
   rng = np.random.default_rng(42)
   ```

2. **Use T/p > 2** minimum, T/p > 5 for reliable estimates

3. **Check multiple metrics** - don't rely on just one number

4. **Cache is your friend** - context automatically caches PCA results

5. **LinearOperator** is automatic - you get O(kp) memory automatically

6. **Plotly is optional** - static plots work fine without it

7. **Test on small problems first** before scaling up

8. **Save your visualizations** - they're useful later

9. **Version your results** - note which package version you used

10. **Read the docstrings** - `help(function)` has great info

---

## 🎬 Complete Example

```python
#!/usr/bin/env python3
"""Complete minimal example."""
import numpy as np
from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import print_verbose_results

# Setup
k, p, T = 3, 50, 200
B = np.random.randn(k, p)
F = np.diag([0.16, 0.09, 0.04])
D = np.diag(np.full(p, 0.01))
model = FactorModelData(B=B, F=F, D=D)

# Simulate
sim = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = sim.simulate(n_periods=T)

# Analyze
context = SimulationContext(model, **results)
manifold = Analyses.manifold_distances().analyze(context)
eigen = Analyses.eigenvalue_analysis(k_top=k).analyze(context)
eigvec = Analyses.eigenvector_comparison(k_components=k).analyze(context)

# Report
all_results = {**manifold, **eigen, **eigvec}
print_verbose_results(all_results, context)

# Done!
print("\n✓ Analysis complete!")
```

---

**Need more help?**
- Full docs: `docs/API.md`
- Math details: `docs/TECHNICAL_MANUAL.md`
- Examples: `demo.py`, `examples/`
- Tests: `tests/` (see usage patterns)

**Version**: 2.2.0 | **Status**: Production Ready | **Tests**: 50/50 ✓
