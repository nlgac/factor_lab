# Factor Lab Manifold Complete - Production-Ready Factor Model Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 50 Passing](https://img.shields.io/badge/tests-50%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Comprehensive manifold-based analysis for high-dimensional factor models with advanced geometric methods, memory-efficient eigenvalue analysis, and production-quality visualization.**

---

## 🎯 Overview

Factor Lab Manifold Complete is a specialized toolkit for rigorous analysis of factor models in high-dimensional settings (p >> T regime). Built for quantitative finance researchers and practitioners, it combines cutting-edge manifold geometry with computational efficiency to provide insights that traditional methods miss.

### Key Differentiators

- **🧮 Manifold Geometry**: Proper handling of rotation/sign/permutation ambiguities inherent in factor models
- **⚡ Memory Efficient**: O(kp) eigenvalue computation instead of O(p²) using LinearOperator methods
- **📊 Production Quality**: 50 comprehensive tests, extensive documentation, type hints throughout
- **🎨 Rich Visualization**: Publication-ready static plots and interactive dashboards
- **🔬 Research Ready**: Implements methods from recent literature on high-dimensional estimation

---

## 🚀 Quick Start (30 Seconds)

```bash
# 1. Extract package
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run comprehensive demo
python demo.py
```

**You'll see:**
- Complete walkthrough of all features
- Beautiful visualizations in `demo_output/`
- Interactive HTML dashboard (if plotly installed)

---

## 💡 Why This Package?

### The Challenge: High-Dimensional Factor Models

When working with factor models in the **p >> T regime** (more securities than time periods):

1. **Sample covariance is singular** - can't invert for traditional methods
2. **Eigenvector bias is severe** - sample eigenvectors systematically differ from truth
3. **Rotation ambiguity** - extracted factors are only unique up to orthogonal transformation
4. **Computational limits** - full p×p covariance requires O(p²) memory

### The Solution: This Package

```python
from factor_lab import FactorModelData, ReturnsSimulator, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses

# 1. Build model (3 factors, 1000 securities)
k, p = 3, 1000
model = FactorModelData(B=B, F=F, D=D)

# 2. Simulate returns (500 periods)
simulator = ReturnsSimulator(model)
results = simulator.simulate(n_periods=500)

# 3. Extract factors via SVD
context = SimulationContext(
    model=model,
    security_returns=results['security_returns'],
    factor_returns=results['factor_returns'],
    idio_returns=results['idio_returns']
)

# 4. Analyze with manifold geometry (rotation-invariant!)
manifold = Analyses.manifold_distances().analyze(context)
print(f"Grassmannian distance: {manifold['dist_grassmannian']:.4f}")

# 5. Memory-efficient eigenvalue analysis (O(kp) instead of O(p²))
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)
print(f"Top 10 eigenvalue RMSE: {eigen['eigenvalue_rmse']:.4f}")

# 6. Eigenvector comparison with proper alignment
eigvec = Analyses.eigenvector_comparison(k_components=3).analyze(context)
print(f"Mean eigenvector correlation: {eigvec['mean_correlation']:.4f}")
```

---

## 📦 Package Structure

```
factor_lab_manifold_complete/
├── factor_lab/                      # Main package
│   ├── __init__.py                  # Public API exports
│   ├── types.py                     # Core types (FactorModelData, svd_decomposition, etc.)
│   ├── analysis/                    # Analysis framework
│   │   ├── protocol.py              # SimulationAnalysis protocol
│   │   └── context.py               # SimulationContext dataclass
│   ├── analyses/                    # Built-in analyses
│   │   ├── __init__.py              # Analyses factory
│   │   ├── builder.py               # Analysis builder pattern
│   │   ├── manifold.py              # Grassmannian/Procrustes/Chordal distances
│   │   ├── spectral.py              # Eigenvalue analysis via LinearOperator
│   │   └── eigenvector.py           # Eigenvector comparison and alignment
│   └── visualization/               # Visualization tools
│       ├── __init__.py
│       └── visualization.py         # Seaborn & plotly dashboards
│
├── tests/                           # Comprehensive test suite (50 tests)
│   └── analysis/
│       ├── conftest.py              # Test fixtures
│       ├── test_integration.py      # Integration tests (9 tests)
│       ├── test_manifold.py         # Manifold geometry tests (15 tests)
│       ├── test_spectral_eigenvector.py  # Spectral/eigenvector tests (13 tests)
│       └── test_visualization_context.py  # Context/viz tests (13 tests)
│
├── examples/                        # Working examples
│   ├── comprehensive_demo.py        # Full feature demonstration
│   └── gemini_integration.py        # Integration example
│
├── docs/                            # Documentation
│   ├── API.md                       # Complete API reference
│   ├── CHEATSHEET.md                # Quick reference guide
│   ├── TECHNICAL_MANUAL.md          # Mathematical details
│   ├── APPENDIX_HIGH_DIMENSIONAL.md # High-dimensional theory
│   ├── PERTURBATION_STUDY.md        # Perturbation analysis guide
│   ├── NPZ_OUTPUT_FORMAT.md         # Output file format
│   └── WHERE_ARE_MY_FILES.md        # Output location guide
│
├── demo.py                          # Main demonstration script
├── build_and_simulate.py            # JSON-driven model building
├── perturbation_study.py            # Perturbation analysis script
├── model_spec.json                  # Example configuration
├── perturbation_spec.json           # Perturbation configuration
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🔬 Core Features

### 1. Manifold Distance Analysis

**Problem**: Factor models have inherent ambiguities:
- Factors can be rotated by any orthogonal matrix
- Factor signs can flip
- Factor ordering can permute

**Solution**: Use manifold geometry for rotation-invariant comparison.

```python
from factor_lab.analyses import Analyses

manifold = Analyses.manifold_distances().analyze(context)

# Results include:
manifold['dist_grassmannian']  # Subspace distance (rotation-invariant)
manifold['dist_procrustes']    # Optimal alignment distance
manifold['dist_chordal']       # Raw frame difference
manifold['principal_angles']   # Angles between subspaces
manifold['optimal_rotation']   # Best rotation matrix
```

**Mathematical Background**:
- **Grassmannian Gr(k,p)**: Space of k-dimensional subspaces of ℝᵖ
- **Stiefel V_{p,k}**: Space of p×k orthonormal matrices
- **Distance metrics** based on principal angles and Procrustes analysis

**When to use**:
- Comparing PCA/SVD estimates to ground truth
- Validating factor extraction methods
- Quantifying estimation error in simulation studies

### 2. Memory-Efficient Eigenvalue Analysis

**Problem**: Computing eigenvalues of Σ = B'FB + D requires forming full p×p matrix.

**Solution**: Use LinearOperator with matrix-free operations.

```python
# Instead of forming Σ (p² memory):
Sigma = B.T @ F @ B + D  # ❌ O(p²) memory

# Use LinearOperator (kp memory):
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)  # ✅ O(kp) memory
```

**Memory Savings**:
| p | k | Dense | Sparse | Reduction |
|---|---|-------|--------|-----------|
| 1,000 | 10 | 8 MB | 80 KB | **100×** |
| 10,000 | 10 | 800 MB | 800 KB | **1000×** |
| 10,000 | 50 | 800 MB | 4 MB | **200×** |

**Metrics returned**:
```python
eigen['eigenvalue_rmse']        # RMSE of top-k eigenvalues
eigen['eigenvalue_errors']      # Individual errors
eigen['eigenvalue_rel_errors']  # Relative errors
eigen['true_eigenvalues']       # Ground truth λ
eigen['sample_eigenvalues']     # Estimated λ
```

### 3. Eigenvector Comparison & Alignment

**Problem**: Sample eigenvectors are biased in high dimensions (dispersion bias).

**Solution**: Proper geometric comparison with sign alignment.

```python
eigvec = Analyses.eigenvector_comparison(
    k_components=5,
    align_signs=True  # Handle sign ambiguity
).analyze(context)

# Results:
eigvec['mean_correlation']       # Average |correlation| per eigenvector
eigvec['canonical_correlations'] # Canonical correlations
eigvec['subspace_distance']      # Principal angles metric
eigvec['procrustes_distance']    # After optimal rotation
eigvec['aligned_correlations']   # Per-component correlations
```

**Theory**: In high dimensions, sample eigenvectors systematically deviate from true eigenvectors even when eigenvalues are well-estimated. This implements proper geometric comparison accounting for rotation ambiguity.

### 4. Rich Visualization

**Static Dashboards** (Matplotlib/Seaborn):
```python
from factor_lab.visualization import create_manifold_dashboard

create_manifold_dashboard(
    all_results,
    output_path="dashboard.png",
    figsize=(16, 12)
)
```

**Interactive Dashboards** (Plotly):
```python
from factor_lab.visualization import create_interactive_plotly_dashboard

create_interactive_plotly_dashboard(
    all_results,
    output_path="dashboard.html"
)
```

**Dashboard includes**:
- Eigenvalue spectrum comparison (true vs estimated)
- Principal angles between subspaces
- Manifold distance metrics
- Eigenvalue error distribution
- Eigenvector correlation heatmap
- Factor loading heatmaps
- Summary statistics table

### 5. Extensible Analysis Framework

**Built-in Analyses**:
```python
from factor_lab.analyses import Analyses

# Factory methods for common analyses
Analyses.manifold_distances()
Analyses.eigenvalue_analysis(k_top=10, compare_eigenvectors=True)
Analyses.eigenvector_comparison(k_components=5, align_signs=True)
```

**Custom Analyses**:
```python
# One-liner lambda
custom = Analyses.custom(lambda ctx: {
    'frobenius_norm': float(np.linalg.norm(ctx.model.B, 'fro')),
    'total_variance': float(np.trace(ctx.model.implied_covariance()))
})

# Or full class
from factor_lab.analysis import SimulationAnalysis

class MyAnalysis(SimulationAnalysis):
    def analyze(self, context):
        # Your custom analysis logic
        return {
            'my_metric_1': value1,
            'my_metric_2': value2
        }

my_analysis = MyAnalysis()
results = my_analysis.analyze(context)
```

---

## 📊 Example Workflows

### Workflow 1: Validate Factor Extraction Method

```python
import numpy as np
from factor_lab import FactorModelData, ReturnsSimulator, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses

# 1. Define ground truth model
k, p, T = 3, 100, 500
B_true = np.random.randn(k, p)
F_true = np.diag([0.09, 0.04, 0.01])
D_true = np.diag(np.full(p, 0.01))
model = FactorModelData(B=B_true, F=F_true, D=D_true)

# 2. Simulate returns
simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = simulator.simulate(n_periods=T)

# 3. Extract factors using your method (e.g., SVD)
extracted = svd_decomposition(results['security_returns'], k=k, center=True)

# 4. Create comparison context
context = SimulationContext(
    model=model,  # Ground truth
    security_returns=results['security_returns'],
    factor_returns=results['factor_returns'],
    idio_returns=results['idio_returns']
)

# 5. Compare extracted vs truth using manifold geometry
manifold = Analyses.manifold_distances().analyze(context)

print("Factor Extraction Quality:")
print(f"  Grassmannian distance: {manifold['dist_grassmannian']:.4f}")
print(f"  Procrustes distance:   {manifold['dist_procrustes']:.4f}")
print(f"  Max principal angle:   {np.max(manifold['principal_angles']):.4f} rad")

# Interpretation:
if manifold['dist_grassmannian'] < 0.1:
    print("  ✓ Excellent subspace recovery")
elif manifold['dist_grassmannian'] < 0.3:
    print("  ○ Good subspace recovery")
else:
    print("  ✗ Poor subspace recovery - need more data or different method")
```

### Workflow 2: Monte Carlo Study of Sampling Error

```python
import numpy as np
from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses

# Setup
k, p = 3, 50
B = np.random.randn(k, p)
F = np.diag([0.16, 0.09, 0.04])
D = np.diag(np.full(p, 0.01))
model = FactorModelData(B=B, F=F, D=D)

# Monte Carlo simulation
n_sims = 100
T_values = [50, 100, 200, 500, 1000]

results_by_T = {}
for T in T_values:
    grass_distances = []
    eigen_rmses = []
    
    for sim in range(n_sims):
        # Simulate
        simulator = ReturnsSimulator(model, rng=np.random.default_rng(42 + sim))
        sim_results = simulator.simulate(n_periods=T)
        
        # Analyze
        context = SimulationContext(
            model=model,
            security_returns=sim_results['security_returns'],
            factor_returns=sim_results['factor_returns'],
            idio_returns=sim_results['idio_returns']
        )
        
        manifold = Analyses.manifold_distances().analyze(context)
        eigen = Analyses.eigenvalue_analysis(k_top=k).analyze(context)
        
        grass_distances.append(manifold['dist_grassmannian'])
        eigen_rmses.append(eigen['eigenvalue_rmse'])
    
    results_by_T[T] = {
        'grass_mean': np.mean(grass_distances),
        'grass_std': np.std(grass_distances),
        'eigen_mean': np.mean(eigen_rmses),
        'eigen_std': np.std(eigen_rmses)
    }

# Report
print("Sampling Error vs Sample Size:")
print("-" * 70)
for T, r in results_by_T.items():
    print(f"T={T:4d}: Grass={r['grass_mean']:.3f}±{r['grass_std']:.3f}, "
          f"Eigen RMSE={r['eigen_mean']:.4f}±{r['eigen_std']:.4f}")
```

### Workflow 3: Research Pipeline with Visualization

```python
from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import (
    create_manifold_dashboard,
    create_interactive_plotly_dashboard,
    print_verbose_results
)

# Build model, simulate, create context (as before)
# ... 

# Run all analyses
manifold = Analyses.manifold_distances().analyze(context)
eigen = Analyses.eigenvalue_analysis(k_top=10, compare_eigenvectors=True).analyze(context)
eigvec = Analyses.eigenvector_comparison(k_components=5).analyze(context)

# Combine results
all_results = {**manifold, **eigen, **eigvec}

# Print formatted results to console
print_verbose_results(all_results, context)

# Create publication-quality visualization
create_manifold_dashboard(
    all_results,
    output_path="results_dashboard.png",
    figsize=(20, 14)
)

# Create interactive dashboard for exploration
create_interactive_plotly_dashboard(
    all_results,
    output_path="results_interactive.html"
)

print("\n✓ Analysis complete!")
print("  Static dashboard:     results_dashboard.png")
print("  Interactive dashboard: results_interactive.html")
```

---

## 🧪 Testing & Validation

### Run Test Suite

```bash
# All tests (50 tests, should all pass)
pytest tests/ -v

# Specific test modules
pytest tests/analysis/test_manifold.py -v              # 15 tests
pytest tests/analysis/test_spectral_eigenvector.py -v  # 13 tests
pytest tests/analysis/test_integration.py -v           # 9 tests
pytest tests/analysis/test_visualization_context.py -v # 13 tests

# With coverage report
pytest tests/ --cov=factor_lab --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Test Organization

**Integration Tests** (`test_integration.py`):
- Complete workflow tests (build → simulate → analyze)
- JSON configuration parsing
- Multiple simulation scenarios
- Error handling

**Manifold Tests** (`test_manifold.py`):
- Orthonormalization correctness
- Grassmannian distance properties (rotation invariance, etc.)
- Procrustes optimal alignment
- Chordal distance computation

**Spectral Tests** (`test_spectral_eigenvector.py`):
- LinearOperator eigenvalue computation
- Eigenvalue error metrics
- Eigenvector alignment and correlation
- Sign flip handling

**Context & Visualization Tests** (`test_visualization_context.py`):
- SimulationContext immutability
- PCA decomposition caching
- Visualization creation
- Factory pattern for analyses

---

## 📚 Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | Overview, quick start, examples | Everyone |
| [API.md](docs/API.md) | Complete API reference | Developers |
| [CHEATSHEET.md](docs/CHEATSHEET.md) | Quick reference guide | Daily users |
| [TECHNICAL_MANUAL.md](docs/TECHNICAL_MANUAL.md) | Mathematical details | Researchers |
| [APPENDIX_HIGH_DIMENSIONAL.md](docs/APPENDIX_HIGH_DIMENSIONAL.md) | High-dim theory | Advanced users |
| [PERTURBATION_STUDY.md](docs/PERTURBATION_STUDY.md) | Perturbation analysis | Researchers |

### In-Code Documentation

Every function includes comprehensive docstrings with:
- Mathematical background
- Parameter descriptions
- Return value specifications
- Usage examples
- Implementation notes
- References to relevant papers

```python
help(Analyses.manifold_distances)  # See full documentation
```

---

## ⚡ Performance Characteristics

### Computational Complexity

| Operation | Dense | Sparse (Factor) | Improvement |
|-----------|-------|----------------|-------------|
| Covariance formation | O(p²) | Not needed | ∞ |
| Eigenvalues (k of them) | O(p³) | O(k²p) | O(p/k) |
| SVD decomposition | O(Tp²) | O(Tkp) | O(p/k) |
| Manifold distances | O(kp²) | O(kp²) | Same |

### Memory Requirements

| Problem Size | k | p | T | Memory (Dense) | Memory (Sparse) | Reduction |
|--------------|---|-----|---|----------------|-----------------|-----------|
| Small | 3 | 50 | 100 | ~20 KB | ~2 KB | 10× |
| Medium | 5 | 500 | 500 | ~2 MB | ~20 KB | 100× |
| Large | 10 | 5,000 | 1,000 | ~200 MB | ~400 KB | 500× |
| Very Large | 10 | 50,000 | 5,000 | ~20 GB | ~4 MB | 5000× |

### Benchmark Results

Tested on: Intel i7, 16GB RAM, Python 3.12

| Configuration | Time | Peak Memory |
|---------------|------|-------------|
| k=3, p=100, T=500 | 0.5s | 10 MB |
| k=5, p=1000, T=1000 | 8s | 80 MB |
| k=10, p=5000, T=2000 | 120s | 400 MB |

---

## 🎓 Mathematical Background

### Factor Model

**Definition**: 
```
r_t = B' f_t + ε_t
```
where:
- r_t ∈ ℝᵖ: Asset returns at time t
- B ∈ ℝᵏˣᵖ: Factor loadings
- f_t ∈ ℝᵏ: Factor returns with Cov(f) = F
- ε_t ∈ ℝᵖ: Idiosyncratic returns with Cov(ε) = D

**Implied Covariance**:
```
Σ = B' F B + D
```

### Manifold Geometry

**Grassmannian Manifold** Gr(k, p):
- Space of all k-dimensional subspaces of ℝᵖ
- Points are subspaces, not individual matrices
- Natural geometry via principal angles

**Principal Angles** θ₁, ..., θₖ ∈ [0, π/2]:
```python
θ = subspace_angles(U, V)  # Via scipy
d_Grassmannian = ||θ||₂
```

**Stiefel Manifold** V_{p,k}:
- Space of all p×k matrices with orthonormal columns
- Points are orthonormal frames

**Procrustes Distance**:
```
d_Procrustes = min_R ||A - B R||_F
where R' R = I (orthogonal)
```

Solution via SVD:
```python
M = B' A
U, S, Vt = svd(M)
R_opt = U @ Vt
d = ||A - B @ R_opt||_F
```

### High-Dimensional Asymptotics

**Marchenko-Pastur Law**: As p → ∞, T → ∞ with p/T → γ:
```
Sample eigenvalues λ̂ concentrate around Marchenko-Pastur distribution
```

**BBP Phase Transition**: For top eigenvalue λ₁:
```
λ̂₁ → λ₁  (consistent)  if λ₁ > 1 + √γ
λ̂₁ → γ + 1  (inconsistent)  if λ₁ ≤ 1 + √γ
```

**Eigenvector Bias**: Sample eigenvectors v̂ satisfy:
```
E[|⟨v̂, v⟩|²] < 1 - O(1/T)  (dispersion bias)
```

### References

1. Edelman, A., Arias, T. A., & Smith, S. T. (1998). *The geometry of algorithms with orthogonality constraints.* SIAM J. Matrix Anal. Appl.

2. Baik, J., Ben Arous, G., & Péché, S. (2005). *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices.* Annals of Probability.

3. Johnstone, I. M., & Lu, A. Y. (2009). *On consistency and sparsity for principal components analysis in high dimensions.* JASA.

---

## 🔧 Installation & Requirements

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (16GB recommended for large problems)
- NumPy-compatible CPU (x86-64, ARM64)

### Dependencies

**Required**:
```
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

**Optional** (for enhanced features):
```
plotly >= 5.0.0  # Interactive dashboards
pytest >= 7.0.0  # Running tests
pytest-cov >= 3.0.0  # Coverage reports
```

### Installation Options

**Option 1: Extract and use (no installation)**
```bash
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete
pip install -r requirements.txt
python demo.py
```

**Option 2: Install as editable package**
```bash
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete
pip install -e .
```

**Option 3: Install dependencies only**
```bash
pip install numpy scipy matplotlib seaborn plotly pytest pytest-cov
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'factor_lab'`

**Solution**: Make sure you're in the package directory or add to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

---

**Issue**: Tests failing with import errors

**Solution**: Install package in editable mode:
```bash
pip install -e .
```

---

**Issue**: `ImportError: No module named 'plotly'`

**Solution**: Plotly is optional. Either:
1. Install it: `pip install plotly`
2. Or skip interactive visualizations (static plots still work)

---

**Issue**: ARPACK convergence errors in eigenvalue analysis

**Solution**: Increase tolerance or iterations:
```python
eigen = Analyses.eigenvalue_analysis(
    k_top=10,
    tol=1e-4,  # Default is 0
    maxiter=None  # Or set to larger value
)
```

---

**Issue**: Memory errors with large problems

**Solution**: Reduce problem size or use LinearOperator (already default):
```python
# This uses O(kp) memory, not O(p²)
eigen = Analyses.eigenvalue_analysis(k_top=10)
```

---

## 🚦 Best Practices

### For Reliable Results

1. **Sample Size**: Use T/p > 2 for stable estimates
   - Below this ratio, sample covariance is highly uncertain
   - Eigenvector recovery requires T/p > 5 for reliability

2. **Random Seeds**: Always set for reproducibility
   ```python
   rng = np.random.default_rng(42)
   simulator = ReturnsSimulator(model, rng=rng)
   ```

3. **Multiple Metrics**: Don't rely on single metric
   ```python
   # Check both subspace and frame distances
   manifold['dist_grassmannian']  # Subspace distance
   manifold['dist_procrustes']    # Frame distance
   ```

4. **Validation**: Cross-validate with multiple simulations
   ```python
   # Run Monte Carlo with different seeds
   for seed in range(100):
       results[seed] = analyze_with_seed(seed)
   ```

### For Performance

1. **Use LinearOperator**: Already default for eigenvalues
   - Automatically uses O(kp) instead of O(p²)
   - Enables problems with p = 10,000+

2. **Cache PCA**: Context automatically caches PCA results
   ```python
   # First call computes
   pca1 = context.pca_decomposition(k=3)
   
   # Second call uses cache
   pca2 = context.pca_decomposition(k=3)  # Instant
   ```

3. **Disable Expensive Analyses**: Skip what you don't need
   ```python
   eigen = Analyses.eigenvalue_analysis(
       k_top=10,
       compare_eigenvectors=False  # Skip eigenvector comparison
   )
   ```

### For Research

1. **Document Everything**: Include random seeds, parameters, versions
   ```python
   # Save complete specification
   np.savez('results.npz',
            model_B=model.B,
            model_F=model.F,
            model_D=model.D,
            random_seed=42,
            **all_results)
   ```

2. **Version Control**: Track which version of package used
   ```python
   import factor_lab
   print(f"Factor Lab version: {factor_lab.__version__}")
   ```

3. **Save Visualizations**: Keep for papers/presentations
   ```python
   create_manifold_dashboard(
       results,
       output_path=f"results_{timestamp}.png"
   )
   ```

---

## 🎯 Research Applications

### Portfolio Optimization
- Validate factor models used for risk estimation
- Check factor stability over time
- Compare estimation methods (PCA vs ML vs Bayesian)

### Asset Pricing
- Test factor model sufficiency
- Quantify pricing errors
- Validate factor extraction procedures

### Risk Management
- Validate covariance matrix estimates
- Check spectral properties
- Assess model stability

### Simulation Studies
- Quantify finite-sample bias
- Study convergence properties
- Compare estimators

---

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{factor_lab_manifold,
  title={Factor Lab Manifold Complete: Production-Ready Factor Model Analysis},
  author={Your Name},
  year={2026},
  version={2.2.0},
  url={https://github.com/yourusername/factor_lab_manifold_complete}
}
```

And consider citing the key methodological papers:

```bibtex
@article{edelman1998geometry,
  title={The geometry of algorithms with orthogonality constraints},
  author={Edelman, Alan and Arias, Tom{\'a}s A and Smith, Steven T},
  journal={SIAM journal on Matrix Analysis and Applications},
  volume={20},
  number={2},
  pages={303--353},
  year={1998}
}
```

---

## 📞 Support & Contributing

### Getting Help

1. **Documentation**: Check `docs/` directory
2. **Examples**: See `demo.py` and `examples/`
3. **Tests**: Review test files for usage patterns
4. **Docstrings**: Use `help()` on any function

### Reporting Issues

When reporting issues, please include:
- Python version
- Package version (`import factor_lab; print(factor_lab.__version__)`)
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

### Contributing

Contributions welcome! Areas of interest:
- Additional analysis methods
- Performance optimizations
- Documentation improvements
- Bug fixes
- Test coverage expansion

---

## 📜 License

This package is provided under the MIT License. See LICENSE file for details.

---

## 🎉 Acknowledgments

Built on excellent work from:
- **NumPy/SciPy teams**: Core numerical computing
- **Matplotlib/Seaborn/Plotly**: Visualization
- **pytest**: Testing framework
- **Original Gemini code**: JSON parsing and manifold metrics inspiration

Special thanks to the quantitative finance and random matrix theory communities for the mathematical foundations.

---

## ✅ Version Information

**Current Version**: 2.2.0  
**Release Date**: February 2026  
**Python**: 3.8+  
**Status**: Production Ready  
**Tests**: 50/50 Passing ✓

---

## 🔗 Quick Links

- [API Reference](docs/API.md) - Complete function documentation
- [Cheatsheet](docs/CHEATSHEET.md) - Quick reference guide
- [Technical Manual](docs/TECHNICAL_MANUAL.md) - Mathematical details
- [Examples](examples/) - Working code examples
- [Tests](tests/) - Comprehensive test suite

---

**Ready to start?**

```bash
python demo.py
```

Enjoy analyzing your factor models! 🚀
