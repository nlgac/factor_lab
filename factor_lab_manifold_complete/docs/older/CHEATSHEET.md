# Factor Lab Manifold Analysis - Quick Reference Cheatsheet

One-page reference for common tasks and usage patterns.

---

## 🚀 Quick Start

```bash
# Extract and run
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete
python demo.py
```

---

## 📦 Installation

```bash
# Dependencies only
pip install numpy scipy matplotlib seaborn plotly pytest

# Or from requirements
pip install -r requirements.txt
```

---

## 💻 Basic Usage Pattern

```python
from factor_lab import FactorModelData, ReturnsSimulator, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import create_manifold_dashboard

# 1. Create/load model
model = FactorModelData(B=B, F=F, D=D)

# 2. Simulate
sim = ReturnsSimulator(model)
results = sim.simulate(n_periods=500)

# 3. Create context
ctx = SimulationContext(model, results['security_returns'],
                        results['factor_returns'], results['idio_returns'])

# 4. Analyze
manifold = Analyses.manifold_distances().analyze(ctx)
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(ctx)
eigvec = Analyses.eigenvector_comparison(k=10).analyze(ctx)

# 5. Visualize
all_results = {**manifold, **eigen, **eigvec}
create_manifold_dashboard(all_results, "dashboard.png")
```

---

## 🏗️ Build from JSON Config

```bash
# Run with model_spec.json
python build_and_simulate.py model_spec.json

# With custom seed
python build_and_simulate.py model_spec.json --seed 123
```

**JSON Format:**
```json
{
  "meta": {"p_assets": 100, "n_periods": 500},
  "factor_loadings": [
    {"distribution": "normal", "params": {"loc": 0, "scale": 1}},
    {"distribution": "normal", "params": {"loc": 0, "scale": 0.5},
     "transform": "gram_schmidt"}
  ],
  "covariance": {
    "F_diagonal": ["0.18^2", "0.10^2"],
    "D_diagonal": "0.16^2"
  },
  "simulations": [
    {"name": "Gaussian", "type": "normal"}
  ]
}
```

---

## 🔬 Built-in Analyses

| Analysis | Factory Method | Key Metrics |
|----------|---------------|-------------|
| **Manifold Distances** | `Analyses.manifold_distances()` | `dist_grassmannian`, `dist_procrustes`, `dist_chordal` |
| **Eigenvalue Analysis** | `Analyses.eigenvalue_analysis(k_top=10)` | `eigenvalue_rmse`, `eigenvalue_errors` |
| **Eigenvector Comparison** | `Analyses.eigenvector_comparison(k=10)` | `mean_correlation`, `subspace_distance` |

---

## 📐 Manifold Metrics Explained

| Metric | Description | Invariant To | Use When |
|--------|-------------|--------------|----------|
| **Grassmannian** | Subspace distance | Rotation, sign, permutation | Compare subspaces |
| **Procrustes** | After optimal alignment | Sign, permutation | Compare frames |
| **Chordal** | Raw difference | Nothing | Debugging |

**Interpretation:**
- Grassmannian < 0.1 → Excellent
- Grassmannian < 0.3 → Good
- Grassmannian ≥ 0.3 → Moderate

---

## 🎯 Common Tasks

### Extract Factors from Returns
```python
model = svd_decomposition(returns, k=3, center=True)
```

### Run Custom Analysis
```python
custom = Analyses.custom(lambda ctx: {
    'frobenius_B': float(np.linalg.norm(ctx.model.B, 'fro')),
    'total_var': float(np.trace(ctx.model.implied_covariance()))
})
results = custom.analyze(context)
```

### Load Saved Results
```python
data = np.load('simulation_gaussian.npz')
true_B = data['true_B']
sample_B = data['sample_B']
```

### Create Visualization
```python
# Static (seaborn)
from factor_lab.visualization import create_manifold_dashboard
create_manifold_dashboard(results, "output.png")

# Interactive (plotly)
from factor_lab.visualization import create_interactive_plotly_dashboard
create_interactive_plotly_dashboard(results, "output.html")
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/analysis/test_manifold.py -v

# With coverage
pytest tests/ --cov=factor_lab --cov-report=html
```

---

## 📊 Key Data Structures

### FactorModelData
```python
model = FactorModelData(
    B=np.ndarray,  # (k, p) loadings
    F=np.ndarray,  # (k, k) factor covariance
    D=np.ndarray   # (p, p) idiosyncratic covariance
)
model.k  # number of factors
model.p  # number of assets
model.implied_covariance()  # Σ = B'FB + D
```

### SimulationContext
```python
context = SimulationContext(
    model=model,
    security_returns=np.ndarray,  # (T, p)
    factor_returns=np.ndarray,     # (T, k)
    idio_returns=np.ndarray        # (T, p)
)
context.T  # periods
context.sample_covariance()  # cached
context.pca_decomposition(k=3)  # cached by k
```

---

## 🎨 Visualization Options

### Static Dashboard (Seaborn)
- 9-panel layout
- Eigenvalue comparison
- Principal angles
- Manifold distances
- Eigenvector correlations
- Loading heatmaps
- Summary statistics

### Interactive Dashboard (Plotly)
- All above + zoom, hover, pan
- HTML output for sharing
- Responsive design

---

## ⚡ Performance Tips

### Memory Optimization
```python
# Use LinearOperator for eigenvalues (O(kp) instead of O(p²))
eigen = Analyses.eigenvalue_analysis(k_top=10)

# For p=10,000, k=10: 100× memory reduction
```

### Speed Optimization
```python
# Disable eigenvector comparison if not needed
eigen = Analyses.eigenvalue_analysis(
    k_top=10,
    compare_eigenvectors=False  # Faster
)

# Skip visualizations
# create_viz=False in build_and_simulate.py
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: factor_lab` | Add parent to path: `sys.path.insert(0, '.')` |
| `ImportError: numpy` | Install deps: `pip install -r requirements.txt` |
| Tests fail | Check Python version (need 3.8+) |
| Plotly missing | Optional: `pip install plotly` |
| ARPACK no convergence | Increase `maxiter` or lower `tol` |

---

## 📁 File Organization

```
project/
├── factor_lab/              # Package
│   ├── types.py            # Core types
│   ├── analysis/           # Framework
│   ├── analyses/           # Built-in analyses
│   └── visualization/      # Plotting
├── build_and_simulate.py   # Main script
├── demo.py                 # Demonstration
├── model_spec.json         # Configuration
├── tests/                  # Test suite
└── output/                 # Results
    ├── *.png              # Static dashboards
    └── *.html             # Interactive dashboards
```

---

## 🔗 Quick Links

- **Full Documentation:** `docs/API.md`
- **Examples:** `examples/`
- **Tests:** `tests/`
- **README:** `README.md`

---

## 💡 Pro Tips

1. **Always set random seed** for reproducibility
2. **Use T/p > 2** for reliable estimates  
3. **Check multiple metrics** (don't rely on just one)
4. **Save visualizations** for later reference
5. **Cache PCA** by using context methods

---

## 🎓 Mathematical Quick Reference

### Grassmannian Distance
```
d_G(U, V) = ||θ||_2
where θ are principal angles from subspace_angles(U, V)
```

### Procrustes Distance
```
d_P(A, B) = min_R ||A - BR||_F
where R is orthogonal (found via SVD)
```

### LinearOperator Eigenvalues
```
Instead of: Σ = B'FB + D (p² memory)
Use: matvec(v) = B'(F(Bv)) + Dv (kp memory)
```

---

## 🚦 Status Symbols

In verbose output:
- ✓ = Excellent (>90%)
- ○ = Good (70-90%)
- ✗ = Needs attention (<70%)

---

## 📞 Getting Help

1. Check `API.md` for detailed documentation
2. See `demo.py` for working examples
3. Run tests: `pytest tests/ -v`
4. Read docstrings in code

---

## 🎯 One-Liners

```python
# Quick manifold distance
Analyses.manifold_distances().analyze(context)['dist_grassmannian']

# Quick eigenvector correlation
Analyses.eigenvector_comparison(k=5).analyze(context)['mean_correlation']

# Quick eigenvalue RMSE
Analyses.eigenvalue_analysis(k_top=10).analyze(context)['eigenvalue_rmse']

# Run all at once
{**Analyses.manifold_distances().analyze(ctx),
 **Analyses.eigenvalue_analysis(k_top=5).analyze(ctx),
 **Analyses.eigenvector_comparison(k=5).analyze(ctx)}
```

---

## ⌨️ Keyboard Shortcuts (Demo)

When running `demo.py`:
- Follow terminal output
- Open PNG in image viewer
- Open HTML in browser (interactive!)

---

## 🔥 Power User Workflow

```bash
# 1. Edit config
vim model_spec.json

# 2. Run analysis
python build_and_simulate.py model_spec.json --seed 42

# 3. Check output
ls -l output/

# 4. Open visualizations
open output/dash_gaussian.html  # macOS
xdg-open output/dash_gaussian.html  # Linux

# 5. Load for further analysis
python -c "import numpy as np; d=np.load('sim_gaussian.npz'); print(d['dist_grassmannian'])"
```

---

## 🎨 Customization Examples

### Custom Color Scheme (Visualization)
```python
import seaborn as sns
sns.set_palette("husl")
create_manifold_dashboard(results, "custom.png")
```

### Custom Analysis
```python
class MyAnalysis:
    def analyze(self, ctx):
        # Your logic here
        return {'my_metric': value}

analysis = MyAnalysis()
results = analysis.analyze(context)
```

---

## 📏 Typical Scales

| Problem Size | k | p | T | Time | Memory |
|--------------|---|-----|-----|------|--------|
| Small | 2-3 | 20-50 | 100-500 | <1s | <10MB |
| Medium | 3-5 | 100-500 | 500-1000 | ~5s | ~50MB |
| Large | 5-10 | 1000-5000 | 1000-5000 | ~60s | ~500MB |
| Very Large | 10-20 | 10000+ | 5000+ | ~10min | ~5GB |

---

## ✅ Pre-Flight Checklist

Before running analysis:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Config file exists (`model_spec.json`)
- [ ] Output directory writable
- [ ] Sufficient memory (estimate: k×p×8 bytes)

---

**Version:** 2.2.0  
**Updated:** 2026-01-30  
**Quick Start:** `python demo.py`
