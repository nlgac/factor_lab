# Factor Lab

**Factor Lab** is a modular, high-performance Python library for financial factor model simulation, analysis, and optimization.

## Key Features

* **Hybrid Simulation Engine**: Automatically switches between O(N) scalar efficiency for diagonal models and O(N^3) Cholesky decomposition for correlated risk models.
* **SVD Factor Extraction**: Direct decomposition of returns matrices ($T \times N$) for superior numerical stability. Automatically caches square roots for instant simulation setup.
* **Smart Logging (Loguru)**: Built-in integration for color-coded debugging. Inspect raw distribution samples via the `debug_intermediate_quantity_length` parameter.
* **Performance PCA**: Support for `scipy` sparse decomposition (`method='scipy'`) for large matrix operations.
* **Convex Optimization**: Efficient Second-Order Cone Programming (SOCP) solver that exploits factor structure.

## Quick Start

```python
from factor_lab import svd_decomposition, ReturnsSimulator
import numpy as np

# 1. Fit Model directly from Returns (SVD Path)
# returns shape: (Time, Assets)
returns_matrix = np.random.normal(0, 0.01, (1000, 50))
model = svd_decomposition(returns_matrix, k=3)

# 2. Initialize Simulator
# Instant setup: Uses cached roots from SVD step (No Cholesky required)
sim = ReturnsSimulator(model)

# 3. Run Simulation
results = sim.simulate(
    n=252,
    f_gens=[...], i_gens=[...],
    debug_intermediate_quantity_length=5
)
```

## Documentation
For detailed class and method definitions, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).