# Factor Lab

**Factor Lab** is a modular, high-performance Python library for financial factor model simulation, analysis, and optimization.

## Key Features

* **Hybrid Simulation Engine**: Automatically switches between O(N) scalar efficiency for diagonal models and O(N^3) Cholesky decomposition for correlated risk models.
* **Abstract Generators**: Decouple the 'shape' of a distribution (Student's t, Beta, Normal) from the 'scale' (Volatility).
* **Granular Control**: Configure factors globally (broadcasting) or individually (explicit lists) for precise modeling.
* **Convex Optimization**: Efficient Second-Order Cone Programming (SOCP) solver that exploits factor structure.

## Installation

1.  Run the installer script:
    ```bash
    python install_factor_lab.py
    ```
2.  Import in your project:
    ```python
    import factor_lab
    ```

## Quick Start

```python
import numpy as np
from factor_lab import DistributionFactory, DataSampler, ReturnsSimulator

# 1. Define Statistical Generators
factory = DistributionFactory()
normal_gen = factory.create_generator('normal', mean=0, std=1)
fat_tail_gen = factory.create_generator('student_t', df=4)
vol_gen = factory.create_generator('uniform', low=0.10, high=0.20)

# 2. Build Factor Model (Heterogeneous Factors)
ds = DataSampler(p=500, k=2)
ds.configure(
    beta=[normal_gen, fat_tail_gen],  # Factor 1 is Normal, Factor 2 is Fat-Tailed
    f_vol=vol_gen,                    # Broadcast: Both factors have uniform vol
    d_vol=factory.create_generator('constant', c=0.05) # All assets have 5% idio vol
)
model_data = ds.generate()

# 3. Run Simulation
sim = ReturnsSimulator(model_data, seed=42)
# ...
```

## Documentation
For detailed class and method definitions, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).