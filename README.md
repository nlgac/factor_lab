# Factor Lab

A Python library for factor model construction, simulation, and portfolio optimization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Factor Lab provides tools for:

- **Factor Extraction**: Extract statistical factor models from returns data using PCA or SVD
- **Returns Simulation**: Generate Monte Carlo samples with proper covariance structure
- **Portfolio Optimization**: Find minimum variance portfolios using factor model covariance
- **Model Validation**: Compare empirical vs. model-implied covariance matrices

The library is designed with:
- Clear, documented APIs
- Dependency injection for testability
- Efficient implementations (O(k²) optimization instead of O(p²))
- Comprehensive test coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/factor_lab.git
cd factor_lab

# Install in development mode (includes CLI)
pip install -e .

# Or install with pandas support for CSV handling
pip install -e ".[pandas]"

# Verify installation
factor-lab --help
```

## Command Line Interface

Factor Lab includes a rich CLI for common workflows:

```bash
# Generate a synthetic model for testing
factor-lab generate --assets 100 --factors 5 --output model.npz

# Fit a model from historical returns
factor-lab fit returns.csv --factors 5 --output model.npz

# Auto-select number of factors for 90% explained variance
factor-lab fit returns.csv --variance 0.90

# Simulate returns
factor-lab simulate model.npz --periods 1000 --output simulated.csv

# Optimize a portfolio
factor-lab optimize model.npz --long-only --max-weight 0.05

# View model information
factor-lab info model.npz --detailed

# Interactive demo
factor-lab demo
```

## Quick Start

```python
import numpy as np
from factor_lab import (
    svd_decomposition,
    ReturnsSimulator,
    minimum_variance_portfolio,
    DistributionFactory,
)

# 1. Extract a factor model from historical returns
returns = np.random.randn(1000, 50) * 0.01  # Example: 1000 days, 50 assets
model = svd_decomposition(returns, k=5)

# 2. Simulate new returns
rng = np.random.default_rng(42)
factory = DistributionFactory(rng=rng)
simulator = ReturnsSimulator(model, rng=rng)

results = simulator.simulate(
    n_periods=500,
    factor_samplers=[factory.create("normal", mean=0, std=1)] * model.k,
    idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
)

# 3. Optimize a portfolio
result = minimum_variance_portfolio(model, long_only=True, max_weight=0.05)
print(f"Optimal risk: {result.risk:.2%}")
```

## Core Concepts

### Factor Models

A factor model decomposes asset returns as:

```
r = B.T @ f + ε
```

Where:
- `r`: (p,) vector of asset returns
- `B`: (k, p) matrix of factor loadings
- `f`: (k,) vector of factor returns with covariance F
- `ε`: (p,) vector of idiosyncratic returns with covariance D

The implied covariance is: `Σ = B.T @ F @ B + D`

### Module Organization

```
factor_lab/
├── types.py          # Core data structures
├── decomposition.py  # PCA/SVD factor extraction
├── simulation.py     # Monte Carlo simulation
├── optimization.py   # Portfolio optimization
└── samplers.py       # Distribution sampling
```

## Usage Guide

### 1. Factor Extraction

#### SVD Decomposition (Recommended)

SVD operates directly on returns data and is more numerically stable:

```python
from factor_lab import svd_decomposition, compute_explained_variance

# Extract 5 factors
model = svd_decomposition(returns, k=5)

# Check explained variance
explained = compute_explained_variance(model)
print(f"Factors explain {explained:.1%} of variance")
```

#### PCA Decomposition

PCA operates on a covariance matrix:

```python
from factor_lab import pca_decomposition
import numpy as np

cov = np.cov(returns, rowvar=False)
B, F = pca_decomposition(cov, k=5)
```

#### Automatic Factor Selection

```python
from factor_lab import select_k_by_variance

# Find k needed to explain 90% of variance
k = select_k_by_variance(returns, target_explained=0.90)
```

### 2. Returns Simulation

#### Basic Simulation

```python
from factor_lab import ReturnsSimulator, DistributionFactory

rng = np.random.default_rng(42)
factory = DistributionFactory(rng=rng)

simulator = ReturnsSimulator(model, rng=rng)
results = simulator.simulate(
    n_periods=1000,
    factor_samplers=[factory.create("normal", mean=0, std=1)] * model.k,
    idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
)

returns = results["security_returns"]  # (1000, p)
```

#### Fat-Tailed Innovations

```python
# Use Student's t for factor innovations (more realistic)
results = simulator.simulate(
    n_periods=1000,
    factor_samplers=[factory.create("student_t", df=5)] * model.k,
    idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
)
```

#### Covariance Validation

```python
from factor_lab import CovarianceValidator

validator = CovarianceValidator(model)
validation = validator.compare(returns)

print(f"Frobenius error: {validation.frobenius_error:.4f}")
print(f"Explained variance: {validation.explained_variance_ratio:.1%}")
```

### 3. Portfolio Optimization

#### Convenience Function

```python
from factor_lab import minimum_variance_portfolio

# Long-only, diversified
result = minimum_variance_portfolio(
    model,
    long_only=True,
    max_weight=0.05
)

if result.solved:
    print(f"Optimal risk: {result.risk:.2%}")
    print(f"Weights: {result.weights}")
```

#### Custom Constraints with ScenarioBuilder

```python
from factor_lab import FactorOptimizer, ScenarioBuilder

# Build a complex scenario
scenario = (ScenarioBuilder(model.p)
    .create("130/30 Strategy")
    .add_fully_invested()
    .add_box_constraints(low=-0.30, high=1.30)
    .build())

# Or with sector neutrality
sectors = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Asset sector assignments
scenario = (ScenarioBuilder(10)
    .create("Sector Neutral")
    .add_fully_invested()
    .add_sector_neutral(sectors)
    .add_box_constraints(low=-0.1, high=0.1)
    .build())

# Optimize
optimizer = FactorOptimizer(model)
optimizer.apply_scenario(scenario)
result = optimizer.solve()
```

### 4. Synthetic Data Generation

```python
from factor_lab import DataSampler, DistributionFactory

rng = np.random.default_rng(42)
factory = DistributionFactory(rng=rng)

# Generate a synthetic factor model
sampler = DataSampler(p=100, k=3, rng=rng)
model = sampler.configure(
    beta=[
        factory.create("normal", mean=1.0, std=0.3),   # Market
        factory.create("normal", mean=0.0, std=0.5),   # Value
        factory.create("normal", mean=0.0, std=0.5),   # Momentum
    ],
    factor_vol=[
        factory.create("constant", value=0.20),
        factory.create("constant", value=0.12),
        factory.create("constant", value=0.10),
    ],
    idio_vol=factory.create("uniform", low=0.05, high=0.15)
).generate()
```

## API Reference

### Types

| Type | Description |
|------|-------------|
| `FactorModelData` | Complete factor model specification (B, F, D) |
| `OptimizationResult` | Portfolio optimization results |
| `Scenario` | Collection of optimization constraints |
| `CovarianceTransform` | Pre-computed covariance square root |
| `CovarianceValidationResult` | Model validation metrics |

### Functions

| Function | Description |
|----------|-------------|
| `svd_decomposition(returns, k)` | Extract k-factor model via SVD |
| `pca_decomposition(cov, k)` | Extract k-factor model via PCA |
| `compute_explained_variance(model)` | Compute variance explained by factors |
| `select_k_by_variance(returns, target)` | Auto-select k for target variance |
| `simulate_returns(model, n)` | Convenience function for simulation |
| `minimum_variance_portfolio(model, ...)` | Convenience function for optimization |

### Classes

| Class | Description |
|-------|-------------|
| `ReturnsSimulator` | Monte Carlo returns generator |
| `CovarianceValidator` | Model validation utility |
| `FactorOptimizer` | SOCP-based portfolio optimizer |
| `ScenarioBuilder` | Fluent constraint builder |
| `DistributionFactory` | Sampler creation factory |
| `DistributionRegistry` | Distribution repository |
| `DataSampler` | Synthetic model generator |

## Design Principles

1. **Dependency Injection**: RNG instances are passed explicitly for reproducibility
2. **Immutable Value Objects**: Results and transforms are frozen dataclasses
3. **Fail-Fast Validation**: Dimensions checked at construction time
4. **Clear Abstractions**: Each module has a single responsibility
5. **Comprehensive Testing**: Unit tests, integration tests, and edge cases

## Performance Notes

- **O(k²) Optimization**: Uses factor structure instead of full covariance
- **Diagonal Fast Path**: Detects diagonal matrices for O(n) simulation
- **Pre-computed Transforms**: SVD provides simulation-ready square roots
- **Memory Efficient**: Economy SVD avoids full matrix storage

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=factor_lab --cov-report=html

# Run specific test file
pytest tests/test_optimization.py -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and migration guides.

## Citation

If you use this library in research, please cite:

```bibtex
@software{factor_lab,
  title = {Factor Lab: A Python Library for Factor Model Construction},
  year = {2024},
  url = {https://github.com/yourusername/factor_lab}
}
```
