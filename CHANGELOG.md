# Changelog

All notable changes to Factor Lab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-16

### Added

#### Command Line Interface
- New `factor-lab` CLI with rich terminal output using Typer and Rich
- Commands: `fit`, `simulate`, `optimize`, `generate`, `info`, `demo`, `version`
- Model persistence in NPZ and JSON formats
- Interactive demo walkthrough
- Beautiful tables, progress bars, and colored output

#### New Classes
- `CovarianceTransform` - Discriminated union for sqrt representations (DIAGONAL vs DENSE)
- `CovarianceValidator` - Extracted from ReturnsSimulator for single responsibility
- `CovarianceValidationResult` - Structured validation metrics
- `DistributionInfo` - Metadata container for registered distributions

#### New Functions
- `simulate_returns()` - Convenience function for quick simulation
- `minimum_variance_portfolio()` - Convenience function for common optimization
- `compute_explained_variance()` - Calculate variance explained by factors
- `select_k_by_variance()` - Auto-select number of factors

#### API Improvements
- `ScenarioBuilder` now uses fluent API (methods return `self`)
- `FactorOptimizer.apply_scenario()` - Apply all constraints from a Scenario
- `FactorOptimizer.reset_constraints()` - Clear constraints for re-solving
- `DistributionRegistry.list_distributions()` - Public method to list available distributions
- `DistributionFactory.list_distributions()` - Delegates to registry

#### New Distributions
- `lognormal(mean, sigma)` - Log-normal distribution
- `chi2(df)` - Chi-squared distribution

### Changed

#### Module Organization
- **Split `computation.py`** into `decomposition.py` and `optimization.py`
- **Deleted `load_tests.py`** - Redundant test installer removed
- **Consolidated fixtures** - All test fixtures now in `conftest.py`

#### Naming Improvements
- `_process()` → `_transform_samples()` - Clearer method name
- `F_tx`, `D_tx` → `factor_transform`, `idio_transform` - Descriptive names
- `F_sqrt`, `D_sqrt` → Wrapped in `CovarianceTransform` with type discrimination
- `GeneratorFunc` → `SamplerCallable` - Avoids confusion with Python generators
- `debug_intermediate_quantity_length` → `sample_log_rows` - Concise parameter name

#### Type Safety
- `FactorModelData.validate()` now called automatically in `__post_init__`
- `CovarianceTransform` uses explicit `TransformType` enum (DIAGONAL, DENSE)
- `OptimizationResult` validates that solved=True requires weights

#### Dependency Injection
- All samplers now accept explicit `rng: np.random.Generator`
- `DistributionFactory` takes RNG in constructor
- `DataSampler` takes RNG in constructor
- `ReturnsSimulator` has `force_dense` parameter for deterministic testing

#### Documentation
- NumPy-style docstrings on all public classes and functions
- Comprehensive README with usage examples
- Full API reference in `docs/API_REFERENCE.md`
- CLI documentation with all commands and options

### Fixed

- Sector-neutral + fully-invested constraints now documented as infeasible
- Test tolerances adjusted for statistical variability
- NPZ model loading handles None transforms correctly

### Removed

- `load_tests.py` - Duplicated test content
- Direct access to `_registry._funcs` - Use `list_distributions()` instead
- Loguru dependency from core library (CLI uses Rich instead)

### Migration Guide

#### From v1.x to v2.0

**ScenarioBuilder now fluent:**
```python
# Old (v1.x)
scen = builder.create("Test")
scen = builder.add_fully_invested(scen)
scen = builder.add_long_only(scen)

# New (v2.0)
scenario = (builder
    .create("Test")
    .add_fully_invested()
    .add_long_only()
    .build())
```

**Applying scenarios to optimizer:**
```python
# Old (v1.x)
for A, b in scen.equality_constraints:
    opt.add_eq(A, b)
for A, b in scen.inequality_constraints:
    opt.add_ineq(A, b)

# New (v2.0)
optimizer.apply_scenario(scenario)
```

**RNG injection:**
```python
# Old (v1.x) - Used global numpy RNG
factory = DistributionFactory()
gen = factory.create_generator('normal', mean=0, std=1)

# New (v2.0) - Explicit RNG for reproducibility
rng = np.random.default_rng(42)
factory = DistributionFactory(rng=rng)
sampler = factory.create("normal", mean=0, std=1)
```

**Pre-computed transforms:**
```python
# Old (v1.x)
model.F_sqrt  # Optional[np.ndarray] - ambiguous shape

# New (v2.0)
model.factor_transform  # Optional[CovarianceTransform] - explicit type
if model.factor_transform:
    if model.factor_transform.is_diagonal:
        stds = model.factor_transform.matrix  # 1D array
    else:
        cholesky = model.factor_transform.matrix  # 2D array
```

**Listing distributions:**
```python
# Old (v1.x) - Accessed private attribute
available = list(factory._registry._funcs.keys())

# New (v2.0) - Public method
available = factory.list_distributions()
```

## [1.0.0] - 2025-01-14

### Added
- Initial release
- SVD and PCA decomposition
- Returns simulation with hybrid diagonal/dense paths
- SOCP portfolio optimization
- Distribution factory and registry
- DataSampler for synthetic models
- Basic test suite
