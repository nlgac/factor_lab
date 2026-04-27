# Adding Custom Distance Metrics

factor_lab measures estimation error on two separate pipelines that use
different distance abstractions. This document explains what each pipeline
does, where its extension point lives, and how to write and register a new
metric for either or both.

---

## The two pipelines

### Simulation pipeline (`factor_sims.py`)

Runs the dispersion-bias accuracy study. For each `(p, sim, radius)` cell it
places random target frames at a known geodesic distance `radius` from the
ground-truth frame, then measures how far the sample estimate sits from those
targets. Every metric in the pipeline needs two things:

- a **distance function** — given two orthonormal frames, return a scalar
- a **target sampler** — given a ground-truth frame and a radius, generate
  target frames at exactly that geodesic distance

Results end up in `SimResults.long_df` as long-form rows keyed by
`metric` (a string like `'grassmann'`).

### Analysis pipeline (`factor_lab/analyses/manifold.py`)

Used by `build_simulate_analyze` and `run_analyses`. After a simulation,
`ManifoldDistanceAnalysis.analyze()` compares the ground-truth loadings
`B_true` (shape `(k, p)`) to the PCA-estimated loadings `B_estimated` and
returns a flat dict of named scalars. Results flow into the NPZ output via
`build_and_simulate.py`.

---

## Simulation pipeline: `MetricSpec` and `register_metric`

### Data structure

```python
# factor_sims.py

@dataclass(frozen=True)
class MetricSpec:
    """One distance metric for the simulation loop."""
    name: str
    distance_fn: Callable[[np.ndarray, np.ndarray], float]
    sampler: Callable  # (U_gt, radius, n, rng, *, Q_full) -> list[ndarray]
```

`name` becomes the `metric` column value in `long_df`. `distance_fn` and
`sampler` are described in detail below.

### Registration

```python
def register_metric(spec: MetricSpec) -> None:
    """Register a custom distance metric. Call before run_simulation."""
    _METRICS[spec.name] = spec
```

Call this once at module level (or before `run_simulation`) with a fully
constructed `MetricSpec`. Registering with an existing name replaces the
previous spec, so you can swap out a built-in if needed.

### Writing a distance function

Signature: `(U1: np.ndarray, U2: np.ndarray) -> float`

Both arrays are orthonormal frames of shape `(p, k)` — columns are already
orthonormal, so `U.T @ U ≈ I_k`. The function must return a non-negative
scalar.

```python
def my_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """Frobenius norm of the cross-covariance deficit."""
    M = U1.T @ U2                        # (k, k) overlap matrix
    return float(np.sqrt(k - np.linalg.norm(M, 'fro')**2))
```

The two built-in distance functions are good references:

| Function | Location | Notes |
|---|---|---|
| `grassmann_distance` | `factor_sims.py:415` | SVD of overlap; L2 norm of principal angles |
| `stiefel_canonical_distance` | `factor_sims.py:428` | Real Schur decomposition of 2k×2k block rotation; ~30× faster than `logm` |

### Writing a target sampler

Signature: `(U_gt, radius, n, rng, *, Q_full=None) -> list[np.ndarray]`

| Argument | Type | Description |
|---|---|---|
| `U_gt` | `ndarray (p, k)` | Ground-truth orthonormal frame |
| `radius` | `float` | Desired geodesic distance from `U_gt` to each target |
| `n` | `int` | Number of target frames to generate |
| `rng` | `np.random.Generator` | RNG (use this; do not seed internally) |
| `Q_full` | `ndarray (p, p)` | Pre-computed orthogonal basis extending `U_gt`; avoids redundant `null_space` calls |

Returns a list of `n` orthonormal frames, each of shape `(p, k)`, each at
geodesic distance `radius` from `U_gt` under the metric being sampled.

The accuracy of the study depends on the sampler placing targets at **exactly**
`radius`. The truth-target reference row records `distance = radius` without
re-measuring, so a sampler that approximates the radius will introduce
systematic bias into the reference distribution.

The built-in samplers are the canonical references:

| Sampler | Location | Notes |
|---|---|---|
| `sample_grassmann_targets` | `factor_sims.py` | Horizontal geodesic (A11=0); exact |
| `sample_stiefel_targets` | `factor_sims.py` | Full canonical geodesic; exact to machine precision (verified in `test_stiefel_tangent_norm_exact`) |

### Complete example

```python
# my_metrics.py
import numpy as np
from factor_sims import MetricSpec, register_metric

# ---- distance function ----

def binet_cauchy_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Binet-Cauchy distance on Gr(k, p).

    d(U1, U2) = sqrt(1 - det(U1.T @ U2)^2)

    Only meaningful for k=1 (reduces to |sin θ|) but generalises via the
    squared determinant of the overlap. For k>1 interpret with care.
    """
    M = U1.T @ U2
    return float(np.sqrt(max(0.0, 1.0 - np.linalg.det(M) ** 2)))


# ---- target sampler ----

def sample_grassmann_targets_bc(U_gt, radius, n, rng, *, Q_full=None):
    """
    Grassmann targets for the Binet-Cauchy metric.

    The Binet-Cauchy distance equals |sin θ| for k=1, where θ is the
    single principal angle. We generate targets by rotating U_gt by angle
    θ = arcsin(radius) in a random direction in the orthogonal complement.
    """
    from factor_sims import _extend_to_orthogonal_basis
    p, k = U_gt.shape
    if Q_full is None:
        Q_full = _extend_to_orthogonal_basis(U_gt)
    complement = Q_full[:, k:]  # (p, p-k)
    targets = []
    for _ in range(n):
        # Pick random unit vector in complement
        v = rng.standard_normal(p - k)
        v /= np.linalg.norm(v)
        direction = complement @ v          # (p,) unit vector ⊥ U_gt
        theta = np.arcsin(np.clip(radius, 0.0, 1.0))
        # Rotate U_gt[:, 0] toward `direction` by angle theta
        target = U_gt.copy()
        target[:, 0] = np.cos(theta) * U_gt[:, 0] + np.sin(theta) * direction
        targets.append(target)
    return targets


# ---- register ----

register_metric(MetricSpec(
    name='binet-cauchy',
    distance_fn=binet_cauchy_distance,
    sampler=sample_grassmann_targets_bc,
))
```

Then in your driver script:

```python
import my_metrics  # registration happens on import
from factor_sims import run_simulation, build_spec

results = run_simulation(build_spec('toy'))
# long_df now has rows where metric == 'binet-cauchy'
bc_rows = results.long_df[results.long_df['metric'] == 'binet-cauchy']
```

---

## Analysis pipeline: `register_manifold_distance`

### How it works

`ManifoldDistanceAnalysis.analyze()` always computes the three built-in
distances (Grassmannian, Procrustes, Chordal) in full — including their
metadata arrays (principal angles, optimal rotation, aligned frame). After
that it iterates `_EXTRA_DISTANCES` and appends any registered scalars:

```python
results = {
    'dist_grassmannian': dist_grass,
    'dist_procrustes': procrustes_result['distance'],
    'dist_chordal': dist_chordal,
    'principal_angles': angles,
    'optimal_rotation': procrustes_result['optimal_rotation'],
    'aligned_frame': procrustes_result['aligned_frame'],
}
for key, fn in _EXTRA_DISTANCES.items():
    results[key] = fn(B_true, B_estimated)
return results
```

### Registration

```python
from factor_lab.analyses import register_manifold_distance

def register_manifold_distance(
    result_key: str,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> None:
```

`result_key` becomes the dict key in the analysis results and, because
`build_and_simulate.py` has a catch-all that forwards any unrecognised scalar
from `analyze()` into the NPZ output, also becomes an array name in the saved
`.npz` file automatically.

### Writing an analysis distance function

Signature: `(B_true: np.ndarray, B_estimated: np.ndarray) -> float`

Unlike the simulation pipeline, these inputs are raw factor loading matrices
of shape `(k, p)` — **not** necessarily orthonormal. The built-in functions
call `orthonormalize()` (QR decomposition) internally. Your function should
do the same if it requires an orthonormal frame.

```python
from factor_lab.analyses.manifold import orthonormalize

def my_analysis_distance(B_true: np.ndarray, B_estimated: np.ndarray) -> float:
    Q_true = orthonormalize(B_true)       # (p, k), columns orthonormal
    Q_est  = orthonormalize(B_estimated)  # (p, k)
    # ... compute and return scalar
```

### Complete example

```python
# my_analysis_metrics.py
import numpy as np
from factor_lab.analyses import register_manifold_distance
from factor_lab.analyses.manifold import orthonormalize


def nuclear_norm_distance(B_true: np.ndarray, B_estimated: np.ndarray) -> float:
    """
    Nuclear-norm distance between orthonormalized frames.

    Equivalent to the sum of singular values of (Q_true - Q_est).
    Lies between Procrustes and chordal distance in sensitivity to rotation.
    """
    Q_true = orthonormalize(B_true)
    Q_est  = orthonormalize(B_estimated)
    return float(np.linalg.norm(Q_true - Q_est, 'nuc'))


register_manifold_distance('dist_nuclear', nuclear_norm_distance)
```

After importing `my_analysis_metrics`:

```python
import my_analysis_metrics  # registers on import
from factor_lab.integration import build_simulate_analyze
from factor_lab.distributions import create_sampler
import numpy as np

rng = np.random.default_rng(42)
factory = lambda name, **p: create_sampler(name, rng, **p)

results = build_simulate_analyze(
    p=100, k=3,
    beta_samplers=factory('normal'),
    idio_vol_sampler=factory('constant', value=0.03),
    factor_variances=[0.04, 0.02, 0.01],
    n_periods=500,
    factor_return_samplers=factory('normal'),
    idio_return_sampler=factory('normal'),
    rng=rng,
)

print(results['dist_nuclear'])   # → float
# Also saved automatically to simulation_*.npz as 'dist_nuclear'
```

---

## Differences between the two pipelines

| | Simulation pipeline | Analysis pipeline |
|---|---|---|
| **File** | `factor_sims.py` | `factor_lab/analyses/manifold.py` |
| **Extension point** | `register_metric(MetricSpec(...))` | `register_manifold_distance(key, fn)` |
| **Input to fn** | Two orthonormal frames `(p, k)` | Two raw loading matrices `(k, p)` |
| **Extra requirement** | Target sampler at exact geodesic radius | None |
| **Output** | Rows in `SimResults.long_df` | Keys in `analyze()` result dict and `.npz` |
| **Built-ins** | `grassmann`, `stiefel-canonical` | `dist_grassmannian`, `dist_procrustes`, `dist_chordal` |

Note the shape convention difference: simulation-pipeline functions receive
frames shaped `(p, k)` (assets × factors, columns orthonormal), while
analysis-pipeline functions receive loading matrices shaped `(k, p)` (factors
× assets) matching `FactorModelData.B`.

---

## Registering a metric in both pipelines

The Grassmannian distance exists in both pipelines under different
implementations (one takes orthonormal frames; the other orthonormalises
internally). If your metric is meaningful in both contexts, write two thin
wrappers sharing a common core:

```python
import numpy as np
from factor_lab.analyses.manifold import orthonormalize
from factor_sims import MetricSpec, register_metric, sample_grassmann_targets
from factor_lab.analyses import register_manifold_distance


def _my_distance_core(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Core computation on orthonormal (p, k) frames."""
    ...


# Simulation pipeline — frames already orthonormal
def my_distance_sim(U1: np.ndarray, U2: np.ndarray) -> float:
    return _my_distance_core(U1, U2)

register_metric(MetricSpec('my-metric', my_distance_sim, sample_grassmann_targets))


# Analysis pipeline — orthonormalise loading matrices first
def my_distance_analysis(B_true: np.ndarray, B_estimated: np.ndarray) -> float:
    return _my_distance_core(orthonormalize(B_true), orthonormalize(B_estimated))

register_manifold_distance('dist_my_metric', my_distance_analysis)
```

---

## Test patterns

The test suite (`tests/analysis/test_factor_sims_registry.py` and
`tests/analysis/test_metric_registry.py`) demonstrates the standard fixture
for isolating module-level state between tests. Copy this pattern for any new
metric tests.

**Simulation pipeline isolation:**

```python
import factor_sims as _sims_module

@pytest.fixture(autouse=True)
def restore_metrics():
    original = dict(_sims_module._METRICS)
    yield
    _sims_module._METRICS.clear()
    _sims_module._METRICS.update(original)
```

**Analysis pipeline isolation:**

```python
from factor_lab.analyses import manifold as _manifold_module

@pytest.fixture(autouse=True)
def isolate_extra_distances():
    original = dict(_manifold_module._EXTRA_DISTANCES)
    yield
    _manifold_module._EXTRA_DISTANCES.clear()
    _manifold_module._EXTRA_DISTANCES.update(original)
```

Key things to assert for any new simulation-pipeline metric:

1. `register_metric(spec)` → `spec.name in _METRICS`
2. After registration, `_measure_one_cell(...)` produces records where `record['metric'] == spec.name`
3. `_sample_truth_records(...)` includes a row for the new metric name
4. `record['distance']` matches the value your `distance_fn` returns

Key things to assert for any new analysis-pipeline metric:

1. `register_manifold_distance(key, fn)` → `key in _manifold_module._EXTRA_DISTANCES`
2. `ManifoldDistanceAnalysis(...).analyze(ctx)` returns a dict containing `key`
3. The value equals `fn(ctx.model.B, ctx.model.B)` when `use_pca_loadings=False`
4. The three built-in keys (`dist_grassmannian`, `dist_procrustes`, `dist_chordal`) are unchanged
