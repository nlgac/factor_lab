# Factor Lab Manifold Analysis - API Reference

Complete API documentation for all modules, classes, and functions.

---

## Table of Contents

1. [Core Types](#core-types)
2. [Analysis Framework](#analysis-framework)
3. [Built-in Analyses](#built-in-analyses)
4. [Visualization](#visualization)
5. [Build & Simulate](#build--simulate)

---

## Core Types

### `FactorModelData`

Represents a factor model: r = B'f + e

```python
@dataclass
class FactorModelData:
    B: np.ndarray  # (k, p) factor loadings
    F: np.ndarray  # (k, k) factor covariance
    D: np.ndarray  # (p, p) idiosyncratic covariance
    
    @property
    def k(self) -> int:
        """Number of factors."""
    
    @property
    def p(self) -> int:
        """Number of assets."""
    
    def implied_covariance(self) -> np.ndarray:
        """Compute Σ = B'FB + D."""
```

**Example:**
```python
B = np.random.randn(3, 100)  # 3 factors, 100 assets
F = np.diag([0.09, 0.04, 0.01])
D = np.diag(np.full(100, 0.01))
model = FactorModelData(B=B, F=F, D=D)
```

### `svd_decomposition`

Extract factor model from returns via SVD.

```python
def svd_decomposition(
    returns: np.ndarray,
    k: int,
    center: bool = True
) -> FactorModelData:
    """
    Parameters
    ----------
    returns : np.ndarray, shape (T, p)
        Returns matrix.
    k : int
        Number of factors to extract.
    center : bool, default=True
        Whether to center returns before SVD.
    
    Returns
    -------
    FactorModelData
        Extracted factor model.
    """
```

**Example:**
```python
returns = np.random.randn(500, 100)  # 500 periods, 100 assets
model = svd_decomposition(returns, k=3, center=True)
```

### `ReturnsSimulator`

Simulate returns from a factor model.

```python
class ReturnsSimulator:
    def __init__(self, model: FactorModelData, rng=None):
        """
        Parameters
        ----------
        model : FactorModelData
            Factor model to simulate from.
        rng : np.random.Generator, optional
            Random number generator.
        """
    
    def simulate(
        self,
        n_periods: int,
        factor_samplers=None,
        idio_samplers=None
    ) -> dict:
        """
        Simulate returns.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate.
        factor_samplers : list, optional
            Custom samplers for factors.
        idio_samplers : list, optional
            Custom samplers for idiosyncratic.
        
        Returns
        -------
        dict
            security_returns : np.ndarray, shape (n_periods, p)
            factor_returns : np.ndarray, shape (n_periods, k)
            idio_returns : np.ndarray, shape (n_periods, p)
        """
```

**Example:**
```python
simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = simulator.simulate(n_periods=500)
returns = results['security_returns']
```

---

## Analysis Framework

### `SimulationContext`

Immutable snapshot of simulation state.

```python
@dataclass(frozen=True)
class SimulationContext:
    # Required
    model: FactorModelData
    security_returns: np.ndarray
    factor_returns: np.ndarray
    idio_returns: np.ndarray
    
    # Optional
    spec: Optional[ModelSpecification] = None
    sim_spec: Optional[SimulationSpec] = None
    test_results: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    
    @property
    def T(self) -> int:
        """Number of time periods."""
    
    @property
    def p(self) -> int:
        """Number of securities."""
    
    @property
    def k(self) -> int:
        """Number of factors."""
    
    def sample_covariance(self, ddof: int = 1) -> np.ndarray:
        """Compute sample covariance (cached)."""
    
    def pca_decomposition(self, n_components: Optional[int] = None) -> FactorModelData:
        """Run PCA decomposition (cached by n_components)."""
    
    def summary(self) -> str:
        """Generate formatted summary."""
```

**Example:**
```python
context = SimulationContext(
    model=model,
    security_returns=returns,
    factor_returns=factors,
    idio_returns=idio,
)

# Cached computations
cov = context.sample_covariance()
pca = context.pca_decomposition(k=3)
```

### `SimulationAnalysis` (Protocol)

Protocol for custom analyses.

```python
@runtime_checkable
class SimulationAnalysis(Protocol):
    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Perform analysis on simulation context.
        
        Parameters
        ----------
        context : SimulationContext
            Immutable snapshot of simulation state.
        
        Returns
        -------
        dict
            Analysis results as key-value pairs.
        """
```

**Example:**
```python
class MyAnalysis(SimulationAnalysis):
    def analyze(self, context):
        return {
            'my_metric': compute_something(context.model)
        }

analysis = MyAnalysis()
results = analysis.analyze(context)
```

---

## Built-in Analyses

### `ManifoldDistanceAnalysis`

Compare factor loadings using manifold geometry.

```python
class ManifoldDistanceAnalysis(SimulationAnalysis):
    def __init__(self, use_pca_loadings: bool = True):
        """
        Parameters
        ----------
        use_pca_loadings : bool, default=True
            If True, compare against PCA-extracted loadings.
            If False, compare model to itself (for testing).
        """
    
    def analyze(self, context: SimulationContext) -> dict:
        """
        Returns
        -------
        dict
            dist_grassmannian : float
                Distance on Grassmannian manifold (rotation-invariant).
            dist_procrustes : float
                Procrustes distance (after optimal alignment).
            dist_chordal : float
                Chordal distance (raw, without alignment).
            principal_angles : np.ndarray
                Principal angles between subspaces.
            optimal_rotation : np.ndarray
                Optimal rotation matrix from Procrustes.
            aligned_frame : np.ndarray
                Estimated frame after alignment.
        """
```

**Example:**
```python
analysis = ManifoldDistanceAnalysis(use_pca_loadings=True)
results = analysis.analyze(context)

print(f"Grassmannian: {results['dist_grassmannian']:.6f}")
print(f"Procrustes:   {results['dist_procrustes']:.6f}")
```

### `ImplicitEigenAnalysis`

Compute eigenvalues via LinearOperator (O(kp) memory).

```python
class ImplicitEigenAnalysis(SimulationAnalysis):
    def __init__(
        self,
        k_top: Optional[int] = None,
        compare_eigenvectors: bool = False,
        tol: float = 1e-10,
        maxiter: int = 10000
    ):
        """
        Parameters
        ----------
        k_top : int, optional
            Number of eigenvalues to compute. If None, uses model.k.
        compare_eigenvectors : bool, default=False
            Whether to also compare eigenvectors.
        tol : float, default=1e-10
            ARPACK convergence tolerance.
        maxiter : int, default=10000
            Maximum ARPACK iterations.
        """
    
    def analyze(self, context: SimulationContext) -> dict:
        """
        Returns
        -------
        dict
            true_eigenvalues : np.ndarray
                Eigenvalues of Σ_true (via LinearOperator).
            sample_eigenvalues : np.ndarray
                Eigenvalues from PCA.
            eigenvalue_errors : np.ndarray
                Difference: true - sample.
            eigenvalue_relative_errors : np.ndarray
                Relative errors: (true - sample) / true.
            eigenvalue_rmse : float
                Root mean squared error.
            eigenvalue_max_error : float
                Maximum absolute error.
            eigenvalue_mean_relative_error : float
                Mean of absolute relative errors.
            
            If compare_eigenvectors=True:
            eigenvector_subspace_distance : float
            eigenvector_principal_angles : np.ndarray
            eigenvector_canonical_correlations : np.ndarray
            eigenvector_mean_correlation : float
        """
```

**Example:**
```python
analysis = ImplicitEigenAnalysis(
    k_top=10,
    compare_eigenvectors=True
)
results = analysis.analyze(context)

print(f"Eigenvalue RMSE: {results['eigenvalue_rmse']:.6f}")
```

### `EigenvectorAlignment`

Compare eigenvectors of Σ_true with PCA eigenvectors.

```python
class EigenvectorAlignment(SimulationAnalysis):
    def __init__(
        self,
        k_components: Optional[int] = None,
        align_signs: bool = True,
        compute_rotation: bool = True,
        tol: float = 1e-10,
        maxiter: int = 10000
    ):
        """
        Parameters
        ----------
        k_components : int, optional
            Number of components to compare. If None, uses model.k.
        align_signs : bool, default=True
            Whether to align signs before comparison.
        compute_rotation : bool, default=True
            Whether to compute optimal rotation matrix.
        tol : float, default=1e-10
            ARPACK convergence tolerance.
        maxiter : int, default=10000
            Maximum ARPACK iterations.
        """
    
    def analyze(self, context: SimulationContext) -> dict:
        """
        Returns
        -------
        dict
            Core metrics:
            - subspace_distance : float
            - principal_angles : np.ndarray
            - max_principal_angle : float
            - grassmann_distance : float
            
            Correlation metrics:
            - vector_correlations : np.ndarray
            - mean_correlation : float
            - min_correlation : float
            - max_correlation : float
            
            If compute_rotation=True:
            - procrustes_distance : float
            - optimal_rotation : np.ndarray
            - aligned_eigenvectors : np.ndarray
            
            Reference data:
            - true_eigenvectors : np.ndarray
            - sample_eigenvectors : np.ndarray
            - k_components : int
        """
```

**Example:**
```python
analysis = EigenvectorAlignment(
    k_components=5,
    align_signs=True,
    compute_rotation=True
)
results = analysis.analyze(context)

print(f"Mean correlation: {results['mean_correlation']:.4f}")
```

### `Analyses` (Factory)

Convenient factory for creating analyses.

```python
class Analyses:
    @staticmethod
    def manifold_distances(use_pca_loadings: bool = True) -> ManifoldDistanceAnalysis:
        """Create manifold distance analysis."""
    
    @staticmethod
    def eigenvalue_analysis(
        k_top: Optional[int] = None,
        compare_eigenvectors: bool = False,
        tol: float = 1e-10,
        maxiter: int = 10000
    ) -> ImplicitEigenAnalysis:
        """Create eigenvalue analysis."""
    
    @staticmethod
    def eigenvector_comparison(
        k: Optional[int] = None,
        align_signs: bool = True,
        compute_rotation: bool = True,
        tol: float = 1e-10,
        maxiter: int = 10000
    ) -> EigenvectorAlignment:
        """Create eigenvector comparison analysis."""
    
    @staticmethod
    def custom(func: Callable[[SimulationContext], Dict[str, Any]]):
        """Wrap arbitrary function as analysis."""
```

**Example:**
```python
# Using factory
analyses = [
    Analyses.manifold_distances(),
    Analyses.eigenvalue_analysis(k_top=10),
    Analyses.eigenvector_comparison(k=5),
]

# Custom analysis
custom = Analyses.custom(lambda ctx: {
    'frobenius_B': float(np.linalg.norm(ctx.model.B, 'fro'))
})
```

---

## Visualization

### `create_manifold_dashboard`

Create comprehensive static dashboard with seaborn.

```python
def create_manifold_dashboard(
    results: Dict[str, Any],
    output_path: Optional[Path] = None
):
    """
    Create 9-panel dashboard with matplotlib/seaborn.
    
    Parameters
    ----------
    results : dict
        Analysis results containing metrics.
    output_path : Path, optional
        Path to save PNG. If None, displays interactively.
    
    Panels
    ------
    1. Eigenvalue spectrum comparison
    2. Principal angles bar chart
    3. Manifold distances
    4. Eigenvalue errors
    5. Eigenvector correlations
    6. Relative errors
    7. Summary statistics
    8. True eigenvector loadings (heatmap)
    9. Sample eigenvector loadings (heatmap)
    """
```

**Example:**
```python
create_manifold_dashboard(results, output_path="dashboard.png")
```

### `create_interactive_plotly_dashboard`

Create interactive dashboard with plotly.

```python
def create_interactive_plotly_dashboard(
    results: Dict[str, Any],
    output_path: Optional[Path] = None
):
    """
    Create interactive dashboard with zoom, hover, pan.
    
    Parameters
    ----------
    results : dict
        Analysis results.
    output_path : Path, optional
        Path to save HTML. If None, displays in browser.
    
    Features
    --------
    - Interactive zoom and pan
    - Hover tooltips with values
    - Responsive design
    - Save as PNG from browser
    """
```

**Example:**
```python
create_interactive_plotly_dashboard(results, output_path="dashboard.html")
```

### `print_verbose_results`

Print formatted results to console (Gemini style).

```python
def print_verbose_results(
    results: Dict[str, Any],
    title: str = "Analysis Results"
):
    """
    Print verbose, formatted results.
    
    Parameters
    ----------
    results : dict
        Analysis results.
    title : str
        Title for output section.
    
    Output Format
    -------------
    - Section headers with emoji
    - Formatted tables
    - Assessment summaries
    - Status symbols (✓, ○, ✗)
    """
```

**Example:**
```python
print_verbose_results(results, "My Analysis")
```

---

## Build & Simulate

### `JsonParser`

Parse JSON specifications with numeric expressions.

```python
class JsonParser:
    @staticmethod
    def _parse_numeric(value: Union[str, float, int]) -> float:
        """
        Parse numeric value with expression support.
        
        Supports:
        - Integers: 5 → 5.0
        - Floats: 0.25 → 0.25
        - Expressions: "0.18^2" → 0.0324
        """
    
    @classmethod
    def parse(cls, filepath: Path) -> ModelSpec:
        """
        Parse JSON specification file.
        
        Parameters
        ----------
        filepath : Path
            Path to JSON file.
        
        Returns
        -------
        ModelSpec
            Parsed model specification.
        """
```

**JSON Format:**
```json
{
  "meta": {
    "p_assets": 500,
    "n_periods": 63
  },
  "factor_loadings": [
    {
      "distribution": "normal",
      "params": {"loc": 1.0, "scale": 0.25}
    },
    {
      "distribution": "normal",
      "params": {"loc": 0.0, "scale": 1.0},
      "transform": "gram_schmidt"
    }
  ],
  "covariance": {
    "F_diagonal": ["0.18^2", "0.05^2"],
    "D_diagonal": "0.16^2"
  },
  "simulations": [
    {
      "name": "Gaussian",
      "type": "normal"
    },
    {
      "name": "Student-t",
      "type": "student_t",
      "params": {"df_factors": 5, "df_idio": 4}
    }
  ]
}
```

### `FactorModelBuilder`

Build factor models from specifications.

```python
class FactorModelBuilder:
    def __init__(self, rng: np.random.Generator):
        """
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        """
    
    def build(self, spec: ModelSpec) -> FactorModelData:
        """
        Build factor model from specification.
        
        Features:
        - Gram-Schmidt orthogonalization
        - Custom distributions
        - Diagonal covariance structures
        
        Parameters
        ----------
        spec : ModelSpec
            Model specification.
        
        Returns
        -------
        FactorModelData
            Constructed factor model.
        """
```

**Example:**
```python
rng = np.random.default_rng(42)
spec = JsonParser.parse("model_spec.json")
builder = FactorModelBuilder(rng)
model = builder.build(spec)
```

### `AnalysisEngine`

Enhanced analysis engine combining all features.

```python
class AnalysisEngine:
    def __init__(
        self,
        model: FactorModelData,
        spec: ModelSpec,
        rng: np.random.Generator
    ):
        """
        Parameters
        ----------
        model : FactorModelData
            Factor model.
        spec : ModelSpec
            Model specification.
        rng : np.random.Generator
            Random number generator.
        """
    
    def run_simulation_and_analyze(
        self,
        sim_config: SimConfig,
        save_outputs: bool = True,
        create_viz: bool = True
    ) -> Dict[str, Any]:
        """
        Run simulation and comprehensive analysis.
        
        Steps:
        1. Simulate returns
        2. Extract factors via SVD
        3. Create context
        4. Run all analyses
        5. Print verbose results
        6. Create visualizations
        7. Save results to .npz
        
        Parameters
        ----------
        sim_config : SimConfig
            Simulation configuration.
        save_outputs : bool, default=True
            Whether to save .npz files.
        create_viz : bool, default=True
            Whether to create visualizations.
        
        Returns
        -------
        dict
            Comprehensive analysis results.
        """
```

**Example:**
```python
engine = AnalysisEngine(model, spec, rng)
results = engine.run_simulation_and_analyze(
    sim_config,
    save_outputs=True,
    create_viz=True
)
```

---

## Utility Functions

### `save_model`

Save model to .npz file.

```python
def save_model(model: FactorModelData, filename: str):
    """Save model parameters to compressed numpy file."""
```

### `orthonormalize`

Orthonormalize factor loadings.

```python
def orthonormalize(B: np.ndarray) -> np.ndarray:
    """
    Project factor loadings onto Stiefel manifold.
    
    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Factor loadings.
    
    Returns
    -------
    np.ndarray, shape (p, k)
        Orthonormal frame with Q.T @ Q = I_k.
    """
```

### `compute_grassmannian_distance`

Compute distance on Grassmannian manifold.

```python
def compute_grassmannian_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute rotation-invariant subspace distance.
    
    Returns
    -------
    distance : float
        L2 norm of principal angles.
    angles : np.ndarray
        Principal angles in radians.
    """
```

### `compute_procrustes_distance`

Compute Procrustes distance (optimal alignment).

```python
def compute_procrustes_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Dict[str, Any]:
    """
    Compute distance after optimal orthogonal rotation.
    
    Returns
    -------
    dict
        distance : float
        optimal_rotation : np.ndarray
        aligned_frame : np.ndarray
    """
```

### `compute_chordal_distance`

Compute chordal distance (no alignment).

```python
def compute_chordal_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> float:
    """
    Compute direct Frobenius distance.
    
    Sensitive to rotation, sign flips, permutation.
    """
```

### `compute_true_eigenvalues`

Compute eigenvalues via LinearOperator.

```python
def compute_true_eigenvalues(
    model: FactorModelData,
    k_top: int,
    tol: float = 1e-10,
    maxiter: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues without forming full covariance.
    
    Memory: O(kp) instead of O(p²).
    
    Returns
    -------
    eigenvalues : np.ndarray, shape (k_top,)
        Top k eigenvalues in descending order.
    eigenvectors : np.ndarray, shape (k_top, p)
        Corresponding eigenvectors as row vectors.
    """
```

---

## Type Hints Reference

```python
# Common types
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Analysis result
AnalysisResult = Dict[str, Any]

# Analysis function
AnalysisFunc = Callable[[SimulationContext], AnalysisResult]
```

---

## Error Handling

### Common Exceptions

- `ValueError` - Invalid parameters (e.g., k > p)
- `ArpackNoConvergence` - Eigenvalue solver didn't converge
- `FileNotFoundError` - Config file not found
- `JSONDecodeError` - Invalid JSON format

### Handling Tips

```python
# Handle partial convergence
try:
    evals, evecs = compute_true_eigenvalues(model, k_top=10)
except ArpackNoConvergence as e:
    # Use partial results
    evals, evecs = e.eigenvalues, e.eigenvectors
    print(f"Warning: Only {len(evals)} converged")

# Validate inputs
if k > p:
    raise ValueError(f"k ({k}) cannot exceed p ({p})")
```

---

## Performance Notes

### Memory Complexity

| Operation | Dense | Sparse/Implicit |
|-----------|-------|-----------------|
| Eigenvalues | O(p²) | O(kp) |
| SVD | O(min(Tp, p²)) | N/A |
| Manifold | O(kp) | O(kp) |

### Time Complexity

| Operation | Complexity |
|-----------|-----------|
| SVD | O(Tp²) |
| Eigenvalues (dense) | O(p³) |
| Eigenvalues (implicit) | O(k²p) |
| Manifold distances | O(kp²) |

### Scalability

Tested scales:
- p up to 10,000 assets
- k up to 20 factors
- T up to 5,000 periods

---

For more examples, see:
- `demo.py` - Comprehensive demonstration
- `examples/` - Working examples
- `tests/` - Test suite with usage patterns
