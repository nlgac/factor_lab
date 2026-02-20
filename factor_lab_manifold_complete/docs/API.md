# Factor Lab Manifold Complete - API Reference

**Version**: 2.2.0  
**Last Updated**: February 2026

Complete API documentation for all modules, classes, and functions in the Factor Lab Manifold Complete package.

---

## Table of Contents

1. [Core Types (`factor_lab.types`)](#core-types)
2. [Analysis Framework (`factor_lab.analysis`)](#analysis-framework)
3. [Built-in Analyses (`factor_lab.analyses`)](#built-in-analyses)
4. [Visualization (`factor_lab.visualization`)](#visualization)
5. [Complete Examples](#complete-examples)

---

## Core Types

Module: `factor_lab.types`

### `FactorModelData`

**Description**: Immutable dataclass representing a factor model.

```python
@dataclass
class FactorModelData:
    """
    Factor model: r = B'f + ε

    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Factor loadings matrix.
    F : np.ndarray, shape (k, k)
        Factor covariance matrix (typically diagonal).
    D : np.ndarray, shape (p, p)
        Idiosyncratic covariance matrix (typically diagonal).
    factor_transform : np.ndarray, optional
        Square root of F for simulations.
    idio_transform : np.ndarray, optional
        Square root of D for simulations.
    """
    B: np.ndarray
    F: np.ndarray
    D: np.ndarray
    factor_transform: Optional[np.ndarray] = None
    idio_transform: Optional[np.ndarray] = None
```

**Properties**:

```python
@property
def k(self) -> int:
    """Number of factors."""
    return self.B.shape[0]

@property
def p(self) -> int:
    """Number of assets/securities."""
    return self.B.shape[1]

def implied_covariance(self) -> np.ndarray:
    """
    Compute implied covariance: Σ = B'FB + D.

    Returns
    -------
    np.ndarray, shape (p, p)
        Covariance matrix of returns.
    """
```

**Example**:

```python
import numpy as np
from factor_lab import FactorModelData

# Create a 3-factor model for 100 assets
k, p = 3, 100
B = np.random.randn(k, p)
F = np.diag([0.09, 0.04, 0.01])  # Factor variances
D = np.diag(np.full(p, 0.01))    # Idiosyncratic variances

model = FactorModelData(B=B, F=F, D=D)

print(f"Model: {model.k} factors, {model.p} assets")
print(f"Implied total variance: {np.trace(model.implied_covariance()):.4f}")
```

---

### `svd_decomposition`

**Description**: Extract factor model from returns using Singular Value Decomposition.

```python
def svd_decomposition(
    returns: np.ndarray,
    k: int,
    center: bool = True
) -> FactorModelData:
    """
    Extract k-factor model from returns via SVD.

    Parameters
    ----------
    returns : np.ndarray, shape (T, p)
        Returns matrix (T time periods, p assets).
    k : int
        Number of factors to extract (must be in [1, min(T-1, p)]).
    center : bool, default=True
        Whether to center (demean) returns before SVD.

    Returns
    -------
    FactorModelData
        Extracted factor model with:
        - B: Factor loadings (k, p)
        - F: Factor covariance (k, k), diagonal
        - D: Residual variance (p, p), diagonal

    Notes
    -----
    The model is estimated as:
    1. Center returns: X = returns - mean(returns) (if center=True)
    2. SVD: X = U Σ V'
    3. Extract top k components:
       - Factor variances: F = diag(σ₁²/(T-1), ..., σₖ²/(T-1))
       - Loadings: B = V[:k, :]
       - Residuals: D = diag(emp_var - model_var), bounded below by 1e-6

    Examples
    --------
    >>> returns = np.random.randn(500, 100)  # 500 periods, 100 assets
    >>> model = svd_decomposition(returns, k=3)
    >>> print(f"Extracted {model.k} factors")
    Extracted 3 factors
    >>> print(f"Explained variance: {np.trace(model.B.T @ model.F @ model.B) / np.trace(np.cov(returns.T)):.2%}")
    """
```

**Example**:

```python
from factor_lab import svd_decomposition
import numpy as np

# Generate synthetic returns
T, p = 500, 100
returns = np.random.randn(T, p)

# Extract 5 factors
model = svd_decomposition(returns, k=5, center=True)

print(f"Factor loadings shape: {model.B.shape}")  # (5, 100)
print(f"Factor variance: {np.diag(model.F)}")
print(f"Mean idio variance: {np.mean(np.diag(model.D)):.6f}")

# Compute explained variance
total_var = np.var(returns, axis=0, ddof=1).sum()
explained_var = np.trace(model.B.T @ model.F @ model.B)
print(f"Explained: {explained_var/total_var:.1%}")
```

---

### `ReturnsSimulator`

**Description**: Simulate returns from a factor model with Gaussian or custom distributions.

```python
class ReturnsSimulator:
    """
    Simulate returns from factor model.

    Parameters
    ----------
    model : FactorModelData
        Factor model to simulate from.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        Default: np.random.default_rng()
    """

    def __init__(self, model: FactorModelData, rng=None):
        self.model = model
        self.rng = rng or np.random.default_rng()

    def simulate(
        self,
        n_periods: int,
        factor_samplers=None,
        idio_samplers=None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns for n_periods.

        Parameters
        ----------
        n_periods : int
            Number of time periods to simulate.
        factor_samplers : callable or list of callables, optional
            Custom samplers for factor returns.
            If None, uses standard normal.
        idio_samplers : callable or list of callables, optional
            Custom samplers for idiosyncratic returns.
            If None, uses standard normal.

        Returns
        -------
        dict with keys:
            'security_returns' : np.ndarray, shape (n_periods, p)
                Simulated security returns.
            'factor_returns' : np.ndarray, shape (n_periods, k)
                Simulated factor returns.
            'idio_returns' : np.ndarray, shape (n_periods, p)
                Simulated idiosyncratic returns.

        Notes
        -----
        The simulation follows:
        1. Generate factor innovations: z_f ~ N(0, I_k)
        2. Transform: f = F^{1/2} z_f
        3. Generate idio innovations: z_ε ~ N(0, I_p)
        4. Transform: ε = D^{1/2} z_ε
        5. Compute returns: r = B'f + ε

        Examples
        --------
        >>> from factor_lab import FactorModelData, ReturnsSimulator
        >>> model = FactorModelData(B=B, F=F, D=D)
        >>> simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
        >>> results = simulator.simulate(n_periods=1000)
        >>> returns = results['security_returns']
        >>> print(f"Simulated shape: {returns.shape}")
        Simulated shape: (1000, 100)
        """
```

**Example**:

```python
from factor_lab import FactorModelData, ReturnsSimulator
import numpy as np

# Create model
k, p = 3, 50
B = np.random.randn(k, p)
F = np.diag([0.16, 0.09, 0.04])
D = np.diag(np.full(p, 0.01))
model = FactorModelData(B=B, F=F, D=D)

# Simulate with fixed seed
simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = simulator.simulate(n_periods=500)

# Access results
security_returns = results['security_returns']  # (500, 50)
factor_returns = results['factor_returns']      # (500, 3)
idio_returns = results['idio_returns']          # (500, 50)

# Verify structure
reconstructed = factor_returns @ B + idio_returns
assert np.allclose(security_returns, reconstructed)
print("✓ Simulation verified")
```

---

### `DistributionFactory`

**Description**: Factory for creating distribution samplers.

```python
class DistributionFactory:
    """
    Create distribution samplers for custom simulations.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def create(self, name: str, **params):
        """
        Create sampler for named distribution.

        Parameters
        ----------
        name : str
            Distribution name (e.g., 'normal', 'uniform', 't').
        **params
            Distribution parameters (e.g., mean=0, std=1).

        Returns
        -------
        callable
            Sampler function that returns random values.
        """
```

---

### `save_model`

**Description**: Save factor model to NPZ file.

```python
def save_model(model: FactorModelData, filename: str):
    """
    Save factor model to NPZ file.

    Parameters
    ----------
    model : FactorModelData
        Model to save.
    filename : str
        Output filename (will append .npz if needed).

    Example
    -------
    >>> from factor_lab import FactorModelData, save_model
    >>> model = FactorModelData(B=B, F=F, D=D)
    >>> save_model(model, 'my_model.npz')
    >>> 
    >>> # Load later
    >>> data = np.load('my_model.npz')
    >>> loaded = FactorModelData(B=data['B'], F=data['F'], D=data['D'])
    """
```

---

## Analysis Framework

Module: `factor_lab.analysis`

### `SimulationContext`

**Description**: Immutable snapshot of simulation state for analyses.

```python
@dataclass(frozen=True)
class SimulationContext:
    """
    Immutable context providing all data for analysis.

    Parameters
    ----------
    model : FactorModelData
        Ground truth factor model.
    security_returns : np.ndarray, shape (T, p)
        Simulated security returns.
    factor_returns : np.ndarray, shape (T, k)
        Simulated factor returns.
    idio_returns : np.ndarray, shape (T, p)
        Simulated idiosyncratic returns.
    spec : optional
        Model specification (if available).
    sim_spec : optional
        Simulation specification (if available).
    test_results : dict, optional
        Normality test results (if run).
    covariance_validation : optional
        Covariance validation result (if run).
    ground_truth_comparison : optional
        Ground truth comparison result (if run).
    timestamp : datetime
        When simulation was run.
    duration : float
        Simulation duration in seconds.
    """

    # Core data (required)
    model: FactorModelData
    security_returns: np.ndarray
    factor_returns: np.ndarray
    idio_returns: np.ndarray

    # Optional metadata (omitted for brevity)
    ...
```

**Properties**:

```python
@property
def T(self) -> int:
    """Number of time periods."""

@property
def p(self) -> int:
    """Number of securities."""

@property
def k(self) -> int:
    """Number of factors."""
```

**Methods**:

```python
def sample_covariance(self, ddof: int = 1) -> np.ndarray:
    """
    Compute sample covariance (cached after first call).

    Parameters
    ----------
    ddof : int, default=1
        Degrees of freedom.

    Returns
    -------
    np.ndarray, shape (p, p)
        Sample covariance matrix.
    """

def pca_decomposition(self, n_components: Optional[int] = None) -> FactorModelData:
    """
    Run PCA decomposition (cached by n_components).

    Parameters
    ----------
    n_components : int, optional
        Number of components. If None, uses model.k.

    Returns
    -------
    FactorModelData
        Extracted factor model from PCA.

    Notes
    -----
    Results cached by n_components for efficiency.
    """

def summary(self) -> str:
    """
    Generate formatted summary string.

    Returns
    -------
    str
        Multi-line summary of context.
    """
```

**Example**:

```python
from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext

# Setup
model = FactorModelData(B=B, F=F, D=D)
simulator = ReturnsSimulator(model)
results = simulator.simulate(n_periods=500)

# Create context
context = SimulationContext(
    model=model,
    security_returns=results['security_returns'],
    factor_returns=results['factor_returns'],
    idio_returns=results['idio_returns']
)

# Use properties
print(f"Problem size: T={context.T}, p={context.p}, k={context.k}")

# Compute sample covariance (cached)
cov = context.sample_covariance()  # Computed and cached
cov2 = context.sample_covariance()  # Retrieved from cache

# Run PCA (cached by k)
pca3 = context.pca_decomposition(n_components=3)  # Computed
pca3_again = context.pca_decomposition(n_components=3)  # From cache
pca5 = context.pca_decomposition(n_components=5)  # Computed (different k)

# Print summary
print(context.summary())
```

---

### `SimulationAnalysis`

**Description**: Protocol defining analysis interface.

```python
class SimulationAnalysis(Protocol):
    """
    Protocol for analysis classes.

    All analyses must implement the analyze method.
    """

    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Analyze simulation context.

        Parameters
        ----------
        context : SimulationContext
            Simulation data and model.

        Returns
        -------
        dict
            Analysis results as string keys to numeric values.
        """
```

---

## Built-in Analyses

Module: `factor_lab.analyses`

### `Analyses` (Factory Class)

**Description**: Factory for creating analysis objects.

```python
class Analyses:
    """Factory for creating built-in analyses."""

    @staticmethod
    def manifold_distances(
        use_pca_loadings: bool = True
    ) -> ManifoldDistanceAnalysis:
        """
        Create manifold distance analysis.

        Parameters
        ----------
        use_pca_loadings : bool, default=True
            If True, extract loadings via PCA and compare to model.B.
            If False, compare model.B to itself (for testing).

        Returns
        -------
        ManifoldDistanceAnalysis

        Example
        -------
        >>> manifold = Analyses.manifold_distances().analyze(context)
        >>> print(f"Grassmannian: {manifold['dist_grassmannian']:.4f}")
        """

    @staticmethod
    def eigenvalue_analysis(
        k_top: int = 10,
        compare_eigenvectors: bool = False,
        tol: float = 0,
        maxiter: Optional[int] = None
    ) -> ImplicitEigenAnalysis:
        """
        Create eigenvalue analysis using LinearOperator.

        Parameters
        ----------
        k_top : int, default=10
            Number of top eigenvalues to compute.
        compare_eigenvectors : bool, default=False
            Whether to also compare eigenvectors.
        tol : float, default=0
            Convergence tolerance for ARPACK.
        maxiter : int, optional
            Maximum iterations for ARPACK.

        Returns
        -------
        ImplicitEigenAnalysis

        Example
        -------
        >>> eigen = Analyses.eigenvalue_analysis(k_top=5).analyze(context)
        >>> print(f"Eigenvalue RMSE: {eigen['eigenvalue_rmse']:.4f}")
        """

    @staticmethod
    def eigenvector_comparison(
        k_components: int,
        align_signs: bool = True,
        compute_canonical_correlations: bool = True
    ) -> EigenvectorAlignment:
        """
        Create eigenvector comparison analysis.

        Parameters
        ----------
        k_components : int
            Number of eigenvectors to compare.
        align_signs : bool, default=True
            Whether to align signs before comparison.
        compute_canonical_correlations : bool, default=True
            Whether to compute canonical correlations.

        Returns
        -------
        EigenvectorAlignment

        Example
        -------
        >>> eigvec = Analyses.eigenvector_comparison(k=3).analyze(context)
        >>> print(f"Mean correlation: {eigvec['mean_correlation']:.4f}")
        """

    @staticmethod
    def custom(func: Callable[[SimulationContext], Dict[str, Any]]):
        """
        Create custom analysis from function.

        Parameters
        ----------
        func : callable
            Function taking SimulationContext and returning dict.

        Returns
        -------
        CustomAnalysis

        Example
        -------
        >>> custom = Analyses.custom(lambda ctx: {
        ...     'total_var': float(np.trace(ctx.model.implied_covariance()))
        ... })
        >>> result = custom.analyze(context)
        """
```

---

### `ManifoldDistanceAnalysis`

**Description**: Computes geometric distances on Grassmannian and Stiefel manifolds.

```python
class ManifoldDistanceAnalysis:
    """
    Manifold-based comparison of factor loadings.

    Computes rotation-invariant distances using manifold geometry.
    """

    def __init__(self, use_pca_loadings: bool = True):
        """
        Parameters
        ----------
        use_pca_loadings : bool, default=True
            Whether to extract PCA loadings for comparison.
        """

    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Compute manifold distances.

        Parameters
        ----------
        context : SimulationContext
            Analysis context.

        Returns
        -------
        dict with keys:
            dist_grassmannian : float
                Grassmannian distance (L2 norm of principal angles).
                Measures subspace distance, invariant to rotation.

            dist_procrustes : float
                Procrustes distance after optimal rotation.
                Measures frame distance after alignment.

            dist_chordal : float
                Chordal distance (raw Frobenius distance).
                Sensitive to rotation, useful for debugging.

            principal_angles : np.ndarray
                Principal angles in radians, shape (k,).

            optimal_rotation : np.ndarray
                Optimal orthogonal rotation matrix, shape (k, k).

            aligned_loadings : np.ndarray
                PCA loadings after optimal rotation, shape (k, p).

        Notes
        -----
        Grassmannian distance is rotation-invariant:
            d_G(U, RU) = 0 for any orthogonal R

        Procrustes finds optimal alignment:
            min_R ||B_true - B_pca @ R||_F

        Interpretation:
            - dist_grassmannian < 0.1: Excellent subspace recovery
            - dist_grassmannian < 0.3: Good subspace recovery
            - dist_grassmannian ≥ 0.3: Poor subspace recovery

        Examples
        --------
        >>> manifold = Analyses.manifold_distances().analyze(context)
        >>> 
        >>> print(f"Subspace distance: {manifold['dist_grassmannian']:.4f}")
        >>> print(f"Procrustes distance: {manifold['dist_procrustes']:.4f}")
        >>> 
        >>> # Check rotation invariance
        >>> Q = scipy.stats.ortho_group.rvs(k)
        >>> B_rotated = Q @ context.model.B
        >>> # Distance to rotated version should be ~0
        """
```

---

### `ImplicitEigenAnalysis`

**Description**: Memory-efficient eigenvalue computation using LinearOperator.

```python
class ImplicitEigenAnalysis:
    """
    Eigenvalue analysis without forming full covariance matrix.

    Uses LinearOperator for O(kp) memory instead of O(p²).
    """

    def __init__(
        self,
        k_top: int = 10,
        compare_eigenvectors: bool = False,
        tol: float = 0,
        maxiter: Optional[int] = None
    ):
        """
        Parameters
        ----------
        k_top : int
            Number of top eigenvalues to compute.
        compare_eigenvectors : bool
            Whether to also extract and compare eigenvectors.
        tol : float
            Convergence tolerance for ARPACK.
        maxiter : int, optional
            Maximum number of iterations.
        """

    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Compute eigenvalues and optionally eigenvectors.

        Parameters
        ----------
        context : SimulationContext
            Analysis context.

        Returns
        -------
        dict with keys:
            eigenvalue_rmse : float
                Root mean squared error of top-k eigenvalues.

            eigenvalue_errors : np.ndarray
                Absolute errors: λ̂ - λ, shape (k_top,).

            eigenvalue_rel_errors : np.ndarray
                Relative errors: (λ̂ - λ)/λ, shape (k_top,).

            true_eigenvalues : np.ndarray
                Ground truth eigenvalues, shape (k_top,).

            sample_eigenvalues : np.ndarray
                Estimated eigenvalues from PCA, shape (k_top,).

            eigenvector_metrics : dict, optional
                If compare_eigenvectors=True, includes:
                - subspace_distance
                - procrustes_distance
                - mean_correlation

        Notes
        -----
        Uses ARPACK (Lanczos iteration) via scipy.sparse.linalg.eigsh.

        Memory: O(kp) instead of O(p²)
        Time: O(k²p) instead of O(p³)

        Examples
        --------
        >>> eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)
        >>> 
        >>> print(f"Top 10 eigenvalue RMSE: {eigen['eigenvalue_rmse']:.6f}")
        >>> print(f"Largest relative error: {np.max(np.abs(eigen['eigenvalue_rel_errors'])):.2%}")
        >>> 
        >>> # Check convergence
        >>> errors = eigen['eigenvalue_errors']
        >>> plt.semilogy(errors)
        >>> plt.xlabel('Eigenvalue index')
        >>> plt.ylabel('Absolute error')
        """
```

---

### `EigenvectorAlignment`

**Description**: Eigenvector comparison with proper geometric alignment.

```python
class EigenvectorAlignment:
    """
    Compare eigenvectors with sign alignment and rotation handling.
    """

    def __init__(
        self,
        k_components: int,
        align_signs: bool = True,
        compute_canonical_correlations: bool = True
    ):
        """
        Parameters
        ----------
        k_components : int
            Number of eigenvectors to compare.
        align_signs : bool
            Whether to align signs before comparison.
        compute_canonical_correlations : bool
            Whether to compute canonical correlations.
        """

    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Compare eigenvectors of true vs sample covariance.

        Parameters
        ----------
        context : SimulationContext
            Analysis context.

        Returns
        -------
        dict with keys:
            mean_correlation : float
                Mean absolute correlation between paired eigenvectors.

            canonical_correlations : np.ndarray, optional
                Canonical correlations between subspaces, shape (k,).

            aligned_correlations : np.ndarray
                Per-component absolute correlations after sign alignment,
                shape (k,).

            subspace_distance : float
                Grassmannian distance between eigenvector subspaces.

            procrustes_distance : float
                Procrustes distance after optimal rotation.

            optimal_rotation : np.ndarray
                Optimal rotation matrix, shape (k, k).

            sign_flips : np.ndarray
                Sign flip indicators (±1), shape (k,).

        Notes
        -----
        Eigenvector comparison is tricky due to:
        1. Sign ambiguity: v and -v are both valid eigenvectors
        2. Rotation: Eigenvectors can be rotated in degenerate subspaces
        3. Permutation: Ordering matters

        This implementation handles all three via:
        - Sign alignment: Choose sign to maximize correlation
        - Procrustes: Find optimal rotation
        - Pairing: Use optimal rotation to determine pairing

        Examples
        --------
        >>> eigvec = Analyses.eigenvector_comparison(
        ...     k_components=5,
        ...     align_signs=True
        ... ).analyze(context)
        >>> 
        >>> print(f"Mean correlation: {eigvec['mean_correlation']:.4f}")
        >>> print(f"Subspace distance: {eigvec['subspace_distance']:.4f}")
        >>> 
        >>> # Visualize per-component correlations
        >>> import matplotlib.pyplot as plt
        >>> plt.bar(range(5), eigvec['aligned_correlations'])
        >>> plt.xlabel('Component')
        >>> plt.ylabel('|Correlation|')
        >>> plt.title('Eigenvector Recovery Quality')
        """
```

---

### Manifold Utility Functions

Module: `factor_lab.analyses.manifold`

```python
def orthonormalize(B: np.ndarray) -> np.ndarray:
    """
    Orthonormalize factor loadings via QR decomposition.

    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Factor loadings.

    Returns
    -------
    Q : np.ndarray, shape (p, k)
        Orthonormal frame with Q.T @ Q = I_k.

    Example
    -------
    >>> B = np.random.randn(3, 100)
    >>> Q = orthonormalize(B)
    >>> print(np.allclose(Q.T @ Q, np.eye(3)))
    True
    """

def compute_grassmannian_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute Grassmannian distance between subspaces.

    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated loadings.

    Returns
    -------
    distance : float
        L2 norm of principal angles.
    angles : np.ndarray
        Principal angles in radians, shape (k,).

    Example
    -------
    >>> # Test rotation invariance
    >>> Q = scipy.stats.ortho_group.rvs(3)
    >>> B_rotated = Q @ B
    >>> dist, angles = compute_grassmannian_distance(B, B_rotated)
    >>> print(f"Distance: {dist:.10f}")  # Should be ~0
    Distance: 0.0000000000
    """

def compute_procrustes_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Procrustes distance with optimal rotation.

    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated loadings.

    Returns
    -------
    dict with keys:
        distance : float
            Frobenius distance after optimal alignment.
        optimal_rotation : np.ndarray
            Optimal orthogonal rotation, shape (k, k).
        aligned_frame : np.ndarray
            Estimated frame after rotation, shape (p, k).

    Example
    -------
    >>> # Test sign flip handling
    >>> B_flipped = B.copy()
    >>> B_flipped[0, :] *= -1  # Flip first factor
    >>> result = compute_procrustes_distance(B, B_flipped)
    >>> print(f"Distance: {result['distance']:.10f}")  # Should be ~0
    Distance: 0.0000000000
    """

def compute_chordal_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> float:
    """
    Compute chordal (Frobenius) distance between frames.

    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated loadings.

    Returns
    -------
    float
        Frobenius distance ||Q_true - Q_estimated||_F.

    Note
    ----
    This is NOT rotation-invariant. Useful for debugging but
    not for comparing factor models.
    """
```

---

## Visualization

Module: `factor_lab.visualization`

### `create_manifold_dashboard`

**Description**: Create publication-quality static dashboard.

```python
def create_manifold_dashboard(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    context: Optional[SimulationContext] = None
) -> None:
    """
    Create comprehensive static visualization dashboard.

    Parameters
    ----------
    results : dict
        Analysis results (typically combined from multiple analyses).
    output_path : str, optional
        Where to save figure. If None, displays interactively.
    figsize : tuple, default=(16, 12)
        Figure size in inches.
    context : SimulationContext, optional
        If provided, adds model information to plots.

    Dashboard Layout
    ----------------
    3×3 grid with panels:
    1. Eigenvalue spectrum comparison
    2. Principal angles bar chart  
    3. Manifold distances comparison
    4. Eigenvalue errors (absolute)
    5. Eigenvalue errors (relative)
    6. Eigenvector correlations
    7. True loadings heatmap
    8. Estimated loadings heatmap
    9. Summary statistics table

    Example
    -------
    >>> # Run all analyses
    >>> manifold = Analyses.manifold_distances().analyze(context)
    >>> eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)
    >>> eigvec = Analyses.eigenvector_comparison(k=5).analyze(context)
    >>> 
    >>> # Combine results
    >>> all_results = {**manifold, **eigen, **eigvec}
    >>> 
    >>> # Create visualization
    >>> create_manifold_dashboard(
    ...     all_results,
    ...     output_path="dashboard.png",
    ...     figsize=(20, 14),
    ...     context=context
    ... )
    """
```

---

### `create_interactive_plotly_dashboard`

**Description**: Create interactive HTML dashboard with Plotly.

```python
def create_interactive_plotly_dashboard(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    context: Optional[SimulationContext] = None
) -> None:
    """
    Create interactive HTML dashboard.

    Parameters
    ----------
    results : dict
        Analysis results.
    output_path : str, optional
        Where to save HTML. If None, displays in browser.
    context : SimulationContext, optional
        If provided, adds model information.

    Features
    --------
    - Zoom, pan, hover tooltips
    - Responsive layout
    - Works in any modern browser
    - Shareable as single HTML file

    Example
    -------
    >>> create_interactive_plotly_dashboard(
    ...     all_results,
    ...     output_path="dashboard.html",
    ...     context=context
    ... )
    >>> # Open in browser: open dashboard.html
    """
```

---

### `print_verbose_results`

**Description**: Print formatted analysis results to console.

```python
def print_verbose_results(
    results: Dict[str, Any],
    context: Optional[SimulationContext] = None
) -> None:
    """
    Print beautifully formatted results to console.

    Parameters
    ----------
    results : dict
        Analysis results.
    context : SimulationContext, optional
        If provided, includes model summary.

    Features
    --------
    - Color-coded output (if terminal supports)
    - Organized by analysis type
    - Clear section headers
    - Quality indicators (✓, ○, ✗)

    Example
    -------
    >>> print_verbose_results(all_results, context)

    ======================================================================
      ANALYSIS RESULTS
    ======================================================================

    Model: 3 factors, 100 assets, 500 periods

    📐 Manifold Distances:
       Grassmannian distance:  0.2134  ✓ Good
       Procrustes distance:    0.0892  ✓ Excellent
       Chordal distance:       0.4567

    📊 Eigenvalue Analysis:
       RMSE (top 10):          0.0234  ✓ Good
       Max relative error:     8.2%    ○ Acceptable

    🎯 Eigenvector Comparison:
       Mean correlation:       0.9234  ✓ Excellent
       Subspace distance:      0.1567  ✓ Good
    """
```

---

## Complete Examples

### Example 1: Basic Workflow

```python
import numpy as np
from factor_lab import FactorModelData, ReturnsSimulator, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import create_manifold_dashboard, print_verbose_results

# 1. Create model
k, p, T = 3, 100, 500
B = np.random.randn(k, p)
F = np.diag([0.09, 0.04, 0.01])
D = np.diag(np.full(p, 0.01))
model = FactorModelData(B=B, F=F, D=D)

# 2. Simulate
simulator = ReturnsSimulator(model, rng=np.random.default_rng(42))
results = simulator.simulate(n_periods=T)

# 3. Create context
context = SimulationContext(
    model=model,
    security_returns=results['security_returns'],
    factor_returns=results['factor_returns'],
    idio_returns=results['idio_returns']
)

# 4. Run analyses
manifold = Analyses.manifold_distances().analyze(context)
eigen = Analyses.eigenvalue_analysis(k_top=10).analyze(context)
eigvec = Analyses.eigenvector_comparison(k_components=k).analyze(context)

# 5. Combine and visualize
all_results = {**manifold, **eigen, **eigvec}
print_verbose_results(all_results, context)
create_manifold_dashboard(all_results, "dashboard.png", context=context)
```

---

### Example 2: Custom Analysis

```python
from factor_lab import FactorModelData
from factor_lab.analysis import SimulationContext, SimulationAnalysis
from factor_lab.analyses import Analyses

class PortfolioConcentration(SimulationAnalysis):
    """Analyze factor loading concentration."""

    def analyze(self, context: SimulationContext):
        B = context.model.B

        # Herfindahl index per factor
        weights_squared = (B ** 2) / (B ** 2).sum(axis=1, keepdims=True)
        herfindahl = (weights_squared ** 2).sum(axis=1)

        return {
            'mean_herfindahl': float(herfindahl.mean()),
            'max_concentration': float(np.max(np.abs(B))),
            'loading_sparsity': float((np.abs(B) < 0.01).mean())
        }

# Use it
concentration = PortfolioConcentration()
results = concentration.analyze(context)
print(f"Mean Herfindahl: {results['mean_herfindahl']:.4f}")

# Or use lambda
quick_analysis = Analyses.custom(lambda ctx: {
    'condition_number': float(np.linalg.cond(ctx.model.B @ ctx.model.B.T))
})
results = quick_analysis.analyze(context)
```

---

### Example 3: Monte Carlo Study

```python
import numpy as np
from factor_lab import FactorModelData, ReturnsSimulator
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses

def monte_carlo_study(model, T_values, n_sims=100):
    """Study how metrics improve with sample size."""

    results_by_T = {}

    for T in T_values:
        metrics = {'grassmannian': [], 'eigenvalue_rmse': []}

        for sim in range(n_sims):
            # Simulate
            simulator = ReturnsSimulator(
                model, 
                rng=np.random.default_rng(42 + sim)
            )
            sim_results = simulator.simulate(n_periods=T)

            # Analyze
            context = SimulationContext(
                model=model,
                security_returns=sim_results['security_returns'],
                factor_returns=sim_results['factor_returns'],
                idio_returns=sim_results['idio_returns']
            )

            manifold = Analyses.manifold_distances().analyze(context)
            eigen = Analyses.eigenvalue_analysis(k_top=model.k).analyze(context)

            metrics['grassmannian'].append(manifold['dist_grassmannian'])
            metrics['eigenvalue_rmse'].append(eigen['eigenvalue_rmse'])

        results_by_T[T] = {
            'grass_mean': np.mean(metrics['grassmannian']),
            'grass_std': np.std(metrics['grassmannian']),
            'eigen_mean': np.mean(metrics['eigenvalue_rmse']),
            'eigen_std': np.std(metrics['eigenvalue_rmse'])
        }

    return results_by_T

# Run study
k, p = 3, 50
model = FactorModelData(
    B=np.random.randn(k, p),
    F=np.diag([0.16, 0.09, 0.04]),
    D=np.diag(np.full(p, 0.01))
)

results = monte_carlo_study(model, T_values=[50, 100, 200, 500, 1000])

# Report
for T, r in results.items():
    print(f"T={T:4d}: Grass={r['grass_mean']:.3f}±{r['grass_std']:.3f}")
```

---

## Quick Reference Tables

### Manifold Metrics Comparison

| Metric       | Invariant To                      | Use Case            | Typical Range |
| ------------ | --------------------------------- | ------------------- | ------------- |
| Grassmannian | Rotation, sign, permutation       | Subspace comparison | [0, π√k/2]    |
| Procrustes   | Sign, permutation (via optimal R) | Frame comparison    | [0, ∞)        |
| Chordal      | Nothing                           | Debugging           | [0, ∞)        |

### Interpretation Guidelines

| Grassmannian Distance | Interpretation | Action                             |
| --------------------- | -------------- | ---------------------------------- |
| < 0.1                 | Excellent      | Proceed confidently                |
| 0.1 - 0.3             | Good           | Acceptable for most uses           |
| 0.3 - 0.5             | Moderate       | Check assumptions                  |
| > 0.5                 | Poor           | Need more data or different method |

### Computational Complexity

| Operation       | Dense Method | Factor Method | Speedup |
| --------------- | ------------ | ------------- | ------- |
| Form Σ          | O(p²) space  | Not needed    | ∞       |
| Eigenvalues (k) | O(p³) time   | O(k²p) time   | O(p/k)  |
| Matvec          | O(p²) time   | O(kp) time    | O(p/k)  |

---

## See Also

- [README.md](../README.md) - Package overview
- [CHEATSHEET.md](CHEATSHEET.md) - Quick reference
- [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md) - Mathematical details
- [Examples](../examples/) - Working code examples

---

**Version**: 2.2.0  
**Last Updated**: February 2026  
**Status**: Production Ready
