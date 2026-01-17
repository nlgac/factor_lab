"""
types.py - Core Data Structures and Type Definitions for Factor Lab

This module defines the fundamental data structures used throughout factor_lab:
- FactorModelData: The central representation of a statistical factor model
- OptimizationResult: Results from portfolio optimization
- Scenario: Constraint definitions for optimization problems
- CovarianceTransform: Discriminated union for sqrt caching

Design Principles:
-----------------
1. Immutability where practical (frozen dataclasses for value objects)
2. Validation at construction time (fail-fast)
3. Clear type discrimination (no ambiguous Optional fields)
4. Numpy-style docstrings throughout

Example Usage:
-------------
    >>> import numpy as np
    >>> from factor_lab.types import FactorModelData
    >>> 
    >>> # Create a simple 2-factor, 10-asset model
    >>> B = np.random.randn(2, 10)  # Factor loadings
    >>> F = np.diag([0.04, 0.09])   # Factor covariance (diagonal)
    >>> D = np.diag(np.full(10, 0.01))  # Idiosyncratic covariance
    >>> 
    >>> model = FactorModelData(B=B, F=F, D=D)
    >>> print(f"Model: {model.k} factors, {model.p} assets")
    Model: 2 factors, 10 assets
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from enum import Enum, auto


# =============================================================================
# TYPE ALIASES
# =============================================================================

# A sampler is a callable that takes a size (int or tuple) and returns samples.
# Named 'Sampler' rather than 'Generator' to avoid confusion with Python generators.
SamplerCallable = Callable[[Union[int, Tuple[int, ...]]], np.ndarray]


# =============================================================================
# COVARIANCE TRANSFORM TYPES
# =============================================================================

class TransformType(Enum):
    """
    Discriminator for covariance matrix square root representations.
    
    DIAGONAL: The covariance is diagonal; sqrt is stored as a 1D vector of 
              standard deviations. Enables O(n) simulation.
    DENSE: The covariance has off-diagonal elements; sqrt is stored as the 
           full Cholesky factor (lower triangular). Requires O(n²) simulation.
    """
    DIAGONAL = auto()
    DENSE = auto()


@dataclass(frozen=True)
class CovarianceTransform:
    """
    Represents the square root of a covariance matrix for efficient simulation.
    
    This is a discriminated union: the `transform_type` field indicates how
    to interpret the `matrix` field.
    
    Parameters
    ----------
    matrix : np.ndarray
        Either a 1D array of standard deviations (DIAGONAL) or a 2D lower
        triangular Cholesky factor (DENSE).
    transform_type : TransformType
        Indicates how `matrix` should be interpreted and applied.
    
    Examples
    --------
    >>> # Diagonal case: just standard deviations
    >>> diag_tx = CovarianceTransform(
    ...     matrix=np.array([0.2, 0.3, 0.1]),
    ...     transform_type=TransformType.DIAGONAL
    ... )
    >>> 
    >>> # Dense case: full Cholesky factor
    >>> L = np.linalg.cholesky(some_covariance_matrix)
    >>> dense_tx = CovarianceTransform(
    ...     matrix=L,
    ...     transform_type=TransformType.DENSE
    ... )
    """
    matrix: np.ndarray
    transform_type: TransformType
    
    @property
    def is_diagonal(self) -> bool:
        """Check if this is a diagonal (O(n)) transform."""
        return self.transform_type == TransformType.DIAGONAL
    
    def apply(self, z: np.ndarray) -> np.ndarray:
        """
        Apply this covariance transform to standardized samples.
        
        Parameters
        ----------
        z : np.ndarray
            Standardized samples with shape (n_samples, dim). Expected to have
            zero mean and unit variance per column.
        
        Returns
        -------
        np.ndarray
            Transformed samples with the target covariance structure.
        
        Notes
        -----
        For DIAGONAL transforms: output = z * diag (element-wise)
        For DENSE transforms: output = z @ L.T (matrix multiplication)
        """
        if self.is_diagonal:
            return z * self.matrix  # Broadcasting: (n, d) * (d,)
        else:
            return z @ self.matrix.T  # (n, d) @ (d, d).T


# =============================================================================
# FACTOR MODEL DATA
# =============================================================================

@dataclass
class FactorModelData:
    """
    Complete specification of a statistical factor model.
    
    A factor model decomposes asset returns as:
        r = B.T @ f + epsilon
    
    where:
        - r: (p,) vector of asset returns
        - B: (k, p) matrix of factor loadings
        - f: (k,) vector of factor returns with covariance F
        - epsilon: (p,) vector of idiosyncratic returns with covariance D
    
    The implied covariance of returns is:
        Σ = B.T @ F @ B + D
    
    Parameters
    ----------
    B : np.ndarray
        Factor loadings matrix with shape (k, p) where k is the number of
        factors and p is the number of assets.
    F : np.ndarray
        Factor covariance matrix with shape (k, k). Often diagonal when
        factors are orthogonal (e.g., from PCA/SVD).
    D : np.ndarray
        Idiosyncratic covariance matrix with shape (p, p). Typically diagonal
        (asset-specific risk not explained by factors).
    factor_transform : Optional[CovarianceTransform]
        Pre-computed square root of F for efficient simulation. If None,
        will be computed on demand by the simulator.
    idio_transform : Optional[CovarianceTransform]
        Pre-computed square root of D for efficient simulation. If None,
        will be computed on demand by the simulator.
    
    Attributes
    ----------
    k : int
        Number of factors (read-only property).
    p : int
        Number of assets (read-only property).
    
    Examples
    --------
    >>> import numpy as np
    >>> from factor_lab.types import FactorModelData
    >>> 
    >>> # Market + Size factor model for 50 assets
    >>> k, p = 2, 50
    >>> B = np.vstack([
    ...     np.ones(p) * 1.0,           # Market betas ~1
    ...     np.linspace(-0.5, 0.5, p)   # Size exposure varies
    ... ])
    >>> F = np.diag([0.04, 0.01])  # Market vol=20%, Size vol=10%
    >>> D = np.diag(np.full(p, 0.0025))  # Idio vol=5% per asset
    >>> 
    >>> model = FactorModelData(B=B, F=F, D=D)
    >>> model.validate()  # Raises if inconsistent
    
    Notes
    -----
    The `validate()` method is automatically called in `__post_init__`.
    To skip validation (e.g., for performance in tight loops), use
    `object.__setattr__` to bypass `__post_init__`.
    """
    B: np.ndarray  # (k, p) Factor Loadings
    F: np.ndarray  # (k, k) Factor Covariance
    D: np.ndarray  # (p, p) Idiosyncratic Covariance
    
    # Pre-computed transforms for efficient simulation
    factor_transform: Optional[CovarianceTransform] = None
    idio_transform: Optional[CovarianceTransform] = None
    
    def __post_init__(self):
        """Validate dimensions on construction."""
        self.validate()
    
    @property
    def k(self) -> int:
        """Number of factors."""
        return self.B.shape[0]
    
    @property
    def p(self) -> int:
        """Number of assets."""
        return self.B.shape[1]
    
    def validate(self) -> None:
        """
        Validate internal consistency of the factor model.
        
        Raises
        ------
        ValueError
            If any dimension mismatches are detected.
        
        Notes
        -----
        Checks performed:
        1. B has shape (k, p) for some k > 0, p > 0
        2. F has shape (k, k)
        3. D has shape (p, p)
        4. If factor_transform provided, its dimension matches k
        5. If idio_transform provided, its dimension matches p
        """
        # Basic shape checks
        if self.B.ndim != 2:
            raise ValueError(f"B must be 2D, got shape {self.B.shape}")
        
        k, p = self.B.shape
        
        if k == 0 or p == 0:
            raise ValueError(f"B must have positive dimensions, got ({k}, {p})")
        
        if self.F.shape != (k, k):
            raise ValueError(
                f"F shape mismatch: expected ({k}, {k}), got {self.F.shape}"
            )
        
        if self.D.shape != (p, p):
            raise ValueError(
                f"D shape mismatch: expected ({p}, {p}), got {self.D.shape}"
            )
        
        # Validate pre-computed transforms if present
        if self.factor_transform is not None:
            self._validate_transform(self.factor_transform, k, "factor_transform")
        
        if self.idio_transform is not None:
            self._validate_transform(self.idio_transform, p, "idio_transform")
    
    def _validate_transform(
        self, 
        transform: CovarianceTransform, 
        expected_dim: int, 
        name: str
    ) -> None:
        """Validate a covariance transform has correct dimensions."""
        if transform.is_diagonal:
            if transform.matrix.shape != (expected_dim,):
                raise ValueError(
                    f"{name} diagonal vector has wrong size: "
                    f"expected ({expected_dim},), got {transform.matrix.shape}"
                )
        else:
            if transform.matrix.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"{name} dense matrix has wrong shape: "
                    f"expected ({expected_dim}, {expected_dim}), "
                    f"got {transform.matrix.shape}"
                )
    
    def implied_covariance(self) -> np.ndarray:
        """
        Compute the full implied covariance matrix Σ = B.T @ F @ B + D.
        
        Returns
        -------
        np.ndarray
            Full covariance matrix with shape (p, p).
        
        Warning
        -------
        This creates a dense (p, p) matrix. For large p, this may be
        memory-intensive. The factor model representation is preferred
        for optimization (O(k²) vs O(p²)).
        """
        return self.B.T @ self.F @ self.B + self.D


# =============================================================================
# OPTIMIZATION TYPES
# =============================================================================

@dataclass(frozen=True)
class OptimizationResult:
    """
    Result container for portfolio optimization.
    
    Parameters
    ----------
    weights : Optional[np.ndarray]
        Optimal portfolio weights with shape (p,). None if optimization failed.
    risk : float
        Portfolio risk (standard deviation) at the optimal point.
    objective : float
        Raw objective value from the solver.
    solved : bool
        True if the optimizer found an optimal solution.
    metadata : Dict[str, Any]
        Additional solver information (status, iterations, etc.).
    
    Examples
    --------
    >>> result = optimizer.solve()
    >>> if result.solved:
    ...     print(f"Optimal risk: {result.risk:.2%}")
    ...     print(f"Top holdings: {np.argsort(result.weights)[-5:]}")
    ... else:
    ...     print(f"Failed: {result.metadata.get('status')}")
    """
    weights: Optional[np.ndarray]
    risk: float
    objective: float
    solved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result consistency."""
        if self.solved and self.weights is None:
            raise ValueError("solved=True but weights is None")


# =============================================================================
# SCENARIO / CONSTRAINT TYPES
# =============================================================================

@dataclass
class Scenario:
    """
    A named collection of optimization constraints.
    
    Scenarios encapsulate the constraints that define a particular
    portfolio construction problem (e.g., "Long Only", "130/30", 
    "Sector Neutral").
    
    Parameters
    ----------
    name : str
        Human-readable identifier for this scenario.
    description : str
        Detailed description of what this scenario represents.
    equality_constraints : List[Tuple[np.ndarray, np.ndarray]]
        List of (A, b) tuples representing A @ w == b constraints.
    inequality_constraints : List[Tuple[np.ndarray, np.ndarray]]
        List of (A, b) tuples representing A @ w <= b constraints.
    
    Examples
    --------
    >>> scenario = Scenario(
    ...     name="Long Only Fully Invested",
    ...     description="Weights sum to 1, no shorting allowed"
    ... )
    >>> # Add constraints via ScenarioBuilder (preferred) or directly:
    >>> scenario.equality_constraints.append(
    ...     (np.ones((1, p)), np.array([1.0]))  # sum(w) = 1
    ... )
    
    Notes
    -----
    Constraints are stored in "A @ w <op> b" form where:
    - A has shape (m, p) for m constraints over p assets
    - b has shape (m,)
    - <op> is == for equality, <= for inequality
    
    Use ScenarioBuilder for a fluent API to construct scenarios.
    """
    name: str
    description: str = ""
    equality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    inequality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    
    def n_equality(self) -> int:
        """Count of equality constraints."""
        return sum(A.shape[0] for A, _ in self.equality_constraints)
    
    def n_inequality(self) -> int:
        """Count of inequality constraints."""
        return sum(A.shape[0] for A, _ in self.inequality_constraints)
    
    def __repr__(self) -> str:
        return (
            f"Scenario(name='{self.name}', "
            f"eq={self.n_equality()}, ineq={self.n_inequality()})"
        )


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================

@dataclass(frozen=True)
class CovarianceValidationResult:
    """
    Results from comparing empirical vs model-implied covariance.
    
    Parameters
    ----------
    frobenius_error : float
        Frobenius norm of (empirical - model) covariance difference.
    mean_absolute_error : float
        Average absolute element-wise error.
    max_absolute_error : float
        Maximum absolute element-wise error.
    explained_variance_ratio : float
        Ratio of factor-explained variance to total variance.
    model_covariance : np.ndarray
        The model-implied covariance matrix B.T @ F @ B + D.
    empirical_covariance : np.ndarray
        Sample covariance computed from returns.
    
    Examples
    --------
    >>> validation = validator.compare(simulated_returns)
    >>> print(f"Model fit: {validation.frobenius_error:.4f} (Frobenius)")
    >>> if validation.frobenius_error > 0.5:
    ...     print("Warning: Poor model fit!")
    """
    frobenius_error: float
    mean_absolute_error: float
    max_absolute_error: float
    explained_variance_ratio: float
    model_covariance: np.ndarray
    empirical_covariance: np.ndarray
