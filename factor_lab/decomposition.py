"""
decomposition.py - Factor Model Extraction via PCA and SVD

This module provides algorithms for extracting factor models from data:
- PCA: Extract factors from a covariance matrix
- SVD: Extract factors directly from a returns matrix (numerically superior)

Mathematical Background:
-----------------------
A factor model represents asset returns r ∈ ℝᵖ as:
    r = B.T @ f + ε

where f ∈ ℝᵏ are factor returns and ε ∈ ℝᵖ is idiosyncratic noise.

The covariance structure is:
    Σ = B.T @ F @ B + D

where F = Cov(f) and D = Cov(ε) (typically diagonal).

**PCA Approach**: Given Σ directly, extract top-k eigenvectors.
    - Pros: Works with any covariance estimate
    - Cons: Numerically less stable (squares the condition number)

**SVD Approach**: Given returns matrix X, decompose directly.
    - Pros: Numerically stable, directly produces simulation-ready sqrt
    - Cons: Requires raw returns data

Example Usage:
-------------
    >>> import numpy as np
    >>> from factor_lab.decomposition import svd_decomposition, pca_decomposition
    >>> 
    >>> # From returns (preferred)
    >>> returns = np.random.randn(1000, 50) * 0.01  # 1000 days, 50 assets
    >>> model_svd = svd_decomposition(returns, k=5)
    >>> 
    >>> # From covariance
    >>> cov = np.cov(returns, rowvar=False)
    >>> B, F = pca_decomposition(cov, k=5)

References:
----------
- Jolliffe, I.T. (2002). Principal Component Analysis. Springer.
- Golub & Van Loan (2013). Matrix Computations. Johns Hopkins.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import Tuple, Literal
from enum import Enum, auto

from .types import (
    FactorModelData, 
    CovarianceTransform, 
    TransformType
)


# =============================================================================
# DECOMPOSITION METHODS ENUM
# =============================================================================

class PCAMethod(Enum):
    """
    Available methods for PCA eigendecomposition.
    
    NUMPY: Uses np.linalg.eigh - general purpose, good for small/medium matrices.
    SCIPY: Uses scipy.linalg.eigh with subset selection - more efficient when
           k << n because it only computes the top k eigenvalues.
    """
    NUMPY = auto()
    SCIPY = auto()


# =============================================================================
# PCA DECOMPOSITION
# =============================================================================

def pca_decomposition(
    covariance: np.ndarray,
    k: int,
    method: Literal["numpy", "scipy"] = "numpy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a k-factor model from a covariance matrix via eigendecomposition.
    
    Decomposes the covariance matrix Σ to find the top-k principal components:
        Σ ≈ V @ Λ @ V.T
    
    where V contains the top-k eigenvectors and Λ is diagonal with top-k 
    eigenvalues.
    
    The factor model representation is:
        B = V.T  (factor loadings, shape k×p)
        F = Λ    (factor covariance, shape k×k, diagonal)
    
    Parameters
    ----------
    covariance : np.ndarray
        Symmetric positive semi-definite covariance matrix with shape (p, p).
    k : int
        Number of factors to extract. Must satisfy 1 <= k <= p.
    method : {"numpy", "scipy"}, default="numpy"
        Eigendecomposition method:
        - "numpy": Full decomposition, then truncate. Simple and robust.
        - "scipy": Partial decomposition of only top-k. More efficient for k << p.
    
    Returns
    -------
    B : np.ndarray
        Factor loadings matrix with shape (k, p). Rows are orthonormal
        eigenvectors (principal components).
    F : np.ndarray
        Factor covariance matrix with shape (k, k). Diagonal matrix of
        eigenvalues (explained variances).
    
    Raises
    ------
    ValueError
        If covariance is not square, or k is out of valid range.
    
    Examples
    --------
    >>> import numpy as np
    >>> from factor_lab.decomposition import pca_decomposition
    >>> 
    >>> # Create a sample covariance with known structure
    >>> p = 100
    >>> true_factors = np.random.randn(3, p)
    >>> cov = true_factors.T @ true_factors + np.eye(p) * 0.1
    >>> 
    >>> # Extract 3 factors
    >>> B, F = pca_decomposition(cov, k=3)
    >>> print(f"Loadings shape: {B.shape}")  # (3, 100)
    >>> print(f"Top eigenvalue: {F[0, 0]:.2f}")
    
    Notes
    -----
    The eigenvalues (diagonal of F) represent the variance explained by each
    factor. They are returned in descending order.
    
    For numerical stability with ill-conditioned covariance matrices,
    consider using `svd_decomposition` on the raw returns instead.
    
    See Also
    --------
    svd_decomposition : More numerically stable alternative using raw returns.
    """
    # Input validation
    if covariance.ndim != 2:
        raise ValueError(f"Covariance must be 2D, got shape {covariance.shape}")
    
    n = covariance.shape[0]
    
    if covariance.shape != (n, n):
        raise ValueError(
            f"Covariance must be square, got shape {covariance.shape}"
        )
    
    if not (1 <= k <= n):
        raise ValueError(
            f"k must be in [1, {n}], got {k}"
        )
    
    # Perform eigendecomposition
    if method == "scipy":
        # scipy.linalg.eigh can compute only a subset of eigenvalues/vectors
        # subset_by_index selects eigenvalues by index (0 = smallest)
        # We want the largest k, so indices [n-k, n-1]
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            covariance,
            subset_by_index=(n - k, n - 1)
        )
        # Results come in ascending order; reverse to descending
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
    elif method == "numpy":
        # numpy computes all eigenvalues, then we select top-k
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # eigh returns ascending order; get indices for descending
        idx = np.argsort(eigenvalues)[::-1][:k]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'numpy' or 'scipy'.")
    
    # Construct factor model components
    # B: (k, p) - rows are eigenvectors (principal directions)
    # F: (k, k) - diagonal matrix of eigenvalues (variances)
    B = eigenvectors.T  # Transpose so rows are factors
    F = np.diag(eigenvalues)
    
    return B, F


# =============================================================================
# SVD DECOMPOSITION
# =============================================================================

def svd_decomposition(
    returns: np.ndarray,
    k: int,
    demean: bool = True
) -> FactorModelData:
    """
    Extract a k-factor model from a returns matrix via Singular Value Decomposition.
    
    This is the preferred method for factor extraction because:
    1. It operates directly on returns (no need to form covariance matrix)
    2. Numerically more stable (condition number not squared)
    3. Automatically produces simulation-ready square roots
    
    Mathematical Details:
    --------------------
    Given centered returns X ∈ ℝ^(T×p), the SVD is:
        X = U @ S @ V.T
    
    where:
        - U: (T, T) orthonormal (left singular vectors = "factor scores")
        - S: (min(T,p),) singular values
        - V: (p, p) orthonormal (right singular vectors = "factor loadings")
    
    The relationship to eigendecomposition of covariance:
        Σ = X.T @ X / (T-1) = V @ (S² / (T-1)) @ V.T
    
    So the factor model is:
        B = V[:, :k].T     (loadings)
        F = diag(s²/(T-1)) (factor variances)
    
    Parameters
    ----------
    returns : np.ndarray
        Returns matrix with shape (T, p) where T is the number of time periods
        and p is the number of assets.
    k : int
        Number of factors to extract. Must satisfy 1 <= k <= min(T, p).
    demean : bool, default=True
        Whether to subtract the mean from each column before SVD.
        Set to False if returns are already centered.
    
    Returns
    -------
    FactorModelData
        Complete factor model with:
        - B: (k, p) factor loadings
        - F: (k, k) factor covariance (diagonal)
        - D: (p, p) idiosyncratic covariance (diagonal)
        - factor_transform: Pre-computed sqrt of F for simulation
        - idio_transform: Pre-computed sqrt of D for simulation
    
    Raises
    ------
    ValueError
        If returns has wrong shape or k is out of valid range.
    
    Examples
    --------
    >>> import numpy as np
    >>> from factor_lab.decomposition import svd_decomposition
    >>> 
    >>> # Simulate some returns with factor structure
    >>> T, p = 1000, 50
    >>> true_factors = np.random.randn(T, 3) * 0.1
    >>> loadings = np.random.randn(3, p)
    >>> idio = np.random.randn(T, p) * 0.02
    >>> returns = true_factors @ loadings + idio
    >>> 
    >>> # Extract the factor model
    >>> model = svd_decomposition(returns, k=3)
    >>> print(f"Extracted {model.k} factors for {model.p} assets")
    >>> print(f"Top factor variance: {model.F[0,0]:.4f}")
    >>> 
    >>> # The model comes with pre-computed transforms for simulation
    >>> assert model.factor_transform is not None
    >>> assert model.idio_transform is not None
    
    Notes
    -----
    The explained variance ratio can be computed as:
        explained = sum(diag(F)) / sum(s² / (T-1))
    
    This is logged during extraction for diagnostic purposes.
    
    Performance Considerations:
    - Uses scipy.linalg.svd with full_matrices=False (economy SVD)
    - Time complexity: O(T * p * min(T, p))
    - Memory: O(T * p) for the centered data
    
    See Also
    --------
    pca_decomposition : Alternative when only covariance matrix is available.
    """
    # Input validation
    if returns.ndim != 2:
        raise ValueError(f"Returns must be 2D, got shape {returns.shape}")
    
    T, p = returns.shape
    
    if T < 2:
        raise ValueError(f"Need at least 2 time periods, got {T}")
    
    max_k = min(T, p)
    if not (1 <= k <= max_k):
        raise ValueError(f"k must be in [1, {max_k}], got {k}")
    
    # Step 1: Center the data if requested
    if demean:
        X = returns - returns.mean(axis=0)
    else:
        X = returns
    
    # Step 2: Compute SVD
    # full_matrices=False gives "economy" SVD: U is (T, min(T,p)), etc.
    # This is more memory efficient when T >> p or p >> T
    U, s, Vt = scipy.linalg.svd(X, full_matrices=False)
    
    # Step 3: Extract top-k components
    s_k = s[:k]       # Top k singular values
    Vt_k = Vt[:k, :]  # Top k right singular vectors (rows of Vt)
    
    # Step 4: Construct factor model
    # Factor loadings are the right singular vectors
    B = Vt_k  # Shape: (k, p)
    
    # Factor variances: eigenvalues of covariance = s² / (T-1)
    factor_variances = (s_k ** 2) / (T - 1)
    F = np.diag(factor_variances)  # Shape: (k, k)
    
    # Step 5: Compute idiosyncratic component
    # Reconstruct the "explained" portion of returns
    # X_explained = U[:, :k] @ diag(s_k) @ Vt_k
    # Efficient computation: (U[:, :k] * s_k) @ Vt_k
    X_explained = (U[:, :k] * s_k) @ Vt_k
    residuals = X - X_explained
    
    # Idiosyncratic variances (per-asset residual variance)
    idio_variances = np.var(residuals, axis=0, ddof=1)
    D = np.diag(idio_variances)  # Shape: (p, p)
    
    # Step 6: Create pre-computed transforms for simulation
    # Since F and D are diagonal, we store just the standard deviations
    factor_transform = CovarianceTransform(
        matrix=np.sqrt(factor_variances),
        transform_type=TransformType.DIAGONAL
    )
    
    idio_transform = CovarianceTransform(
        matrix=np.sqrt(idio_variances),
        transform_type=TransformType.DIAGONAL
    )
    
    # Compute explained variance ratio for diagnostics
    total_variance = np.sum(s ** 2) / (T - 1)
    explained_variance = np.sum(factor_variances)
    explained_ratio = explained_variance / total_variance if total_variance > 0 else 0.0
    
    # Note: We don't log here to keep the module dependency-free.
    # The caller can compute this from the returned model if needed.
    
    # Step 7: Construct and return the model
    # We need to bypass __post_init__ validation temporarily because
    # we're setting transforms that would fail validation before B, F, D are set.
    # Actually, dataclass handles this correctly - let's just create normally.
    
    return FactorModelData(
        B=B,
        F=F,
        D=D,
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_explained_variance(model: FactorModelData) -> float:
    """
    Compute the fraction of total variance explained by the factors.
    
    Parameters
    ----------
    model : FactorModelData
        A fitted factor model.
    
    Returns
    -------
    float
        Ratio in [0, 1] of variance explained by factors vs total variance.
    
    Examples
    --------
    >>> model = svd_decomposition(returns, k=5)
    >>> ratio = compute_explained_variance(model)
    >>> print(f"Factors explain {ratio:.1%} of variance")
    
    Notes
    -----
    Computed as: trace(B.T @ F @ B) / trace(B.T @ F @ B + D)
    """
    factor_var = np.trace(model.B.T @ model.F @ model.B)
    idio_var = np.trace(model.D)
    total_var = factor_var + idio_var
    
    if total_var == 0:
        return 0.0
    
    return factor_var / total_var


def select_k_by_variance(
    returns: np.ndarray,
    target_explained: float = 0.90,
    max_k: int = None
) -> int:
    """
    Determine the number of factors needed to explain a target variance fraction.
    
    Parameters
    ----------
    returns : np.ndarray
        Returns matrix with shape (T, p).
    target_explained : float, default=0.90
        Target fraction of variance to explain (e.g., 0.90 for 90%).
    max_k : int, optional
        Maximum number of factors to consider. Defaults to min(T, p).
    
    Returns
    -------
    int
        Minimum k such that the top-k factors explain at least `target_explained`
        fraction of total variance.
    
    Examples
    --------
    >>> # Find how many factors to explain 95% of variance
    >>> k = select_k_by_variance(returns, target_explained=0.95)
    >>> model = svd_decomposition(returns, k=k)
    
    Notes
    -----
    This performs a full SVD to compute all singular values, then finds
    the cumulative explained variance. For very large matrices, consider
    using randomized SVD (not implemented here).
    """
    if returns.ndim != 2:
        raise ValueError(f"Returns must be 2D, got shape {returns.shape}")
    
    T, p = returns.shape
    
    if max_k is None:
        max_k = min(T - 1, p)
    
    # Center and compute SVD
    X = returns - returns.mean(axis=0)
    _, s, _ = scipy.linalg.svd(X, full_matrices=False)
    
    # Compute cumulative explained variance
    variances = (s ** 2) / (T - 1)
    total_var = variances.sum()
    cumulative_ratio = np.cumsum(variances) / total_var
    
    # Find minimum k
    for k in range(1, min(len(s), max_k) + 1):
        if cumulative_ratio[k - 1] >= target_explained:
            return k
    
    return max_k
