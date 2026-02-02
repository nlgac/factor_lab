"""
decomposition.py - Efficient Factor Extraction Algorithms
=========================================================

Provides high-performance algorithms for extracting factor models.
Uses loguru for diagnostics and supports Truncated SVD/LinearOperators.
"""

from __future__ import annotations
from enum import Enum
from typing import Tuple, Optional, Union
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
from loguru import logger

from .types import FactorModelData, CovarianceTransform, TransformType

# =============================================================================
# ENUMS (Required by __init__.py)
# =============================================================================

class PCAMethod(str, Enum):
    """Available methods for PCA decomposition."""
    SVD = "svd"
    EIG = "eig"
    NUMPY = "numpy"  # Added for compatibility
    SCIPY = "scipy"  # Added for compatibility

# =============================================================================
# HELPER: LOW-LEVEL SOLVERS
# =============================================================================

def _compute_svd(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dispatches to the most efficient SVD solver. 
    Uses Truncated SVD (ARPACK) for large k << p problems.
    """
    min_dim = min(X.shape)
    
    # Use Iterative solver if we want few factors from a large matrix
    if k < 0.1 * min_dim and min_dim > 500:
        logger.debug(f"Using Truncated SVD (ARPACK) | Shape: {X.shape}, k: {k}")
        # svds returns ascending order, we flip for PCA consistency
        u, s, vt = scipy.sparse.linalg.svds(X, k=k)
        return u[:, ::-1], s[::-1], vt[::-1, :]
    
    logger.debug(f"Using Dense SVD (LAPACK) | Shape: {X.shape}, k: {k}")
    u, s, vt = scipy.linalg.svd(X, full_matrices=False)
    return u[:, :k], s[:k], vt[:k, :]

# =============================================================================
# MAIN DECOMPOSITION FUNCTIONS
# =============================================================================

def svd_decomposition(
    returns: np.ndarray, 
    k: int, 
    center: bool = True,
    demean: Optional[bool] = None  # If None, uses center value
) -> FactorModelData:
    """
    Extract factor model from returns via SVD. (Best Practice)
    
    Parameters
    ----------
    returns : ndarray (T, p)
        Return matrix with T periods and p assets
    k : int
        Number of factors to extract
    center : bool, default=True
        Whether to center returns (same as demean)
    demean : bool, default=True
        Whether to demean returns (alias for center)
        
    Returns
    -------
    model : FactorModelData
        Factor model with B, F, D, and transforms
        
    Raises
    ------
    ValueError
        If returns is not 2D
    """
    # Validate input
    if returns.ndim != 2:
        raise ValueError(f"Returns must be 2D array, got shape {returns.shape}")
    
    T, p = returns.shape
    
    # Validate minimum dimensions
    if T < 2:
        raise ValueError(f"Returns must have at least 2 time periods, got T={T}")
    
    # Validate k
    max_k = min(T, p)
    if k < 1 or k > max_k:
        raise ValueError(f"k must be in range [1, {max_k}], got k={k}")
    
    logger.info(f"Starting SVD Decomposition: {T} periods, {p} assets, k={k}")

    # demean parameter overrides center if explicitly provided
    should_center = demean if demean is not None else center
    
    # 1. Centering
    X = returns - returns.mean(axis=0) if should_center else returns

    # 2. Perform SVD
    try:
        U, s, Vt = _compute_svd(X, k=k)
    except Exception as e:
        logger.exception("SVD solver failed. Check for NaNs or Infinite values in returns.")
        raise

    # 3. Construct Components
    factor_variances = (s**2) / (T - 1)
    B = Vt  # (k, p)
    F = np.diag(factor_variances)
    
    # 4. Efficient Residuals (O(p) memory)
    # D_ii = Var(r_i) - (B.T @ F @ B)_ii
    emp_var = np.var(X, axis=0, ddof=1)
    
    # Diag(B.T @ F @ B) = sum(B_ij^2 * var_j)
    model_var = np.sum((B.T ** 2) * factor_variances, axis=1)
    
    d_diag = np.maximum(emp_var - model_var, 1e-6)
    D = np.diag(d_diag)
    
    # 5. Create transforms
    # Factor transform: sqrt of diagonal elements (1D vector)
    factor_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(F)),  # Extract diagonal as 1D vector (k,)
        transform_type=TransformType.DIAGONAL
    )
    
    # Idiosyncratic transform: sqrt of diagonal elements (1D vector)
    idio_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(D)),  # Extract diagonal as 1D vector (p,)
        transform_type=TransformType.DIAGONAL
    )

    explained_var = np.sum(factor_variances) / np.sum(emp_var)
    logger.success(f"SVD Complete. Explained Variance: {explained_var:.2%}")
    
    return FactorModelData(
        B=B, 
        F=F, 
        D=D,
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )


def pca_decomposition(
    covariance: Union[np.ndarray, LinearOperator], 
    k: int,
    method: str = "numpy"  # Added method parameter
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract factors from Covariance or an implicit LinearOperator.
    
    Parameters
    ----------
    covariance : ndarray or LinearOperator (p, p)
        Covariance matrix or linear operator
    k : int
        Number of factors to extract
    method : str, default='numpy'
        Method to use: 'numpy', 'scipy', or auto-detect
        
    Returns
    -------
    B : ndarray (k, p)
        Factor loadings (eigenvectors as rows)
    F : ndarray (k, k)
        Factor covariance (diagonal of eigenvalues)
        
    Raises
    ------
    ValueError
        If method is invalid or k is out of range
    """
    p = covariance.shape[0]
    
    # Validate k
    if k < 1 or k > p:
        raise ValueError(f"k must be in range [1, {p}], got k={k}")
    
    # Validate method
    valid_methods = ['numpy', 'scipy', 'auto']
    if method not in valid_methods:
        raise ValueError(f"Unknown method: '{method}'. Valid methods are: {valid_methods}")
    
    # Auto-detect or use specified method
    use_iterative = (
        method == 'scipy' or 
        (method == 'numpy' and (isinstance(covariance, LinearOperator) or p > 2000))
    )
    
    if use_iterative or isinstance(covariance, LinearOperator):
        logger.info(f"Using Iterative Eigensolver (Lanczos) for p={p}")
        vals, vecs = scipy.sparse.linalg.eigsh(covariance, k=k, which='LM')
        vals, vecs = vals[::-1], vecs[:, ::-1]
    else:
        logger.info(f"Using Dense Eigensolver (numpy/scipy) for p={p}")
        vals, vecs = scipy.linalg.eigh(covariance)
        idx = np.argsort(vals)[::-1][:k]
        vals, vecs = vals[idx], vecs[:, idx]

    return vecs.T, np.diag(vals)

# =============================================================================
# UTILITIES (Restored)
# =============================================================================

def compute_explained_variance(model: FactorModelData) -> float:
    """
    Calculates the ratio of Systematic Variance to Total Variance.
    
    Ratio = Trace(Systematic) / (Trace(Systematic) + Idiosyncratic))
    """
    # Trace(B.T @ F @ B) = Trace(F @ B @ B.T)
    # Since F is diagonal and B is orthonormal (from SVD), B @ B.T is approx Identity (in subspace)
    # More robustly: sum(diag(F) * norm(B, axis=1)^2) ? 
    # Actually, for SVD B is orthonormal rows, so sum(diag(F)) is the trace of Systematic.
    
    sys_var = np.sum(np.diag(model.F))
    idio_var = np.sum(np.diag(model.D))
    
    total = sys_var + idio_var
    if total == 0:
        return 0.0
        
    return sys_var / total


def select_k_by_variance(
    returns: np.ndarray, 
    target_explained: float = 0.90,
    max_k: Optional[int] = None
) -> int:
    """
    Finds minimum k using Singular Value spectrum.
    
    Parameters
    ----------
    returns : ndarray (T, p)
        Return matrix
    target_explained : float, default=0.90
        Target explained variance ratio
    max_k : int, optional
        Maximum k to consider
        
    Returns
    -------
    k : int
        Number of factors needed to reach target
        
    Raises
    ------
    ValueError
        If returns is not 2D
    """
    # Validate input
    if returns.ndim != 2:
        raise ValueError(f"Returns must be 2D array, got shape {returns.shape}")
    
    T, p = returns.shape
    if max_k is None:
        max_k = min(T, p) - 1

    X = returns - returns.mean(axis=0)
    
    # Just get singular values (fast)
    s = scipy.linalg.svdvals(X)
    
    vars_ = (s ** 2) / (T - 1)
    cum_ratio = np.cumsum(vars_) / np.sum(vars_)
    
    k = np.searchsorted(cum_ratio, target_explained) + 1
    k = min(k, max_k)
    
    logger.info(f"Target {target_explained:.0%} variance requires k={k}")
    return int(k)
