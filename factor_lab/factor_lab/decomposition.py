"""
decomposition.py - Factor Model Decomposition

SVD-based factor extraction from returns data.
"""

import numpy as np
from .factor_types import FactorModelData

__all__ = ['svd_decomposition']


def svd_decomposition(
    returns: np.ndarray,
    k: int,
    center: bool = True
) -> FactorModelData:
    """
    Extract factor model via SVD decomposition.
    
    Parameters
    ----------
    returns : np.ndarray, shape (T, p)
        Return matrix (T periods, p assets)
    k : int
        Number of factors to extract
    center : bool, default=True
        Whether to center returns before decomposition
    
    Returns
    -------
    FactorModelData
        Extracted factor model with B, F, D matrices
    
    Notes
    -----
    Sign normalization: Each factor loading is normalized to have
    positive mean, ensuring consistent sign convention.
    """
    T, p = returns.shape
    
    if k > min(T, p):
        raise ValueError(f"k={k} cannot exceed min(T={T}, p={p})")
    
    # Center returns if requested
    X = returns - returns.mean(axis=0) if center else returns
    
    # SVD decomposition
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Extract top k components
    factor_variances = (s[:k] ** 2) / (T - 1)
    B = Vt[:k, :]  # Shape: (k, p)
    
    # Sign normalization: ensure each factor has positive mean
    for i in range(k):
        if B[i, :].mean() < 0:
            B[i, :] *= -1
    
    # Factor covariance matrix (diagonal)
    F = np.diag(factor_variances)
    
    # Idiosyncratic variances
    emp_var = np.var(X, axis=0, ddof=1)
    model_var = np.sum((B.T ** 2) * factor_variances, axis=1)
    d_diag = np.maximum(emp_var - model_var, 1e-6)
    D = np.diag(d_diag)
    
    return FactorModelData(B=B, F=F, D=D)
