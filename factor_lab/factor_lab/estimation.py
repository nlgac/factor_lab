"""
estimation.py - Factor Model Estimation

Algorithms for estimating factor models from return data.
"""

import numpy as np
try:
    from .factor_types import FactorModelData
except ImportError:
    from factor_types import FactorModelData


def svd_decomposition(
    returns: np.ndarray, 
    k: int, 
    center: bool = True
) -> FactorModelData:
    """
    Extract factor model via Singular Value Decomposition (SVD).
    
    Estimates a k-factor model from historical returns using PCA/SVD.
    The method extracts the top k principal components as factors.
    
    Parameters
    ----------
    returns : np.ndarray, shape (T, p)
        Historical returns matrix
        - T: number of time periods
        - p: number of assets
    k : int
        Number of factors to extract
    center : bool, default=True
        Whether to center returns (subtract mean) before decomposition
        
    Returns
    -------
    model : FactorModelData
        Estimated factor model with:
        - B: (k, p) factor loadings
        - F: (k, k) factor covariance (diagonal)
        - D: (p, p) idiosyncratic covariance (diagonal)
        
    Examples
    --------
    Estimate 3-factor model from returns:
        >>> import numpy as np
        >>> T, p, k = 1000, 100, 3
        >>> returns = np.random.randn(T, p) * 0.01  # 1% vol
        >>> model = svd_decomposition(returns, k=k)
        >>> print(model.B.shape)  # (3, 100)
        >>> print(model.F.shape)  # (3, 3)
        >>> print(model.D.shape)  # (100, 100)
    
    Without centering (for already-centered data):
        >>> model = svd_decomposition(returns, k=3, center=False)
    
    Notes
    -----
    Algorithm:
    1. Center returns (if center=True): X = returns - mean
    2. Compute SVD: X = U @ diag(s) @ V'
    3. Extract top k components:
       - B = V[:k, :] (first k right singular vectors)
       - F = diag(s[:k]^2 / (T-1)) (factor variances)
       - D = diag(residual variances) (idiosyncratic variances)
    4. Sign normalization: flip factors with negative mean loadings
    
    The sign normalization ensures that factors tend to have positive
    average loadings across assets, making interpretation easier.
    
    References
    ----------
    - Bai, J., & Ng, S. (2002). Determining the number of factors in 
      approximate factor models. Econometrica, 70(1), 191-221.
    """
    T, p = returns.shape
    
    # Center returns if requested
    X = returns - returns.mean(axis=0) if center else returns
    
    # SVD: X = U @ diag(s) @ V'
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Extract top k components
    factor_variances = (s[:k]**2) / (T - 1)
    B = Vt[:k, :]  # First k rows of V'
    
    # Sign normalization: flip factors with negative mean loadings
    row_means = B.mean(axis=1)  # Mean across assets for each factor
    sign_flips = np.where(row_means < 0, -1, 1)  # -1 if mean < 0, else +1
    B = B * sign_flips[:, np.newaxis]  # Broadcast and flip
    
    # Factor covariance (diagonal)
    F = np.diag(factor_variances)
    
    # Idiosyncratic variance (residual after factor model)
    emp_var = np.var(X, axis=0, ddof=1)  # Empirical variance
    model_var = np.sum((B.T ** 2) * factor_variances, axis=1)  # Explained variance
    d_diag = np.maximum(emp_var - model_var, 1e-6)  # Residual (with floor)
    D = np.diag(d_diag)
    
    return FactorModelData(B=B, F=F, D=D)
