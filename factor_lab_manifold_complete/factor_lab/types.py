"""Minimal types for standalone operation."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FactorModelData:
    """Factor model: r = B'f + e"""
    B: np.ndarray  # (k, p) loadings
    F: np.ndarray  # (k, k) factor covariance  
    D: np.ndarray  # (p, p) idiosyncratic covariance
    factor_transform: Optional[np.ndarray] = None
    idio_transform: Optional[np.ndarray] = None
    
    @property
    def k(self) -> int:
        return self.B.shape[0]
    
    @property
    def p(self) -> int:
        return self.B.shape[1]
    
    def implied_covariance(self) -> np.ndarray:
        return self.B.T @ self.F @ self.B + self.D

def svd_decomposition(returns: np.ndarray, k: int, center: bool = True) -> FactorModelData:
    """Extract factor model via SVD."""
    T, p = returns.shape
    X = returns - returns.mean(axis=0) if center else returns
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    factor_variances = (s[:k]**2) / (T - 1)
    B = Vt[:k, :]
    
    # Sign normalization: flip factors with negative mean loadings
    row_means = B.mean(axis=1)  # Mean across assets for each factor
    sign_flips = np.where(row_means < 0, -1, 1)  # -1 if mean < 0, else +1
    B = B * sign_flips[:, np.newaxis]  # Broadcast and flip
    
    F = np.diag(factor_variances)
    emp_var = np.var(X, axis=0, ddof=1)
    model_var = np.sum((B.T ** 2) * factor_variances, axis=1)
    d_diag = np.maximum(emp_var - model_var, 1e-6)
    D = np.diag(d_diag)
    return FactorModelData(B=B, F=F, D=D)

class ReturnsSimulator:
    """Minimal simulator for standalone operation."""
    def __init__(self, model: FactorModelData, rng=None):
        self.model = model
        self.rng = rng or np.random.default_rng()
    
    def simulate(self, n_periods: int, factor_samplers=None, idio_samplers=None):
        """Simulate returns."""
        k, p = self.model.k, self.model.p
        F_chol = np.linalg.cholesky(self.model.F)
        D_sqrt = np.sqrt(np.diag(self.model.D))
        
        factor_returns = self.rng.normal(0, 1, (n_periods, k)) @ F_chol.T
        idio_returns = self.rng.normal(0, 1, (n_periods, p)) * D_sqrt
        security_returns = factor_returns @ self.model.B + idio_returns
        
        return {
            'security_returns': security_returns,
            'factor_returns': factor_returns,
            'idio_returns': idio_returns
        }

class DistributionFactory:
    """Minimal distribution factory."""
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
    
    def create(self, name: str, **params):
        return lambda: self.rng.normal(params.get('mean', 0), params.get('std', 1))

def save_model(model: FactorModelData, filename: str):
    """Save model to file."""
    np.savez(filename, B=model.B, F=model.F, D=model.D)
