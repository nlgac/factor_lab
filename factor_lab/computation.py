# computation.py: Heavy mathematical lifting.
# Includes PCA decomposition and Convex Optimization logic.

import numpy as np
import cvxpy as cp
import scipy.linalg
from .types import FactorModelData, OptimizationResult

def pca_decomposition(M: np.ndarray, k: int, method: str = 'numpy') -> tuple[np.ndarray, np.ndarray]:
    """
    Rank-k PCA decomposition of symmetric matrix M.
    Returns (B, F) such that M ≈ B.T @ F @ B.
    
    Args:
        M: Symmetric matrix (p, p)
        k: Number of factors to recover
        method: 'numpy' (dense, O(N^3)) or 'scipy' (sparse/subset, O(N^2*k))
        
    Returns:
        B: Factor Loadings (k, p)
        F: Factor Covariance (k, k) - Diagonal
    """
    if M.shape[0] != M.shape[1]: 
        raise ValueError("M must be square")
    
    n = M.shape[0]

    if method == 'scipy':
        # Scipy Optimization: Only compute top k eigenvectors.
        # eigh returns ascending, so we want indices [n-k, n-1]
        vals, vecs = scipy.linalg.eigh(M, subset_by_index=(n - k, n - 1))
        
        # Reverse to get Descending order (Largest -> Smallest)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        
    elif method == 'numpy':
        # Numpy Standard: Compute all eigenvalues (O(N^3))
        vals, vecs = np.linalg.eigh(M)
        
        # Sort Descending and slice top k
        idx = np.argsort(vals)[::-1][:k]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'numpy' or 'scipy'.")

    # Construct Matrices
    F = np.diag(vals)      # (k, k)
    B = vecs.T             # (k, p)
    
    return B, F

class FactorOptimizer:
    # Efficient solver for min w.T(B.T F B + D)w.
    # Uses SOCP reformulation to avoid forming dense covariance.
    def __init__(self, data: FactorModelData):
        self.data = data
        self.w = cp.Variable(data.p)
        self.y = cp.Variable(data.k) # Aux variable y = Bw
        self._constraints = []

    def add_eq(self, A: np.ndarray, b: np.ndarray):
        self._constraints.append(A @ self.w == b)

    def add_ineq(self, A: np.ndarray, b: np.ndarray):
        self._constraints.append(A @ self.w <= b)

    def solve(self) -> OptimizationResult:
        # Objective: 0.5 * (||D^0.5 w||^2 + ||F^0.5 y||^2)
        D_sqrt = np.sqrt(np.diag(self.data.D))
        F_sqrt = np.sqrt(np.diag(self.data.F))
        
        obj = cp.Minimize(0.5 * (cp.sum_squares(cp.multiply(D_sqrt, self.w)) + 
                                 cp.sum_squares(cp.multiply(F_sqrt, self.y))))
        
        cons = [self.y == self.data.B @ self.w] + self._constraints
        prob = cp.Problem(obj, cons)
        
        # Use default solver (auto-selection) for maximum compatibility
        try:
            prob.solve()
        except cp.SolverError:
            return OptimizationResult(None, 0, 0, False)

        return OptimizationResult(
            weights=self.w.value,
            risk=np.sqrt(2 * prob.value),
            objective=prob.value,
            solved=(prob.status == 'optimal')
        )