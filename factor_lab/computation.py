# computation.py: Heavy mathematical lifting.

from loguru import logger
import numpy as np
import cvxpy as cp
import scipy.linalg
from .types import FactorModelData, OptimizationResult

def pca_decomposition(M: np.ndarray, k: int, method: str = 'numpy') -> tuple[np.ndarray, np.ndarray]:
    logger.info(f"Starting PCA Decomposition (k={k}, method='{method}')")
    
    if M.shape[0] != M.shape[1]: 
        logger.error("PCA failed: Matrix M must be square.")
        raise ValueError("M must be square")
    
    n = M.shape[0]

    if method == 'scipy':
        vals, vecs = scipy.linalg.eigh(M, subset_by_index=(n - k, n - 1))
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        
    elif method == 'numpy':
        vals, vecs = np.linalg.eigh(M)
        idx = np.argsort(vals)[::-1][:k]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
    else:
        logger.error(f"Invalid PCA method: {method}")
        raise ValueError(f"Unknown method '{method}'")

    F = np.diag(vals)
    B = vecs.T
    logger.debug(f"PCA complete. Recovered Variance: {vals.sum():.4f}")
    return B, F

def svd_decomposition(returns: np.ndarray, k: int) -> FactorModelData:
    """
    Decomposes an (N_time, N_assets) returns matrix into a Factor Model
    using Singular Value Decomposition (SVD).
    
    Why SVD? 
    1. Numerically more stable than PCA on Covariance matrix (Condition number).
    2. Directly produces square roots needed for simulation.

    Args:
        returns: (T, p) array of asset returns.
        k: Number of factors to extract.

    Returns:
        FactorModelData with pre-calculated F_sqrt and D_sqrt.
    """
    T, p = returns.shape
    logger.info(f"Starting SVD Decomposition on Returns ({T}x{p}, k={k})...")

    # 1. Center the data (Mean = 0)
    mu = returns.mean(axis=0)
    X = returns - mu

    # 2. Run SVD
    # X = U @ S @ Vt
    # We use scipy.linalg.svd for robust implementation
    # full_matrices=False makes it efficient (economy SVD)
    U, s, Vt = scipy.linalg.svd(X, full_matrices=False)

    # 3. Truncate to top k factors
    # S contains singular values. Eigenvalues of Cov = (s^2) / (T-1)
    # We only need the top k components.
    
    s_k = s[:k]          # (k,)
    Vt_k = Vt[:k, :]     # (k, p)

    # 4. Construct Factor Terms
    # Factor Loadings (B) = Vt_k
    # Factor Covariance (F) is diagonal of variances.
    # Variance = s^2 / (T-1)
    
    factor_variances = (s_k ** 2) / (T - 1)
    F = np.diag(factor_variances)
    B = Vt_k

    # Pre-compute F_sqrt (It is just the std devs on the diagonal)
    F_sqrt = np.sqrt(factor_variances) # Vector (k,)

    # 5. Construct Idiosyncratic Terms (D)
    # Residuals E = X - (Factors @ Loadings)
    # Ideally, reconstruct X_approx = U_k @ diag(s_k) @ Vt_k
    # But strictly, E = X - (Projected X).
    
    # Efficient calculation of residual variance:
    # Total Variance - Explained Variance is a good approximation, 
    # but calculating element-wise residuals is safer for correct D diagonal.
    
    # Reconstruct the 'explained' part of returns using top k
    # X_explained = U[:, :k] * s_k @ Vt_k
    X_explained = (U[:, :k] * s_k) @ Vt_k
    residuals = X - X_explained
    
    # Calculate variance of residuals for each asset
    d_variances = np.var(residuals, axis=0, ddof=1)
    D = np.diag(d_variances)
    D_sqrt = np.sqrt(d_variances) # Vector (p,)

    logger.success(f"SVD Complete. Explained Variance Ratio: {np.sum(factor_variances) / np.sum(s**2/(T-1)):.2%}")
    
    return FactorModelData(
        B=B, F=F, D=D,
        F_sqrt=F_sqrt, # Pass as vector (Diagonal optimization)
        D_sqrt=D_sqrt  # Pass as vector (Diagonal optimization)
    )

class FactorOptimizer:
    def __init__(self, data: FactorModelData):
        self.data = data
        self.w = cp.Variable(data.p)
        self.y = cp.Variable(data.k)
        self._constraints = []
        logger.debug(f"Initialized FactorOptimizer for p={data.p} assets.")

    def add_eq(self, A: np.ndarray, b: np.ndarray):
        self._constraints.append(A @ self.w == b)

    def add_ineq(self, A: np.ndarray, b: np.ndarray):
        self._constraints.append(A @ self.w <= b)

    def solve(self) -> OptimizationResult:
        logger.info(f"Solving SOCP Optimization with {len(self._constraints)} constraints...")
        
        D_sqrt = np.sqrt(np.diag(self.data.D))
        F_sqrt = np.sqrt(np.diag(self.data.F))
        
        obj = cp.Minimize(0.5 * (cp.sum_squares(cp.multiply(D_sqrt, self.w)) + 
                                 cp.sum_squares(cp.multiply(F_sqrt, self.y))))
        
        cons = [self.y == self.data.B @ self.w] + self._constraints
        prob = cp.Problem(obj, cons)
        
        try:
            prob.solve()
        except cp.SolverError as e:
            logger.exception("Solver crashed!")
            return OptimizationResult(None, 0, 0, False)

        if prob.status == 'optimal':
            logger.success("Optimization Solved.")
        else:
            logger.warning(f"Optimization finished with status: {prob.status}")

        return OptimizationResult(
            weights=self.w.value,
            risk=np.sqrt(2 * prob.value),
            objective=prob.value,
            solved=(prob.status == 'optimal')
        )