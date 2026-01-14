# simulation.py: Time-series generation and Scenario orchestration.
# Features Hybrid (Diagonal/Cholesky) returns simulation.

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from .types import FactorModelData, GeneratorFunc, Scenario, OptimizationResult
from .computation import FactorOptimizer

class ReturnsSimulator:
    # Simulates security returns: R = f @ B + epsilon.
    # Automatically detects diagonal F/D for O(N) efficiency, falls back to Cholesky.
    def __init__(self, data: FactorModelData, seed: Optional[int] = None):
        self.data = data
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute transforms (Choice A/B logic)
        self.F_tx, self.F_diag = self._prep_transform(data.F)
        self.D_tx, self.D_diag = self._prep_transform(data.D)

    def _prep_transform(self, M: np.ndarray) -> Tuple[np.ndarray, bool]:
        # Returns (Transform, IsDiagonal).
        if np.allclose(M, np.diag(np.diag(M))):
            return np.sqrt(np.diag(M)), True # Vector (Choice B)
        return np.linalg.cholesky(M), False  # Matrix (Choice A)

    def _process(self, gens: List[GeneratorFunc], n: int, tx: np.ndarray, is_diag: bool) -> np.ndarray:
        # Generate -> Standardize -> Scale.
        # 1. Generate
        raw = np.column_stack([g(n) for g in gens])
        
        # 2. Standardize (Force Mean=0, Var=1 to respect model vols)
        std = raw.std(axis=0)
        std[std == 0] = 1.0
        z_score = (raw - raw.mean(axis=0)) / std
        
        # 3. Scale
        if is_diag: return z_score * tx        # O(N)
        return z_score @ tx.T                  # O(N^3)

    def simulate(self, n: int, f_gens: List[GeneratorFunc], i_gens: List[GeneratorFunc]) -> Dict[str, np.ndarray]:
        f_ret = self._process(f_gens, n, self.F_tx, self.F_diag)
        i_ret = self._process(i_gens, n, self.D_tx, self.D_diag)
        return {
            'security_returns': f_ret @ self.data.B + i_ret,
            'factor_returns': f_ret,
            'idio_returns': i_ret
        }

    def validate_covariance(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Compares the empirical covariance of the simulated returns against
        the theoretical 'Ground Truth' covariance of the factor model.

        Ground Truth: Sigma = B.T @ F @ B + D
        Empirical:    Cov(returns)

        Returns:
            Dictionary containing:
            - 'frobenius_error': The Euclidean distance between matrices.
            - 'model_cov': The theoretical matrix (p, p).
            - 'empirical_cov': The sample matrix (p, p).
        """
        if returns.shape[1] != self.data.p:
            raise ValueError(f"Returns columns {returns.shape[1]} != Model Assets {self.data.p}")

        # 1. Calculate Theoretical Covariance (Ground Truth)
        # Sigma = B'FB + D
        # Note: B is (k, p), so B.T is (p, k)
        cov_model = (self.data.B.T @ self.data.F @ self.data.B) + self.data.D

        # 2. Calculate Empirical Covariance
        # rowvar=False because rows are Time, cols are Assets
        cov_empirical = np.cov(returns, rowvar=False)

        # 3. Compute Error (Frobenius Norm)
        diff = cov_empirical - cov_model
        error = np.linalg.norm(diff, ord='fro')

        return {
            'frobenius_error': error,
            'avg_abs_error': np.mean(np.abs(diff)),
            'model_cov': cov_model,
            'empirical_cov': cov_empirical
        }
    def __repr__(self):
            return f"{self.data.p}, {self.data.k}"

class ScenarioBuilder:
    # Fluent interface for constructing constraints.
    def __init__(self, p: int):
        self.p = p

    def create(self, name: str) -> Scenario:
        return Scenario(name)

    def add_fully_invested(self, s: Scenario) -> Scenario:
        s.equality_constraints.append((np.ones((1, self.p)), np.array([1.0])))
        return s

    def add_long_only(self, s: Scenario) -> Scenario:
        s.inequality_constraints.append((-np.eye(self.p), np.zeros(self.p)))
        return s
        
    def add_box_constraints(self, s: Scenario, low: float, high: float) -> Scenario:
        # w <= high  -> I w <= high
        s.inequality_constraints.append((np.eye(self.p), np.full(self.p, high)))
        # w >= low   -> -I w <= -low
        s.inequality_constraints.append((-np.eye(self.p), np.full(self.p, -low)))
        return s