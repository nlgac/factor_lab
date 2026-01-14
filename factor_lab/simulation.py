# simulation.py: Time-series generation and Scenario orchestration.

import sys
from loguru import logger
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from .types import FactorModelData, GeneratorFunc, Scenario, OptimizationResult

class ReturnsSimulator:
    def __init__(self, data: FactorModelData, seed: Optional[int] = None):
        self.data = data
        self.rng = np.random.default_rng(seed)
        
        # 1. Setup Factor Transform (F_sqrt)
        if data.F_sqrt is not None:
            logger.debug("Simulator: Using pre-computed F_sqrt from model.")
            # Check if vector (diag) or matrix
            if data.F_sqrt.ndim == 1:
                self.F_tx, self.F_diag = data.F_sqrt, True
            else:
                self.F_tx, self.F_diag = data.F_sqrt, False
        else:
            self.F_tx, self.F_diag = self._prep_transform(data.F, "Factor Cov")

        # 2. Setup Idio Transform (D_sqrt)
        if data.D_sqrt is not None:
            logger.debug("Simulator: Using pre-computed D_sqrt from model.")
            if data.D_sqrt.ndim == 1:
                self.D_tx, self.D_diag = data.D_sqrt, True
            else:
                self.D_tx, self.D_diag = data.D_sqrt, False
        else:
            self.D_tx, self.D_diag = self._prep_transform(data.D, "Idio Cov")

    def _prep_transform(self, M: np.ndarray, name: str) -> Tuple[np.ndarray, bool]:
        if np.allclose(M, np.diag(np.diag(M))):
            logger.debug(f"{name}: Detected Diagonal Matrix. Using O(N) scalar transform.")
            return np.sqrt(np.diag(M)), True 
        
        logger.debug(f"{name}: Detected Dense Matrix. Using O(N^3) Cholesky transform.")
        return np.linalg.cholesky(M), False 

    def _process(self, gens: List[GeneratorFunc], n: int, tx: np.ndarray, is_diag: bool, context: str, debug_intermediate_quantity_length: int) -> np.ndarray:
        # 1. Generate Raw
        raw = np.column_stack([g(n) for g in gens])
        
        # DEBUG LOGGING
        try:
            if debug_intermediate_quantity_length > 0:
                subset = raw[:debug_intermediate_quantity_length]
                logger.debug(f"[{context}] Generated Raw Samples (Shape: {raw.shape})\nFirst {debug_intermediate_quantity_length} rows:\n{subset}")
            else:
                logger.debug(f"[{context}] Generated Raw Samples (Shape: {raw.shape})")
        except Exception:
            pass 

        # 2. Standardize
        std = raw.std(axis=0)
        std[std == 0] = 1.0
        z_score = (raw - raw.mean(axis=0)) / std
        
        # 3. Scale
        if is_diag: return z_score * tx
        return z_score @ tx.T

    def simulate(self, n: int, f_gens: List[GeneratorFunc], i_gens: List[GeneratorFunc], debug_intermediate_quantity_length: int = 100) -> Dict[str, np.ndarray]:
        logger.info(f"Starting simulation: n={n}")
        
        f_ret = self._process(f_gens, n, self.F_tx, self.F_diag, "Factor", debug_intermediate_quantity_length)
        i_ret = self._process(i_gens, n, self.D_tx, self.D_diag, "Idio", debug_intermediate_quantity_length)
        
        sec_returns = f_ret @ self.data.B + i_ret
        logger.success("Simulation complete.")
        
        return {
            'security_returns': sec_returns,
            'factor_returns': f_ret,
            'idio_returns': i_ret
        }

    def validate_covariance(self, returns: np.ndarray) -> Dict[str, Any]:
        if returns.shape[1] != self.data.p:
            logger.error(f"Validation Mismatch: Returns {returns.shape[1]} != Assets {self.data.p}")
            raise ValueError("Returns/Asset count mismatch")

        cov_model = (self.data.B.T @ self.data.F @ self.data.B) + self.data.D
        cov_empirical = np.cov(returns, rowvar=False)
        diff = cov_empirical - cov_model
        error = np.linalg.norm(diff, ord='fro')
        
        logger.info(f"Covariance Validation Error (Frobenius): {error:.6f}")
        return {
            'frobenius_error': error,
            'avg_abs_error': np.mean(np.abs(diff)),
            'model_cov': cov_model,
            'empirical_cov': cov_empirical
        }

class ScenarioBuilder:
    def __init__(self, p: int):
        self.p = p

    def create(self, name: str) -> Scenario:
        logger.debug(f"Creating scenario: '{name}'")
        return Scenario(name)

    def add_fully_invested(self, s: Scenario) -> Scenario:
        s.equality_constraints.append((np.ones((1, self.p)), np.array([1.0])))
        return s

    def add_long_only(self, s: Scenario) -> Scenario:
        s.inequality_constraints.append((-np.eye(self.p), np.zeros(self.p)))
        return s
        
    def add_box_constraints(self, s: Scenario, low: float, high: float) -> Scenario:
        s.inequality_constraints.append((np.eye(self.p), np.full(self.p, high)))
        s.inequality_constraints.append((-np.eye(self.p), np.full(self.p, -low)))
        return s