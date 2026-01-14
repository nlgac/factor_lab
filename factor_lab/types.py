# types.py: Shared data structures and type definitions.

from loguru import logger
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np

# Type alias for a generator function: f(n) -> array
GeneratorFunc = Callable[[Union[int, tuple]], np.ndarray]

@dataclass
class FactorModelData:
    B: np.ndarray  # (k, p) Factor Loadings
    F: np.ndarray  # (k, k) Factor Covariance
    D: np.ndarray  # (p, p) Idiosyncratic Covariance
    
    # Pre-computed roots for efficient simulation (Optional)
    # If provided, simulator skips Cholesky/Sqrt steps.
    F_sqrt: Optional[np.ndarray] = None 
    D_sqrt: Optional[np.ndarray] = None

    @property
    def k(self) -> int: return self.B.shape[0]
    
    @property
    def p(self) -> int: return self.B.shape[1]

    def validate(self):
        logger.debug(f"Validating FactorModelData: k={self.k}, p={self.p}")
        assert self.B.shape == (self.k, self.p), "B shape mismatch"
        assert self.F.shape == (self.k, self.k), "F shape mismatch"
        assert self.D.shape == (self.p, self.p), "D shape mismatch"
        
        if self.F_sqrt is not None:
            # Verify F_sqrt shape
            if self.F_sqrt.ndim == 1: # Diagonal vector
                 assert self.F_sqrt.shape[0] == self.k
            else: # Full matrix
                 assert self.F_sqrt.shape == (self.k, self.k)
                 
        logger.success("FactorModelData validation successful.")

@dataclass
class OptimizationResult:
    weights: np.ndarray
    risk: float
    objective: float
    solved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scenario:
    name: str
    description: str = ""
    equality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    inequality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)