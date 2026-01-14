# types.py: Shared data structures and type definitions.
# Acts as the contract between Sampling, Computation, and Simulation modules.

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np

# Type alias for a generator function: f(n) -> array
GeneratorFunc = Callable[[Union[int, tuple]], np.ndarray]

@dataclass
class FactorModelData:
    # Immutable container for Factor Model matrices.
    # Model: R = f @ B + epsilon
    B: np.ndarray  # (k, p) Factor Loadings
    F: np.ndarray  # (k, k) Factor Covariance
    D: np.ndarray  # (p, p) Idiosyncratic Covariance

    @property
    def k(self) -> int: return self.B.shape[0]
    
    @property
    def p(self) -> int: return self.B.shape[1]

    def validate(self):
        assert self.B.shape == (self.k, self.p), "B shape mismatch"
        assert self.F.shape == (self.k, self.k), "F shape mismatch"
        assert self.D.shape == (self.p, self.p), "D shape mismatch"

@dataclass
class OptimizationResult:
    # Standardized output for portfolio optimization.
    weights: np.ndarray
    risk: float
    objective: float
    solved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scenario:
    # Defines constraints for an optimization run.
    name: str
    description: str = ""
    # Constraints in form (A, b) for Ax = b or Ax <= b
    equality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    inequality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)