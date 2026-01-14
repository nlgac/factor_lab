import os
from pathlib import Path

# ==============================================================================
# REFACTOR: samplers.py
# ==============================================================================

SAMPLERS_CONTENT = r'''
# samplers.py: Statistical distribution management.
# Uses Composition to separate Registry (Storage), Validator (Logic), and Factory (Creation).

import numpy as np
import inspect
from typing import Callable, Dict, Union, Optional, List, Any
from .types import GeneratorFunc, FactorModelData

class DistributionRegistry:
    # Storage for distribution functions.
    def __init__(self):
        self._funcs: Dict[str, Callable] = {
            'normal': lambda n, mean, std: np.random.normal(mean, std, n),
            'uniform': lambda n, low, high: np.random.uniform(low, high, n),
            'constant': lambda n, c: np.full(n, c, dtype=float),
            'student_t': lambda n, df: np.random.standard_t(df, size=n),
            'beta': lambda n, a, b: np.random.beta(a, b, size=n),
        }

    def register(self, name: str, func: Callable):
        self._funcs[name.lower()] = func

    def get(self, name: str) -> Callable:
        name = name.lower()
        if name not in self._funcs:
            raise ValueError(f"Unknown dist '{name}'. Available: {list(self._funcs.keys())}")
        return self._funcs[name]

class SignatureValidator:
    # Validates parameters against function signatures.
    @staticmethod
    def validate(name: str, func: Callable, params: dict):
        sig = inspect.signature(func)
        # Skip 'n' (first param)
        required = {p.name for p in list(sig.parameters.values())[1:] 
                   if p.default == inspect.Parameter.empty}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"'{name}' missing params: {missing}")

class DistributionFactory:
    # Orchestrator for creating samplers.
    _registry = DistributionRegistry()
    _validator = SignatureValidator()

    @classmethod
    def register(cls, name: str, func: Callable):
        cls._registry.register(name, func)

    @classmethod
    def create_generator(cls, dist_name: str, **params) -> GeneratorFunc:
        # Returns a function f(n) -> array.
        func = cls._registry.get(dist_name)
        cls._validator.validate(dist_name, func, params)
        return lambda n: func(n, **params)

class DataSampler:
    # Generates synthetic FactorModelData (B, F, D) using generators.
    # Updated to support explicit per-factor and per-asset configuration.
    
    def __init__(self, p: int, k: int, n_samples: int = 1000):
        self.p = p
        self.k = k
        self.n = n_samples
        
        # Internal storage is ALWAYS lists of generators
        self._beta_gens: List[GeneratorFunc] = []
        self._f_vol_gens: List[GeneratorFunc] = []
        self._d_vol_gens: List[GeneratorFunc] = []

    def configure(self, 
                  beta: Union[GeneratorFunc, List[GeneratorFunc]], 
                  f_vol: Union[GeneratorFunc, List[GeneratorFunc]], 
                  d_vol: Union[GeneratorFunc, List[GeneratorFunc]]):
        """
        Configures the sampler. 
        Arguments can be single generators (broadcasted) or explicit lists.
        """
        self._beta_gens = self._resolve(beta, self.k, "Beta")
        self._f_vol_gens = self._resolve(f_vol, self.k, "Factor Vol")
        self._d_vol_gens = self._resolve(d_vol, self.p, "Idio Vol")

    def _resolve(self, gen_or_list: Union[GeneratorFunc, List], target_len: int, name: str) -> List[GeneratorFunc]:
        """Helper to standardize single vs list inputs."""
        if isinstance(gen_or_list, list):
            if len(gen_or_list) != target_len:
                raise ValueError(f"{name} list length {len(gen_or_list)} != required {target_len}")
            return gen_or_list
        elif callable(gen_or_list):
            # Broadcast the single generator
            return [gen_or_list for _ in range(target_len)]
        else:
            raise TypeError(f"{name} must be a generator function or list of functions.")

    def generate(self) -> FactorModelData:
        # Builds B, F, D matrices using the configured generator lists.
        if not self._beta_gens: raise RuntimeError("Not configured")
        
        # 1. B: (k, p) - Each factor draws p loadings
        # We iterate through the k generators
        B = np.vstack([g(self.p) for g in self._beta_gens])
        
        # 2. F: (k, k) Diagonal
        # Each generator produces 1 scalar vol.
        f_vols = np.array([g(1)[0] for g in self._f_vol_gens])
        F = np.diag(f_vols ** 2)
        
        # 3. D: (p, p) Diagonal
        # Each generator produces 1 scalar vol.
        d_vols = np.array([g(1)[0] for g in self._d_vol_gens])
        D = np.diag(d_vols ** 2)
        
        return FactorModelData(B, F, D)
'''

def install_patch():
    base_dir = Path.cwd()
    path = base_dir / "factor_lab" / "samplers.py"
    
    print(f"Refactoring {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLERS_CONTENT.strip())
    print("  + Success: DataSampler now supports explicit list configuration.")

if __name__ == "__main__":
    install_patch()