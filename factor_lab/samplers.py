# samplers.py: Statistical distribution management.

from loguru import logger
import numpy as np
import inspect
from typing import Callable, Dict, Union, Optional, List, Any
from .types import GeneratorFunc, FactorModelData

class DistributionRegistry:
    def __init__(self):
        self._funcs: Dict[str, Callable] = {
            'normal': lambda n, mean, std: np.random.normal(mean, std, n),
            'uniform': lambda n, low, high: np.random.uniform(low, high, n),
            'constant': lambda n, c: np.full(n, c, dtype=float),
            'student_t': lambda n, df: np.random.standard_t(df, size=n),
            'beta': lambda n, a, b: np.random.beta(a, b, size=n),
        }

    def register(self, name: str, func: Callable):
        logger.info(f"Registering custom distribution: '{name}'")
        self._funcs[name.lower()] = func

    def get(self, name: str) -> Callable:
        name = name.lower()
        if name not in self._funcs:
            logger.error(f"Distribution '{name}' not found. Available: {list(self._funcs.keys())}")
            raise ValueError(f"Unknown dist '{name}'")
        return self._funcs[name]

class SignatureValidator:
    @staticmethod
    def validate(name: str, func: Callable, params: dict):
        sig = inspect.signature(func)
        required = {p.name for p in list(sig.parameters.values())[1:] 
                   if p.default == inspect.Parameter.empty}
        missing = required - set(params.keys())
        if missing:
            logger.error(f"Generator validation failed for '{name}'. Missing: {missing}")
            raise ValueError(f"'{name}' missing params: {missing}")

class DistributionFactory:
    _registry = DistributionRegistry()
    _validator = SignatureValidator()

    @classmethod
    def register(cls, name: str, func: Callable):
        cls._registry.register(name, func)

    @classmethod
    def create_generator(cls, dist_name: str, **params) -> GeneratorFunc:
        # We use .bind() to add context if needed, but simple debug is sufficient here
        logger.debug(f"Creating generator: '{dist_name}' | params={params}")
        func = cls._registry.get(dist_name)
        cls._validator.validate(dist_name, func, params)
        return lambda n: func(n, **params)

class DataSampler:
    def __init__(self, p: int, k: int, n_samples: int = 1000):
        self.p = p
        self.k = k
        self.n = n_samples
        self._beta_gens: List[GeneratorFunc] = []
        self._f_vol_gens: List[GeneratorFunc] = []
        self._d_vol_gens: List[GeneratorFunc] = []

    def configure(self, 
                  beta: Union[GeneratorFunc, List[GeneratorFunc]], 
                  f_vol: Union[GeneratorFunc, List[GeneratorFunc]], 
                  d_vol: Union[GeneratorFunc, List[GeneratorFunc]]):
        
        logger.info(f"Configuring DataSampler (p={self.p}, k={self.k})")
        self._beta_gens = self._resolve(beta, self.k, "Beta")
        self._f_vol_gens = self._resolve(f_vol, self.k, "Factor Vol")
        self._d_vol_gens = self._resolve(d_vol, self.p, "Idio Vol")

    def _resolve(self, gen_or_list: Union[GeneratorFunc, List], target_len: int, name: str) -> List[GeneratorFunc]:
        if isinstance(gen_or_list, list):
            if len(gen_or_list) != target_len:
                logger.error(f"{name} list mismatch: Got {len(gen_or_list)}, expected {target_len}")
                raise ValueError(f"{name} list length mismatch")
            logger.debug(f"{name}: Using explicit list of {len(gen_or_list)} generators.")
            return gen_or_list
        elif callable(gen_or_list):
            logger.debug(f"{name}: Broadcasting single generator to {target_len} elements.")
            return [gen_or_list for _ in range(target_len)]
        else:
            raise TypeError(f"{name} must be a generator function or list of functions.")

    def generate(self) -> FactorModelData:
        if not self._beta_gens: raise RuntimeError("Not configured")
        
        logger.info("Generating Factor Model Data...")
        B = np.vstack([g(self.p) for g in self._beta_gens])
        f_vols = np.array([g(1)[0] for g in self._f_vol_gens])
        F = np.diag(f_vols ** 2)
        d_vols = np.array([g(1)[0] for g in self._d_vol_gens])
        D = np.diag(d_vols ** 2)
        
        logger.success("Model generation complete.")
        return FactorModelData(B, F, D)