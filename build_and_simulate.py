"""
Factor Model Builder & Simulator (JSON Configured)
==================================================
A modular framework that parses JSON specifications to build,
simulate, analyze, and SAVE factor models and returns.

Usage:
    python build_and_simulate.py model_spec.json
"""

import json
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

import numpy as np
import scipy.linalg

# Standard Factor Lab Imports
from factor_lab import (
    FactorModelData,
    ReturnsSimulator,
    DistributionFactory,
    save_model,
    svd_decomposition
)

# =============================================================================
# 1. DATA CONTRACTS (The Spec)
# =============================================================================

@dataclass
class DistConfig:
    name: str
    params: Dict[str, float]
    transform: Optional[str] = None

@dataclass
class SimConfig:
    name: str
    dist_type: str
    params: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelSpec:
    p_assets: int
    factor_loadings: List[DistConfig]
    factor_variances: List[float]
    idio_variance: float
    sim_n_periods: int
    simulations: List[SimConfig]

    @property
    def k_factors(self) -> int:
        return len(self.factor_loadings)

# =============================================================================
# 2. JSON PARSER
# =============================================================================

class JsonParser:
    """Parses JSON configuration files for Factor Models."""

    @staticmethod
    def _parse_numeric(value: Union[str, float, int]) -> float:
        """Safely evaluates strings like '0.18^2' or returns floats directly."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if "^" in value:
                base, exp = value.split("^")
                return float(base.strip()) ** float(exp.strip())
            return float(value)
        raise ValueError(f"Invalid numeric format: {value}")

    @classmethod
    def parse(cls, filepath: Path) -> ModelSpec:
        with open(filepath, 'r') as f:
            data = json.load(f)

        meta = data.get("meta", {})
        
        # Factors
        factors = []
        for f_conf in data.get("factor_loadings", []):
            factors.append(DistConfig(
                name=f_conf.get("distribution", "normal"),
                params=f_conf.get("params", {}),
                transform=f_conf.get("transform")
            ))

        # Covariances
        cov_data = data.get("covariance", {})
        f_diag_raw = cov_data.get("F_diagonal", [])
        f_vars = [cls._parse_numeric(x) for x in f_diag_raw]
        d_var = cls._parse_numeric(cov_data.get("D_diagonal", 0.1))

        # Simulations
        sims = []
        for s_conf in data.get("simulations", []):
            sims.append(SimConfig(
                name=s_conf.get("name", "Unnamed"),
                dist_type=s_conf.get("type", "normal"),
                params=s_conf.get("params", {})
            ))

        return ModelSpec(
            p_assets=meta.get("p_assets", 100),
            sim_n_periods=meta.get("n_periods", 100),
            factor_loadings=factors,
            factor_variances=f_vars,
            idio_variance=d_var,
            simulations=sims
        )

# =============================================================================
# 3. BUILDER LAYER
# =============================================================================

class FactorModelBuilder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _gram_schmidt_project(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        return v - (np.dot(u, v) / np.dot(u, u)) * u

    def build(self, spec: ModelSpec) -> FactorModelData:
        print(f"🏗️  Building Model: p={spec.p_assets}, k={spec.k_factors}")
        
        B = np.zeros((spec.k_factors, spec.p_assets))
        
        for i, config in enumerate(spec.factor_loadings):
            loc = config.params.get("loc", 0.0)
            scale = config.params.get("scale", 1.0)
            raw_vec = self.rng.normal(loc, scale, spec.p_assets)
            
            if config.transform == "gram_schmidt":
                if i == 0:
                    B[i, :] = raw_vec
                else:
                    print(f"   ↳ Factor {i+1}: Applying Gram-Schmidt...")
                    B[i, :] = self._gram_schmidt_project(raw_vec, B[0, :])
            else:
                B[i, :] = raw_vec

        F = np.diag(spec.factor_variances)
        D = np.diag(np.full(spec.p_assets, spec.idio_variance))
        
        return FactorModelData(B=B, F=F, D=D)

# =============================================================================
# 4. ANALYSIS & PERSISTENCE ENGINE
# =============================================================================

class AnalysisEngine:
    def __init__(self, model: FactorModelData, spec: ModelSpec, rng: np.random.Generator):
        self.model = model
        self.spec = spec
        self.rng = rng
        self.factory = DistributionFactory(rng)

    def compute_subspace_distance(self, returns: np.ndarray) -> float:
        """Computes geodesic distance between True Model B and Sample B."""
        # 1. Get True B (Orthonormalized for comparison)
        B_true = self.model.B.T # (p, k)
        B_true_orth, _ = scipy.linalg.qr(B_true, mode='economic')

        # 2. Get Sample B via SVD
        sample_model = svd_decomposition(returns, k=self.model.k)
        B_sample = sample_model.B.T # (p, k)

        # 3. Compute Principal Angles
        angles = scipy.linalg.subspace_angles(B_true_orth, B_sample)
        
        return float(np.linalg.norm(angles))

    def run_all_simulations(self) -> Dict[str, Dict[str, np.ndarray]]:
        results_collection = {}
        sim = ReturnsSimulator(self.model, rng=self.rng)

        for sim_config in self.spec.simulations:
            print(f"\n▶ Simulation: {sim_config.name} ({sim_config.dist_type})")
            
            # FIX: Explicitly create samplers for ALL cases.
            # Passing None caused simulation.py to crash on len(None).
            
            if sim_config.dist_type == "normal":
                # Create standard normal samplers explicitly
                f_samplers = [
                    self.factory.create("normal", mean=0.0, std=1.0) 
                    for _ in range(self.model.k)
                ]
                d_samplers = [
                    self.factory.create("normal", mean=0.0, std=1.0) 
                    for _ in range(self.model.p)
                ]
                
            elif "student" in sim_config.dist_type:
                df_f = int(sim_config.params.get('df_factors', 5))
                df_d = int(sim_config.params.get('df_idio', 5))
                f_samplers = [self.factory.create("student_t", df=df_f) for _ in range(self.model.k)]
                d_samplers = [self.factory.create("student_t", df=df_d) for _ in range(self.model.p)]
            
            else:
                # Fallback to Normal
                print(f"   ⚠ Unknown type '{sim_config.dist_type}', defaulting to Normal.")
                f_samplers = [self.factory.create("normal", mean=0, std=1) for _ in range(self.model.k)]
                d_samplers = [self.factory.create("normal", mean=0, std=1) for _ in range(self.model.p)]

            # 2. Run Simulation
            res = sim.simulate(
                n_periods=self.spec.sim_n_periods,
                factor_samplers=f_samplers,
                idio_samplers=d_samplers
            )
            results_collection[sim_config.name] = res

            # 3. Analyze
            dist = self.compute_subspace_distance(res['security_returns'])
            print(f"   ✓ Subspace Geodesic Distance: {dist:.4f}")

        return results_collection

    def save_results(self, results: Dict[str, Dict[str, np.ndarray]]):
        print("\n💾 Saving Results...")
        
        # 1. Save the Model
        model_filename = "factor_model.npz"
        save_model(self.model, model_filename)
        print(f"   • Model saved to: {model_filename}")

        # 2. Save Simulation Data
        for name, data_dict in results.items():
            safe_name = name.replace(" ", "_").lower()
            filename = f"simulation_{safe_name}.npz"
            np.savez_compressed(filename, **data_dict)
            print(f"   • {name} results saved to: {filename}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Factor Lab Builder (JSON)")
    parser.add_argument("config_file", type=Path, help="Path to .json model spec")
    args = parser.parse_args()

    if not args.config_file.exists():
        sys.exit(f"Error: File {args.config_file} not found.")

    rng = np.random.default_rng(42)
    
    try:
        spec = JsonParser.parse(args.config_file)
    except Exception as e:
        sys.exit(f"Parsing Error: {e}")

    builder = FactorModelBuilder(rng)
    model = builder.build(spec)
    
    engine = AnalysisEngine(model, spec, rng)
    all_results = engine.run_all_simulations()
    
    engine.save_results(all_results)
    
    print("\n✅ Process Complete.")

if __name__ == "__main__":
    main()