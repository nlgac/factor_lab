"""
Factor Model Builder & Simulator (Manifold Analysis Version)
==========================================================
Parses JSON specs to build, simulate, and perform Deep Spectral Analysis.

New Features:
- Computes Grassmannian Distance (Subspace fit)
- Computes Stiefel Distance (Frame fit, Chordal Metric)
- Computes Procrustes Distance (Frame fit, Rotation Invariant)
"""

import json
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator

# Standard Factor Lab Imports
from factor_lab import (
    FactorModelData,
    ReturnsSimulator,
    DistributionFactory,
    save_model,
    svd_decomposition
)

# =============================================================================
# 1. DATA CONTRACTS
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
    @staticmethod
    def _parse_numeric(value: Union[str, float, int]) -> float:
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
        factors = [
            DistConfig(
                name=f.get("distribution", "normal"), 
                params=f.get("params", {}), 
                transform=f.get("transform")
            ) for f in data.get("factor_loadings", [])
        ]
        
        cov_data = data.get("covariance", {})
        f_vars = [cls._parse_numeric(x) for x in cov_data.get("F_diagonal", [])]
        d_var = cls._parse_numeric(cov_data.get("D_diagonal", 0.1))

        sims = [
            SimConfig(
                name=s.get("name"), 
                dist_type=s.get("type", "normal"), 
                params=s.get("params", {})
            ) for s in data.get("simulations", [])
        ]

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
            raw = self.rng.normal(loc, scale, spec.p_assets)
            
            if config.transform == "gram_schmidt" and i > 0:
                print(f"   ↳ Factor {i+1}: Applying Gram-Schmidt...")
                B[i, :] = self._gram_schmidt_project(raw, B[0, :])
            else:
                B[i, :] = raw

        F = np.diag(spec.factor_variances)
        D = np.diag(np.full(spec.p_assets, spec.idio_variance))
        return FactorModelData(B=B, F=F, D=D)

# =============================================================================
# 4. DEEP ANALYSIS ENGINE
# =============================================================================

class AnalysisEngine:
    def __init__(self, model: FactorModelData, spec: ModelSpec, rng: np.random.Generator):
        self.model = model
        self.spec = spec
        self.rng = rng
        self.factory = DistributionFactory(rng)

    def _compute_true_eigenvalues(self, k_top: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes eigenpairs of True Matrix Sigma = B.T @ F @ B + D."""
        B, F, D = self.model.B, self.model.F, self.model.D
        p = self.model.p
        D_diag = np.diag(D)

        def matvec(v):
            return B.T @ (F @ (B @ v)) + D_diag * v

        op = LinearOperator((p, p), matvec=matvec, dtype=float)
        vals, vecs = scipy.sparse.linalg.eigsh(op, k=k_top, which='LM')
        return vals[::-1], vecs[:, ::-1].T

    def _compute_manifold_distances(self, B_true: np.ndarray, B_sample: np.ndarray) -> Dict[str, Any]:
        """
        Computes distances on Grassmannian and Stiefel Manifolds.
        """
        # 1. ORTHONORMALIZE FRAMES (Project to Stiefel V_{p,k})
        # Q matrices are (p, k) with orthonormal columns
        Q_true, _ = scipy.linalg.qr(B_true.T, mode='economic')
        ortho_B = Q_true.T # (k, p) Stiefel frame
        
        Q_sample, _ = scipy.linalg.qr(B_sample.T, mode='economic')

        # 2. GRASSMANNIAN METRIC (Subspace Distance)
        # Invariant to rotation of the frame
        angles = scipy.linalg.subspace_angles(Q_true, Q_sample)
        dist_grassmann = float(np.linalg.norm(angles))

        # 3. STIEFEL METRIC 1: Chordal Distance
        # Measures direct difference between frames: || U - V ||_F
        # Sensitive to sign flips and permutation
        dist_stiefel_chordal = float(np.linalg.norm(Q_true - Q_sample))

        # 4. STIEFEL METRIC 2: Procrustes Geodesic
        # Finds optimal rotation R to align Q_sample to Q_true, then measures distance.
        # This handles the "Sign Flip" and "Permutation" issues common in simulations.
        # Solves: min_R || Q_true - Q_sample @ R ||_F
        M = Q_sample.T @ Q_true
        U_m, _, Vt_m = scipy.linalg.svd(M)
        R_opt = U_m @ Vt_m
        Q_aligned = Q_sample @ R_opt
        dist_stiefel_procrustes = float(np.linalg.norm(Q_true - Q_aligned))
        
        return {
            "dist_grassmannian": dist_grassmann,
            "dist_stiefel_chordal": dist_stiefel_chordal,
            "dist_stiefel_procrustes": dist_stiefel_procrustes,
            "principal_angles": angles,
            "ortho_B": ortho_B 
        }

    def run_simulation_and_analyze(self, sim_config: SimConfig) -> Dict[str, Any]:
        print(f"\n▶ Simulation: {sim_config.name} ({sim_config.dist_type})")
        
        # 1. Setup Samplers
        if "student" in sim_config.dist_type:
            df_f = int(sim_config.params.get('df_factors', 5))
            df_d = int(sim_config.params.get('df_idio', 5))
            f_s = [self.factory.create("student_t", df=df_f) for _ in range(self.model.k)]
            d_s = [self.factory.create("student_t", df=df_d) for _ in range(self.model.p)]
        else:
            f_s = [self.factory.create("normal", mean=0, std=1) for _ in range(self.model.k)]
            d_s = [self.factory.create("normal", mean=0, std=1) for _ in range(self.model.p)]

        # 2. Simulate
        sim = ReturnsSimulator(self.model, rng=self.rng)
        sim_res = sim.simulate(self.spec.sim_n_periods, factor_samplers=f_s, idio_samplers=d_s)
        returns = sim_res['security_returns']

        # 3. Extract Estimated Model (SVD)
        est_model = svd_decomposition(returns, k=self.model.k)

        # 4. Compute Spectral Comparison
        print("   ↳ Computing True Eigenvalues (Implicitly)...")
        true_evals, true_evecs = self._compute_true_eigenvalues(k_top=self.model.k)
        
        sample_evecs = est_model.B 
        sample_evals = np.diag(est_model.F) + np.mean(np.diag(est_model.D))
        
        # 5. Compute Manifold Distances
        print("   ↳ Computing Manifold Distances (Grassmann & Stiefel)...")
        manifold_res = self._compute_manifold_distances(self.model.B, est_model.B)
        ortho_B = manifold_res.pop("ortho_B")

        # 6. Package Results
        return {
            "returns": returns,
            "matrices": {
                "true": {
                    "B": self.model.B,
                    "ortho_B": ortho_B,
                    "F": self.model.F,
                    "D": self.model.D,
                    "eigenvalues": true_evals,
                    "eigenvectors_top_k": true_evecs
                },
                "sample": {
                    "B": est_model.B,
                    "F": est_model.F,
                    "D": est_model.D,
                    "eigenvalues": sample_evals,
                    "eigenvectors_top_k": sample_evecs 
                }
            },
            "metrics": manifold_res
        }

    def save_results(self, all_results: Dict[str, Any]):
        print("\n💾 Saving Results...")
        save_model(self.model, "factor_model.npz")
        
        for name, data in all_results.items():
            safe_name = name.replace(" ", "_").lower()
            fname = f"simulation_{safe_name}.npz"
            
            save_dict = {
                "security_returns": data['returns'],
                
                # TRUE
                "true_B": data['matrices']['true']['B'],
                "true_ortho_B": data['matrices']['true']['ortho_B'],
                "true_F": data['matrices']['true']['F'],
                "true_D": data['matrices']['true']['D'],
                "true_eigenvalues": data['matrices']['true']['eigenvalues'],
                "true_eigenvectors": data['matrices']['true']['eigenvectors_top_k'],
                
                # SAMPLE
                "sample_B": data['matrices']['sample']['B'],
                "sample_F": data['matrices']['sample']['F'],
                "sample_D": data['matrices']['sample']['D'],
                "sample_eigenvalues": data['matrices']['sample']['eigenvalues'],
                "sample_eigenvectors": data['matrices']['sample']['eigenvectors_top_k'],
                
                # METRICS
                "dist_grassmannian": data['metrics']['dist_grassmannian'],
                "dist_stiefel_chordal": data['metrics']['dist_stiefel_chordal'],
                "dist_stiefel_procrustes": data['metrics']['dist_stiefel_procrustes'],
                "principal_angles": data['metrics']['principal_angles']
            }
            np.savez_compressed(fname, **save_dict)
            print(f"   • {name} -> {fname}")
            print(f"     Grassmann Dist (Subspace): {data['metrics']['dist_grassmannian']:.4f}")
            print(f"     Stiefel Dist (Chordal):    {data['metrics']['dist_stiefel_chordal']:.4f}")
            print(f"     Stiefel Dist (Procrustes): {data['metrics']['dist_stiefel_procrustes']:.4f}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()

    if not args.config_file.exists():
        sys.exit(f"Error: {args.config_file} not found.")

    rng = np.random.default_rng(42)
    spec = JsonParser.parse(args.config_file)
    model = FactorModelBuilder(rng).build(spec)
    
    engine = AnalysisEngine(model, spec, rng)
    results = {}
    
    for sim_conf in spec.simulations:
        results[sim_conf.name] = engine.run_simulation_and_analyze(sim_conf)
        
    engine.save_results(results)
    print("\n✅ Deep Analysis Complete.")

if __name__ == "__main__":
    main()