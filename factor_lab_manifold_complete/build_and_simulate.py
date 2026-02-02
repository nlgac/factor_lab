#!/usr/bin/env python3
"""
build_and_simulate.py - Integrated Factor Model Builder & Analyzer
===================================================================

Usage: python build_and_simulate.py model_spec.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np, scipy.linalg

from factor_lab import (FactorModelData, svd_decomposition, ReturnsSimulator,
                        DistributionFactory, save_model)
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from factor_lab.visualization import (create_manifold_dashboard, 
                                      create_interactive_plotly_dashboard,
                                      print_verbose_results)

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
        factors = [DistConfig(name=f.get("distribution", "normal"), params=f.get("params", {}),
                             transform=f.get("transform")) for f in data.get("factor_loadings", [])]
        cov_data = data.get("covariance", {})
        f_vars = [cls._parse_numeric(x) for x in cov_data.get("F_diagonal", [])]
        d_var = cls._parse_numeric(cov_data.get("D_diagonal", 0.1))
        sims = [SimConfig(name=s.get("name"), dist_type=s.get("type", "normal"),
                         params=s.get("params", {})) for s in data.get("simulations", [])]
        return ModelSpec(p_assets=meta.get("p_assets", 100), sim_n_periods=meta.get("n_periods", 100),
                        factor_loadings=factors, factor_variances=f_vars,
                        idio_variance=d_var, simulations=sims)

class FactorModelBuilder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def _gram_schmidt_project(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        return v - (np.dot(u, v) / np.dot(u, u)) * u
    
    def build(self, spec: ModelSpec) -> FactorModelData:
        print(f"\n🏗️  Building: p={spec.p_assets}, k={spec.k_factors}")
        B = np.zeros((spec.k_factors, spec.p_assets))
        for i, config in enumerate(spec.factor_loadings):
            loc, scale = config.params.get("loc", 0.0), config.params.get("scale", 1.0)
            raw = self.rng.normal(loc, scale, spec.p_assets)
            if config.transform == "gram_schmidt" and i > 0:
                print(f"   ↳ Factor {i+1}: Gram-Schmidt")
                B[i, :] = self._gram_schmidt_project(raw, B[0, :])
            else:
                B[i, :] = raw
        F, D = np.diag(spec.factor_variances), np.diag(np.full(spec.p_assets, spec.idio_variance))
        print(f"   ✓ Built successfully")
        return FactorModelData(B=B, F=F, D=D)

class AnalysisEngine:
    def __init__(self, model, spec, rng):
        self.model, self.spec, self.rng = model, spec, rng
        self.factory = DistributionFactory(rng)
    
    def run_simulation_and_analyze(self, sim_config):
        print(f"\n{'='*70}\n  {sim_config.name} ({sim_config.dist_type})\n{'='*70}")
        
        simulator = ReturnsSimulator(self.model, rng=self.rng)
        results = simulator.simulate(self.spec.sim_n_periods)
        returns = results['security_returns']
        
        print(f"\n✓ Simulated: {returns.shape}")
        est_model = svd_decomposition(returns, k=self.model.k)
        
        context = SimulationContext(model=self.model, security_returns=returns,
                                   factor_returns=results['factor_returns'],
                                   idio_returns=results['idio_returns'])
        
        print(f"\n🔬 Analyzing...")
        analyses = [("Manifold", Analyses.manifold_distances()),
                   ("Eigenvalue", Analyses.eigenvalue_analysis(k_top=self.model.k, compare_eigenvectors=True)),
                   ("Eigenvector", Analyses.eigenvector_comparison(k=self.model.k))]
        
        all_results = {}
        for name, analysis in analyses:
            all_results.update(analysis.analyze(context))
        
        print_verbose_results(all_results, sim_config.name)
        
        # Visualizations
        Path("output").mkdir(exist_ok=True)
        safe_name = sim_config.name.replace(" ", "_").lower()
        create_manifold_dashboard(all_results, Path(f"output/dash_{safe_name}.png"))
        try:
            create_interactive_plotly_dashboard(all_results, Path(f"output/dash_{safe_name}.html"))
        except: pass
        
        # Save comprehensive NPZ file (matching Gemini's format)
        print(f"\n💾 SAVING COMPREHENSIVE RESULTS")
        
        # Compute orthonormalized B (true_ortho_B)
        Q_true, _ = scipy.linalg.qr(self.model.B.T, mode='economic')
        true_ortho_B = Q_true.T  # (k, p)
        
        # Extract eigenvalues and eigenvectors from results
        true_eigenvalues = all_results.get('true_eigenvalues', np.array([]))
        true_eigenvectors = all_results.get('true_eigenvectors', np.array([]))
        sample_eigenvalues = all_results.get('sample_eigenvalues', np.array([]))
        sample_eigenvectors = all_results.get('sample_eigenvectors', np.array([]))
        
        # Build comprehensive save dict (Gemini's format)
        save_dict = {
            # Raw data
            'security_returns': returns,
            
            # True model
            'true_B': self.model.B,
            'true_ortho_B': true_ortho_B,  # Orthonormalized version
            'true_F': self.model.F,
            'true_D': self.model.D,
            'true_eigenvalues': true_eigenvalues,
            'true_eigenvectors': true_eigenvectors,
            
            # Sample model
            'sample_B': est_model.B,
            'sample_F': est_model.F,
            'sample_D': est_model.D,
            'sample_eigenvalues': sample_eigenvalues,
            'sample_eigenvectors': sample_eigenvectors,
            
            # Manifold distances (with Gemini's naming)
            'dist_grassmannian': all_results.get('dist_grassmannian', np.nan),
            'dist_stiefel_procrustes': all_results.get('dist_procrustes', np.nan),
            'dist_stiefel_chordal': all_results.get('dist_chordal', np.nan),
            'principal_angles': all_results.get('principal_angles', np.array([])),
        }
        
        # Add any other numeric results
        for key, value in all_results.items():
            if key not in save_dict and isinstance(value, (int, float, np.ndarray)):
                save_dict[key] = value
        
        # Save
        fname = f"simulation_{safe_name}.npz"
        np.savez_compressed(fname, **save_dict)
        
        save_model(self.model, "factor_model.npz")
        
        print(f"   ✓ Comprehensive data: {fname}")
        print(f"   ✓ Model: factor_model.npz")
        print(f"\n   📊 NPZ file contains {len(save_dict)} arrays:")
        for key in sorted(save_dict.keys()):
            val = save_dict[key]
            if isinstance(val, np.ndarray):
                print(f"      • {key:30s} {str(val.shape):20s}")
            else:
                print(f"      • {key:30s} scalar: {val:.6f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if not args.config_file.exists():
        sys.exit(f"❌ File not found: {args.config_file}")
    
    print("="*70 + "\n  FACTOR MODEL BUILDER & SIMULATOR\n" + "="*70)
    rng = np.random.default_rng(args.seed)
    spec = JsonParser.parse(args.config_file)
    model = FactorModelBuilder(rng).build(spec)
    engine = AnalysisEngine(model, spec, rng)
    
    for sim_config in spec.simulations:
        engine.run_simulation_and_analyze(sim_config)
    
    print(f"\n✅ Complete! Check output/ directory.\n")

if __name__ == "__main__":
    main()
