"""
factor_lab_adapter.py

The 'Brain' of the UI.
Acts as a bridge between the stateless factor_lab library and the stateful UI.
Responsible for:
1. Managing Session State (Current Model, Last Simulation, etc.)
2. Formatting NumPy data for UI display (DataTables, Logs).
3. Error handling and validation.
"""

import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger

# Import Core Library
from factor_lab import (
    DistributionFactory, DataSampler, ReturnsSimulator, 
    FactorModelData, FactorOptimizer, ScenarioBuilder,
    svd_decomposition, pca_decomposition
)

class FactorLabAdapter:
    def __init__(self):
        self.factory = DistributionFactory()
        
        # Session State
        self.model: Optional[FactorModelData] = None
        self.sim_results: Optional[Dict[str, np.ndarray]] = None
        self.optimization_result: Optional[Any] = None
        
        # Defaults
        self.p_assets = 100
        self.k_factors = 3
        
    # --- 1. Generator Management ---
    
    def get_available_distributions(self) -> List[str]:
        """Returns list of supported distributions for dropdowns."""
        # Introspect the factory registry (accessing private _funcs for discovery)
        return list(self.factory._registry._funcs.keys())

    def create_generative_model(self, 
                              p: int, k: int, 
                              beta_dist: str, beta_params: dict,
                              f_vol_dist: str, f_vol_params: dict,
                              d_vol_dist: str, d_vol_params: dict) -> bool:
        """
        Builds a factor model using the DistributionFactory.
        """
        logger.info(f"Building Generative Model: P={p}, K={k}")
        try:
            self.p_assets = p
            self.k_factors = k
            
            # Create Generators
            gen_beta = self.factory.create_generator(beta_dist, **beta_params)
            gen_f = self.factory.create_generator(f_vol_dist, **f_vol_params)
            gen_d = self.factory.create_generator(d_vol_dist, **d_vol_params)
            
            # Configure Sampler
            ds = DataSampler(p, k)
            ds.configure(beta=gen_beta, f_vol=gen_f, d_vol=gen_d)
            
            self.model = ds.generate()
            logger.success("Model generated successfully.")
            return True
        except Exception as e:
            logger.exception(f"Failed to generate model: {e}")
            return False

    def create_svd_model(self, n_samples: int, p: int, k: int) -> bool:
        """
        Creates a model by simulating random returns and running SVD.
        (Simulates the 'Empirical' workflow).
        """
        logger.info(f"Running SVD Extraction on synthetic history (T={n_samples})...")
        try:
            # Generate dummy history
            raw_returns = np.random.normal(0, 0.01, (n_samples, p))
            
            # Run Library SVD
            self.model = svd_decomposition(raw_returns, k=k)
            self.p_assets = p
            self.k_factors = k
            
            logger.success(f"SVD Model Extracted. Explained Variance: {np.trace(self.model.F):.4f}")
            return True
        except Exception as e:
            logger.exception(f"SVD Failed: {e}")
            return False

    # --- 2. Simulation ---

    def run_simulation(self, n: int, debug_len: int = 0) -> bool:
        """Runs the simulation using the active model."""
        if not self.model:
            logger.error("No model defined. Create or load a model first.")
            return False
            
        logger.info(f"Starting Simulation (N={n})...")
        try:
            # For simplicity in TUI, use standard normal for driving factors
            # In a full app, we would expose these config options too.
            gen_std = self.factory.create_generator('normal', mean=0, std=1)
            
            sim = ReturnsSimulator(self.model)
            
            self.sim_results = sim.simulate(
                n=n,
                f_gens=[gen_std] * self.model.k,
                i_gens=[gen_std] * self.model.p,
                debug_intermediate_quantity_length=debug_len
            )
            return True
        except Exception as e:
            logger.exception(f"Simulation failed: {e}")
            return False

    def get_returns_preview(self, rows: int = 20) -> List[List[str]]:
        """Formats the first N rows of returns for a DataTable."""
        if not self.sim_results:
            return []
            
        # Get Returns
        R = self.sim_results['security_returns'][:rows]
        
        # Format as strings
        table_data = []
        for t in range(R.shape[0]):
            row_label = f"T+{t}"
            # Show first 5 assets max to fit screen
            vals = [f"{x:.4f}" for x in R[t, :8]] 
            if R.shape[1] > 8:
                vals.append("...")
            table_data.append([row_label] + vals)
            
        return table_data

    # --- 3. Optimization ---

    def optimize_portfolio(self, target_risk: float = None, long_only: bool = True) -> bool:
        """Runs the optimizer on the active model."""
        if not self.model:
            logger.error("No model available for optimization.")
            return False
            
        logger.info("Setting up Optimization Problem...")
        try:
            opt = FactorOptimizer(self.model)
            builder = ScenarioBuilder(self.p_assets)
            
            scen = builder.create("TUI Scenario")
            scen = builder.add_fully_invested(scen)
            
            if long_only:
                scen = builder.add_long_only(scen)
                
            # Apply constraints
            for A, b in scen.equality_constraints: opt.add_eq(A, b)
            for A, b in scen.inequality_constraints: opt.add_ineq(A, b)
            
            self.optimization_result = opt.solve()
            return self.optimization_result.solved
            
        except Exception as e:
            logger.exception(f"Optimization failed: {e}")
            return False

    def get_optimization_summary(self) -> str:
        if not self.optimization_result:
            return "No results."
        
        res = self.optimization_result
        w = res.weights
        
        top_n_idx = np.argsort(w)[::-1][:5]
        top_n_str = ", ".join([f"Asset {i}: {w[i]:.1%}" for i in top_n_idx])
        
        return (
            f"Status: {'Solved' if res.solved else 'Failed'}\n"
            f"Portfolio Risk: {res.risk:.4f}\n"
            f"Objective Val:  {res.objective:.4f}\n"
            f"Top Allocations: {top_n_str}..."
        )