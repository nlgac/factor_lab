# demo.py: Comprehensive showcase of factor_lab features.

import sys
import numpy as np
from loguru import logger
from factor_lab import (
    DistributionFactory, DataSampler, ReturnsSimulator, 
    ScenarioBuilder, FactorOptimizer, pca_decomposition, svd_decomposition
)

# --- LOGGING SETUP ---
# Configure Loguru to show DEBUG messages (needed for 'debug_intermediate_quantity_length')
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def main():
    logger.info("=== 1. Setup Heterogeneous Model (Generative) ===")
    p_assets, k_factors = 100, 2
    factory = DistributionFactory()
    
    # Create a synthetic model structure
    ds = DataSampler(p_assets, k_factors)
    ds.configure(
        beta=[factory.create_generator('normal', mean=1, std=0.5), 
              factory.create_generator('normal', mean=0, std=1)],
        f_vol=[factory.create_generator('constant', c=0.20), 
               factory.create_generator('constant', c=0.05)],
        d_vol=factory.create_generator('constant', c=0.10)
    )
    model = ds.generate()
    
    logger.info("\n=== 2. Generate Synthetic Returns ===")
    # We use the simulator to create a "History" of returns
    sim = ReturnsSimulator(model, seed=42)
    
    # We log the first 5 rows of the raw (pre-scaled) distributions for inspection
    results = sim.simulate(n=2000, 
        f_gens=[factory.create_generator('student_t', df=4)] * 2, 
        i_gens=[factory.create_generator('normal', mean=0, std=1)] * p_assets,
        debug_intermediate_quantity_length=5
    )
    history = results['security_returns']
    logger.success(f"Generated {history.shape} return matrix.")

    logger.info("\n=== 3. SVD Decomposition (Path B) ===")
    # "Path B": Fit a model directly to the raw returns history.
    svd_model = svd_decomposition(history, k=k_factors)
    
    logger.info(f"SVD Model Extracted. Factors: {svd_model.k}")
    logger.info(f"Top Factor Variance: {svd_model.F[0,0]:.4f}")

    logger.info("\n=== 4. Re-Simulation using SVD Model ===")
    # Initialize a new simulator using the model we just extracted.
    sim_svd = ReturnsSimulator(svd_model)
    
    val_res = sim_svd.simulate(n=1000,
        f_gens=[factory.create_generator('normal', mean=0, std=1)] * k_factors,
        i_gens=[factory.create_generator('normal', mean=0, std=1)] * p_assets
    )
    
    # Validate that our SVD model captures the risk structure
    val = sim_svd.validate_covariance(val_res['security_returns'])
    logger.info(f"SVD Model Validation Error (Frobenius): {val['frobenius_error']:.4f}")

    logger.info("\n=== 5. Optimization ===")
    # Optimize a Long-Only portfolio using the SVD-derived risk model
    opt = FactorOptimizer(svd_model)
    builder = ScenarioBuilder(p_assets)
    
    # CORRECTED: Builder methods return the Scenario, they are not chained on the Scenario itself.
    scen = builder.create("Long Only")
    scen = builder.add_fully_invested(scen)
    scen = builder.add_long_only(scen)
    
    for A, b in scen.equality_constraints: opt.add_eq(A, b)
    for A, b in scen.inequality_constraints: opt.add_ineq(A, b)
    
    res = opt.solve()
    logger.info(f"Solved: {res.solved}, Risk: {res.risk:.4f}")

if __name__ == "__main__":
    main()