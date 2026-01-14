# demo.py: Showcases the full pipeline of 'factor_lab'.

import numpy as np
from factor_lab import (
    DistributionFactory, DataSampler, ReturnsSimulator, 
    ScenarioBuilder, FactorOptimizer, pca_decomposition
)

def main():
    print("=== 1. Setup Data Generation ===")
    p_assets, k_factors = 100, 3
    
    # Use Factory to create generators
    factory = DistributionFactory()
    
    # Create a DataSampler with specific distributions
    ds = DataSampler(p_assets, k_factors)
    ds.configure(
        beta_gen=factory.create_generator('normal', mean=0, std=1),
        f_vol_gen=factory.create_generator('uniform', low=0.1, high=0.2),
        d_vol_gen=factory.create_generator('uniform', low=0.05, high=0.15)
    )
    model_data = ds.generate()
    print(f"Generated Model: B{model_data.B.shape}, F{model_data.F.shape}")

    print("\n=== 2. Simulate Returns (Hybrid Method) ===")
    # Register a Fat-Tailed distribution
    factory.register('student_t', lambda n, df: np.random.standard_t(df, n))
    
    # Factor 1 is Fat-Tailed, others Normal
    f_gens = [factory.create_generator('student_t', df=4)] + \
             [factory.create_generator('normal', mean=0, std=1) for _ in range(k_factors-1)]
    i_gens = [factory.create_generator('normal', mean=0, std=1) for _ in range(p_assets)]
    
    sim = ReturnsSimulator(model_data, seed=42)
    results = sim.simulate(252, f_gens, i_gens)
    
    returns = results['security_returns']
    print(f"Simulated 1 Year Returns: {returns.shape}")
    print(f"Worst Day: {returns.min():.2%}")

    print("\n=== 3. PCA Reconstruction (Verification) ===")
    # Compute covariance of simulated returns
    cov_matrix = np.cov(returns, rowvar=False)
    B_pca, F_pca = pca_decomposition(cov_matrix, k=k_factors)
    print(f"PCA Recovered F (Diagonal Elements):\n{np.diag(F_pca)[:3]}")

    print("\n=== 4. Optimization Scenarios ===")
    builder = ScenarioBuilder(p_assets)
    
    # Scenario A: Long Only
    scen_a = builder.create("Long Only")
    scen_a = builder.add_fully_invested(scen_a)
    scen_a = builder.add_long_only(scen_a)
    
    # Solve
    opt = FactorOptimizer(model_data)
    for A, b in scen_a.equality_constraints: opt.add_eq(A, b)
    for A, b in scen_a.inequality_constraints: opt.add_ineq(A, b)
    
    res = opt.solve()
    print(f"Scenario: {scen_a.name}")
    print(f"Solved: {res.solved}, Risk: {res.risk:.4f}")
    if res.weights is not None:
        print(f"Net Exposure: {res.weights.sum():.4f}")

if __name__ == "__main__":
    main()