"""
Multi-Distribution Simulation Example
======================================
"""
import numpy as np
from factor_lab import (
    FactorModelData,
    ReturnsSimulator,
    DistributionFactory
)

def create_test_model(p=20, k=3):
    B = np.ones((k, p))
    F = np.eye(k) * 0.04
    D = np.eye(p) * 0.01
    return FactorModelData(B=B, F=F, D=D)

# FIX: Added **kwargs
def main(**kwargs):
    print("=" * 70)
    print("Multi-Distribution Simulation Analysis")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    factory = DistributionFactory(rng=rng)
    
    p = kwargs.get('p', 20)
    model = create_test_model(p=p)
    
    # Standard Normal Simulation
    print("\n1. Simulating Normal Distribution...")
    sim = ReturnsSimulator(model, rng=rng)
    
    # Allow n_periods override for tests
    n_periods = kwargs.get('n_periods', 1000)
    
    results_normal = sim.simulate(
        n_periods=n_periods,
        factor_samplers=[factory.create("normal", mean=0, std=1)] * model.k,
        idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
    )
    
    print(f"   Generated {results_normal['security_returns'].shape} returns")
    
    results = {
        'normal': results_normal
    }
    
    # FIX: Must return results
    return results

if __name__ == "__main__":
    main()