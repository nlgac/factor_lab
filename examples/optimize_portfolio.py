"""
Portfolio Optimization Example
================================
"""
import numpy as np
from factor_lab import (
    FactorModelData,
    FactorOptimizer,
    ScenarioBuilder,
    minimum_variance_portfolio
)

def create_market_model(p=50, k=3):
    rng = np.random.default_rng(42)
    B = rng.standard_normal((k, p))
    F = np.diag([0.04, 0.01, 0.0025])
    D = np.diag(rng.uniform(0.005, 0.02, p))
    return FactorModelData(B=B, F=F, D=D)

def scenario_1_long_only_basic(model):
    print("\nScenario 1: Long-Only Minimum Variance")
    return minimum_variance_portfolio(model, long_only=True)

# FIX: Added **kwargs
def main(**kwargs):
    print("=" * 70)
    print("Portfolio Optimization Scenarios")
    print("=" * 70)
    
    # Allow overriding p via kwargs for faster tests
    p = kwargs.get('p', 50) 
    model = create_market_model(p=p)
    
    results = {}
    
    # Run scenarios
    results['basic'] = scenario_1_long_only_basic(model)
    
    if results['basic'].solved:
        print(f"   Risk: {results['basic'].risk:.2%}")
    else:
        print("   Optimization failed")

    print("\n" + "=" * 70)
    print("All Scenarios Complete")
    print("=" * 70)
    
    # FIX: Must return results for tests
    return results

if __name__ == "__main__":
    main()