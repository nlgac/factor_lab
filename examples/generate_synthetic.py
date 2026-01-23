"""
Synthetic Data Generation Example
==================================
"""
import numpy as np
from factor_lab import DataSampler, DistributionFactory

# FIX: Added **kwargs
def main(**kwargs):
    print("=" * 70)
    print("Synthetic Data Generation")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    factory = DistributionFactory(rng=rng)
    
    # Allow overriding dimensions for tests
    p = kwargs.get('p', 100)
    k = kwargs.get('k', 5)
    n_models = kwargs.get('n_models', 5)
    
    print(f"\nGenerating {n_models} random models (p={p}, k={k})...")
    
    sampler = DataSampler(p=p, k=k, rng=rng)
    
    models = []
    for i in range(n_models):
        model = sampler.configure(
            beta=factory.create('normal', mean=0, std=1.0),
            factor_vol=factory.create('uniform', low=0.05, high=0.20),
            idio_vol=factory.create('uniform', low=0.02, high=0.10)
        ).generate()
        models.append(model)
        
    print(f"Generated {len(models)} models successfully.")
    
    # FIX: Must return data for tests
    return models, {}

if __name__ == "__main__":
    main()