"""
Basic Factor Extraction Example
================================
"""
import numpy as np
import matplotlib.pyplot as plt
from factor_lab import (
    svd_decomposition,
    pca_decomposition,
    compute_explained_variance,
    select_k_by_variance
)

def generate_sample_returns(T=1000, p=50, k_true=3, seed=42):
    rng = np.random.default_rng(seed)
    true_factors = rng.standard_normal((T, k_true))
    true_factors[:, 0] *= 0.20
    true_factors[:, 1] *= 0.10
    true_factors[:, 2] *= 0.05
    true_loadings = rng.standard_normal((k_true, p))
    true_idio = rng.standard_normal((T, p)) * 0.05
    return true_factors @ true_loadings + true_idio

# FIX: Added arguments and **kwargs to support test overrides
def main(T=1000, p=50, k_true=3, seed=42, **kwargs):
    print("=" * 70)
    print(f"Running Factor Extraction Example (T={T}, p={p})")
    print("=" * 70)

    # 1. Generate data
    returns = generate_sample_returns(T=T, p=p, k_true=k_true, seed=seed)
    
    # 2. SVD Decomposition
    k_selected = select_k_by_variance(returns, target_explained=0.90)
    print(f"\n1. Automatic Factor Selection (Target 90%): k={k_selected}")
    
    model_svd = svd_decomposition(returns, k=k_selected)
    
    # 3. Validation
    model_cov = model_svd.implied_covariance()
    empirical_cov = np.cov(returns, rowvar=False)
    frobenius_error = np.linalg.norm(model_cov - empirical_cov, ord='fro')
    
    print(f"   Frobenius error: {frobenius_error:.4f}")
    
    print("\n" + "=" * 70)
    print("Factor extraction complete!")
    print("=" * 70)
    
    # FIX: Must return the model for tests to pass
    return model_svd

if __name__ == "__main__":
    main()