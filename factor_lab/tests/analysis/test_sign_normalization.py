"""
test_sign_normalization.py  (tests/analysis/)

Verifies that sign normalization is applied correctly to:
- Factor loadings extracted by SVD (in decomposition.py)
- True eigenvectors from eigendecomposition of Sigma = B'FB + D
- Sample eigenvectors from PCA on simulated returns
"""

import numpy as np
import pytest

from factor_lab import FactorModelData, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from conftest import make_simulator


class TestSignNormalization:

    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        k, p, T = 3, 50, 200
        returns = (np.random.randn(T, k) @ np.random.randn(k, p)
                   + np.random.randn(T, p) * 0.1)
        return returns, k

    def _context_from_model(self, model, seed=42):
        rng = np.random.default_rng(seed)
        results = make_simulator(model, rng).simulate(n_periods=200)
        return SimulationContext(
            model=model,
            security_returns=results['security_returns'],
            factor_returns=results['factor_returns'],
            idio_returns=results['idio_returns'],
        )

    def test_svd_factor_loadings_positive_mean(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        means = model.B.mean(axis=1)
        assert model.B.shape == (k, returns.shape[1])
        assert np.all(means >= -1e-10), f"Negative means: {means}"

    def test_true_eigenvectors_positive_mean(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        ctx = self._context_from_model(model)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(ctx)
        means = eigvec['true_eigenvectors'].mean(axis=1)
        assert eigvec['true_eigenvectors'].shape[0] == k
        assert np.all(means >= -1e-10), f"Negative means: {means}"

    def test_sample_eigenvectors_positive_mean(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        ctx = self._context_from_model(model)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(ctx)
        means = eigvec['sample_eigenvectors'].mean(axis=1)
        assert eigvec['sample_eigenvectors'].shape[0] == k
        assert np.all(means >= -1e-10), f"Negative means: {means}"

    def test_all_components_consistent_signs(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        ctx = self._context_from_model(model)
        eigvec = Analyses.eigenvector_comparison(k=k).analyze(ctx)
        for label, arr in [
            ("factor loadings",      model.B),
            ("true eigenvectors",    eigvec['true_eigenvectors']),
            ("sample eigenvectors",  eigvec['sample_eigenvectors']),
        ]:
            means = arr.mean(axis=1)
            assert np.all(means >= -1e-10), f"{label} have negative means: {means}"

    def test_sign_normalization_preserves_subspace(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        flipped = FactorModelData(B=-model.B, F=model.F, D=model.D)
        assert np.allclose(model.implied_covariance(), flipped.implied_covariance())

    def test_sign_normalization_idempotent(self, synthetic_data):
        returns, k = synthetic_data
        model = svd_decomposition(returns, k=k, center=True)
        B = model.B.copy()
        signs = np.where(B.mean(axis=1) < 0, -1, 1)
        assert np.allclose(B, B * signs[:, np.newaxis])
        assert np.all(signs == 1)


def test_sign_normalization_comprehensive():
    """Module-level regression: full workflow positive-mean check."""
    np.random.seed(42)
    k, p, T = 3, 50, 200
    returns = (np.random.randn(T, k) @ np.random.randn(k, p)
               + np.random.randn(T, p) * 0.1)

    model = svd_decomposition(returns, k=k, center=True)
    assert np.all(model.B.mean(axis=1) >= -1e-10)

    rng = np.random.default_rng(42)
    results = make_simulator(model, rng).simulate(n_periods=T)
    ctx = SimulationContext(
        model=model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns'],
    )
    eigvec = Analyses.eigenvector_comparison(k=k).analyze(ctx)
    assert np.all(eigvec['true_eigenvectors'].mean(axis=1)   >= -1e-10)
    assert np.all(eigvec['sample_eigenvectors'].mean(axis=1) >= -1e-10)
