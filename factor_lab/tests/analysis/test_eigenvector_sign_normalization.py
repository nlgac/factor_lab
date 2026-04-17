"""
test_eigenvector_sign_normalization.py  (tests/analysis/)

Verifies sign normalization consistency across all three components:
factor loadings, true eigenvectors, and sample eigenvectors.
"""

import numpy as np
import pytest

from factor_lab import FactorModelData, svd_decomposition
from factor_lab.analysis import SimulationContext
from factor_lab.analyses import Analyses
from conftest import make_simulator


@pytest.fixture(scope="module")
def svd_model():
    np.random.seed(42)
    k, p, T = 3, 50, 200
    returns = (np.random.randn(T, k) @ np.random.randn(k, p)
               + np.random.randn(T, p) * 0.1)
    return svd_decomposition(returns, k=k, center=True)


@pytest.fixture(scope="module")
def eigvec_results(svd_model):
    rng = np.random.default_rng(42)
    results = make_simulator(svd_model, rng).simulate(n_periods=200)
    ctx = SimulationContext(
        model=svd_model,
        security_returns=results['security_returns'],
        factor_returns=results['factor_returns'],
        idio_returns=results['idio_returns'],
    )
    return Analyses.eigenvector_comparison(k=svd_model.k).analyze(ctx)


class TestSignNormalization:

    def test_factor_loadings_positive_mean(self, svd_model):
        means = svd_model.B.mean(axis=1)
        assert np.all(means >= -1e-10), f"Negative loading means: {means}"

    def test_true_eigenvectors_positive_mean(self, eigvec_results):
        means = eigvec_results['true_eigenvectors'].mean(axis=1)
        assert np.all(means >= -1e-10), f"Negative true eigvec means: {means}"

    def test_sample_eigenvectors_positive_mean(self, eigvec_results):
        means = eigvec_results['sample_eigenvectors'].mean(axis=1)
        assert np.all(means >= -1e-10), f"Negative sample eigvec means: {means}"

    def test_normalization_idempotent(self, svd_model):
        signs = np.where(svd_model.B.mean(axis=1) < 0, -1, 1)
        assert np.all(signs == 1), f"Normalization not idempotent: {signs}"

    def test_sign_flip_preserves_covariance(self, svd_model):
        Sigma_orig    = svd_model.implied_covariance()
        Sigma_flipped = FactorModelData(
            B=-svd_model.B, F=svd_model.F, D=svd_model.D
        ).implied_covariance()
        assert np.allclose(Sigma_orig, Sigma_flipped)

    def test_eigvec_shape(self, svd_model, eigvec_results):
        k = svd_model.k
        assert eigvec_results['true_eigenvectors'].shape[0] == k
        assert eigvec_results['sample_eigenvectors'].shape[0] == k
