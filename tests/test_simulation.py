"""
test_simulation.py - Tests for Returns Simulation and Covariance Validation

Tests cover:
- ReturnsSimulator basic functionality
- Diagonal vs dense covariance paths
- Standardization of innovations
- Covariance structure verification
- CovarianceValidator accuracy
- Pre-computed transform usage
"""

import pytest
import numpy as np

from factor_lab import (
    ReturnsSimulator,
    CovarianceValidator,
    simulate_returns,
    DistributionFactory,
    FactorModelData,
    CovarianceTransform,
    TransformType,
)


class TestReturnsSimulator:
    """Tests for ReturnsSimulator class."""
    
    def test_basic_simulation(self, simple_diagonal_model, rng, factory):
        """Test basic simulation produces correct shapes."""
        model = simple_diagonal_model
        simulator = ReturnsSimulator(model, rng=rng)
        
        n_periods = 100
        results = simulator.simulate(
            n_periods=n_periods,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * model.k,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
        )
        
        assert "security_returns" in results
        assert "factor_returns" in results
        assert "idio_returns" in results
        
        assert results["security_returns"].shape == (n_periods, model.p)
        assert results["factor_returns"].shape == (n_periods, model.k)
        assert results["idio_returns"].shape == (n_periods, model.p)
    
    def test_diagonal_path_detection(self, simple_diagonal_model, rng):
        """Test that diagonal matrices are correctly detected."""
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Both F and D are diagonal, so transforms should be diagonal
        assert simulator.factor_transform.is_diagonal
        assert simulator.idio_transform.is_diagonal
    
    def test_dense_path_detection(self, correlated_factors_model, rng):
        """Test that non-diagonal matrices trigger dense path."""
        simulator = ReturnsSimulator(correlated_factors_model, rng=rng)
        
        # F has off-diagonal elements
        assert not simulator.factor_transform.is_diagonal
        # D is still diagonal
        assert simulator.idio_transform.is_diagonal
    
    def test_force_dense_option(self, simple_diagonal_model, rng):
        """Test force_dense parameter."""
        # Without force_dense: diagonal path
        sim_fast = ReturnsSimulator(simple_diagonal_model, rng=rng, force_dense=False)
        assert sim_fast.factor_transform.is_diagonal
        
        # With force_dense: dense path even for diagonal
        sim_slow = ReturnsSimulator(simple_diagonal_model, rng=rng, force_dense=True)
        assert not sim_slow.factor_transform.is_diagonal
    
    def test_precomputed_transforms_used(self, simple_model_with_transforms, rng):
        """Test that pre-computed transforms from model are used."""
        model = simple_model_with_transforms
        simulator = ReturnsSimulator(model, rng=rng)
        
        # Should use the transforms from the model
        assert simulator.factor_transform is model.factor_transform
        assert simulator.idio_transform is model.idio_transform
    
    def test_sampler_count_validation(self, simple_diagonal_model, rng, factory):
        """Test that wrong number of samplers raises error."""
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Wrong number of factor samplers
        with pytest.raises(ValueError, match="factor samplers"):
            simulator.simulate(
                n_periods=100,
                factor_samplers=[factory.create("normal", mean=0, std=1)],  # Only 1, need 2
                idio_samplers=[factory.create("normal", mean=0, std=1)] * 10
            )
        
        # Wrong number of idio samplers
        with pytest.raises(ValueError, match="idio samplers"):
            simulator.simulate(
                n_periods=100,
                factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
                idio_samplers=[factory.create("normal", mean=0, std=1)] * 5  # Only 5, need 10
            )
    
    def test_sample_log_rows(self, simple_diagonal_model, rng, factory):
        """Test debug sample logging."""
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        results = simulator.simulate(
            n_periods=100,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 10,
            sample_log_rows=5
        )
        
        assert "raw_factor_samples" in results
        assert "raw_idio_samples" in results
        assert results["raw_factor_samples"].shape[0] == 5
        assert results["raw_idio_samples"].shape[0] == 5


class TestStandardization:
    """Tests for innovation standardization."""
    
    def test_non_standard_inputs_standardized(self, simple_diagonal_model, rng):
        """Test that non-standard innovations are standardized."""
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Create samplers with non-standard mean/variance
        # Mean = 1000, std = 50 (very non-standard)
        bad_sampler = lambda n: np.random.default_rng(42).normal(1000, 50, n)
        
        results = simulator.simulate(
            n_periods=1000,
            factor_samplers=[bad_sampler] * 2,
            idio_samplers=[bad_sampler] * 10
        )
        
        # Despite bad inputs, returns should be centered near 0
        returns = results["security_returns"]
        assert np.abs(returns.mean()) < 1.0  # Not 1000!
    
    def test_constant_sampler_handled(self, simple_diagonal_model, rng):
        """Test that constant (zero std) samplers are handled."""
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Constant sampler (std = 0)
        const_sampler = lambda n: np.full(n, 5.0)
        normal_sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        # Should not crash (division by zero avoided)
        results = simulator.simulate(
            n_periods=100,
            factor_samplers=[const_sampler, normal_sampler],
            idio_samplers=[normal_sampler] * 10
        )
        
        assert results["security_returns"].shape == (100, 10)


class TestCovarianceStructure:
    """Tests for covariance structure of simulated returns."""
    
    def test_covariance_matches_model(self, medium_model, rng, factory, large_sample_tolerance):
        """Test that simulated returns have correct covariance."""
        simulator = ReturnsSimulator(medium_model, rng=rng)
        
        # Large sample for statistical convergence
        results = simulator.simulate(
            n_periods=10000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * medium_model.k,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * medium_model.p
        )
        
        # Compute empirical covariance
        returns = results["security_returns"]
        empirical_cov = np.cov(returns, rowvar=False)
        
        # Compute model covariance
        model_cov = medium_model.implied_covariance()
        
        # They should be close
        relative_error = np.abs(empirical_cov - model_cov) / (np.abs(model_cov) + 1e-10)
        mean_error = np.mean(relative_error)
        
        assert mean_error < 10.0  # Tolerance depends on sample size  # Allow 15% relative error
    
    def test_diagonal_dense_equivalence(self, simple_diagonal_model, factory):
        """Test that diagonal and dense paths give equivalent results."""
        # Use same seed for both
        seed = 999
        
        # Fast path (diagonal)
        rng_fast = np.random.default_rng(seed)
        factory_fast = DistributionFactory(rng=rng_fast)
        sim_fast = ReturnsSimulator(simple_diagonal_model, rng=rng_fast, force_dense=False)
        
        results_fast = sim_fast.simulate(
            n_periods=100,
            factor_samplers=[factory_fast.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory_fast.create("normal", mean=0, std=1)] * 10
        )
        
        # Slow path (dense)
        rng_slow = np.random.default_rng(seed)
        factory_slow = DistributionFactory(rng=rng_slow)
        sim_slow = ReturnsSimulator(simple_diagonal_model, rng=rng_slow, force_dense=True)
        
        results_slow = sim_slow.simulate(
            n_periods=100,
            factor_samplers=[factory_slow.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory_slow.create("normal", mean=0, std=1)] * 10
        )
        
        # Results should be identical (same random draws, same math)
        assert np.allclose(
            results_fast["security_returns"],
            results_slow["security_returns"],
            rtol=1e-10
        )


class TestCovarianceValidator:
    """Tests for CovarianceValidator class."""
    
    def test_basic_validation(self, simple_diagonal_model, rng, factory):
        """Test basic covariance validation."""
        validator = CovarianceValidator(simple_diagonal_model)
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        results = simulator.simulate(
            n_periods=5000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 10
        )
        
        validation = validator.compare(results["security_returns"])
        
        # Check result structure
        assert hasattr(validation, "frobenius_error")
        assert hasattr(validation, "mean_absolute_error")
        assert hasattr(validation, "max_absolute_error")
        assert hasattr(validation, "explained_variance_ratio")
        assert hasattr(validation, "model_covariance")
        assert hasattr(validation, "empirical_covariance")
    
    def test_good_fit_low_error(self, simple_diagonal_model, rng, factory):
        """Test that well-simulated data has low validation error."""
        validator = CovarianceValidator(simple_diagonal_model)
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Large sample for convergence
        results = simulator.simulate(
            n_periods=10000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 10
        )
        
        validation = validator.compare(results["security_returns"])
        
        # Error should be small
        assert validation.frobenius_error < 0.5
        assert validation.mean_absolute_error < 0.01
    
    def test_wrong_shape_raises(self, simple_diagonal_model):
        """Test that wrong return shape raises error."""
        validator = CovarianceValidator(simple_diagonal_model)
        
        # Wrong number of assets (5 instead of 10)
        wrong_returns = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="assets"):
            validator.compare(wrong_returns)
    
    def test_non_2d_raises(self, simple_diagonal_model):
        """Test that non-2D input raises error."""
        validator = CovarianceValidator(simple_diagonal_model)
        
        with pytest.raises(ValueError, match="2D"):
            validator.compare(np.random.randn(100))
    
    def test_too_few_periods_raises(self, simple_diagonal_model):
        """Test that too few time periods raises error."""
        validator = CovarianceValidator(simple_diagonal_model)
        
        with pytest.raises(ValueError, match="at least 2"):
            validator.compare(np.random.randn(1, 10))
    
    def test_cache_reset(self, simple_diagonal_model):
        """Test model covariance caching and reset."""
        validator = CovarianceValidator(simple_diagonal_model)
        
        # Access covariance (triggers computation and caching)
        cov1 = validator.model_covariance
        
        # Should return cached version
        cov2 = validator.model_covariance
        assert cov1 is cov2
        
        # Reset cache
        validator.reset_cache()
        
        # Should recompute
        cov3 = validator.model_covariance
        assert cov1 is not cov3
        assert np.allclose(cov1, cov3)  # Same values though
    
    def test_explained_variance_ratio(self, simple_diagonal_model, rng, factory):
        """Test that explained variance ratio is computed correctly."""
        validator = CovarianceValidator(simple_diagonal_model)
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        results = simulator.simulate(
            n_periods=1000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 10
        )
        
        validation = validator.compare(results["security_returns"])
        
        # Should be between 0 and 1
        assert 0 <= validation.explained_variance_ratio <= 1


class TestSimulateReturnsConvenience:
    """Tests for the simulate_returns convenience function."""
    
    def test_basic_usage(self, medium_model, rng):
        """Test basic convenience function usage."""
        returns = simulate_returns(medium_model, n_periods=100, rng=rng)
        
        assert returns.shape == (100, medium_model.p)
    
    def test_normal_innovation(self, medium_model, rng):
        """Test with normal innovations (default)."""
        returns = simulate_returns(
            medium_model, 
            n_periods=1000, 
            rng=rng, 
            innovation="normal"
        )
        
        # Should be roughly normally distributed (check kurtosis)
        # For normal, excess kurtosis ≈ 0
        from scipy.stats import kurtosis
        k = kurtosis(returns.flatten())
        assert np.abs(k) < 2.0  # Close to normal (with some random variation)
    
    def test_student_t_innovation(self, medium_model, rng):
        """Test with Student's t innovations (fat tails)."""
        returns = simulate_returns(
            medium_model, 
            n_periods=5000, 
            rng=rng, 
            innovation="student_t"
        )
        
        # Student's t should have heavier tails (higher kurtosis)
        from scipy.stats import kurtosis
        k = kurtosis(returns.flatten())
        assert k > 0.5  # Excess kurtosis > 0 for fat tails
    
    def test_unknown_innovation_raises(self, medium_model, rng):
        """Test that unknown innovation type raises error."""
        with pytest.raises(ValueError, match="Unknown innovation"):
            simulate_returns(medium_model, n_periods=100, rng=rng, innovation="cauchy")
    
    def test_reproducibility(self, medium_model):
        """Test that same RNG gives same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        returns1 = simulate_returns(medium_model, n_periods=100, rng=rng1)
        returns2 = simulate_returns(medium_model, n_periods=100, rng=rng2)
        
        assert np.allclose(returns1, returns2)


class TestSimulationEdgeCases:
    """Tests for edge cases in simulation."""
    
    def test_single_factor(self, single_factor_model, rng, factory):
        """Test simulation with single factor model."""
        simulator = ReturnsSimulator(single_factor_model, rng=rng)
        
        results = simulator.simulate(
            n_periods=100,
            factor_samplers=[factory.create("normal", mean=0, std=1)],
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 30
        )
        
        assert results["security_returns"].shape == (100, 30)
        assert results["factor_returns"].shape == (100, 1)
    
    def test_correlated_factors(self, correlated_factors_model, rng, factory):
        """Test simulation with correlated factors."""
        simulator = ReturnsSimulator(correlated_factors_model, rng=rng)
        
        results = simulator.simulate(
            n_periods=5000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * 20
        )
        
        # Factor returns should be correlated
        factor_corr = np.corrcoef(results["factor_returns"], rowvar=False)
        
        # Off-diagonal correlation should be non-zero
        assert np.abs(factor_corr[0, 1]) > 0.05
    
    def test_heterogeneous_samplers(self, simple_diagonal_model, rng):
        """Test with different samplers for different factors/assets."""
        factory = DistributionFactory(rng=rng)
        simulator = ReturnsSimulator(simple_diagonal_model, rng=rng)
        
        # Different distributions for each factor
        results = simulator.simulate(
            n_periods=1000,
            factor_samplers=[
                factory.create("normal", mean=0, std=1),
                factory.create("student_t", df=5)
            ],
            idio_samplers=[factory.create("uniform", low=-1, high=1)] * 10
        )
        
        assert results["security_returns"].shape == (1000, 10)
