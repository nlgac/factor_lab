"""
test_samplers.py - Tests for Distribution Sampling and Data Generation

Tests cover:
- DistributionRegistry registration and retrieval
- DistributionFactory sampler creation
- Parameter validation
- DataSampler configuration and model generation
- RNG injection and reproducibility
"""

import pytest
import numpy as np

from factor_lab import (
    DistributionFactory,
    DistributionRegistry,
    DataSampler,
    DistributionInfo,
    FactorModelData,
)


class TestDistributionRegistry:
    """Tests for DistributionRegistry."""
    
    def test_list_distributions(self):
        """Test listing available distributions."""
        registry = DistributionRegistry()
        distributions = registry.list_distributions()
        
        # Check built-ins are present
        assert "normal" in distributions
        assert "uniform" in distributions
        assert "constant" in distributions
        assert "student_t" in distributions
        assert "beta" in distributions
    
    def test_get_distribution(self):
        """Test retrieving a distribution."""
        registry = DistributionRegistry()
        info = registry.get("normal")
        
        assert info.name == "normal"
        assert "mean" in info.required_params
        assert "std" in info.required_params
    
    def test_get_case_insensitive(self):
        """Test that get is case-insensitive."""
        registry = DistributionRegistry()
        
        info1 = registry.get("NORMAL")
        info2 = registry.get("Normal")
        info3 = registry.get("normal")
        
        assert info1.name == info2.name == info3.name
    
    def test_get_unknown_raises(self):
        """Test that unknown distribution raises KeyError."""
        registry = DistributionRegistry()
        
        with pytest.raises(KeyError, match="Unknown distribution"):
            registry.get("nonexistent")
    
    def test_register_custom(self):
        """Test registering a custom distribution."""
        registry = DistributionRegistry()
        
        # Register a truncated normal (simplified)
        registry.register(
            name="truncated_normal",
            func=lambda rng, n, mean, std: np.clip(
                rng.normal(mean, std, n), -2*std, 2*std
            ),
            required_params={"mean", "std"},
            description="Truncated normal distribution"
        )
        
        assert "truncated_normal" in registry.list_distributions()
        info = registry.get("truncated_normal")
        assert info.description == "Truncated normal distribution"
    
    def test_get_info(self):
        """Test get_info returns correct structure."""
        registry = DistributionRegistry()
        info = registry.get_info("beta")
        
        assert "name" in info
        assert "required_params" in info
        assert "optional_params" in info
        assert "description" in info
        assert info["name"] == "beta"


class TestDistributionFactory:
    """Tests for DistributionFactory."""
    
    def test_create_normal(self, rng):
        """Test creating a normal sampler."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("normal", mean=5.0, std=2.0)
        
        samples = sampler(10000)
        
        assert len(samples) == 10000
        assert np.isclose(samples.mean(), 5.0, atol=0.1)
        assert np.isclose(samples.std(), 2.0, atol=0.1)
    
    def test_create_uniform(self, rng):
        """Test creating a uniform sampler."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("uniform", low=0.0, high=1.0)
        
        samples = sampler(10000)
        
        assert np.all(samples >= 0.0)
        assert np.all(samples < 1.0)
        assert np.isclose(samples.mean(), 0.5, atol=0.05)
    
    def test_create_constant(self, rng):
        """Test creating a constant sampler."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("constant", value=42.0)
        
        samples = sampler(100)
        
        assert np.all(samples == 42.0)
    
    def test_create_student_t(self, rng):
        """Test creating a Student's t sampler."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("student_t", df=5)
        
        samples = sampler(10000)
        
        # Mean should be ~0, variance should be > 1 (fat tails)
        assert np.isclose(samples.mean(), 0.0, atol=0.1)
        assert samples.std() > 1.0  # Heavier tails than normal
    
    def test_missing_params_raises(self, rng):
        """Test that missing required params raises ValueError."""
        factory = DistributionFactory(rng=rng)
        
        with pytest.raises(ValueError, match="requires parameters"):
            factory.create("normal", mean=0.0)  # Missing std
    
    def test_unknown_dist_raises(self, rng):
        """Test that unknown distribution raises KeyError."""
        factory = DistributionFactory(rng=rng)
        
        with pytest.raises(KeyError, match="Unknown distribution"):
            factory.create("nonexistent", param=1.0)
    
    def test_list_distributions(self, rng):
        """Test listing distributions through factory."""
        factory = DistributionFactory(rng=rng)
        distributions = factory.list_distributions()
        
        assert "normal" in distributions
        assert "uniform" in distributions
    
    def test_tuple_size_support(self, rng):
        """Test that samplers support tuple sizes."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("normal", mean=0.0, std=1.0)
        
        samples = sampler((100, 5))
        
        assert samples.shape == (100, 5)
    
    def test_reproducibility(self):
        """Test that same RNG seed gives same samples."""
        rng1 = np.random.default_rng(seed=12345)
        rng2 = np.random.default_rng(seed=12345)
        
        factory1 = DistributionFactory(rng=rng1)
        factory2 = DistributionFactory(rng=rng2)
        
        sampler1 = factory1.create("normal", mean=0.0, std=1.0)
        sampler2 = factory2.create("normal", mean=0.0, std=1.0)
        
        samples1 = sampler1(100)
        samples2 = sampler2(100)
        
        assert np.allclose(samples1, samples2)
    
    def test_different_seeds_different_samples(self):
        """Test that different seeds give different samples."""
        rng1 = np.random.default_rng(seed=111)
        rng2 = np.random.default_rng(seed=222)
        
        factory1 = DistributionFactory(rng=rng1)
        factory2 = DistributionFactory(rng=rng2)
        
        sampler1 = factory1.create("normal", mean=0.0, std=1.0)
        sampler2 = factory2.create("normal", mean=0.0, std=1.0)
        
        samples1 = sampler1(100)
        samples2 = sampler2(100)
        
        assert not np.allclose(samples1, samples2)


class TestDataSampler:
    """Tests for DataSampler model generation."""
    
    def test_basic_generation(self, rng, factory):
        """Test basic model generation."""
        sampler = DataSampler(p=50, k=3, rng=rng)
        
        model = sampler.configure(
            beta=factory.create("normal", mean=0.0, std=1.0),
            factor_vol=factory.create("constant", value=0.2),
            idio_vol=factory.create("constant", value=0.05)
        ).generate()
        
        assert isinstance(model, FactorModelData)
        assert model.k == 3
        assert model.p == 50
    
    def test_dimension_validation(self, rng):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="positive"):
            DataSampler(p=0, k=3, rng=rng)
        
        with pytest.raises(ValueError, match="positive"):
            DataSampler(p=50, k=-1, rng=rng)
    
    def test_configure_returns_self(self, rng, factory):
        """Test that configure returns self for chaining."""
        sampler = DataSampler(p=10, k=2, rng=rng)
        
        result = sampler.configure(
            beta=factory.create("normal", mean=0.0, std=1.0),
            factor_vol=factory.create("constant", value=0.1),
            idio_vol=factory.create("constant", value=0.05)
        )
        
        assert result is sampler
    
    def test_generate_without_configure_raises(self, rng):
        """Test that generate without configure raises error."""
        sampler = DataSampler(p=10, k=2, rng=rng)
        
        with pytest.raises(RuntimeError, match="configure"):
            sampler.generate()
    
    def test_broadcast_single_sampler(self, rng, factory):
        """Test that single sampler is broadcast to all dimensions."""
        sampler = DataSampler(p=20, k=3, rng=rng)
        
        # Single sampler for beta (broadcast to k factors)
        # Single sampler for idio_vol (broadcast to p assets)
        model = sampler.configure(
            beta=factory.create("normal", mean=0.0, std=1.0),
            factor_vol=factory.create("uniform", low=0.1, high=0.3),
            idio_vol=factory.create("uniform", low=0.02, high=0.08)
        ).generate()
        
        assert model.B.shape == (3, 20)
        assert model.F.shape == (3, 3)
        assert model.D.shape == (20, 20)
    
    def test_explicit_sampler_list(self, rng, factory):
        """Test using explicit list of samplers per factor."""
        sampler = DataSampler(p=10, k=2, rng=rng)
        
        model = sampler.configure(
            beta=[
                factory.create("normal", mean=1.0, std=0.2),  # Market factor
                factory.create("normal", mean=0.0, std=0.5),  # Size factor
            ],
            factor_vol=[
                factory.create("constant", value=0.20),
                factory.create("constant", value=0.10),
            ],
            idio_vol=factory.create("constant", value=0.05)
        ).generate()
        
        # Check factor variances match what we specified
        assert np.isclose(model.F[0, 0], 0.20**2)  # Variance = vol^2
        assert np.isclose(model.F[1, 1], 0.10**2)
    
    def test_sampler_list_length_mismatch(self, rng, factory):
        """Test that wrong list length raises error."""
        sampler = DataSampler(p=10, k=3, rng=rng)
        
        with pytest.raises(ValueError, match="length"):
            sampler.configure(
                beta=[
                    factory.create("normal", mean=0.0, std=1.0),
                    factory.create("normal", mean=0.0, std=1.0),
                    # Missing third sampler for k=3
                ],
                factor_vol=factory.create("constant", value=0.1),
                idio_vol=factory.create("constant", value=0.05)
            )
    
    def test_non_callable_raises(self, rng):
        """Test that non-callable argument raises TypeError."""
        sampler = DataSampler(p=10, k=2, rng=rng)
        
        with pytest.raises(TypeError, match="callable"):
            sampler.configure(
                beta="not a callable",
                factor_vol=lambda n: np.ones(n),
                idio_vol=lambda n: np.ones(n)
            )
    
    def test_generated_model_is_valid(self, rng, factory):
        """Test that generated model passes validation."""
        sampler = DataSampler(p=30, k=4, rng=rng)
        
        model = sampler.configure(
            beta=factory.create("normal", mean=0.0, std=1.0),
            factor_vol=factory.create("uniform", low=0.1, high=0.3),
            idio_vol=factory.create("uniform", low=0.02, high=0.1)
        ).generate()
        
        # Should not raise
        model.validate()
    
    def test_covariance_structure(self, rng, factory):
        """Test that F and D are diagonal with correct variances."""
        sampler = DataSampler(p=5, k=2, rng=rng)
        
        model = sampler.configure(
            beta=factory.create("constant", value=1.0),
            factor_vol=factory.create("constant", value=0.15),
            idio_vol=factory.create("constant", value=0.08)
        ).generate()
        
        # F should be diagonal with 0.15^2 on diagonal
        expected_F = np.diag([0.15**2, 0.15**2])
        assert np.allclose(model.F, expected_F)
        
        # D should be diagonal with 0.08^2 on diagonal
        expected_D = np.diag([0.08**2] * 5)
        assert np.allclose(model.D, expected_D)


class TestSamplerEdgeCases:
    """Tests for edge cases and special distributions."""
    
    def test_exponential_distribution(self, rng):
        """Test exponential distribution."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("exponential", scale=2.0)
        
        samples = sampler(10000)
        
        assert np.all(samples >= 0)
        assert np.isclose(samples.mean(), 2.0, atol=0.1)
    
    def test_lognormal_distribution(self, rng):
        """Test lognormal distribution."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("lognormal", mean=0.0, sigma=0.5)
        
        samples = sampler(10000)
        
        assert np.all(samples > 0)
        # Log of samples should be normal
        log_samples = np.log(samples)
        assert np.isclose(log_samples.mean(), 0.0, atol=0.05)
        assert np.isclose(log_samples.std(), 0.5, atol=0.05)
    
    def test_chi2_distribution(self, rng):
        """Test chi-squared distribution."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("chi2", df=5)
        
        samples = sampler(10000)
        
        assert np.all(samples >= 0)
        # Mean should be df
        assert np.isclose(samples.mean(), 5.0, atol=0.2)
    
    def test_beta_distribution(self, rng):
        """Test beta distribution."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("beta", a=2.0, b=5.0)
        
        samples = sampler(10000)
        
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
        # Mean should be a/(a+b) = 2/7
        expected_mean = 2.0 / 7.0
        assert np.isclose(samples.mean(), expected_mean, atol=0.02)
    
    def test_zero_samples(self, rng):
        """Test requesting zero samples."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("normal", mean=0.0, std=1.0)
        
        samples = sampler(0)
        
        assert len(samples) == 0
    
    def test_single_sample(self, rng):
        """Test requesting single sample."""
        factory = DistributionFactory(rng=rng)
        sampler = factory.create("constant", value=3.14)
        
        samples = sampler(1)
        
        assert len(samples) == 1
        assert samples[0] == 3.14
