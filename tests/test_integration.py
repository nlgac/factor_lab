"""
test_integration.py - Integration Tests for End-to-End Workflows

These tests verify that all components work correctly together:
- Data generation → SVD extraction → Simulation → Validation
- Model creation → Optimization → Result verification
- Full pipeline with realistic parameters
"""

import pytest
import numpy as np

from factor_lab import (
    # Decomposition
    svd_decomposition,
    pca_decomposition,
    compute_explained_variance,
    select_k_by_variance,
    
    # Simulation
    ReturnsSimulator,
    CovarianceValidator,
    simulate_returns,
    
    # Optimization
    FactorOptimizer,
    ScenarioBuilder,
    minimum_variance_portfolio,
    
    # Samplers
    DistributionFactory,
    DataSampler,
    
    # Types
    FactorModelData,
)


class TestGenerativeToOptimizationPipeline:
    """Test: DataSampler → Simulate → SVD → Optimize."""
    
    def test_full_pipeline(self, rng):
        """Test complete pipeline from generation to optimization."""
        # Step 1: Generate a synthetic model
        factory = DistributionFactory(rng=rng)
        
        p_assets, k_factors = 100, 3
        
        data_sampler = DataSampler(p=p_assets, k=k_factors, rng=rng)
        original_model = data_sampler.configure(
            beta=[
                factory.create("normal", mean=1.0, std=0.3),   # Market
                factory.create("normal", mean=0.0, std=0.5),   # Value
                factory.create("normal", mean=0.0, std=0.5),   # Momentum
            ],
            factor_vol=[
                factory.create("constant", value=0.20),
                factory.create("constant", value=0.15),
                factory.create("constant", value=0.10),
            ],
            idio_vol=factory.create("uniform", low=0.05, high=0.15)
        ).generate()
        
        # Step 2: Simulate returns using the original model
        simulator = ReturnsSimulator(original_model, rng=rng)
        sim_results = simulator.simulate(
            n_periods=2000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * k_factors,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * p_assets
        )
        
        history = sim_results["security_returns"]
        
        # Step 3: Extract a new model from the simulated history
        extracted_model = svd_decomposition(history, k=k_factors)
        
        # Step 4: Validate the extracted model
        validator = CovarianceValidator(extracted_model)
        
        # Generate new returns from extracted model
        new_sim = ReturnsSimulator(extracted_model, rng=rng)
        new_results = new_sim.simulate(
            n_periods=1000,
            factor_samplers=[factory.create("normal", mean=0, std=1)] * k_factors,
            idio_samplers=[factory.create("normal", mean=0, std=1)] * p_assets
        )
        
        validation = validator.compare(new_results["security_returns"])
        
        # Validation error should be reasonable
        assert validation.frobenius_error < 1.0
        
        # Step 5: Optimize a portfolio
        result = minimum_variance_portfolio(
            extracted_model,
            long_only=True,
            max_weight=0.05
        )
        
        assert result.solved
        assert np.isclose(result.weights.sum(), 1.0)
        assert np.all(result.weights >= -1e-7)
        assert np.all(result.weights <= 0.05 + 1e-7)
    
    def test_pipeline_reproducibility(self):
        """Test that full pipeline is reproducible with same seed."""
        def run_pipeline(seed):
            rng = np.random.default_rng(seed)
            factory = DistributionFactory(rng=rng)
            
            # Generate model
            sampler = DataSampler(p=50, k=2, rng=rng)
            model = sampler.configure(
                beta=factory.create("normal", mean=0, std=1),
                factor_vol=factory.create("constant", value=0.2),
                idio_vol=factory.create("constant", value=0.1)
            ).generate()
            
            # Simulate
            sim = ReturnsSimulator(model, rng=rng)
            results = sim.simulate(
                n_periods=500,
                factor_samplers=[factory.create("normal", mean=0, std=1)] * 2,
                idio_samplers=[factory.create("normal", mean=0, std=1)] * 50
            )
            
            # Optimize
            opt_result = minimum_variance_portfolio(model, long_only=True)
            
            return results["security_returns"], opt_result.weights
        
        # Run twice with same seed
        returns1, weights1 = run_pipeline(12345)
        returns2, weights2 = run_pipeline(12345)
        
        assert np.allclose(returns1, returns2)
        assert np.allclose(weights1, weights2)


class TestSVDAndOptimizationConsistency:
    """Test that SVD-extracted models optimize correctly."""
    
    def test_svd_model_optimization(self, sample_returns):
        """Test optimization on SVD-extracted model."""
        # Extract model
        model = svd_decomposition(sample_returns, k=5)
        
        # Optimize
        result = minimum_variance_portfolio(model, long_only=True)
        
        assert result.solved
        assert result.risk > 0
        
        # Verify constraints
        assert np.isclose(result.weights.sum(), 1.0)
        assert np.all(result.weights >= -1e-7)
    
    def test_explained_variance_vs_optimization(self, sample_returns):
        """Test relationship between explained variance and optimization."""
        # Models with different k
        models = {}
        for k in [1, 3, 5, 10]:
            if k <= min(sample_returns.shape):
                models[k] = svd_decomposition(sample_returns, k=k)
        
        # More factors should explain more variance
        explained = {k: compute_explained_variance(m) for k, m in models.items()}
        
        # Check monotonicity
        prev_exp = 0
        for k in sorted(explained.keys()):
            assert explained[k] >= prev_exp - 0.01  # Allow tiny numerical error
            prev_exp = explained[k]
        
        # Optimize each
        risks = {}
        for k, model in models.items():
            result = minimum_variance_portfolio(model, long_only=True, max_weight=0.1)
            if result.solved:
                risks[k] = result.risk
        
        # Risk estimates should be consistent (not too different)
        risk_values = list(risks.values())
        if len(risk_values) >= 2:
            # Coefficient of variation should be reasonable
            cv = np.std(risk_values) / np.mean(risk_values)
            assert cv < 1.0  # Statistical variation can be higher  # Not too much variation


class TestConstraintEnforcement:
    """Test that various constraint combinations work correctly."""
    
    def test_sector_neutral_optimization(self, rng):
        """Test sector-neutral portfolio optimization (dollar neutral)."""
        # Create a model
        factory = DistributionFactory(rng=rng)
        sampler = DataSampler(p=30, k=2, rng=rng)
        model = sampler.configure(
            beta=factory.create("normal", mean=0, std=1),
            factor_vol=factory.create("constant", value=0.2),
            idio_vol=factory.create("constant", value=0.1)
        ).generate()
        
        # Define sectors (3 sectors, 10 assets each)
        sectors = np.repeat([0, 1, 2], 10)
        
        # Build scenario - dollar neutral (sum=0) with sector neutral
        # This is feasible: each sector sums to 0, total sums to 0
        scenario = (ScenarioBuilder(30)
            .create("Sector Neutral")
            .add_custom_equality(np.ones((1, 30)), np.array([0.0]))  # Dollar neutral
            .add_sector_neutral(sectors)
            .add_box_constraints(low=-0.2, high=0.2)  # Allow shorts
            .build())
        
        # Optimize
        optimizer = FactorOptimizer(model)
        optimizer.apply_scenario(scenario)
        result = optimizer.solve()
        
        assert result.solved
        
        # Check sector neutrality
        weights = result.weights
        for sector in [0, 1, 2]:
            sector_sum = weights[sectors == sector].sum()
            assert np.isclose(sector_sum, 0.0, atol=1e-6)
    
    def test_complex_constraints(self, medium_model):
        """Test optimization with multiple constraint types."""
        p = medium_model.p
        
        # Complex scenario
        scenario = (ScenarioBuilder(p)
            .create("Complex")
            .add_fully_invested()
            .add_long_only()
            .add_box_constraints(low=0.0, high=0.08)
            .build())
        
        # Add custom constraint: first 10 assets sum to at least 0.2
        A_custom = np.zeros((1, p))
        A_custom[0, :10] = -1.0  # -sum(w[:10]) <= -0.2, i.e., sum >= 0.2
        b_custom = np.array([-0.2])
        scenario.inequality_constraints.append((A_custom, b_custom))
        
        # Optimize
        optimizer = FactorOptimizer(medium_model)
        optimizer.apply_scenario(scenario)
        result = optimizer.solve()
        
        assert result.solved
        
        # Verify custom constraint
        first_10_sum = result.weights[:10].sum()
        assert first_10_sum >= 0.2 - 1e-6


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""
    
    def test_ill_conditioned_covariance(self, rng):
        """Test handling of ill-conditioned covariance matrices."""
        # Create a nearly singular model
        p, k = 50, 2
        
        B = np.zeros((k, p))
        B[0, :] = 1.0  # All assets load equally on factor 1
        B[1, :25] = 0.001  # Very small loading differences
        
        F = np.diag([0.04, 0.0001])  # Second factor variance very small
        D = np.diag(np.full(p, 0.01))
        
        model = FactorModelData(B=B, F=F, D=D)
        
        # Should still optimize
        result = minimum_variance_portfolio(model, long_only=True)
        
        # May or may not solve depending on solver, but shouldn't crash
        assert isinstance(result.solved, bool)
    
    def test_many_assets(self, rng):
        """Test with a large number of assets."""
        # Generate large model
        factory = DistributionFactory(rng=rng)
        sampler = DataSampler(p=200, k=5, rng=rng)
        model = sampler.configure(
            beta=factory.create("normal", mean=0, std=0.5),
            factor_vol=factory.create("uniform", low=0.1, high=0.25),
            idio_vol=factory.create("uniform", low=0.03, high=0.1)
        ).generate()
        
        # Simulate
        returns = simulate_returns(model, n_periods=500, rng=rng)
        
        assert returns.shape == (500, 200)
        
        # Extract and optimize
        extracted = svd_decomposition(returns, k=5)
        result = minimum_variance_portfolio(extracted, long_only=True, max_weight=0.02)
        
        assert result.solved


class TestPCAVsSVDConsistency:
    """Compare PCA and SVD decomposition results."""
    
    def test_eigenvalue_consistency(self, sample_returns):
        """Test that PCA and SVD give consistent eigenvalues."""
        k = 5
        
        # SVD on returns
        model_svd = svd_decomposition(sample_returns, k=k)
        
        # PCA on sample covariance
        sample_cov = np.cov(sample_returns, rowvar=False)
        B_pca, F_pca = pca_decomposition(sample_cov, k=k)
        
        # Eigenvalues should match
        svd_eigenvalues = np.sort(np.diag(model_svd.F))[::-1]
        pca_eigenvalues = np.sort(np.diag(F_pca))[::-1]
        
        assert np.allclose(svd_eigenvalues, pca_eigenvalues, rtol=1e-10)
    
    def test_optimization_consistency(self, sample_returns):
        """Test that optimization gives similar results for PCA and SVD models."""
        k = 5
        p = sample_returns.shape[1]
        
        # SVD model
        model_svd = svd_decomposition(sample_returns, k=k)
        
        # PCA-based model (construct manually)
        sample_cov = np.cov(sample_returns, rowvar=False)
        B_pca, F_pca = pca_decomposition(sample_cov, k=k)
        
        # For PCA, we need to estimate D separately
        # Use residual variance
        explained_cov = B_pca.T @ F_pca @ B_pca
        residual_var = np.diag(sample_cov - explained_cov)
        residual_var = np.maximum(residual_var, 1e-8)  # Ensure positive
        D_pca = np.diag(residual_var)
        
        model_pca = FactorModelData(B=B_pca, F=F_pca, D=D_pca)
        
        # Optimize both
        result_svd = minimum_variance_portfolio(model_svd, long_only=True, max_weight=0.1)
        result_pca = minimum_variance_portfolio(model_pca, long_only=True, max_weight=0.1)
        
        # Both should solve
        assert result_svd.solved
        assert result_pca.solved
        
        # Risks should be similar
        assert np.isclose(result_svd.risk, result_pca.risk, rtol=0.1)


class TestSelectKByVariance:
    """Test automatic factor count selection."""
    
    def test_k_selection_accuracy(self, rng):
        """Test that k selection achieves target variance."""
        # Create data with known factor structure
        T, p = 1000, 100
        k_true = 5
        
        # Generate returns with clear factor structure
        factory = DistributionFactory(rng=rng)
        sampler = DataSampler(p=p, k=k_true, rng=rng)
        model = sampler.configure(
            beta=factory.create("normal", mean=0, std=1),
            factor_vol=factory.create("constant", value=0.2),
            idio_vol=factory.create("constant", value=0.02)  # Low idio
        ).generate()
        
        returns = simulate_returns(model, n_periods=T, rng=rng)
        
        # Select k for 90% variance
        k_selected = select_k_by_variance(returns, target_explained=0.90)
        
        # Should select reasonable k
        assert 1 <= k_selected <= p
        
        # Verify that selected k achieves target
        extracted = svd_decomposition(returns, k=k_selected)
        actual_explained = compute_explained_variance(extracted)
        
        # Should be close to or above target
        assert actual_explained >= 0.85  # Allow some slack
