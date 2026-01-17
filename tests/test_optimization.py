"""
test_optimization.py - Tests for Portfolio Optimization

Tests cover:
- ScenarioBuilder fluent API
- Constraint construction (fully invested, long only, box, etc.)
- FactorOptimizer solving
- Constraint enforcement verification
- Convenience functions
"""

import pytest
import numpy as np

from factor_lab import (
    FactorOptimizer,
    ScenarioBuilder,
    minimum_variance_portfolio,
    Scenario,
)


class TestScenarioBuilder:
    """Tests for ScenarioBuilder fluent API."""
    
    def test_create(self):
        """Test scenario creation."""
        builder = ScenarioBuilder(p=50)
        builder.create("Test Scenario", "A test description")
        scenario = builder.build()
        
        assert scenario.name == "Test Scenario"
        assert scenario.description == "A test description"
    
    def test_fluent_chaining(self):
        """Test that methods can be chained."""
        scenario = (ScenarioBuilder(p=50)
            .create("Chained")
            .add_fully_invested()
            .add_long_only()
            .build())
        
        assert scenario.name == "Chained"
        assert scenario.n_equality() == 1  # fully invested
        assert scenario.n_inequality() == 50  # long only (p constraints)
    
    def test_build_without_create_raises(self):
        """Test that build without create raises error."""
        builder = ScenarioBuilder(p=50)
        
        with pytest.raises(RuntimeError, match="create()"):
            builder.build()
    
    def test_invalid_p(self):
        """Test that p <= 0 raises error."""
        with pytest.raises(ValueError, match="positive"):
            ScenarioBuilder(p=0)
        
        with pytest.raises(ValueError, match="positive"):
            ScenarioBuilder(p=-5)


class TestConstraints:
    """Tests for individual constraint types."""
    
    def test_fully_invested(self):
        """Test fully invested constraint."""
        p = 10
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_fully_invested()
            .build())
        
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        
        assert A.shape == (1, p)
        assert np.allclose(A, np.ones((1, p)))
        assert b.shape == (1,)
        assert b[0] == 1.0
    
    def test_long_only(self):
        """Test long only constraint."""
        p = 10
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_long_only()
            .build())
        
        assert len(scenario.inequality_constraints) == 1
        A, b = scenario.inequality_constraints[0]
        
        assert A.shape == (p, p)
        assert np.allclose(A, -np.eye(p))
        assert np.allclose(b, np.zeros(p))
    
    def test_box_constraints(self):
        """Test box constraints."""
        p = 10
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_box_constraints(low=-0.1, high=0.2)
            .build())
        
        # Should add two inequality constraints
        assert len(scenario.inequality_constraints) == 2
        
        # Upper bound: w <= 0.2
        A_upper, b_upper = scenario.inequality_constraints[0]
        assert np.allclose(A_upper, np.eye(p))
        assert np.allclose(b_upper, 0.2)
        
        # Lower bound: -w <= 0.1 (i.e., w >= -0.1)
        A_lower, b_lower = scenario.inequality_constraints[1]
        assert np.allclose(A_lower, -np.eye(p))
        assert np.allclose(b_lower, 0.1)
    
    def test_box_constraints_validation(self):
        """Test that low > high raises error."""
        builder = ScenarioBuilder(10).create("Test")
        
        with pytest.raises(ValueError, match="low.*high"):
            builder.add_box_constraints(low=0.5, high=0.1)
    
    def test_sector_neutral(self):
        """Test sector neutrality constraints."""
        p = 12
        # 3 sectors with 4 assets each
        sectors = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_sector_neutral(sectors)
            .build())
        
        # Should have 3 equality constraints (one per sector)
        assert len(scenario.equality_constraints) == 3
        
        # Each constraint should sum weights in that sector to 0
        for A, b in scenario.equality_constraints:
            assert A.shape == (1, p)
            assert b[0] == 0.0
    
    def test_sector_neutral_validation(self):
        """Test sector assignment validation."""
        builder = ScenarioBuilder(10).create("Test")
        
        # Wrong size
        with pytest.raises(ValueError, match="shape"):
            builder.add_sector_neutral(np.array([0, 1, 2]))  # Wrong length
    
    def test_custom_equality(self):
        """Test custom equality constraint."""
        p = 10
        A = np.random.randn(2, p)
        b = np.array([0.0, 1.0])
        
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_custom_equality(A, b)
            .build())
        
        assert len(scenario.equality_constraints) == 1
        A_stored, b_stored = scenario.equality_constraints[0]
        assert np.allclose(A_stored, A)
        assert np.allclose(b_stored, b)
    
    def test_custom_inequality(self):
        """Test custom inequality constraint."""
        p = 10
        A = np.random.randn(3, p)
        b = np.ones(3)
        
        scenario = (ScenarioBuilder(p)
            .create("Test")
            .add_custom_inequality(A, b)
            .build())
        
        assert len(scenario.inequality_constraints) == 1
    
    def test_custom_constraint_validation(self):
        """Test custom constraint validation."""
        builder = ScenarioBuilder(10).create("Test")
        
        # A not 2D
        with pytest.raises(ValueError, match="2D"):
            builder.add_custom_equality(np.ones(10), np.ones(1))
        
        # A wrong number of columns
        with pytest.raises(ValueError, match="columns"):
            builder.add_custom_equality(np.ones((1, 5)), np.ones(1))
        
        # b not 1D
        with pytest.raises(ValueError, match="1D"):
            builder.add_custom_equality(np.ones((1, 10)), np.ones((1, 1)))
        
        # A and b dimension mismatch
        with pytest.raises(ValueError, match="dimension mismatch"):
            builder.add_custom_equality(np.ones((2, 10)), np.ones(3))


class TestFactorOptimizer:
    """Tests for FactorOptimizer."""
    
    def test_solve_basic(self, simple_diagonal_model):
        """Test basic optimization."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        # Add fully invested constraint
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        
        result = optimizer.solve()
        
        assert result.solved
        assert result.weights is not None
        assert len(result.weights) == model.p
        assert np.isclose(result.weights.sum(), 1.0)
    
    def test_solve_long_only(self, simple_diagonal_model):
        """Test long only optimization."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        # Fully invested
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        # Long only
        optimizer.add_inequality(-np.eye(model.p), np.zeros(model.p))
        
        result = optimizer.solve()
        
        assert result.solved
        assert np.all(result.weights >= -1e-7)  # Allow tiny numerical error
    
    def test_solve_box_constraints(self, medium_model):
        """Test box constraints are enforced."""
        model = medium_model
        optimizer = FactorOptimizer(model)
        
        # Apply scenario with box constraints
        scenario = (ScenarioBuilder(model.p)
            .create("Test")
            .add_fully_invested()
            .add_long_only()
            .add_box_constraints(low=0.0, high=0.10)
            .build())
        
        optimizer.apply_scenario(scenario)
        result = optimizer.solve()
        
        assert result.solved
        assert np.all(result.weights >= -1e-7)
        assert np.all(result.weights <= 0.10 + 1e-7)
    
    def test_apply_scenario(self, simple_diagonal_model):
        """Test apply_scenario method."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        scenario = (ScenarioBuilder(model.p)
            .create("Test")
            .add_fully_invested()
            .add_long_only()
            .build())
        
        optimizer.apply_scenario(scenario)
        result = optimizer.solve()
        
        assert result.solved
    
    def test_reset_constraints(self, simple_diagonal_model):
        """Test constraint reset."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        # First solve: long only
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        optimizer.add_inequality(-np.eye(model.p), np.zeros(model.p))
        result1 = optimizer.solve()
        
        # Reset and solve again (unconstrained except fully invested)
        optimizer.reset_constraints()
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        result2 = optimizer.solve()
        
        # Without long-only, result might have negative weights
        # (depends on model structure)
        assert result2.solved
    
    def test_risk_calculation(self, simple_diagonal_model):
        """Test that returned risk is accurate."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        result = optimizer.solve()
        
        assert result.solved
        
        # Verify risk using manual calculation
        w = result.weights
        cov = model.implied_covariance()
        expected_variance = w @ cov @ w
        expected_risk = np.sqrt(expected_variance)
        
        assert np.isclose(result.risk, expected_risk, rtol=1e-4)
    
    def test_infeasible_problem(self, simple_diagonal_model):
        """Test handling of infeasible problems."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        # Conflicting constraints: sum = 1 and sum = 0
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        optimizer.add_equality(np.ones((1, model.p)), np.array([0.0]))
        
        result = optimizer.solve()
        
        assert not result.solved
        assert result.weights is None
    
    def test_metadata(self, simple_diagonal_model):
        """Test that metadata is populated."""
        model = simple_diagonal_model
        optimizer = FactorOptimizer(model)
        
        optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
        result = optimizer.solve()
        
        assert "status" in result.metadata


class TestMinimumVariancePortfolio:
    """Tests for the convenience function."""
    
    def test_long_only(self, medium_model):
        """Test long only minimum variance."""
        result = minimum_variance_portfolio(medium_model, long_only=True)
        
        assert result.solved
        assert np.all(result.weights >= -1e-7)
        assert np.isclose(result.weights.sum(), 1.0)
    
    def test_with_max_weight(self, medium_model):
        """Test with max weight constraint."""
        result = minimum_variance_portfolio(
            medium_model, 
            long_only=True,
            max_weight=0.05
        )
        
        assert result.solved
        assert np.all(result.weights <= 0.05 + 1e-7)
    
    def test_short_allowed(self, medium_model):
        """Test allowing short positions."""
        result = minimum_variance_portfolio(medium_model, long_only=False)
        
        assert result.solved
        # May have some negative weights
        # Just check it's feasible


class TestOptimizationNumericalStability:
    """Tests for numerical edge cases."""
    
    def test_equal_risk_assets(self):
        """Test with assets of equal risk."""
        p, k = 10, 1
        
        B = np.ones((k, p))
        F = np.array([[0.04]])
        D = np.diag(np.full(p, 0.01))
        
        from factor_lab import FactorModelData
        model = FactorModelData(B=B, F=F, D=D)
        
        result = minimum_variance_portfolio(model, long_only=True)
        
        assert result.solved
        # With equal risk and exposures, should be equally weighted
        expected = np.full(p, 0.1)
        assert np.allclose(result.weights, expected, atol=1e-5)
    
    def test_large_problem(self, large_model):
        """Test optimization on large problem."""
        result = minimum_variance_portfolio(
            large_model,
            long_only=True,
            max_weight=0.01  # Diversified
        )
        
        assert result.solved
        assert np.all(result.weights >= -1e-6)
        assert np.all(result.weights <= 0.01 + 1e-6)
