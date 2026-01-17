"""
test_types.py - Tests for Core Data Structures

Tests cover:
- FactorModelData validation and properties
- CovarianceTransform application
- OptimizationResult validation
- Scenario constraint counting
"""

import pytest
import numpy as np

from factor_lab import (
    FactorModelData,
    OptimizationResult,
    Scenario,
    CovarianceTransform,
    TransformType,
    CovarianceValidationResult,
)


class TestFactorModelData:
    """Tests for FactorModelData dataclass."""
    
    def test_basic_creation(self, simple_diagonal_model):
        """Test that a valid model can be created."""
        model = simple_diagonal_model
        assert model.k == 2
        assert model.p == 10
    
    def test_dimension_properties(self):
        """Test k and p property computation."""
        B = np.random.randn(3, 50)
        F = np.eye(3)
        D = np.eye(50)
        
        model = FactorModelData(B=B, F=F, D=D)
        assert model.k == 3
        assert model.p == 50
    
    def test_validation_on_construction(self):
        """Test that validation runs automatically on construction."""
        B = np.random.randn(2, 10)
        F = np.eye(3)  # Wrong size!
        D = np.eye(10)
        
        with pytest.raises(ValueError, match="F shape mismatch"):
            FactorModelData(B=B, F=F, D=D)
    
    def test_validation_b_shape(self):
        """Test validation catches non-2D B matrix."""
        B = np.random.randn(10)  # 1D, should be 2D
        F = np.eye(1)
        D = np.eye(10)
        
        with pytest.raises(ValueError, match="B must be 2D"):
            FactorModelData(B=B, F=F, D=D)
    
    def test_validation_d_shape(self):
        """Test validation catches D dimension mismatch."""
        B = np.random.randn(2, 10)
        F = np.eye(2)
        D = np.eye(5)  # Wrong size!
        
        with pytest.raises(ValueError, match="D shape mismatch"):
            FactorModelData(B=B, F=F, D=D)
    
    def test_validation_positive_dimensions(self):
        """Test validation catches zero dimensions."""
        B = np.empty((0, 10))
        F = np.empty((0, 0))
        D = np.eye(10)
        
        with pytest.raises(ValueError, match="positive dimensions"):
            FactorModelData(B=B, F=F, D=D)
    
    def test_implied_covariance(self, simple_diagonal_model):
        """Test implied covariance computation."""
        model = simple_diagonal_model
        cov = model.implied_covariance()
        
        # Should be (p, p)
        assert cov.shape == (model.p, model.p)
        
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)
    
    def test_transform_validation_diagonal(self):
        """Test validation of diagonal transform dimensions."""
        B = np.random.randn(2, 10)
        F = np.eye(2)
        D = np.eye(10)
        
        # Wrong size diagonal transform
        bad_transform = CovarianceTransform(
            matrix=np.ones(5),  # Should be 2 for factors
            transform_type=TransformType.DIAGONAL
        )
        
        with pytest.raises(ValueError, match="factor_transform diagonal"):
            FactorModelData(B=B, F=F, D=D, factor_transform=bad_transform)
    
    def test_transform_validation_dense(self):
        """Test validation of dense transform dimensions."""
        B = np.random.randn(2, 10)
        F = np.eye(2)
        D = np.eye(10)
        
        # Wrong size dense transform
        bad_transform = CovarianceTransform(
            matrix=np.eye(5),  # Should be (2, 2) for factors
            transform_type=TransformType.DENSE
        )
        
        with pytest.raises(ValueError, match="factor_transform dense"):
            FactorModelData(B=B, F=F, D=D, factor_transform=bad_transform)


class TestCovarianceTransform:
    """Tests for CovarianceTransform operations."""
    
    def test_diagonal_is_diagonal(self):
        """Test is_diagonal property for diagonal transforms."""
        transform = CovarianceTransform(
            matrix=np.array([0.1, 0.2, 0.3]),
            transform_type=TransformType.DIAGONAL
        )
        assert transform.is_diagonal is True
    
    def test_dense_is_not_diagonal(self):
        """Test is_diagonal property for dense transforms."""
        transform = CovarianceTransform(
            matrix=np.eye(3),
            transform_type=TransformType.DENSE
        )
        assert transform.is_diagonal is False
    
    def test_diagonal_apply(self):
        """Test diagonal transform application (element-wise scaling)."""
        stds = np.array([0.1, 0.2])
        transform = CovarianceTransform(
            matrix=stds,
            transform_type=TransformType.DIAGONAL
        )
        
        z = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, 0.5]
        ])
        
        result = transform.apply(z)
        
        # Each column should be scaled by corresponding std
        expected = z * stds
        assert np.allclose(result, expected)
    
    def test_dense_apply(self):
        """Test dense transform application (matrix multiplication)."""
        # Create a simple lower triangular matrix
        L = np.array([
            [1.0, 0.0],
            [0.5, 1.0]
        ])
        transform = CovarianceTransform(
            matrix=L,
            transform_type=TransformType.DENSE
        )
        
        z = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        result = transform.apply(z)
        
        # result = z @ L.T
        expected = z @ L.T
        assert np.allclose(result, expected)
    
    def test_frozen_dataclass(self):
        """Test that CovarianceTransform is immutable."""
        transform = CovarianceTransform(
            matrix=np.ones(3),
            transform_type=TransformType.DIAGONAL
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            transform.matrix = np.zeros(3)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_solved_with_weights(self):
        """Test valid solved result."""
        result = OptimizationResult(
            weights=np.array([0.5, 0.3, 0.2]),
            risk=0.15,
            objective=0.0225,
            solved=True
        )
        assert result.solved is True
        assert result.weights is not None
    
    def test_unsolved_without_weights(self):
        """Test valid unsolved result."""
        result = OptimizationResult(
            weights=None,
            risk=0.0,
            objective=0.0,
            solved=False,
            metadata={"status": "infeasible"}
        )
        assert result.solved is False
        assert result.weights is None
    
    def test_validation_solved_requires_weights(self):
        """Test that solved=True requires weights."""
        with pytest.raises(ValueError, match="solved=True but weights is None"):
            OptimizationResult(
                weights=None,
                risk=0.15,
                objective=0.0225,
                solved=True
            )
    
    def test_frozen_dataclass(self):
        """Test that OptimizationResult is immutable."""
        result = OptimizationResult(
            weights=np.array([0.5, 0.5]),
            risk=0.1,
            objective=0.01,
            solved=True
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            result.risk = 0.2


class TestScenario:
    """Tests for Scenario constraint container."""
    
    def test_empty_scenario(self):
        """Test scenario with no constraints."""
        scenario = Scenario(name="Empty")
        assert scenario.n_equality() == 0
        assert scenario.n_inequality() == 0
    
    def test_equality_count(self):
        """Test equality constraint counting."""
        scenario = Scenario(name="Test")
        
        # Add two constraints (1 row each)
        scenario.equality_constraints.append((np.ones((1, 10)), np.array([1.0])))
        scenario.equality_constraints.append((np.ones((1, 10)), np.array([0.0])))
        
        assert scenario.n_equality() == 2
    
    def test_inequality_count_multi_row(self):
        """Test inequality constraint counting with multi-row constraints."""
        scenario = Scenario(name="Test")
        
        # Add one constraint with 3 rows
        scenario.inequality_constraints.append((np.eye(3, 10), np.zeros(3)))
        
        assert scenario.n_inequality() == 3
    
    def test_repr(self):
        """Test string representation."""
        scenario = Scenario(name="Test", description="A test scenario")
        scenario.equality_constraints.append((np.ones((1, 10)), np.array([1.0])))
        scenario.inequality_constraints.append((np.eye(10), np.zeros(10)))
        
        repr_str = repr(scenario)
        assert "Test" in repr_str
        assert "eq=1" in repr_str
        assert "ineq=10" in repr_str


class TestCovarianceValidationResult:
    """Tests for CovarianceValidationResult dataclass."""
    
    def test_creation(self):
        """Test basic creation of validation result."""
        model_cov = np.eye(3)
        emp_cov = np.eye(3) * 1.1
        
        result = CovarianceValidationResult(
            frobenius_error=0.173,
            mean_absolute_error=0.033,
            max_absolute_error=0.1,
            explained_variance_ratio=0.85,
            model_covariance=model_cov,
            empirical_covariance=emp_cov
        )
        
        assert result.frobenius_error == pytest.approx(0.173)
        assert result.explained_variance_ratio == pytest.approx(0.85)
    
    def test_frozen(self):
        """Test that result is immutable."""
        result = CovarianceValidationResult(
            frobenius_error=0.1,
            mean_absolute_error=0.01,
            max_absolute_error=0.05,
            explained_variance_ratio=0.9,
            model_covariance=np.eye(3),
            empirical_covariance=np.eye(3)
        )
        
        with pytest.raises(Exception):
            result.frobenius_error = 0.5
