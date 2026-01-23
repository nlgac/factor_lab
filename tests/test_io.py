"""
test_io.py - Tests for Model Serialization and Deserialization

Tests cover:
- NPZ format save/load
- JSON format save/load
- Transform preservation
- Error handling
- Edge cases
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from factor_lab import (
    FactorModelData,
    CovarianceTransform,
    TransformType,
)
from factor_lab.io import (
    save_model,
    load_model,
    ModelFormat,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple model without transforms."""
    B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    F = np.diag([0.04, 0.09])  # (2, 2)
    D = np.diag([0.01, 0.02, 0.03])  # (3, 3)
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def model_with_diagonal_transforms():
    """Model with diagonal covariance transforms."""
    B = np.random.randn(3, 10)
    F = np.diag([0.04, 0.09, 0.01])
    D = np.diag(np.random.uniform(0.001, 0.01, 10))
    
    # Add diagonal transforms
    factor_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(F)),
        transform_type=TransformType.DIAGONAL
    )
    idio_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(D)),
        transform_type=TransformType.DIAGONAL
    )
    
    return FactorModelData(
        B=B, F=F, D=D,
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )


@pytest.fixture
def model_with_dense_transforms():
    """Model with dense (Cholesky) covariance transforms."""
    B = np.random.randn(2, 5)
    
    # Create non-diagonal F
    F = np.array([[0.04, 0.01], [0.01, 0.09]])
    
    # D is diagonal
    D = np.diag([0.01, 0.02, 0.03, 0.01, 0.02])
    
    # Add transforms
    factor_transform = CovarianceTransform(
        matrix=np.linalg.cholesky(F),
        transform_type=TransformType.DENSE
    )
    idio_transform = CovarianceTransform(
        matrix=np.sqrt(np.diag(D)),
        transform_type=TransformType.DIAGONAL
    )
    
    return FactorModelData(
        B=B, F=F, D=D,
        factor_transform=factor_transform,
        idio_transform=idio_transform
    )


# =============================================================================
# NPZ Format Tests
# =============================================================================

class TestNPZFormat:
    """Tests for NPZ format save/load."""
    
    def test_save_and_load_simple(self, simple_model, tmp_path):
        """Test basic save/load without transforms."""
        file_path = tmp_path / "model.npz"
        
        save_model(simple_model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.k == simple_model.k
        assert loaded.p == simple_model.p
        np.testing.assert_array_almost_equal(loaded.B, simple_model.B)
        np.testing.assert_array_almost_equal(loaded.F, simple_model.F)
        np.testing.assert_array_almost_equal(loaded.D, simple_model.D)
        assert loaded.factor_transform is None
        assert loaded.idio_transform is None
    
    def test_save_and_load_with_diagonal_transforms(
        self, model_with_diagonal_transforms, tmp_path
    ):
        """Test save/load preserves diagonal transforms."""
        file_path = tmp_path / "model.npz"
        
        save_model(model_with_diagonal_transforms, file_path)
        loaded = load_model(file_path)
        
        # Check basic structure
        assert loaded.k == model_with_diagonal_transforms.k
        assert loaded.p == model_with_diagonal_transforms.p
        
        # Check transforms are preserved
        assert loaded.factor_transform is not None
        assert loaded.idio_transform is not None
        assert loaded.factor_transform.is_diagonal
        assert loaded.idio_transform.is_diagonal
        
        # Check transform matrices
        np.testing.assert_array_almost_equal(
            loaded.factor_transform.matrix,
            model_with_diagonal_transforms.factor_transform.matrix
        )
        np.testing.assert_array_almost_equal(
            loaded.idio_transform.matrix,
            model_with_diagonal_transforms.idio_transform.matrix
        )
    
    def test_save_and_load_with_dense_transforms(
        self, model_with_dense_transforms, tmp_path
    ):
        """Test save/load preserves dense (Cholesky) transforms."""
        file_path = tmp_path / "model.npz"
        
        save_model(model_with_dense_transforms, file_path)
        loaded = load_model(file_path)
        
        # Check transforms are preserved
        assert loaded.factor_transform is not None
        assert not loaded.factor_transform.is_diagonal
        
        # Check transform matrices
        np.testing.assert_array_almost_equal(
            loaded.factor_transform.matrix,
            model_with_dense_transforms.factor_transform.matrix
        )
    
    def test_explicit_npz_format(self, simple_model, tmp_path):
        """Test explicit NPZ format specification."""
        file_path = tmp_path / "model.npz"
        
        save_model(simple_model, file_path, format=ModelFormat.NPZ)
        loaded = load_model(file_path)
        
        assert loaded.k == simple_model.k
        assert loaded.p == simple_model.p


# =============================================================================
# JSON Format Tests
# =============================================================================

class TestJSONFormat:
    """Tests for JSON format save/load."""
    
    def test_save_and_load_simple(self, simple_model, tmp_path):
        """Test basic JSON save/load."""
        file_path = tmp_path / "model.json"
        
        save_model(simple_model, file_path, format=ModelFormat.JSON)
        loaded = load_model(file_path)
        
        assert loaded.k == simple_model.k
        assert loaded.p == simple_model.p
        np.testing.assert_array_almost_equal(loaded.B, simple_model.B)
        np.testing.assert_array_almost_equal(loaded.F, simple_model.F)
        np.testing.assert_array_almost_equal(loaded.D, simple_model.D)
    
    def test_transforms_not_preserved_in_json(
        self, model_with_diagonal_transforms, tmp_path
    ):
        """JSON format does not preserve transforms."""
        file_path = tmp_path / "model.json"
        
        save_model(model_with_diagonal_transforms, file_path, format=ModelFormat.JSON)
        loaded = load_model(file_path)
        
        # Transforms should be None after JSON round-trip
        assert loaded.factor_transform is None
        assert loaded.idio_transform is None
        
        # But matrices should be preserved
        np.testing.assert_array_almost_equal(
            loaded.B, model_with_diagonal_transforms.B
        )
        np.testing.assert_array_almost_equal(
            loaded.F, model_with_diagonal_transforms.F
        )
        np.testing.assert_array_almost_equal(
            loaded.D, model_with_diagonal_transforms.D
        )
    
    def test_json_is_human_readable(self, simple_model, tmp_path):
        """JSON output should be human-readable."""
        file_path = tmp_path / "model.json"
        
        save_model(simple_model, file_path, format=ModelFormat.JSON)
        
        # Verify it's valid JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert "B" in data
        assert "F" in data
        assert "D" in data
        assert isinstance(data["B"], list)


# =============================================================================
# Format Detection Tests
# =============================================================================

class TestFormatDetection:
    """Tests for automatic format detection."""
    
    def test_load_detects_npz(self, simple_model, tmp_path):
        """load_model should detect NPZ from extension."""
        file_path = tmp_path / "model.npz"
        save_model(simple_model, file_path)
        
        # Should load successfully without specifying format
        loaded = load_model(file_path)
        assert loaded.k == simple_model.k
    
    def test_load_detects_json(self, simple_model, tmp_path):
        """load_model should detect JSON from extension."""
        file_path = tmp_path / "model.json"
        save_model(simple_model, file_path, format=ModelFormat.JSON)
        
        # Should load successfully without specifying format
        loaded = load_model(file_path)
        assert loaded.k == simple_model.k
    
    def test_unknown_extension_raises(self, simple_model, tmp_path):
        """Unknown file extension should raise ValueError."""
        file_path = tmp_path / "model.xyz"
        
        # Create a dummy file
        file_path.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unknown model format"):
            load_model(file_path)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_load_nonexistent_file_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model("nonexistent_file.npz")
    
    def test_invalid_format_raises(self, simple_model, tmp_path):
        """Invalid format specification should raise ValueError."""
        file_path = tmp_path / "model.npz"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            save_model(simple_model, file_path, format="invalid")


# =============================================================================
# Path Handling Tests
# =============================================================================

class TestPathHandling:
    """Tests for path handling."""
    
    def test_string_path(self, simple_model, tmp_path):
        """Should accept string paths."""
        file_path = str(tmp_path / "model.npz")
        
        save_model(simple_model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.k == simple_model.k
    
    def test_path_object(self, simple_model, tmp_path):
        """Should accept Path objects."""
        file_path = tmp_path / "model.npz"
        
        save_model(simple_model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.k == simple_model.k


# =============================================================================
# Round-Trip Tests
# =============================================================================

class TestRoundTrip:
    """Tests for round-trip consistency."""
    
    def test_npz_round_trip_exact(self, simple_model, tmp_path):
        """NPZ round-trip should be exact (within float precision)."""
        file_path = tmp_path / "model.npz"
        
        save_model(simple_model, file_path)
        loaded = load_model(file_path)
        
        # Should be exact (or very close)
        np.testing.assert_allclose(loaded.B, simple_model.B, rtol=1e-10)
        np.testing.assert_allclose(loaded.F, simple_model.F, rtol=1e-10)
        np.testing.assert_allclose(loaded.D, simple_model.D, rtol=1e-10)
    
    def test_multiple_saves_same_result(self, simple_model, tmp_path):
        """Multiple saves should produce identical results."""
        file_path1 = tmp_path / "model1.npz"
        file_path2 = tmp_path / "model2.npz"
        
        save_model(simple_model, file_path1)
        save_model(simple_model, file_path2)
        
        loaded1 = load_model(file_path1)
        loaded2 = load_model(file_path2)
        
        np.testing.assert_array_equal(loaded1.B, loaded2.B)
        np.testing.assert_array_equal(loaded1.F, loaded2.F)
        np.testing.assert_array_equal(loaded1.D, loaded2.D)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_factor_single_asset(self, tmp_path):
        """Minimal model: 1 factor, 1 asset."""
        B = np.array([[1.5]])  # (1, 1)
        F = np.array([[0.04]])  # (1, 1)
        D = np.array([[0.01]])  # (1, 1)
        model = FactorModelData(B=B, F=F, D=D)
        
        file_path = tmp_path / "tiny_model.npz"
        save_model(model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.k == 1
        assert loaded.p == 1
        np.testing.assert_array_almost_equal(loaded.B, B)
    
    def test_large_model(self, tmp_path):
        """Large model: many factors and assets."""
        k, p = 50, 500
        B = np.random.randn(k, p)
        F = np.diag(np.random.uniform(0.001, 0.1, k))
        D = np.diag(np.random.uniform(0.001, 0.01, p))
        model = FactorModelData(B=B, F=F, D=D)
        
        file_path = tmp_path / "large_model.npz"
        save_model(model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.k == k
        assert loaded.p == p
    
    def test_mixed_transform_types(self, tmp_path):
        """Model with diagonal factor transform and dense idio transform."""
        B = np.random.randn(2, 3)
        F = np.diag([0.04, 0.09])
        
        # Create non-diagonal D
        D = np.array([[0.01, 0.001, 0.0], 
                      [0.001, 0.02, 0.001],
                      [0.0, 0.001, 0.03]])
        
        factor_transform = CovarianceTransform(
            matrix=np.sqrt(np.diag(F)),
            transform_type=TransformType.DIAGONAL
        )
        idio_transform = CovarianceTransform(
            matrix=np.linalg.cholesky(D),
            transform_type=TransformType.DENSE
        )
        
        model = FactorModelData(
            B=B, F=F, D=D,
            factor_transform=factor_transform,
            idio_transform=idio_transform
        )
        
        file_path = tmp_path / "mixed_transforms.npz"
        save_model(model, file_path)
        loaded = load_model(file_path)
        
        assert loaded.factor_transform.is_diagonal
        assert not loaded.idio_transform.is_diagonal
