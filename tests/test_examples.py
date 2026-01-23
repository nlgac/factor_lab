"""
Test Suite for Examples Package - FIXED V2
==========================================
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from examples import run_example
except ImportError as e:
    raise ImportError(
        f"Failed to import examples package from {project_root}.\n"
        f"Error: {e}\n"
        "Ensure your directory structure is: project_root/examples/"
    )

from factor_lab import FactorModelData, OptimizationResult

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_valid_factor_model(model):
    """Helper to validate a FactorModelData object."""
    assert isinstance(model, FactorModelData)
    assert model.k > 0
    assert model.p > 0
    assert model.B.shape == (model.k, model.p)
    assert model.F.shape == (model.k, model.k)
    assert model.D.shape == (model.p, model.p)
    assert not np.any(np.isnan(model.B))

# =============================================================================
# TESTS
# =============================================================================

class TestRunExample:
    """Test that examples run end-to-end via the wrapper."""
    
    def test_run_example_extract_factors(self):
        result = run_example("extract_factors")
        assert result is not None
        assert_valid_factor_model(result)

    def test_run_example_simulate_distributions(self):
        result = run_example("simulate_distributions")
        assert isinstance(result, dict)
        
        # FIX: The example returns nested results keyed by distribution name
        # Structure: {'normal': {'security_returns': array(...), ...}}
        assert "normal" in result
        assert "security_returns" in result["normal"]

    def test_run_example_optimize_portfolio(self):
        result = run_example("optimize_portfolio")
        assert isinstance(result, dict)
        assert "basic" in result
        assert isinstance(result["basic"], OptimizationResult)

    def test_run_example_generate_synthetic(self):
        result = run_example("generate_synthetic")
        # Returns (models, ensemble)
        assert isinstance(result, tuple) 
        assert len(result) == 2

class TestExampleEdgeCases:
    """
    Test examples with custom parameters to verify they accept **kwargs.
    """
    
    def test_extract_factors_small_data(self):
        # Pass custom args to speed up test
        result = run_example("extract_factors", T=50, p=10)
        assert result is not None
        assert result.p == 10

    def test_optimize_portfolio_edge_weights(self):
        # If the example supports configuring constraints via args
        result = run_example("optimize_portfolio")
        assert result is not None

class TestExampleIntegration:
    """Test that outputs from one example can feed another."""
    
    def test_extract_then_optimize(self):
        # 1. Get model from extraction
        model = run_example("extract_factors", T=100, p=20)
        
        # 2. Optimize it directly
        from factor_lab import minimum_variance_portfolio
        result = minimum_variance_portfolio(model)
        
        assert result.solved
        assert len(result.weights) == 20

if __name__ == "__main__":
    pytest.main([__file__, '-v'])