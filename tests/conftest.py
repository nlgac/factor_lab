import sys
import os
import pytest
import numpy as np

# Automatically add the project root to sys.path for all tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from factor_lab import FactorModelData

@pytest.fixture
def simple_model():
    """Returns a simple 10-asset, 2-factor model for math checks."""
    p, k = 10, 2
    # Loadings: Block structure to make factors distinct
    B = np.zeros((k, p))
    B[0, :5] = 1.0   # Factor 1 affects first 5
    B[1, 5:] = 1.0   # Factor 2 affects last 5
    
    F = np.diag([0.04, 0.09]) # Vols: 0.20, 0.30
    D = np.diag(np.full(p, 0.01)) # Idio Vol: 0.10
    return FactorModelData(B, F, D)