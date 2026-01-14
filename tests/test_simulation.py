import pytest
import numpy as np
import sys
import os

# --- PATH FIX ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from factor_lab import (
    FactorModelData, ReturnsSimulator, DistributionFactory
)

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def diagonal_model():
    """Creates a simple model with Diagonal F and D."""
    p, k = 10, 2
    B = np.ones((k, p))
    F = np.diag([0.04, 0.04])
    D = np.diag(np.full(p, 0.01))
    return FactorModelData(B, F, D)

@pytest.fixture
def factory():
    return DistributionFactory()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_hybrid_path_selection(diagonal_model):
    """Verify simulator correctly identifies diagonal matrices."""
    sim = ReturnsSimulator(diagonal_model)
    assert sim.F_diag is True, "Should detect Diagonal F"
    assert sim.D_diag is True, "Should detect Diagonal D"

    # Make F dense
    diagonal_model.F[0, 1] = 0.01
    diagonal_model.F[1, 0] = 0.01
    sim_dense = ReturnsSimulator(diagonal_model)
    assert sim_dense.F_diag is False, "Should detect Dense F"

def test_standardization_logic(diagonal_model, factory):
    """
    CRITICAL: Simulator must standardize raw samples (Mean=0, Std=1)
    BEFORE applying model volatility.
    """
    # Generator centered at 1000.0
    bad_gen = lambda n: np.random.normal(loc=1000.0, scale=1.0, size=n)
    
    sim = ReturnsSimulator(diagonal_model, seed=42)
    f_gens = [bad_gen, bad_gen]
    i_gens = [bad_gen] * diagonal_model.p
    
    # We don't care about global seed here because we check magnitude
    results = sim.simulate(1000, f_gens, i_gens)
    
    # Mean should be near 0, NOT 1000
    mu = results['security_returns'].mean()
    assert np.abs(mu) < 0.1, f"Standardization failed. Mean is {mu}, expected ~0"

def test_hybrid_equivalence(diagonal_model, factory):
    """
    Ensure 'Path A' (Diagonal) yields identical results to 'Path B' (Cholesky).
    """
    n_samples = 100
    seed_val = 999
    
    gen = factory.create_generator('normal', mean=0, std=1)
    f_gens = [gen] * diagonal_model.k
    i_gens = [gen] * diagonal_model.p

    # --- Run 1: Fast (Diagonal) ---
    sim_fast = ReturnsSimulator(diagonal_model)
    
    # FORCE GLOBAL SEED RESET
    # The factory generators use the global numpy state.
    np.random.seed(seed_val) 
    res_fast = sim_fast.simulate(n_samples, f_gens, i_gens)

    # --- Run 2: Slow (Forced Dense) ---
    sim_slow = ReturnsSimulator(diagonal_model)
    sim_slow.F_diag = False 
    sim_slow.D_diag = False
    sim_slow.F_tx = np.linalg.cholesky(diagonal_model.F)
    sim_slow.D_tx = np.linalg.cholesky(diagonal_model.D)
    
    # FORCE GLOBAL SEED RESET to same value
    np.random.seed(seed_val)
    res_slow = sim_slow.simulate(n_samples, f_gens, i_gens)

    diff = np.abs(res_fast['security_returns'] - res_slow['security_returns'])
    assert np.all(diff < 1e-10), "Diagonal and Cholesky paths diverged!"