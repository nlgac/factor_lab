import pytest
import numpy as np
import sys
from loguru import logger
from factor_lab import (
    ReturnsSimulator, FactorOptimizer, ScenarioBuilder, 
    FactorModelData, DistributionFactory
)

@pytest.fixture
def simple_model():
    p, k = 20, 2
    B = np.random.normal(0, 1, (k, p))
    F = np.diag([0.04, 0.04])
    D = np.diag(np.full(p, 0.01))
    return FactorModelData(B, F, D)

def test_covariance_validation(simple_model):
    """
    Verify that the validate_covariance method correctly identifies 
    discrepancies between empirical and theoretical covariance.
    """
    # 1. Simulate a large sample to get convergence
    factory = DistributionFactory()
    gen = factory.create_generator('normal', mean=0, std=1)
    
    sim = ReturnsSimulator(simple_model, seed=42)
    # Use 5000 samples to ensure statistical convergence
    res = sim.simulate(n=5000, 
                       f_gens=[gen]*simple_model.k, 
                       i_gens=[gen]*simple_model.p)
    
    # 2. Validate
    validation = sim.validate_covariance(res['security_returns'])
    
    # Error should be small (e.g. < 0.10 Frobenius norm for this size)
    assert validation['frobenius_error'] < 0.1, \
        f"Validation error too high: {validation['frobenius_error']}"

    # 3. Test mismatch detection
    # Pass in garbage data (wrong shape)
    with pytest.raises(ValueError, match="Returns/Asset count mismatch"):
        # Transpose returns so shape is (Assets, Time) instead of (Time, Assets)
        sim.validate_covariance(res['security_returns'].T)


def test_optimization_constraints(simple_model):
    """
    Verify that Long Only and Box constraints are strictly enforced.
    """
    opt = FactorOptimizer(simple_model)
    builder = ScenarioBuilder(simple_model.p)
    
    # Create scenario: Long Only + Box (Max 20% per asset)
    scen = builder.create("Constrained")
    scen = builder.add_fully_invested(scen)
    scen = builder.add_long_only(scen)
    scen = builder.add_box_constraints(scen, low=0.0, high=0.20)
    
    # Apply constraints
    for A, b in scen.equality_constraints: opt.add_eq(A, b)
    for A, b in scen.inequality_constraints: opt.add_ineq(A, b)
    
    res = opt.solve()
    
    assert res.solved is True
    w = res.weights
    
    # Check 1: Sum to 1
    assert np.isclose(w.sum(), 1.0)
    
    # Check 2: Long Only (Allow tiny numerical error -1e-7)
    assert np.all(w >= -1e-7), f"Long Only failed: Min weight {w.min()}"
    
    # Check 3: Max Weight 0.20
    assert np.all(w <= 0.20 + 1e-7), f"Box Constraint failed: Max weight {w.max()}"

def test_logging_safety(simple_model, capsys):
    """
    Ensure that enabling debug logging does not crash the simulation
    and correctly handles array slicing.
    """
    factory = DistributionFactory()
    gen = factory.create_generator('normal', mean=0, std=1)
    sim = ReturnsSimulator(simple_model)
    
    # Check for crashes during logging
    try:
        sim.simulate(n=10, 
                     f_gens=[gen]*simple_model.k, 
                     i_gens=[gen]*simple_model.p,
                     debug_intermediate_quantity_length=5) # Test the slicing logic
    except Exception as e:
        pytest.fail(f"Simulation crashed with logging enabled: {e}")