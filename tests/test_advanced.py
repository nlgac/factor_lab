import pytest
import numpy as np
import sys
from factor_lab import (
    ReturnsSimulator, FactorOptimizer, ScenarioBuilder, 
    FactorModelData, DistributionFactory, CovarianceValidator
)

@pytest.fixture
def simple_model():
    p, k = 20, 2
    B = np.random.normal(0, 1, (k, p))
    F = np.diag([0.04, 0.04])
    D = np.diag(np.full(p, 0.01))
    return FactorModelData(B, F, D)

def test_covariance_validation(simple_model, rng, factory):
    """
    Verify that CovarianceValidator correctly identifies 
    discrepancies between empirical and theoretical covariance.
    """
    # 1. Simulate a large sample to get convergence
    normal_sampler = factory.create('normal', mean=0, std=1)
    
    sim = ReturnsSimulator(simple_model, rng=rng)
    # Use 5000 samples to ensure statistical convergence
    results = sim.simulate(
        n_periods=5000, 
        factor_samplers=[normal_sampler] * simple_model.k, 
        idio_samplers=[normal_sampler] * simple_model.p
    )
    
    # 2. Validate
    validator = CovarianceValidator(simple_model)
    validation = validator.compare(results['security_returns'])
    
    # Error should be small (e.g. < 0.10 Frobenius norm for this size)
    assert validation.frobenius_error < 0.1, \
        f"Validation error too high: {validation.frobenius_error}"

    # 3. Test mismatch detection
    # Pass in garbage data (wrong shape)
    with pytest.raises(ValueError, match="assets"):
        # Transpose returns so shape is (Assets, Time) instead of (Time, Assets)
        validator.compare(results['security_returns'].T)


def test_optimization_constraints(simple_model):
    """
    Verify that Long Only and Box constraints are strictly enforced.
    """
    opt = FactorOptimizer(simple_model)
    builder = ScenarioBuilder(simple_model.p)
    
    # Create scenario: Long Only + Box (Max 20% per asset)
    scenario = (builder
        .create("Constrained")
        .add_fully_invested()
        .add_long_only()
        .add_box_constraints(low=0.0, high=0.20)
        .build())
    
    # Apply constraints
    opt.apply_scenario(scenario)
    
    res = opt.solve()
    
    assert res.solved is True
    w = res.weights
    
    # Check 1: Sum to 1
    assert np.isclose(w.sum(), 1.0)
    
    # Check 2: Long Only (Allow tiny numerical error -1e-7)
    assert np.all(w >= -1e-7), f"Long Only failed: Min weight {w.min()}"
    
    # Check 3: Max Weight 0.20
    assert np.all(w <= 0.20 + 1e-7), f"Box Constraint failed: Max weight {w.max()}"

def test_logging_safety(simple_model, rng, factory, capsys):
    """
    Ensure that enabling debug logging does not crash the simulation
    and correctly handles array slicing.
    """
    normal_sampler = factory.create('normal', mean=0, std=1)
    sim = ReturnsSimulator(simple_model, rng=rng)
    
    # Check for crashes during logging
    try:
        results = sim.simulate(
            n_periods=10, 
            factor_samplers=[normal_sampler] * simple_model.k, 
            idio_samplers=[normal_sampler] * simple_model.p,
            sample_log_rows=5  # Test the slicing logic
        )
        # If we got results with the optional log, that's success
        assert 'security_returns' in results
    except Exception as e:
        pytest.fail(f"Simulation crashed with logging enabled: {e}")
