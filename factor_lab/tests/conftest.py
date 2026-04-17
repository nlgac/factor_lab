"""Pytest configuration and shared fixtures."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from factor_lab import FactorModelData
from factor_lab.flexible_simulator import ReturnsSimulator as FlexibleReturnsSimulator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_model():
    """Small 2-factor, 30-asset model for fast tests."""
    np.random.seed(42)
    k, p = 2, 30
    B = np.random.randn(k, p)
    F = np.diag([0.1, 0.05])
    D = np.diag(np.full(p, 0.01))
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def simulated_returns(simple_model, rng):
    """100-period returns from simple_model."""
    sim = FlexibleReturnsSimulator(rng=rng)
    normal = lambda n: rng.standard_normal(n)
    return sim.simulate(simple_model, n_periods=100,
                        factor_return_samplers=normal,
                        idio_return_sampler=normal)


def make_simulator(model, rng_inst):
    """Create a FlexibleReturnsSimulator and helper that wraps it for tests
    that still use the legacy call pattern (model, n_periods)."""
    sim = FlexibleReturnsSimulator(rng=rng_inst)
    normal = lambda n: rng_inst.standard_normal(n)

    class _Adapter:
        def simulate(self, n_periods):
            return sim.simulate(model, n_periods,
                                factor_return_samplers=normal,
                                idio_return_sampler=normal)
    return _Adapter()
