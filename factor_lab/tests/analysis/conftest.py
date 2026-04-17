"""
conftest.py for tests/analysis/

Shared fixtures and helpers. pytest makes these available to all tests
in this directory automatically — no import needed in test files.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package root is on the path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factor_lab import FactorModelData
from factor_lab.flexible_simulator import ReturnsSimulator as FlexibleReturnsSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_model():
    """Small 2-factor, 30-asset model for fast tests."""
    rng = np.random.default_rng(42)
    k, p = 2, 30
    B = rng.standard_normal((k, p))
    F = np.diag([0.1, 0.05])
    D = np.diag(np.full(p, 0.01))
    return FactorModelData(B=B, F=F, D=D)


@pytest.fixture
def simulated_returns(simple_model):
    """100-period returns from simple_model."""
    rng = np.random.default_rng(0)
    return _run_sim(simple_model, rng, n_periods=100)


# ---------------------------------------------------------------------------
# Helper (importable by tests that need it directly)
# ---------------------------------------------------------------------------

def make_simulator(model, rng_inst):
    """
    Return an adapter whose .simulate(n_periods) mirrors the legacy API
    but uses FlexibleReturnsSimulator internally.

    Usage in tests:
        sim = make_simulator(model, np.random.default_rng(42))
        results = sim.simulate(100)
    """
    sim = FlexibleReturnsSimulator(rng=rng_inst)
    normal = lambda n: rng_inst.standard_normal(n)

    class _Adapter:
        def simulate(self, n_periods):
            return sim.simulate(
                model, n_periods,
                factor_return_samplers=normal,
                idio_return_sampler=normal,
            )
    return _Adapter()


def _run_sim(model, rng_inst, n_periods):
    return make_simulator(model, rng_inst).simulate(n_periods)
