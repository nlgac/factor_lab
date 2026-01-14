import pytest
import numpy as np
from factor_lab import DataSampler, DistributionFactory

@pytest.fixture
def factory():
    return DistributionFactory()

def test_broadcasting_configuration(factory):
    """Test that single generators are correctly broadcast to all factors/assets."""
    ds = DataSampler(p=10, k=2)
    
    gen_beta = factory.create_generator('constant', c=1.0)
    gen_vol = factory.create_generator('constant', c=0.1)
    
    ds.configure(beta=gen_beta, f_vol=gen_vol, d_vol=gen_vol)
    
    model = ds.generate()
    
    # Check broadcasting
    assert np.allclose(model.B, 1.0), "Beta broadcasting failed"
    assert np.allclose(np.diag(model.F), 0.1**2), "Factor Vol broadcasting failed"
    assert np.allclose(np.diag(model.D), 0.1**2), "Idio Vol broadcasting failed"

def test_explicit_list_configuration(factory):
    """Test that explicit lists assign specific generators to specific slots."""
    ds = DataSampler(p=4, k=2)
    
    # Factor 1 -> 1.0, Factor 2 -> 2.0
    beta_gens = [
        factory.create_generator('constant', c=1.0),
        factory.create_generator('constant', c=2.0)
    ]
    # Factor Vol 1 -> 0.1, Factor Vol 2 -> 0.2
    f_vol_gens = [
        factory.create_generator('constant', c=0.1),
        factory.create_generator('constant', c=0.2)
    ]
    # Idio Vol -> Constant 0.05 (Broadcast)
    d_vol_gen = factory.create_generator('constant', c=0.05)
    
    ds.configure(beta=beta_gens, f_vol=f_vol_gens, d_vol=d_vol_gen)
    model = ds.generate()
    
    # Check Factor 1 (Row 0)
    assert np.allclose(model.B[0, :], 1.0)
    assert np.isclose(model.F[0, 0], 0.1**2)
    
    # Check Factor 2 (Row 1)
    assert np.allclose(model.B[1, :], 2.0)
    assert np.isclose(model.F[1, 1], 0.2**2)

def test_validation_errors(factory):
    """Ensure incorrect list lengths raise ValueError."""
    ds = DataSampler(p=10, k=2)
    gen = factory.create_generator('constant', c=1.0)
    
    # Pass 3 generators for 2 factors
    with pytest.raises(ValueError, match="Beta list length 3 != required 2"):
        ds.configure(beta=[gen, gen, gen], f_vol=gen, d_vol=gen)