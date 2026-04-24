"""
model_io.py - File I/O for Factor Models

Save and load factor models to/from disk.
"""

import numpy as np
from .factor_types import FactorModelData


def save_model(model: FactorModelData, filename: str) -> None:
    """
    Save factor model to NPZ file.

    Saves B, F, and D matrices.  Optional transforms are not included;
    use save_model_full() for those.

    Example
    -------
    >>> save_model(model, 'my_model.npz')
    """
    np.savez(filename, B=model.B, F=model.F, D=model.D)


def load_model(filename: str) -> FactorModelData:
    """
    Load factor model from NPZ file saved by save_model().

    Example
    -------
    >>> model = load_model('my_model.npz')
    >>> print(model.k, model.p)
    """
    data = np.load(filename)
    return FactorModelData(B=data['B'], F=data['F'], D=data['D'])


def save_model_full(model: FactorModelData, filename: str) -> None:
    """
    Save factor model including optional transforms to NPZ file.

    Example
    -------
    >>> save_model_full(model, 'full_model.npz')
    """
    save_dict = {'B': model.B, 'F': model.F, 'D': model.D}
    if model.factor_transform is not None:
        save_dict['factor_transform'] = model.factor_transform
    if model.idio_transform is not None:
        save_dict['idio_transform'] = model.idio_transform
    np.savez(filename, **save_dict)


def load_model_full(filename: str) -> FactorModelData:
    """
    Load factor model including optional transforms from NPZ file.

    Example
    -------
    >>> model = load_model_full('full_model.npz')
    >>> if model.factor_transform is not None:
    ...     print("Has factor transform")
    """
    data = np.load(filename)
    return FactorModelData(
        B=data['B'],
        F=data['F'],
        D=data['D'],
        factor_transform=data['factor_transform'] if 'factor_transform' in data else None,
        idio_transform=data['idio_transform'] if 'idio_transform' in data else None,
    )
