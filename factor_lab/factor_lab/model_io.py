"""
io.py - File I/O for Factor Models

Save and load factor models to/from disk.
"""

import numpy as np
try:
    from .factor_types import FactorModelData
except ImportError:
    from factor_types import FactorModelData


def save_model(model: FactorModelData, filename: str) -> None:
    """
    Save factor model to NPZ file.
    
    Saves the factor model's B, F, and D matrices to a compressed
    NumPy archive file.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model to save
    filename : str
        Path to save file (e.g., 'model.npz')
        
    Examples
    --------
    >>> model = FactorModelData(B, F, D)
    >>> save_model(model, 'my_model.npz')
    
    Notes
    -----
    The NPZ file will contain three arrays:
    - 'B': Factor loadings (k, p)
    - 'F': Factor covariance (k, k)
    - 'D': Idiosyncratic covariance (p, p)
    
    Optional transforms are not saved in the basic version.
    For full serialization including transforms, use save_model_full().
    """
    np.savez(filename, B=model.B, F=model.F, D=model.D)


def load_model(filename: str) -> FactorModelData:
    """
    Load factor model from NPZ file.
    
    Loads a factor model previously saved with save_model().
    
    Parameters
    ----------
    filename : str
        Path to NPZ file (e.g., 'model.npz')
        
    Returns
    -------
    model : FactorModelData
        Loaded factor model
        
    Examples
    --------
    >>> model = load_model('my_model.npz')
    >>> print(model.k, model.p)
    
    Notes
    -----
    The NPZ file must contain arrays named 'B', 'F', and 'D'.
    If transforms were saved separately, use load_model_full().
    """
    data = np.load(filename)
    return FactorModelData(
        B=data['B'],
        F=data['F'],
        D=data['D']
    )


def save_model_full(model: FactorModelData, filename: str) -> None:
    """
    Save factor model with optional transforms to NPZ file.
    
    Saves all components of the factor model including optional transforms.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model to save
    filename : str
        Path to save file
        
    Examples
    --------
    >>> model = FactorModelData(B, F, D, factor_transform=T1, idio_transform=T2)
    >>> save_model_full(model, 'full_model.npz')
    """
    save_dict = {
        'B': model.B,
        'F': model.F,
        'D': model.D
    }
    
    # Add transforms if present
    if model.factor_transform is not None:
        save_dict['factor_transform'] = model.factor_transform
    if model.idio_transform is not None:
        save_dict['idio_transform'] = model.idio_transform
    
    np.savez(filename, **save_dict)


def load_model_full(filename: str) -> FactorModelData:
    """
    Load factor model with optional transforms from NPZ file.
    
    Loads a factor model including transforms if they were saved.
    
    Parameters
    ----------
    filename : str
        Path to NPZ file
        
    Returns
    -------
    model : FactorModelData
        Loaded factor model with all components
        
    Examples
    --------
    >>> model = load_model_full('full_model.npz')
    >>> if model.factor_transform is not None:
    ...     print("Has factor transform")
    """
    data = np.load(filename)
    
    return FactorModelData(
        B=data['B'],
        F=data['F'],
        D=data['D'],
        factor_transform=data.get('factor_transform'),
        idio_transform=data.get('idio_transform')
    )
