"""
io.py - Model Serialization and Deserialization

This module handles saving and loading FactorModelData to/from disk.
Supported formats:
- NPZ: NumPy's compressed archive format (default, preserves transforms)
- JSON: Human-readable format (no transform preservation)

Example Usage:
-------------
    >>> from factor_lab.io import save_model, load_model
    >>> 
    >>> # Save a model
    >>> save_model(model, "my_model.npz")
    >>> 
    >>> # Load it back
    >>> loaded = load_model("my_model.npz")
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np

from .types import (
    FactorModelData,
    CovarianceTransform,
    TransformType,
)


class ModelFormat(str, Enum):
    """Supported model file formats."""
    NPZ = "npz"
    JSON = "json"


def save_model(
    model: FactorModelData,
    path: Union[str, Path],
    format: ModelFormat = ModelFormat.NPZ
) -> None:
    """
    Save a factor model to disk.

    Parameters
    ----------
    model : FactorModelData
        The factor model to save.
    path : str or Path
        Destination file path.
    format : ModelFormat, default=ModelFormat.NPZ
        Output format. NPZ preserves covariance transforms; JSON does not.

    Examples
    --------
    >>> save_model(model, "model.npz")
    >>> save_model(model, "model.json", format=ModelFormat.JSON)
    """
    path = Path(path)

    if format == ModelFormat.NPZ:
        _save_npz(model, path)
    elif format == ModelFormat.JSON:
        _save_json(model, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_model(path: Union[str, Path]) -> FactorModelData:
    """
    Load a factor model from disk.

    Parameters
    ----------
    path : str or Path
        Source file path. Format is inferred from extension.

    Returns
    -------
    FactorModelData
        The loaded factor model.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not recognized.

    Examples
    --------
    >>> model = load_model("model.npz")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix == ".npz":
        return _load_npz(path)
    elif path.suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unknown model format: {path.suffix}")


def _save_npz(model: FactorModelData, path: Path) -> None:
    """Save model to NPZ format with optional transforms."""
    data = {
        "B": model.B,
        "F": model.F,
        "D": model.D,
    }

    # Save transforms if present
    if model.factor_transform is not None:
        data["factor_transform_matrix"] = model.factor_transform.matrix
        data["factor_transform_type"] = model.factor_transform.transform_type.name
    else:
        data["factor_transform_matrix"] = None
        data["factor_transform_type"] = None

    if model.idio_transform is not None:
        data["idio_transform_matrix"] = model.idio_transform.matrix
        data["idio_transform_type"] = model.idio_transform.transform_type.name
    else:
        data["idio_transform_matrix"] = None
        data["idio_transform_type"] = None

    np.savez(path, **data)


def _load_npz(path: Path) -> FactorModelData:
    """Load model from NPZ format."""
    data = np.load(path, allow_pickle=True)

    # Reconstruct factor transform if present
    factor_transform = _reconstruct_transform(
        data.get("factor_transform_matrix"),
        data.get("factor_transform_type")
    )

    # Reconstruct idio transform if present
    idio_transform = _reconstruct_transform(
        data.get("idio_transform_matrix"),
        data.get("idio_transform_type")
    )

    return FactorModelData(
        B=data["B"],
        F=data["F"],
        D=data["D"],
        factor_transform=factor_transform,
        idio_transform=idio_transform,
    )


def _save_json(model: FactorModelData, path: Path) -> None:
    """
    Save model to JSON format.
    
    Warning: Transforms are not preserved in JSON format.
    """
    data = {
        "B": model.B.tolist(),
        "F": model.F.tolist(),
        "D": model.D.tolist(),
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_json(path: Path) -> FactorModelData:
    """Load model from JSON format."""
    with open(path, 'r') as f:
        data = json.load(f)

    return FactorModelData(
        B=np.array(data["B"]),
        F=np.array(data["F"]),
        D=np.array(data["D"]),
        factor_transform=None,
        idio_transform=None,
    )


def _reconstruct_transform(
    matrix, 
    transform_type_name
) -> Union[CovarianceTransform, None]:
    """
    Reconstruct a CovarianceTransform from saved components.
    
    Parameters
    ----------
    matrix : np.ndarray or None
        The transform matrix.
    transform_type_name : str or None
        Name of the TransformType enum value.
    
    Returns
    -------
    CovarianceTransform or None
        Reconstructed transform, or None if inputs are None.
    """
    # Handle None cases
    if matrix is None:
        return None
    
    # Handle numpy array containing None (from .npz file)
    if isinstance(matrix, np.ndarray):
        if matrix.dtype == object and matrix.size == 1:
            # Single-element object array, check if it contains None
            if matrix.item() is None:
                return None
    
    if transform_type_name is None:
        return None
    
    # Handle numpy scalar arrays (from .npz file)
    if isinstance(transform_type_name, np.ndarray):
        if transform_type_name.size == 1:
            transform_type_name = transform_type_name.item()
        else:
            return None
    
    # Convert string name back to enum
    transform_type = TransformType[transform_type_name]
    
    return CovarianceTransform(
        matrix=matrix,
        transform_type=transform_type
    )
