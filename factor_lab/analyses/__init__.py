"""
Built-in Analyses for Factor Models
====================================

Provides ready-to-use analysis modules for factor model validation:

- **Manifold Analysis**: Geometric distances (Grassmannian, Procrustes, Chordal)
- **Spectral Analysis**: Eigenvalue comparison via LinearOperator
- **Eigenvector Analysis**: Eigenvector alignment and correlation
- **Builder API**: Convenient factory for creating analyses

Examples
--------
>>> from factor_lab.analyses import Analyses
>>> 
>>> # Create analyses using factory
>>> analyses = [
...     Analyses.manifold_distances(),
...     Analyses.eigenvector_comparison(k=10),
...     Analyses.eigenvalue_analysis(),
... ]
>>> 
>>> # Or import specific classes
>>> from factor_lab.analyses import ManifoldDistanceAnalysis
>>> analysis = ManifoldDistanceAnalysis()
"""

# Import all analysis classes
from .manifold import (
    ManifoldDistanceAnalysis,
    compute_grassmannian_distance,
    compute_procrustes_distance,
    compute_chordal_distance,
    orthonormalize,
)

from .spectral import (
    ImplicitEigenAnalysis,
    compute_true_eigenvalues,
)

from .eigenvector import EigenvectorAlignment

from .builder import Analyses

__all__ = [
    # Analysis classes
    'ManifoldDistanceAnalysis',
    'ImplicitEigenAnalysis',
    'EigenvectorAlignment',
    
    # Builder/Factory
    'Analyses',
    
    # Utility functions
    'compute_grassmannian_distance',
    'compute_procrustes_distance',
    'compute_chordal_distance',
    'orthonormalize',
    'compute_true_eigenvalues',
]
