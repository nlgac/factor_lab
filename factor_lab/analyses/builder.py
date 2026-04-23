"""
builder.py - Convenience Builder for Common Analyses
=====================================================

Provides a simple factory API for creating standard analyses without
needing to import individual classes or remember their parameters.

Examples
--------
>>> from factor_lab.analyses import Analyses
>>> 
>>> # Create analyses using factory methods
>>> results = run_simulation(
...     model, spec, sim_spec, rng,
...     custom_analyses=[
...         Analyses.manifold_distances(),
...         Analyses.eigenvector_comparison(k=10),
...         Analyses.eigenvalue_analysis(),
...     ]
... )
"""

from typing import Optional, Callable, Dict, Any

from .manifold import ManifoldDistanceAnalysis
from .spectral import ImplicitEigenAnalysis
from .eigenvector import EigenvectorAlignment
from ..analysis import SimulationContext

__all__ = ['Analyses']


class Analyses:
    """
    Factory for creating common analysis objects.
    
    Provides convenient static methods for creating standard analyses
    without needing to import individual classes or remember constructor
    parameters.
    
    Methods
    -------
    manifold_distances(use_pca_loadings=True)
        Compare factor loadings using manifold geometry.
        
    eigenvalue_analysis(k_top=None, compare_eigenvectors=False)
        Compute true eigenvalues via implicit operator.
        
    eigenvector_comparison(k=None, align_signs=True)
        Compare ground truth and PCA eigenvectors.
        
    custom(func)
        Wrap arbitrary function as analysis.
    
    Examples
    --------
    >>> # Basic usage
    >>> manifold_analysis = Analyses.manifold_distances()
    >>> eigen_analysis = Analyses.eigenvalue_analysis(k_top=10)
    >>> 
    >>> # In simulation pipeline
    >>> from factor_lab.analyses import Analyses
    >>> 
    >>> results = run_simulation(
    ...     model, spec, sim_spec, rng,
    ...     custom_analyses=[
    ...         Analyses.manifold_distances(),
    ...         Analyses.eigenvector_comparison(k=5, align_signs=True),
    ...         Analyses.custom(lambda ctx: {
    ...             'custom_metric': np.linalg.norm(ctx.model.B)
    ...         })
    ...     ]
    ... )
    """
    
    @staticmethod
    def manifold_distances(use_pca_loadings: bool = True) -> ManifoldDistanceAnalysis:
        """
        Create manifold distance analysis.
        
        Computes Grassmannian, Procrustes, and Chordal distances between
        ground truth and estimated factor loadings.
        
        Parameters
        ----------
        use_pca_loadings : bool, default=True
            If True, compare against PCA-extracted loadings.
            If False, compare model to itself (for testing).
        
        Returns
        -------
        ManifoldDistanceAnalysis
            Analysis object ready to run.
        
        Examples
        --------
        >>> analysis = Analyses.manifold_distances()
        >>> results = analysis.analyze(context)
        >>> print(results['dist_grassmannian'])
        """
        return ManifoldDistanceAnalysis(use_pca_loadings=use_pca_loadings)
    
    @staticmethod
    def eigenvalue_analysis(
        k_top: Optional[int] = None,
        compare_eigenvectors: bool = False,
        tol: float = 1e-10,
        maxiter: int = 10000
    ) -> ImplicitEigenAnalysis:
        """
        Create eigenvalue analysis via implicit operator.
        
        Computes eigenvalues of true covariance Σ = B'FB + D using
        LinearOperator (memory efficient: O(kp) instead of O(p²)).
        
        Parameters
        ----------
        k_top : int, optional
            Number of eigenvalues to compute.
            If None, uses model.k.
        compare_eigenvectors : bool, default=False
            Whether to also compare eigenvector alignment.
        tol : float, default=1e-10
            ARPACK convergence tolerance.
        maxiter : int, default=10000
            Maximum ARPACK iterations.
        
        Returns
        -------
        ImplicitEigenAnalysis
            Analysis object ready to run.
        
        Examples
        --------
        >>> # Just eigenvalues
        >>> analysis = Analyses.eigenvalue_analysis(k_top=10)
        >>> 
        >>> # Including eigenvectors
        >>> analysis = Analyses.eigenvalue_analysis(
        ...     k_top=10,
        ...     compare_eigenvectors=True
        ... )
        """
        return ImplicitEigenAnalysis(
            k_top=k_top,
            compare_eigenvectors=compare_eigenvectors,
            tol=tol,
            maxiter=maxiter
        )
    
    @staticmethod
    def eigenvector_comparison(
        k: Optional[int] = None,
        align_signs: bool = True,
        compute_rotation: bool = True,
        tol: float = 1e-10,
        maxiter: int = 10000
    ) -> EigenvectorAlignment:
        """
        Create eigenvector alignment analysis.
        
        Compares ground truth eigenvectors (from Σ = B'FB + D) with
        PCA eigenvectors from sample covariance.
        
        Computes alignment metrics including Procrustes distance,
        canonical correlations, and principal angles.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvectors to compare.
            If None, uses model.k.
        align_signs : bool, default=True
            Whether to align signs before comparison.
        compute_rotation : bool, default=True
            Whether to compute optimal rotation matrix.
        tol : float, default=1e-10
            ARPACK convergence tolerance.
        maxiter : int, default=10000
            Maximum ARPACK iterations.
        
        Returns
        -------
        EigenvectorAlignment
            Analysis object ready to run.
        
        Examples
        --------
        >>> analysis = Analyses.eigenvector_comparison(k=5)
        >>> results = analysis.analyze(context)
        >>> print(f"Mean correlation: {results['mean_correlation']:.4f}")
        """
        return EigenvectorAlignment(
            k_components=k,
            align_signs=align_signs,
            compute_rotation=compute_rotation,
            tol=tol,
            maxiter=maxiter
        )
    
    @staticmethod
    def custom(func: Callable[[SimulationContext], Dict[str, Any]]):
        """
        Wrap arbitrary function as analysis.
        
        Allows quick creation of custom analyses without defining
        a full class. Useful for one-off computations.
        
        Parameters
        ----------
        func : callable
            Function that takes SimulationContext and returns
            dict of results.
        
        Returns
        -------
        object
            Analysis object implementing the protocol.
        
        Examples
        --------
        >>> # Simple custom analysis
        >>> custom_analysis = Analyses.custom(
        ...     lambda ctx: {
        ...         'frobenius_B': np.linalg.norm(ctx.model.B, 'fro'),
        ...         'trace_F': np.trace(ctx.model.F),
        ...     }
        ... )
        >>> 
        >>> # More complex
        >>> def my_analysis(ctx):
        ...     pca = ctx.pca_decomposition(n_components=3)
        ...     error = np.linalg.norm(ctx.model.B - pca.B)
        ...     return {'loading_error': error}
        >>> 
        >>> analysis = Analyses.custom(my_analysis)
        """
        class CustomAnalysis:
            def analyze(self, context: SimulationContext):
                return func(context)
        
        return CustomAnalysis()
