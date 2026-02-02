"""
context.py - Simulation Context Dataclass
==========================================

Provides an immutable snapshot of simulation state for custom analyses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
import numpy as np

if TYPE_CHECKING:
    from ..types import FactorModelData
    from ..model_spec_parser import ModelSpecification, SimulationSpec
    from ..covariance_comparison import CovarianceComparisonResult
    from ..normality_tests import TestSuiteResults

__all__ = ['SimulationContext']


@dataclass(frozen=True)
class SimulationContext:
    """
    Immutable snapshot of simulation state.
    
    Provides all data needed for custom analyses without coupling to
    internal implementation details. The context is frozen (immutable)
    to prevent accidental modification and enable safe caching.
    
    Parameters
    ----------
    model : FactorModelData
        The factor model (B, F, D matrices).
    security_returns : np.ndarray, shape (T, p)
        Simulated security returns.
    factor_returns : np.ndarray, shape (T, k)
        Simulated factor returns.
    idio_returns : np.ndarray, shape (T, p)
        Simulated idiosyncratic returns.
    spec : ModelSpecification, optional
        Model specification (if available).
    sim_spec : SimulationSpec, optional
        Simulation specification (if available).
    test_results : Dict[str, TestSuiteResults], optional
        Normality test results (if run).
    covariance_validation : Any, optional
        Covariance validation result (if run).
    ground_truth_comparison : CovarianceComparisonResult, optional
        Ground truth comparison result (if run).
    timestamp : datetime
        When the simulation was run.
    duration : float
        Simulation duration in seconds.
    
    Examples
    --------
    >>> context = SimulationContext(
    ...     model=model,
    ...     security_returns=returns,
    ...     factor_returns=factors,
    ...     idio_returns=idio
    ... )
    >>> 
    >>> # Access cached sample covariance
    >>> cov = context.sample_covariance()
    >>> 
    >>> # Run PCA (cached)
    >>> pca = context.pca_decomposition(n_components=3)
    
    Notes
    -----
    The context uses lazy evaluation for expensive computations like
    PCA decomposition and sample covariance. Results are cached after
    first computation.
    """
    
    # Core data (required)
    model: 'FactorModelData'
    security_returns: np.ndarray
    factor_returns: np.ndarray
    idio_returns: np.ndarray
    
    # Optional metadata
    spec: Optional['ModelSpecification'] = None
    sim_spec: Optional['SimulationSpec'] = None
    
    # Optional validation results
    test_results: Optional[Dict[str, 'TestSuiteResults']] = None
    covariance_validation: Optional[Any] = None
    ground_truth_comparison: Optional['CovarianceComparisonResult'] = None
    
    # Timing metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    
    # Private cached computations (not included in repr/comparison)
    _sample_cov: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False, init=False
    )
    _pca_cache: Dict[int, Any] = field(
        default_factory=dict, repr=False, compare=False, init=False
    )
    
    @property
    def T(self) -> int:
        """Number of time periods."""
        return self.security_returns.shape[0]
    
    @property
    def p(self) -> int:
        """Number of securities."""
        return self.security_returns.shape[1]
    
    @property
    def k(self) -> int:
        """Number of factors."""
        return self.model.k
    
    def sample_covariance(self, ddof: int = 1) -> np.ndarray:
        """
        Compute sample covariance matrix (cached).
        
        Parameters
        ----------
        ddof : int, default=1
            Degrees of freedom for covariance computation.
        
        Returns
        -------
        np.ndarray, shape (p, p)
            Sample covariance matrix.
        
        Notes
        -----
        Result is cached after first computation. The matrix is
        computed as: (1/(T-ddof)) * (X - μ)' (X - μ)
        """
        if self._sample_cov is None:
            cov = np.cov(self.security_returns, rowvar=False, ddof=ddof)
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(self, '_sample_cov', cov)
        return self._sample_cov
    
    def pca_decomposition(self, n_components: Optional[int] = None):
        """
        Run PCA decomposition (cached by n_components).
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to extract.
            If None, uses model.k.
        
        Returns
        -------
        FactorModelData
            Extracted factor model from PCA.
        
        Notes
        -----
        Results are cached by n_components. Subsequent calls with
        the same n_components return the cached result.
        """
        from ..types import svd_decomposition
        
        n_components = n_components or self.model.k
        
        if n_components not in self._pca_cache:
            result = svd_decomposition(
                self.security_returns,
                k=n_components,
                center=True
            )
            # Store in cache (need to work around frozen dataclass)
            cache = dict(self._pca_cache)
            cache[n_components] = result
            object.__setattr__(self, '_pca_cache', cache)
        
        return self._pca_cache[n_components]
    
    def summary(self) -> str:
        """
        Generate a formatted summary of the context.
        
        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "SimulationContext Summary",
            "=" * 60,
            f"Model: {self.k} factors, {self.p} securities",
            f"Periods: {self.T}",
            f"Duration: {self.duration:.2f}s",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if self.spec:
            lines.append(f"Specification: Available")
        if self.test_results:
            lines.append(f"Test Results: {len(self.test_results)} test(s)")
        if self.covariance_validation:
            lines.append(f"Covariance Validation: Available")
        if self.ground_truth_comparison:
            lines.append(f"Ground Truth Comparison: Available")
        
        return "\n".join(lines)
