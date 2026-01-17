"""
simulation.py - Monte Carlo Simulation for Factor Models

This module provides tools for simulating returns from factor models:
- ReturnsSimulator: Generate synthetic returns with proper covariance structure
- CovarianceValidator: Compare empirical vs model-implied covariance

Mathematical Background:
-----------------------
Given a factor model:
    r = B.T @ f + ε

where f ~ N(0, F) and ε ~ N(0, D), we simulate returns by:
1. Drawing standardized factor innovations z_f ~ N(0, I_k)
2. Drawing standardized idio innovations z_ε ~ N(0, I_p)
3. Scaling: f = F^(1/2) @ z_f, ε = D^(1/2) @ z_ε
4. Combining: r = B.T @ f + ε

The simulator supports non-Gaussian innovations (e.g., Student's t) via
custom sampler functions, while ensuring the correct covariance structure.

Example Usage:
-------------
    >>> from factor_lab.simulation import ReturnsSimulator, CovarianceValidator
    >>> from factor_lab.samplers import DistributionFactory
    >>> 
    >>> # Setup
    >>> rng = np.random.default_rng(42)
    >>> factory = DistributionFactory(rng=rng)
    >>> 
    >>> # Create samplers for innovations
    >>> factor_samplers = [factory.create("student_t", df=5)] * model.k
    >>> idio_samplers = [factory.create("normal", mean=0, std=1)] * model.p
    >>> 
    >>> # Simulate
    >>> simulator = ReturnsSimulator(model, rng=rng)
    >>> results = simulator.simulate(
    ...     n_periods=1000,
    ...     factor_samplers=factor_samplers,
    ...     idio_samplers=idio_samplers
    ... )
    >>> 
    >>> # Validate
    >>> validator = CovarianceValidator(model)
    >>> validation = validator.compare(results["security_returns"])
    >>> print(f"Frobenius error: {validation.frobenius_error:.4f}")
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from .types import (
    FactorModelData, 
    SamplerCallable, 
    CovarianceTransform, 
    TransformType,
    CovarianceValidationResult
)


# =============================================================================
# COVARIANCE VALIDATOR
# =============================================================================

class CovarianceValidator:
    """
    Validator for comparing empirical and model-implied covariance matrices.
    
    This class provides diagnostic tools to assess how well simulated
    returns match the theoretical factor model covariance structure.
    
    Parameters
    ----------
    model : FactorModelData
        The factor model to validate against.
    
    Examples
    --------
    >>> validator = CovarianceValidator(model)
    >>> 
    >>> # After simulation
    >>> results = simulator.simulate(...)
    >>> validation = validator.compare(results["security_returns"])
    >>> 
    >>> if validation.frobenius_error > 0.1:
    ...     print("Warning: Large discrepancy between model and empirical cov")
    
    Notes
    -----
    The model covariance is computed once and cached. Call `reset_cache()`
    if the model is modified (though models should typically be immutable).
    """
    
    def __init__(self, model: FactorModelData):
        """
        Initialize the validator.
        
        Parameters
        ----------
        model : FactorModelData
            The reference factor model.
        """
        self.model = model
        self._model_cov: Optional[np.ndarray] = None
    
    @property
    def model_covariance(self) -> np.ndarray:
        """
        Compute and cache the model-implied covariance matrix.
        
        Returns
        -------
        np.ndarray
            The covariance matrix Σ = B.T @ F @ B + D.
        """
        if self._model_cov is None:
            self._model_cov = self.model.implied_covariance()
        return self._model_cov
    
    def reset_cache(self) -> None:
        """Clear the cached model covariance."""
        self._model_cov = None
    
    def compare(self, returns: np.ndarray) -> CovarianceValidationResult:
        """
        Compare empirical covariance of returns to model-implied covariance.
        
        Parameters
        ----------
        returns : np.ndarray
            Returns matrix with shape (T, p) where T is the number of
            time periods and p is the number of assets.
        
        Returns
        -------
        CovarianceValidationResult
            Detailed comparison metrics including:
            - frobenius_error: L2 norm of covariance difference
            - mean_absolute_error: Average absolute element-wise error
            - max_absolute_error: Worst element-wise error
            - explained_variance_ratio: Fraction explained by factors
            - Both covariance matrices for inspection
        
        Raises
        ------
        ValueError
            If returns shape is incompatible with the model.
        
        Examples
        --------
        >>> validation = validator.compare(simulated_returns)
        >>> print(f"Model fit: {validation.frobenius_error:.4f}")
        >>> print(f"Explained variance: {validation.explained_variance_ratio:.1%}")
        """
        # Validate input shape
        if returns.ndim != 2:
            raise ValueError(
                f"Returns must be 2D array, got shape {returns.shape}"
            )
        
        T, p = returns.shape
        
        if p != self.model.p:
            raise ValueError(
                f"Returns have {p} assets, model has {self.model.p}. "
                "Did you transpose the returns matrix?"
            )
        
        if T < 2:
            raise ValueError(
                f"Need at least 2 time periods for covariance, got {T}"
            )
        
        # Compute empirical covariance
        empirical_cov = np.cov(returns, rowvar=False)
        
        # Get model covariance
        model_cov = self.model_covariance
        
        # Compute error metrics
        diff = empirical_cov - model_cov
        
        frobenius_error = np.linalg.norm(diff, ord='fro')
        mean_abs_error = np.mean(np.abs(diff))
        max_abs_error = np.max(np.abs(diff))
        
        # Compute explained variance ratio
        # Factor variance = trace(B.T @ F @ B)
        # Total variance = trace(Σ) = trace(B.T @ F @ B + D)
        factor_cov = self.model.B.T @ self.model.F @ self.model.B
        factor_var = np.trace(factor_cov)
        total_var = np.trace(model_cov)
        
        explained_ratio = factor_var / total_var if total_var > 0 else 0.0
        
        return CovarianceValidationResult(
            frobenius_error=frobenius_error,
            mean_absolute_error=mean_abs_error,
            max_absolute_error=max_abs_error,
            explained_variance_ratio=explained_ratio,
            model_covariance=model_cov,
            empirical_covariance=empirical_cov
        )


# =============================================================================
# RETURNS SIMULATOR
# =============================================================================

class ReturnsSimulator:
    """
    Monte Carlo simulator for factor model returns.
    
    This class generates synthetic returns that match the covariance
    structure implied by a factor model. It supports:
    - Automatic detection of diagonal covariance matrices (O(n) path)
    - Custom innovation distributions (e.g., fat-tailed)
    - Pre-computed transforms from SVD decomposition
    
    Parameters
    ----------
    model : FactorModelData
        The factor model specifying the return distribution.
    rng : np.random.Generator, optional
        Random number generator for internal operations (fallback sampling).
        If None, creates a new default RNG.
    force_dense : bool, default=False
        If True, always use Cholesky decomposition even for diagonal matrices.
        Primarily useful for testing that diagonal and dense paths are equivalent.
    
    Examples
    --------
    >>> from factor_lab.simulation import ReturnsSimulator
    >>> from factor_lab.samplers import DistributionFactory
    >>> 
    >>> # Basic usage with normal innovations
    >>> rng = np.random.default_rng(42)
    >>> factory = DistributionFactory(rng=rng)
    >>> 
    >>> simulator = ReturnsSimulator(model, rng=rng)
    >>> results = simulator.simulate(
    ...     n_periods=1000,
    ...     factor_samplers=[factory.create("normal", mean=0, std=1)] * model.k,
    ...     idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
    ... )
    >>> 
    >>> # Access results
    >>> security_returns = results["security_returns"]  # (1000, p)
    >>> factor_returns = results["factor_returns"]       # (1000, k)
    >>> idio_returns = results["idio_returns"]          # (1000, p)
    
    Notes
    -----
    **Covariance Transform Selection:**
    
    The simulator automatically selects between two paths:
    
    1. **Diagonal path (O(n))**: When F and D are diagonal, we simply scale
       by standard deviations. This is the common case after SVD decomposition.
    
    2. **Dense path (O(n²))**: When F or D have off-diagonal elements, we use
       Cholesky decomposition to properly correlate the innovations.
    
    **Pre-computed Transforms:**
    
    If the model includes `factor_transform` and/or `idio_transform`, these
    are used directly without recomputation. SVD decomposition automatically
    provides these.
    
    **Innovation Standardization:**
    
    Raw samples from the provided samplers are standardized (mean=0, std=1)
    before applying the covariance transform. This ensures the output has
    the correct covariance regardless of the innovation distribution.
    """
    
    def __init__(
        self,
        model: FactorModelData,
        rng: Optional[np.random.Generator] = None,
        force_dense: bool = False
    ):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        model : FactorModelData
            The factor model to simulate from.
        rng : np.random.Generator, optional
            Random number generator.
        force_dense : bool, default=False
            Force Cholesky decomposition even for diagonal matrices.
        """
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self._force_dense = force_dense
        
        # Setup covariance transforms
        self._factor_transform = self._setup_transform(
            model.factor_transform,
            model.F,
            "factor",
            force_dense
        )
        
        self._idio_transform = self._setup_transform(
            model.idio_transform,
            model.D,
            "idio",
            force_dense
        )
    
    @property
    def factor_transform(self) -> CovarianceTransform:
        """The transform used for factor covariance."""
        return self._factor_transform
    
    @property
    def idio_transform(self) -> CovarianceTransform:
        """The transform used for idiosyncratic covariance."""
        return self._idio_transform
    
    def _setup_transform(
        self,
        precomputed: Optional[CovarianceTransform],
        cov_matrix: np.ndarray,
        name: str,
        force_dense: bool
    ) -> CovarianceTransform:
        """
        Setup a covariance transform, using precomputed if available.
        
        Parameters
        ----------
        precomputed : CovarianceTransform or None
            Pre-computed transform from the model (e.g., from SVD).
        cov_matrix : np.ndarray
            The covariance matrix (F or D).
        name : str
            Name for logging/debugging ("factor" or "idio").
        force_dense : bool
            If True, always compute Cholesky even if diagonal.
        
        Returns
        -------
        CovarianceTransform
            The transform to use for simulation.
        """
        # Use precomputed if available and not forcing dense
        if precomputed is not None and not force_dense:
            return precomputed
        
        # Otherwise, compute from the covariance matrix
        if not force_dense and self._is_diagonal(cov_matrix):
            # Diagonal case: extract standard deviations
            stds = np.sqrt(np.diag(cov_matrix))
            return CovarianceTransform(
                matrix=stds,
                transform_type=TransformType.DIAGONAL
            )
        else:
            # Dense case: compute Cholesky factor
            L = np.linalg.cholesky(cov_matrix)
            return CovarianceTransform(
                matrix=L,
                transform_type=TransformType.DENSE
            )
    
    @staticmethod
    def _is_diagonal(M: np.ndarray) -> bool:
        """Check if a matrix is diagonal (off-diagonals are zero)."""
        return np.allclose(M, np.diag(np.diag(M)))
    
    def simulate(
        self,
        n_periods: int,
        factor_samplers: List[SamplerCallable],
        idio_samplers: List[SamplerCallable],
        sample_log_rows: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns from the factor model.
        
        Parameters
        ----------
        n_periods : int
            Number of time periods (rows) to simulate.
        factor_samplers : List[SamplerCallable]
            List of k samplers, one for each factor's innovations.
            Each sampler is called with n_periods to get raw samples.
        idio_samplers : List[SamplerCallable]
            List of p samplers, one for each asset's idiosyncratic innovations.
        sample_log_rows : int, default=0
            If > 0, store the first `sample_log_rows` rows of raw (pre-transform)
            samples in the result dict for debugging.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - "security_returns": (n_periods, p) array of asset returns
            - "factor_returns": (n_periods, k) array of factor returns
            - "idio_returns": (n_periods, p) array of idiosyncratic returns
            - "raw_factor_samples": (optional) raw samples before transform
            - "raw_idio_samples": (optional) raw samples before transform
        
        Raises
        ------
        ValueError
            If sampler list lengths don't match model dimensions.
        
        Examples
        --------
        >>> # Using Student's t innovations for fat tails
        >>> results = simulator.simulate(
        ...     n_periods=5000,
        ...     factor_samplers=[factory.create("student_t", df=4)] * model.k,
        ...     idio_samplers=[factory.create("normal", mean=0, std=1)] * model.p
        ... )
        >>> 
        >>> # Check covariance
        >>> returns = results["security_returns"]
        >>> empirical_cov = np.cov(returns, rowvar=False)
        
        Notes
        -----
        The simulation process:
        1. Draw raw samples from each sampler
        2. Standardize samples (subtract mean, divide by std)
        3. Apply covariance transform (scale by sqrt(F) or sqrt(D))
        4. Combine: security_returns = factor_returns @ B + idio_returns
        """
        # Validate sampler counts
        if len(factor_samplers) != self.model.k:
            raise ValueError(
                f"Expected {self.model.k} factor samplers, got {len(factor_samplers)}"
            )
        
        if len(idio_samplers) != self.model.p:
            raise ValueError(
                f"Expected {self.model.p} idio samplers, got {len(idio_samplers)}"
            )
        
        # Process factor innovations
        factor_returns, raw_factor = self._transform_samples(
            samplers=factor_samplers,
            n_samples=n_periods,
            transform=self._factor_transform,
            context="factor"
        )
        
        # Process idiosyncratic innovations
        idio_returns, raw_idio = self._transform_samples(
            samplers=idio_samplers,
            n_samples=n_periods,
            transform=self._idio_transform,
            context="idio"
        )
        
        # Combine: r = B.T @ f + epsilon
        # factor_returns: (n_periods, k)
        # B: (k, p)
        # So: factor_returns @ B gives (n_periods, p)
        security_returns = factor_returns @ self.model.B + idio_returns
        
        # Build result dictionary
        result: Dict[str, np.ndarray] = {
            "security_returns": security_returns,
            "factor_returns": factor_returns,
            "idio_returns": idio_returns
        }
        
        # Add debug samples if requested
        if sample_log_rows > 0:
            result["raw_factor_samples"] = raw_factor[:sample_log_rows]
            result["raw_idio_samples"] = raw_idio[:sample_log_rows]
        
        return result
    
    def _transform_samples(
        self,
        samplers: List[SamplerCallable],
        n_samples: int,
        transform: CovarianceTransform,
        context: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate samples, standardize them, and apply covariance transform.
        
        Parameters
        ----------
        samplers : List[SamplerCallable]
            One sampler per dimension.
        n_samples : int
            Number of samples to draw from each sampler.
        transform : CovarianceTransform
            The covariance transform to apply.
        context : str
            Name for debugging ("factor" or "idio").
        
        Returns
        -------
        transformed : np.ndarray
            Transformed samples with correct covariance structure.
        raw : np.ndarray
            Original samples before standardization (for debugging).
        """
        # Step 1: Draw raw samples from each sampler
        # Each sampler returns (n_samples,), we stack into (n_samples, d)
        raw = np.column_stack([sampler(n_samples) for sampler in samplers])
        
        # Step 2: Standardize each column (mean=0, std=1)
        # This ensures the innovation distribution doesn't affect covariance
        means = raw.mean(axis=0)
        stds = raw.std(axis=0)
        
        # Handle zero std (constant columns) to avoid division by zero
        stds = np.where(stds == 0, 1.0, stds)
        
        z_scores = (raw - means) / stds
        
        # Step 3: Apply covariance transform
        transformed = transform.apply(z_scores)
        
        return transformed, raw


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def simulate_returns(
    model: FactorModelData,
    n_periods: int,
    rng: Optional[np.random.Generator] = None,
    innovation: str = "normal"
) -> np.ndarray:
    """
    Convenience function to simulate returns with standard innovations.
    
    This is a simplified interface when you don't need custom innovation
    distributions or detailed control over the simulation process.
    
    Parameters
    ----------
    model : FactorModelData
        The factor model to simulate from.
    n_periods : int
        Number of time periods to simulate.
    rng : np.random.Generator, optional
        Random number generator. If None, creates a new one.
    innovation : {"normal", "student_t"}, default="normal"
        Innovation distribution type. "student_t" uses df=5 for moderate
        fat tails.
    
    Returns
    -------
    np.ndarray
        Simulated returns with shape (n_periods, p).
    
    Examples
    --------
    >>> returns = simulate_returns(model, n_periods=1000, rng=rng)
    >>> print(f"Shape: {returns.shape}")
    """
    from .samplers import DistributionFactory
    
    if rng is None:
        rng = np.random.default_rng()
    
    factory = DistributionFactory(rng=rng)
    
    if innovation == "normal":
        factor_samplers = [
            factory.create("normal", mean=0.0, std=1.0) 
            for _ in range(model.k)
        ]
        idio_samplers = [
            factory.create("normal", mean=0.0, std=1.0)
            for _ in range(model.p)
        ]
    elif innovation == "student_t":
        factor_samplers = [
            factory.create("student_t", df=5)
            for _ in range(model.k)
        ]
        idio_samplers = [
            factory.create("student_t", df=5)
            for _ in range(model.p)
        ]
    else:
        raise ValueError(f"Unknown innovation type: {innovation}")
    
    simulator = ReturnsSimulator(model, rng=rng)
    results = simulator.simulate(
        n_periods=n_periods,
        factor_samplers=factor_samplers,
        idio_samplers=idio_samplers
    )
    
    return results["security_returns"]
