"""
spectral.py - Eigenvalue/Eigenvector Analysis via LinearOperator
=================================================================

Computes eigenvalues of the true covariance matrix Σ = B'FB + D using
implicit LinearOperator for memory efficiency. Never forms the full p×p
covariance matrix.

Based on Gemini's implementation with enhancements for robustness and
error handling.

Complexity Analysis
-------------------
Dense Method:
- Memory: O(p²) - must store full covariance matrix
- Time: O(p³) - eigendecomposition

LinearOperator Method:
- Memory: O(kp) - only store B, F, D
- Time: O(k²p) - iterative solver

For p=10,000, k=10:
- Dense: 800 MB memory, ~10 minutes
- Sparse: 800 KB memory, ~10 seconds
"""

from typing import Dict, Any, Optional, Tuple
import warnings
import numpy as np
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence

from ..analysis import SimulationAnalysis, SimulationContext
from ..types import FactorModelData

__all__ = [
    'compute_true_eigenvalues',
    'ImplicitEigenAnalysis',
]


def compute_true_eigenvalues(
    model: FactorModelData,
    k_top: int,
    tol: float = 1e-10,
    maxiter: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of Σ = B'FB + D via implicit operator.
    
    Never forms the full p×p covariance matrix. Instead, defines
    a LinearOperator that performs matrix-vector products:
        Σ @ v = B'(F(Bv)) + Dv
    
    Then uses ARPACK (Lanczos iteration) to find the top k eigenvalues.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model with B, F, D matrices.
    k_top : int
        Number of top eigenvalues to compute.
    tol : float, default=1e-10
        Convergence tolerance for ARPACK.
    maxiter : int, default=10000
        Maximum number of Lanczos iterations.
    verbose : bool, default=False
        Whether to print convergence information.
    
    Returns
    -------
    eigenvalues : np.ndarray, shape (k_top,)
        Top k eigenvalues in descending order.
    eigenvectors : np.ndarray, shape (k_top, p)
        Corresponding eigenvectors as row vectors.
    
    Raises
    ------
    Warning
        If ARPACK does not converge, returns partial results
        with a warning.
    
    Notes
    -----
    **Why this is efficient:**
    
    The true covariance matrix Σ has special structure:
        Σ = B'FB + D
    
    where B is k×p, F is k×k, D is p×p diagonal.
    
    For k << p, we can compute Σv efficiently without forming Σ:
        Σv = B'(F(Bv)) + Dv
    
    This reduces:
    - Memory: O(p²) → O(kp)
    - Computation: O(p²) → O(kp)
    
    **Convergence:**
    
    ARPACK typically converges quickly when eigenvalues are well-separated.
    Convergence can be slow if:
    - Eigenvalues are clustered
    - Factor variances are very small
    - High condition number
    
    Examples
    --------
    >>> import numpy as np
    >>> from factor_lab import FactorModelData
    >>> 
    >>> # Create model
    >>> B = np.random.randn(3, 1000)
    >>> F = np.diag([0.09, 0.04, 0.01])
    >>> D = np.diag(np.full(1000, 0.01))
    >>> model = FactorModelData(B=B, F=F, D=D)
    >>> 
    >>> # Compute top eigenvalues
    >>> evals, evecs = compute_true_eigenvalues(model, k_top=10)
    >>> 
    >>> # Compare to dense method (only for small p!)
    >>> Sigma = B.T @ F @ B + D
    >>> evals_dense = np.linalg.eigvalsh(Sigma)[::-1][:10]
    >>> np.allclose(evals, evals_dense)
    True
    """
    B, F, D = model.B, model.F, model.D
    p = model.p
    D_diag = np.diag(D)
    
    # Define matrix-vector product
    def matvec(v):
        """
        Compute Σ @ v = B'FB @ v + D @ v efficiently.
        
        Breakdown:
        1. Bv: (k, p) @ (p,) = (k,) - O(kp)
        2. F(Bv): (k, k) @ (k,) = (k,) - O(k²)
        3. B'(F(Bv)): (p, k) @ (k,) = (p,) - O(kp)
        4. Dv: (p,) * (p,) = (p,) - O(p)
        
        Total: O(kp + k²) << O(p²) when k << p
        """
        return B.T @ (F @ (B @ v)) + D_diag * v
    
    # Create LinearOperator
    op = LinearOperator((p, p), matvec=matvec, dtype=float)
    
    # Solve eigenvalue problem
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(
            op, k=k_top, which='LM',  # LM = largest magnitude
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=True
        )
        
        if verbose:
            print(f"✓ ARPACK converged: {k_top}/{k_top} eigenvalues")
            
    except ArpackNoConvergence as e:
        # Partial convergence - use what we got
        warnings.warn(
            f"ARPACK did not fully converge. "
            f"Obtained {e.eigenvalues.shape[0]}/{k_top} eigenvalues. "
            f"Try increasing maxiter or decreasing tol.",
            RuntimeWarning
        )
        vals, vecs = e.eigenvalues, e.eigenvectors
        
        # Sort by magnitude
        idx = np.argsort(np.abs(vals))[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        
        if verbose:
            print(f"⚠ ARPACK partial: {len(vals)}/{k_top} eigenvalues")
    
    # eigsh returns ascending order; we want descending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    
    # Return eigenvectors as row vectors (k_top, p)
    return vals, vecs.T


class ImplicitEigenAnalysis(SimulationAnalysis):
    """
    Compute true eigenvalues via implicit LinearOperator.
    
    Compares eigenvalues (and optionally eigenvectors) of the ground
    truth covariance Σ_true = B'FB + D with eigenvalues from PCA
    decomposition of sample returns.
    
    Memory efficient: O(kp) instead of O(p²).
    
    Parameters
    ----------
    k_top : int, optional
        Number of eigenvalues to compute.
        If None, uses model.k.
    compare_eigenvectors : bool, default=False
        Whether to also compare eigenvector alignment.
        Adds principal angles and canonical correlations.
    tol : float, default=1e-10
        ARPACK convergence tolerance.
    maxiter : int, default=10000
        Maximum ARPACK iterations.
    
    Examples
    --------
    >>> from factor_lab.analyses import ImplicitEigenAnalysis
    >>> 
    >>> # Basic usage
    >>> analysis = ImplicitEigenAnalysis(k_top=10)
    >>> results = analysis.analyze(context)
    >>> 
    >>> print(f"Eigenvalue RMSE: {results['eigenvalue_rmse']:.6f}")
    >>> print(f"Max error: {results['eigenvalue_max_error']:.6f}")
    >>> 
    >>> # With eigenvector comparison
    >>> analysis = ImplicitEigenAnalysis(
    ...     k_top=5,
    ...     compare_eigenvectors=True
    ... )
    >>> results = analysis.analyze(context)
    >>> print(f"Eigenvector alignment: {results['eigenvector_mean_correlation']:.4f}")
    
    Notes
    -----
    **Eigenvalues vs Eigenvectors:**
    
    - Eigenvalues measure variance explained by each direction
    - Eigenvectors are the directions themselves
    
    For factor models:
    - True eigenvalues come from Σ = B'FB + D
    - Sample eigenvalues come from PCA on returns
    - Both should be similar if model is correctly specified
    
    **Error Sources:**
    
    1. **Sampling error**: Σ_sample ≠ Σ_true due to finite T
       - Decreases as O(1/√T)
       - Larger for smaller eigenvalues
       
    2. **Model misspecification**: If true model is not exactly
       a k-factor model, errors will persist even with T→∞
    """
    
    def __init__(
        self,
        k_top: Optional[int] = None,
        compare_eigenvectors: bool = False,
        tol: float = 1e-10,
        maxiter: int = 10000
    ):
        self.k_top = k_top
        self.compare_eigenvectors = compare_eigenvectors
        self.tol = tol
        self.maxiter = maxiter
    
    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Perform eigenvalue analysis.
        
        Parameters
        ----------
        context : SimulationContext
            Simulation context with model and returns.
        
        Returns
        -------
        dict
            true_eigenvalues : np.ndarray
                Eigenvalues of Σ_true (implicit computation).
            sample_eigenvalues : np.ndarray
                Eigenvalues from PCA on sample returns.
            eigenvalue_errors : np.ndarray
                Difference: true - sample.
            eigenvalue_relative_errors : np.ndarray
                Relative errors: (true - sample) / true.
            eigenvalue_rmse : float
                Root mean squared error.
            eigenvalue_max_error : float
                Maximum absolute error.
            eigenvalue_mean_relative_error : float
                Mean of absolute relative errors.
                
            If compare_eigenvectors=True, also includes:
            eigenvector_subspace_distance : float
                L2 norm of principal angles.
            eigenvector_principal_angles : np.ndarray
                Principal angles between eigenvector subspaces.
            eigenvector_canonical_correlations : np.ndarray
                Per-vector correlations.
            eigenvector_mean_correlation : float
                Mean correlation across all vectors.
        """
        k = self.k_top or context.model.k
        
        # Compute true eigenvalues implicitly
        true_evals, true_evecs = compute_true_eigenvalues(
            context.model, k,
            tol=self.tol,
            maxiter=self.maxiter
        )
        
        # Get sample eigenvalues from PCA
        pca_result = context.pca_decomposition(k)
        sample_evals = np.diag(pca_result.F)
        
        # Ensure same length (in case of partial convergence)
        n = min(len(true_evals), len(sample_evals))
        true_evals = true_evals[:n]
        sample_evals = sample_evals[:n]
        true_evecs = true_evecs[:n, :]
        
        # Compute errors
        errors = true_evals - sample_evals
        relative_errors = errors / np.abs(true_evals)
        
        results = {
            'true_eigenvalues': true_evals,
            'sample_eigenvalues': sample_evals,
            'eigenvalue_errors': errors,
            'eigenvalue_relative_errors': relative_errors,
            'eigenvalue_rmse': float(np.sqrt((errors ** 2).mean())),
            'eigenvalue_max_error': float(np.abs(errors).max()),
            'eigenvalue_mean_relative_error': float(np.abs(relative_errors).mean()),
        }
        
        # Optionally compare eigenvectors
        if self.compare_eigenvectors:
            sample_evecs = pca_result.B[:n, :]  # (k, p)
            evec_comp = self._compare_eigenvectors(true_evecs, sample_evecs)
            results.update(evec_comp)
        
        return results
    
    def _compare_eigenvectors(
        self,
        true_evecs: np.ndarray,
        sample_evecs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare eigenvector alignment.
        
        Note: These are eigenvectors of the covariance matrix Σ,
        not the factor loadings B!
        
        Eigenvectors of Σ are directions of maximum variance in
        the return space, while B are factor loadings.
        """
        # Compute subspace distance via principal angles
        angles = scipy.linalg.subspace_angles(true_evecs.T, sample_evecs.T)
        
        # Compute canonical correlations (per-vector similarity)
        k = true_evecs.shape[0]
        canonical_corrs = np.zeros(k)
        
        for i in range(k):
            # Correlation between i-th eigenvectors
            # Take absolute value to handle sign ambiguity
            corr = np.corrcoef(true_evecs[i, :], sample_evecs[i, :])[0, 1]
            canonical_corrs[i] = np.abs(corr)
        
        return {
            'eigenvector_subspace_distance': float(np.linalg.norm(angles)),
            'eigenvector_principal_angles': angles,
            'eigenvector_canonical_correlations': canonical_corrs,
            'eigenvector_mean_correlation': float(canonical_corrs.mean()),
        }
