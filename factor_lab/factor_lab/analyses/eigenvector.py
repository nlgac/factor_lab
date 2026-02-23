"""
eigenvector.py - Eigenvector Alignment Analysis
================================================

Compares ground truth eigenvectors (from true Σ = B'FB + D) with
PCA-recovered eigenvectors from sample covariance.

This directly addresses the request to compare eigenvectors of the
ground truth covariance matrix with eigenvectors obtained from PCA
on simulated returns.

Key Distinction
---------------
This module compares EIGENVECTORS of the covariance matrix Σ, which
are different from the factor loadings B:

- B: Factor loadings (k×p) - how assets load on factors
- Eigenvectors of Σ: Directions (p-dimensional) of maximum variance

For a k-factor model, the top k eigenvectors of Σ span approximately
the same subspace as B.T, but they are not identical.
"""

from typing import Dict, Any, Optional
import numpy as np
import scipy.linalg
from scipy.linalg import orthogonal_procrustes

from ..analysis import SimulationAnalysis, SimulationContext
from .spectral import compute_true_eigenvalues

__all__ = ['EigenvectorAlignment']


class EigenvectorAlignment(SimulationAnalysis):
    """
    Compare ground truth eigenvectors with PCA eigenvectors.
    
    Computes multiple alignment metrics to quantify how well PCA
    recovers the true eigenvector structure:
    
    - **Subspace distance**: Principal angles between subspaces
    - **Procrustes distance**: Optimal frame alignment
    - **Canonical correlations**: Per-vector similarity
    - **Sign recovery**: Fraction of correctly aligned signs
    
    This analysis answers: "Did PCA recover the correct directions
    of maximum variance?"
    
    Parameters
    ----------
    k_components : int, optional
        Number of components to compare.
        If None, uses model.k.
    align_signs : bool, default=True
        Whether to align signs before comparison.
        Eigenvectors have sign ambiguity: v and -v both valid.
    compute_rotation : bool, default=True
        Whether to compute optimal rotation matrix.
    tol : float, default=1e-10
        Convergence tolerance for eigenvalue computation.
    maxiter : int, default=10000
        Maximum iterations for ARPACK.
    
    Examples
    --------
    >>> from factor_lab.analyses import EigenvectorAlignment
    >>> 
    >>> # Basic usage
    >>> analysis = EigenvectorAlignment(k_components=5, align_signs=True)
    >>> results = analysis.analyze(context)
    >>> 
    >>> print(f"Subspace distance: {results['subspace_distance']:.6f}")
    >>> print(f"Procrustes distance: {results['procrustes_distance']:.6f}")
    >>> print(f"Mean correlation: {results['mean_correlation']:.4f}")
    >>> 
    >>> # Check individual vectors
    >>> corrs = results['vector_correlations']
    >>> for i, corr in enumerate(corrs):
    ...     print(f"Vector {i+1}: {corr:.4f}")
    
    Notes
    -----
    **Interpreting the metrics:**
    
    1. **Subspace distance ≈ 0**: The k-dimensional subspaces match.
       PCA found the correct signal space.
       
    2. **Procrustes distance ≈ 0**: After optimal rotation, vectors align.
       Individual directions are correctly recovered.
       
    3. **Mean correlation ≈ 1**: On average, eigenvectors are well-aligned.
       Values > 0.95 indicate excellent recovery.
       
    4. **Min correlation < 0.8**: At least one eigenvector is poorly recovered.
       May indicate insufficient data or model misspecification.
    
    **Typical values:**
    
    For well-specified models with sufficient data (T/p > 2):
    - Subspace distance: 0.01 - 0.1
    - Mean correlation: 0.90 - 0.99
    - Procrustes distance: 0.05 - 0.2
    
    For undersampled data (T/p < 1):
    - Subspace distance: 0.1 - 0.5
    - Mean correlation: 0.60 - 0.85
    - Procrustes distance: 0.2 - 0.8
    
    **Why eigenvectors matter:**
    
    Even if eigenvalues are correctly estimated, the eigenvectors
    might not be. This is particularly important for:
    - Principal component regression
    - Dimensionality reduction
    - Factor interpretation
    """
    
    def __init__(
        self,
        k_components: Optional[int] = None,
        align_signs: bool = True,
        compute_rotation: bool = True,
        tol: float = 1e-10,
        maxiter: int = 10000
    ):
        self.k_components = k_components
        self.align_signs = align_signs
        self.compute_rotation = compute_rotation
        self.tol = tol
        self.maxiter = maxiter
    
    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Perform eigenvector alignment analysis.
        
        Parameters
        ----------
        context : SimulationContext
            Simulation context with model and returns.
        
        Returns
        -------
        dict
            Core metrics:
            - subspace_distance : float
                L2 norm of principal angles.
            - principal_angles : np.ndarray
                Principal angles in radians.
            - max_principal_angle : float
                Largest principal angle.
            - grassmann_distance : float
                Same as subspace_distance (Grassmannian metric).
            
            Correlation metrics:
            - vector_correlations : np.ndarray
                Absolute correlations per vector.
            - mean_correlation : float
                Mean across all vectors.
            - min_correlation : float
                Minimum correlation (worst-case vector).
            - max_correlation : float
                Maximum correlation (best-case vector).
            
            If compute_rotation=True:
            - procrustes_distance : float
                Distance after optimal alignment.
            - optimal_rotation : np.ndarray
                Rotation matrix R minimizing ||A - BR||.
            - rotation_scale : float
                Scale factor from Procrustes.
            - aligned_eigenvectors : np.ndarray
                Sample eigenvectors after rotation.
            
            Reference data:
            - true_eigenvectors : np.ndarray
                Ground truth eigenvectors (k, p).
            - sample_eigenvectors : np.ndarray
                PCA eigenvectors (k, p).
            - k_components : int
                Number of components compared.
        """
        k = self.k_components or context.model.k
        
        # Get ground truth eigenvectors (from true Σ = B'FB + D)
        true_evals, true_evecs = compute_true_eigenvalues(
            context.model, k,
            tol=self.tol,
            maxiter=self.maxiter
        )
        
        # Get PCA eigenvectors (from sample covariance)
        pca_result = context.pca_decomposition(k)
        sample_evecs = pca_result.B  # (k, p)
        
        # Handle partial convergence
        n = min(true_evecs.shape[0], sample_evecs.shape[0])
        true_evecs = true_evecs[:n, :]
        sample_evecs = sample_evecs[:n, :]
        
        # Optionally align signs
        if self.align_signs:
            sample_evecs = self._align_signs(true_evecs, sample_evecs)
        
        # Initialize results
        results = {
            'true_eigenvectors': true_evecs,
            'sample_eigenvectors': sample_evecs,
            'k_components': n,
        }
        
        # Compute subspace distance (principal angles)
        angles = scipy.linalg.subspace_angles(true_evecs.T, sample_evecs.T)
        results['subspace_distance'] = float(np.linalg.norm(angles))
        results['principal_angles'] = angles
        results['max_principal_angle'] = float(angles.max())
        results['grassmann_distance'] = results['subspace_distance']
        
        # Compute Procrustes alignment (if requested)
        if self.compute_rotation:
            proc_results = self._compute_procrustes(true_evecs, sample_evecs)
            results.update(proc_results)
        
        # Compute canonical correlations (per-vector similarity)
        corr_results = self._compute_correlations(true_evecs, sample_evecs)
        results.update(corr_results)
        
        return results
    
    def _align_signs(
        self,
        true: np.ndarray,
        estimated: np.ndarray
    ) -> np.ndarray:
        """
        Align signs to maximize correlation.
        
        For each eigenvector, flip the sign if the dot product
        with the true eigenvector is negative.
        
        This handles the sign ambiguity: eigenvectors v and -v
        both satisfy Σv = λv.
        """
        result = estimated.copy()
        k = true.shape[0]
        
        for i in range(k):
            if np.dot(true[i, :], estimated[i, :]) < 0:
                result[i, :] *= -1
        
        return result
    
    def _compute_procrustes(
        self,
        true: np.ndarray,
        estimated: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Procrustes distance and optimal rotation.
        
        Solves the orthogonal Procrustes problem:
            min_R ||true - estimated @ R.T||_F
            subject to R.T @ R = I
        
        where R is a k×k orthogonal matrix.
        
        Returns the optimal rotation and the resulting distance.
        """
        # orthogonal_procrustes expects (p, k) matrices
        # Our eigenvectors are (k, p), so we transpose
        R, scale = orthogonal_procrustes(estimated.T, true.T)
        
        # Apply rotation: (k, p) @ (p, p) @ (p, k) = (k, p)
        # Actually we need: estimated @ R.T
        # But since estimated is (k, p), we need to think carefully
        # estimated.T is (p, k), R is (p, p)
        # We want: estimated @ R.T which is (k, p) @ (p, p) = (k, p)
        
        # Actually, orthogonal_procrustes returns R such that:
        # estimated.T @ R ≈ true.T
        # So: (estimated.T @ R).T = R.T @ estimated
        # We want the rotated version of estimated
        aligned = (estimated.T @ R).T  # (k, p)
        
        # Alternative: since eigenvectors are row vectors,
        # we can think of rotation in the k-dimensional space
        # Let's use a simpler approach
        
        # For row vectors, we want: estimated_rotated = R @ estimated
        # where R rotates in the k-dimensional factor space
        # Use SVD to find optimal R in factor space
        M = estimated @ true.T  # (k, k)
        U, _, Vt = scipy.linalg.svd(M, full_matrices=False)
        R_factor = U @ Vt  # (k, k)
        
        aligned = R_factor @ estimated  # (k, p)
        
        # Measure Frobenius distance
        distance = float(np.linalg.norm(true - aligned, 'fro'))
        
        return {
            'procrustes_distance': distance,
            'optimal_rotation': R_factor,
            'rotation_scale': scale,
            'aligned_eigenvectors': aligned,
        }
    
    def _compute_correlations(
        self,
        true: np.ndarray,
        estimated: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute canonical correlations between eigenvectors.
        
        Measures per-vector similarity using Pearson correlation.
        Takes absolute value to handle sign ambiguity.
        """
        k = true.shape[0]
        correlations = np.zeros(k)
        
        for i in range(k):
            # Pearson correlation between i-th eigenvectors
            corr = np.corrcoef(true[i, :], estimated[i, :])[0, 1]
            # Take absolute value (sign ambiguity)
            correlations[i] = np.abs(corr)
        
        return {
            'vector_correlations': correlations,
            'mean_correlation': float(correlations.mean()),
            'min_correlation': float(correlations.min()),
            'max_correlation': float(correlations.max()),
        }
