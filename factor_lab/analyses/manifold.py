"""
manifold.py - Geometric Distance Metrics for Factor Models
===========================================================

Implements distances on Grassmannian and Stiefel manifolds for comparing
factor model estimates. This code is based on Gemini's implementation in
build_and_simulate.py with enhancements for reusability and testing.

The key insight is that factor models have inherent ambiguities:
- Factors can be rotated (any orthogonal transformation)
- Factors can be sign-flipped
- Factors can be permuted

Manifold geometry provides rotation-invariant comparison metrics.

References
----------
Edelman, A., Arias, T. A., & Smith, S. T. (1998).
The geometry of algorithms with orthogonality constraints.
SIAM journal on Matrix Analysis and Applications, 20(2), 303-353.
"""

from typing import Callable, Dict, Any, Tuple, Optional
import numpy as np
import scipy.linalg

from ..analysis import SimulationAnalysis, SimulationContext

__all__ = [
    'orthonormalize',
    'compute_grassmannian_distance',
    'compute_procrustes_distance',
    'compute_chordal_distance',
    'compute_sine_alignment',
    'register_manifold_distance',
    'ManifoldDistanceAnalysis',
]


def orthonormalize(B: np.ndarray) -> np.ndarray:
    """
    Orthonormalize factor loadings via QR decomposition.
    
    Projects the factor loadings onto the Stiefel manifold V_{p,k},
    the space of p×k matrices with orthonormal columns.
    
    Parameters
    ----------
    B : np.ndarray, shape (k, p)
        Factor loadings (k factors, p assets).
    
    Returns
    -------
    Q : np.ndarray, shape (p, k)
        Orthonormal frame with Q.T @ Q = I_k.
    
    Examples
    --------
    >>> B = np.random.randn(3, 100)
    >>> Q = orthonormalize(B)
    >>> np.allclose(Q.T @ Q, np.eye(3))
    True
    """
    Q, _ = scipy.linalg.qr(B.T, mode='economic')
    return Q


def compute_grassmannian_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute distance on Grassmannian manifold Gr(k, p).
    
    The Grassmannian distance measures how similar two k-dimensional
    subspaces of R^p are. It is invariant to rotation, sign flips,
    and permutation of factors.
    
    The distance is defined as the L2 norm of the principal angles
    between the two subspaces.
    
    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth factor loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated factor loadings from PCA/SVD.
    
    Returns
    -------
    distance : float
        Grassmannian distance (L2 norm of principal angles).
    angles : np.ndarray, shape (k,)
        Principal angles between subspaces in radians.
    
    Notes
    -----
    The Grassmannian Gr(k, p) is the space of k-dimensional subspaces
    of R^p. Two subspaces span the same space if and only if their
    Grassmannian distance is zero.
    
    Principal angles θ_i ∈ [0, π/2] satisfy:
    - θ_i = 0 means the i-th directions are aligned
    - θ_i = π/2 means the i-th directions are orthogonal
    
    Examples
    --------
    >>> B = np.random.randn(3, 100)
    >>> # Rotated version (same subspace)
    >>> Q = scipy.stats.ortho_group.rvs(3)
    >>> B_rot = Q @ B
    >>> dist, angles = compute_grassmannian_distance(B, B_rot)
    >>> dist < 1e-10
    True
    """
    Q_true = orthonormalize(B_true)
    Q_estimated = orthonormalize(B_estimated)
    
    angles = scipy.linalg.subspace_angles(Q_true, Q_estimated)
    distance = float(np.linalg.norm(angles))
    
    return distance, angles


def compute_procrustes_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Procrustes distance on Stiefel manifold V_{p,k}.
    
    Finds the optimal orthogonal rotation R that aligns B_estimated
    to B_true, then measures the Frobenius distance after alignment.
    
    This handles sign flips and permutations inherent in SVD/PCA,
    where factors can be recovered up to an orthogonal transformation.
    
    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth factor loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated factor loadings from PCA/SVD.
    
    Returns
    -------
    dict
        distance : float
            Frobenius distance after optimal alignment:
            ||Q_true - Q_estimated @ R||_F
        optimal_rotation : np.ndarray, shape (k, k)
            Optimal orthogonal rotation matrix R.
        aligned_frame : np.ndarray, shape (p, k)
            Estimated frame after applying R: Q_estimated @ R.
    
    Notes
    -----
    Solves the orthogonal Procrustes problem:
        min_R ||Q_true - Q_estimated @ R||_F^2
        subject to R.T @ R = I
    
    The solution is R = U @ V.T where U, S, V.T = svd(Q_estimated.T @ Q_true).
    
    This is the "best case" distance after finding the optimal alignment.
    
    Examples
    --------
    >>> B = np.random.randn(3, 100)
    >>> B_flipped = B.copy()
    >>> B_flipped[0, :] *= -1  # Flip first factor
    >>> result = compute_procrustes_distance(B, B_flipped)
    >>> result['distance'] < 1e-10  # Should be ~0 after alignment
    True
    """
    Q_true = orthonormalize(B_true)
    Q_estimated = orthonormalize(B_estimated)
    
    # Solve for optimal rotation via SVD of the cross-product matrix
    M = Q_estimated.T @ Q_true
    U, _, Vt = scipy.linalg.svd(M, full_matrices=False)
    R_opt = U @ Vt
    
    # Apply rotation
    Q_aligned = Q_estimated @ R_opt
    
    # Measure Frobenius distance after alignment
    distance = float(np.linalg.norm(Q_true - Q_aligned, 'fro'))
    
    return {
        'distance': distance,
        'optimal_rotation': R_opt,
        'aligned_frame': Q_aligned,
    }


def compute_chordal_distance(
    B_true: np.ndarray,
    B_estimated: np.ndarray
) -> float:
    """
    Compute chordal distance on Stiefel manifold.
    
    Direct Frobenius distance between orthonormalized frames.
    Sensitive to rotation, sign flips, and permutation.
    
    This is the "worst case" distance without any alignment.
    Compare to Procrustes distance to see the benefit of alignment.
    
    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Ground truth factor loadings.
    B_estimated : np.ndarray, shape (k, p)
        Estimated factor loadings from PCA/SVD.
    
    Returns
    -------
    float
        Chordal distance: ||Q_true - Q_estimated||_F
    
    Notes
    -----
    Unlike Procrustes distance, this does NOT attempt to align
    the frames before measuring distance.
    
    Examples
    --------
    >>> B = np.random.randn(3, 100)
    >>> Q = scipy.stats.ortho_group.rvs(3)
    >>> B_rot = Q @ B
    >>> # Chordal distance will be large despite same subspace
    >>> dist = compute_chordal_distance(B, B_rot)
    >>> dist > 0.1
    True
    >>> # But Grassmannian distance is ~0
    >>> grass_dist, _ = compute_grassmannian_distance(B, B_rot)
    >>> grass_dist < 1e-10
    True
    """
    Q_true = orthonormalize(B_true)
    Q_estimated = orthonormalize(B_estimated)
    
    return float(np.linalg.norm(Q_true - Q_estimated, 'fro'))


def compute_sine_alignment(
    B_true: np.ndarray,
    B_estimated: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Per-factor sin²∠(hⱼ, bⱼ) with j matched by row order.

    Computes sin²∠(hⱼ, bⱼ) for j = 1, …, k, where the j-th row of
    B_true is population direction bⱼ and the j-th row of B_estimated
    is sample direction hⱼ.  Matching is by **row order** (eigenvalue
    rank), not by optimal alignment — this is what Theorem 1, Eq. (20)
    predicts.

    This differs from the Grassmannian distance, which uses principal
    angles (optimally matched).  d_Gr ≤ d_sine always; equality holds
    iff the rank-order matching is already optimal.

    Parameters
    ----------
    B_true : np.ndarray, shape (k, p)
        Population loading directions (rows need not be unit-norm).
    B_estimated : np.ndarray, shape (k, p)
        Sample loading directions (rows need not be unit-norm).

    Returns
    -------
    sin2 : np.ndarray, shape (k,)
        sin²∠(hⱼ, bⱼ) for each factor j, values in [0, 1].
    dist_sine : float
        Aggregate scalar sqrt(Σⱼ sin²∠(hⱼ, bⱼ)).

    Examples
    --------
    >>> B = np.random.randn(3, 100)
    >>> sin2, d = compute_sine_alignment(B, B)   # perfect alignment
    >>> np.allclose(sin2, 0.0)
    True
    """
    b    = B_true      / np.linalg.norm(B_true,      axis=1, keepdims=True)
    h    = B_estimated / np.linalg.norm(B_estimated, axis=1, keepdims=True)
    cos2 = np.einsum('jp,jp->j', b, h) ** 2      # (bⱼ · hⱼ)²; sign-invariant
    sin2 = np.clip(1.0 - cos2, 0.0, 1.0)
    return sin2, float(np.sqrt(float(sin2.sum())))


# Extra scalar distances registered at runtime.
# Signature: (B_true: ndarray, B_estimated: ndarray) -> float
# Add new metrics here — nowhere else.
_EXTRA_DISTANCES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}


def register_manifold_distance(
    result_key: str,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    """Register an extra scalar distance for ManifoldDistanceAnalysis.

    Example:
        register_manifold_distance('dist_my_metric', my_distance_fn)
        # → 'dist_my_metric' appears in every subsequent analyze() result
    """
    _EXTRA_DISTANCES[result_key] = fn


class ManifoldDistanceAnalysis(SimulationAnalysis):
    """
    Compare factor loadings using manifold geometry.
    
    Computes three complementary distance metrics:
    
    1. **Grassmannian Distance** - Measures if the subspaces match.
       Invariant to rotation, sign flips, permutation.
       
    2. **Procrustes Distance** - Measures frame similarity after
       optimal alignment. Handles sign flips and rotations.
       
    3. **Chordal Distance** - Raw frame difference without alignment.
       Sensitive to all transformations.
    
    Parameters
    ----------
    use_pca_loadings : bool, default=True
        If True, compare against PCA-extracted loadings from sample.
        If False, uses model.B directly (for testing/validation).
    
    Attributes
    ----------
    use_pca_loadings : bool
        Whether to extract loadings via PCA.
    
    Examples
    --------
    >>> from factor_lab.analyses import ManifoldDistanceAnalysis
    >>> 
    >>> analysis = ManifoldDistanceAnalysis()
    >>> results = analysis.analyze(context)
    >>> 
    >>> print(f"Subspace match: {results['dist_grassmannian']:.6f}")
    >>> print(f"Frame match (aligned): {results['dist_procrustes']:.6f}")
    >>> print(f"Frame match (raw): {results['dist_chordal']:.6f}")
    
    Notes
    -----
    **Interpreting the distances:**
    
    - If `dist_grassmannian ≈ 0`: The subspaces match perfectly.
      PCA recovered the correct signal space.
      
    - If `dist_procrustes ≈ 0` but `dist_chordal > 0`: Factors are
      recovered correctly but with sign flips or permutations.
      
    - If all distances > 0: Estimation error or insufficient data.
    
    **Typical values for simulated data:**
    
    - T=100, p=50, k=3: dist_grassmannian ~ 0.1-0.3
    - T=500, p=50, k=3: dist_grassmannian ~ 0.01-0.05
    - T=1000, p=50, k=3: dist_grassmannian ~ 0.005-0.02
    """
    
    def __init__(self, use_pca_loadings: bool = True):
        self.use_pca_loadings = use_pca_loadings
    
    def analyze(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Perform manifold distance analysis.
        
        Parameters
        ----------
        context : SimulationContext
            Simulation context containing model and returns.
        
        Returns
        -------
        dict
                Distance on Grassmannian manifold.
            dist_procrustes : float
                Procrustes distance (after alignment).
            dist_chordal : float
                Chordal distance (without alignment).
            principal_angles : np.ndarray
                Principal angles between subspaces.
            optimal_rotation : np.ndarray
                Optimal rotation matrix from Procrustes.
            aligned_frame : np.ndarray
                Estimated frame after alignment.
        """
        # Get ground truth loadings
        B_true = context.model.B  # (k, p)
        
        # Get estimated loadings
        if self.use_pca_loadings:
            pca_result = context.pca_decomposition(context.model.k)
            B_estimated = pca_result.B  # (k, p)
        else:
            # For testing: compare model to itself
            B_estimated = context.model.B
        
        # Compute all three distances
        dist_grass, angles = compute_grassmannian_distance(B_true, B_estimated)
        procrustes_result = compute_procrustes_distance(B_true, B_estimated)
        dist_chordal = compute_chordal_distance(B_true, B_estimated)
        
        results = {
            'dist_grassmannian': dist_grass,
            'dist_procrustes': procrustes_result['distance'],
            'dist_chordal': dist_chordal,
            'principal_angles': angles,
            'optimal_rotation': procrustes_result['optimal_rotation'],
            'aligned_frame': procrustes_result['aligned_frame'],
        }
        for key, fn in _EXTRA_DISTANCES.items():
            results[key] = fn(B_true, B_estimated)
        return results
