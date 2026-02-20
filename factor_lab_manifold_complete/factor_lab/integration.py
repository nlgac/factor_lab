"""
integration.py - Integration Layer for Refactored Design

Bridges FactorModelBuilder + ReturnsSimulator with existing manifold distance
analysis infrastructure.

This module provides the glue between the new two-class design and the existing
factor_lab_manifold_complete analysis framework.

Design
------
Complete pipeline in one function:
    Build Model → Simulate Returns → Estimate from Returns → Analyze Distances

Usage
-----
>>> from integration import build_simulate_analyze
>>> from distributions import create_sampler
>>> import numpy as np
>>> 
>>> rng = np.random.default_rng(42)
>>> factory = lambda name, **p: create_sampler(name, rng, **p)
>>> 
>>> results = build_simulate_analyze(
...     # Building parameters
...     p=100, k=2,
...     beta_samplers=[
...         factory("normal", loc=1.0, scale=0.2),
...         factory("student_t", df=5)
...     ],
...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
...     factor_variances=[0.04, 0.01],
...     # Simulation parameters
...     n_periods=1000,
...     factor_return_samplers=factory("normal", loc=0, scale=1),
...     idio_return_sampler=factory("normal", loc=0, scale=1),
...     # Analysis parameters
...     analyses=['manifold', 'eigenvalue', 'eigenvector'],
...     rng=rng
... )
>>> 
>>> # Access results
>>> print(f"Grassmannian distance: {results['grassmannian_distance']:.6f}")
>>> print(f"Procrustes distance: {results['procrustes_distance']:.6f}")
"""

from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import time
import numpy as np

# Import refactored components
from .model_builder import FactorModelBuilder
from .flexible_simulator import ReturnsSimulator as FlexibleReturnsSimulator
from .decomposition import svd_decomposition
from .factor_types import FactorModelData
from .distributions import Sampler

__all__ = [
    'build_simulate_analyze',
    'create_simulation_context',
    'run_analyses',
]


def build_simulate_analyze(
    # Model building parameters
    p: int,
    k: int,
    beta_samplers: Union[Sampler, List[Sampler]],
    idio_vol_sampler: Sampler,
    factor_variances: List[float],
    # Simulation parameters
    n_periods: int,
    factor_return_samplers: Union[Sampler, List[Sampler]],
    idio_return_sampler: Sampler,
    # Analysis parameters
    analyses: Optional[List[str]] = None,
    # Optional parameters
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline: Build → Simulate → Estimate → Analyze
    
    This is the main entry point for the integrated system. It combines
    the new refactored design (FactorModelBuilder + ReturnsSimulator) with
    the existing analysis framework (manifold distances, etc.).
    
    Parameters
    ----------
    p : int
        Number of assets/securities
    k : int
        Number of factors
    beta_samplers : Sampler or List[Sampler]
        Distribution(s) for factor loadings
    idio_vol_sampler : Sampler
        Distribution for idiosyncratic volatilities
    factor_variances : List[float]
        Variance for each factor (diagonal of F)
    n_periods : int
        Number of time periods to simulate
    factor_return_samplers : Sampler or List[Sampler]
        Distribution(s) for factor returns
    idio_return_sampler : Sampler
        Distribution for idiosyncratic returns
    analyses : List[str], optional
        Which analyses to run. Options:
        - 'manifold': Grassmannian, Procrustes, Chordal distances
        - 'eigenvalue': Eigenvalue analysis via implicit operator
        - 'eigenvector': Eigenvector alignment analysis
        - 'all': Run all analyses
        If None, defaults to ['manifold']
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    verbose : bool, default=True
        Whether to print progress messages
    
    Returns
    -------
    results : dict
        Dictionary containing:
        
        **Core Results**:
        - 'true_model': FactorModelData (built model)
        - 'estimated_model': FactorModelData (from SVD of returns)
        - 'simulation_results': dict from ReturnsSimulator
        - 'context': SimulationContext (for further analysis)
        
        **Analysis Results** (depending on `analyses` parameter):
        
        If 'manifold' in analyses:
        - 'grassmannian_distance': float
        - 'procrustes_distance': float
        - 'chordal_distance': float
        
        If 'eigenvalue' in analyses:
        - 'true_eigenvalues': array
        - 'sample_eigenvalues': array
        - 'eigenvalue_errors': array
        
        If 'eigenvector' in analyses:
        - 'eigenvector_correlations': array
        - 'mean_correlation': float
        - 'procrustes_distance_eigvec': float
        - 'principal_angles': array
        
        **Metadata**:
        - 'duration': float (total time in seconds)
        - 'timestamp': datetime
    
    Examples
    --------
    Basic usage with normal distributions:
        >>> from distributions import create_sampler
        >>> rng = np.random.default_rng(42)
        >>> factory = lambda name, **p: create_sampler(name, rng, **p)
        >>> 
        >>> results = build_simulate_analyze(
        ...     p=100, k=2,
        ...     beta_samplers=factory("normal", loc=0, scale=1),
        ...     idio_vol_sampler=factory("constant", value=0.03),
        ...     factor_variances=[0.04, 0.01],
        ...     n_periods=1000,
        ...     factor_return_samplers=factory("normal", loc=0, scale=1),
        ...     idio_return_sampler=factory("normal", loc=0, scale=1),
        ...     rng=rng
        ... )
        >>> print(f"Grassmannian: {results['grassmannian_distance']:.6f}")
    
    Heavy-tailed distributions:
        >>> results = build_simulate_analyze(
        ...     p=100, k=2,
        ...     beta_samplers=[
        ...         factory("normal", loc=1.0, scale=0.2),
        ...         factory("student_t", df=5)
        ...     ],
        ...     idio_vol_sampler=factory("uniform", low=0.02, high=0.05),
        ...     factor_variances=[0.04, 0.01],
        ...     n_periods=1000,
        ...     factor_return_samplers=factory("student_t", df=5),
        ...     idio_return_sampler=factory("student_t", df=7),
        ...     analyses=['all'],
        ...     rng=rng
        ... )
    
    Reuse model with different return distributions:
        >>> # Build model once
        >>> builder = FactorModelBuilder(rng=rng)
        >>> model = builder.build(...)
        >>> 
        >>> # Simulate with different return distributions
        >>> for dist_name, samplers in experiments.items():
        ...     results = build_simulate_analyze_from_model(
        ...         model=model,
        ...         n_periods=1000,
        ...         factor_return_samplers=samplers['factor'],
        ...         idio_return_sampler=samplers['idio'],
        ...         analyses=['manifold'],
        ...         rng=rng
        ...     )
        ...     print(f"{dist_name}: {results['grassmannian_distance']:.6f}")
    
    Notes
    -----
    This function is the primary integration point between the refactored
    design and the existing analysis framework. It:
    
    1. Uses FactorModelBuilder to create the model
    2. Uses ReturnsSimulator to simulate returns
    3. Uses svd_decomposition to estimate model from returns
    4. Creates SimulationContext for analysis
    5. Runs requested analyses using existing framework
    
    The function is designed to be compatible with the existing
    factor_lab_manifold_complete infrastructure while using the new
    two-class design internally.
    """
    start_time = time.time()
    timestamp = datetime.now()
    
    if analyses is None:
        analyses = ['manifold']
    elif 'all' in analyses:
        analyses = ['manifold', 'eigenvalue', 'eigenvector']
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Step 1: Build model
    if verbose:
        print(f"Building model: p={p}, k={k}...")
    
    builder = FactorModelBuilder(rng=rng)
    true_model = builder.build(
        p=p,
        k=k,
        beta_samplers=beta_samplers,
        idio_vol_sampler=idio_vol_sampler,
        factor_variances=factor_variances
    )
    
    if verbose:
        print(f"  ✓ Model built")
    
    # Step 2: Simulate returns
    if verbose:
        print(f"Simulating {n_periods} periods...")
    
    simulator = FlexibleFlexibleFlexibleFlexibleReturnsSimulator(rng=rng)
    sim_results = simulator.simulate(
        model=true_model,
        n_periods=n_periods,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler
    )
    
    if verbose:
        print(f"  ✓ Simulation complete")
    
    # Step 3: Estimate model from returns (SVD)
    if verbose:
        print(f"Estimating model from returns (SVD)...")
    
    estimated_model = svd_decomposition(
        sim_results['security_returns'],
        k=k,
        center=True
    )
    
    if verbose:
        print(f"  ✓ Model estimated")
    
    # Step 4: Create SimulationContext
    if verbose:
        print(f"Creating context for analysis...")
    
    context = create_simulation_context(
        model=true_model,
        sim_results=sim_results,
        timestamp=timestamp
    )
    
    # Step 5: Run analyses
    if verbose:
        print(f"Running analyses: {analyses}...")
    
    analysis_results = run_analyses(
        context=context,
        analyses=analyses,
        verbose=verbose
    )
    
    # Compile final results
    duration = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Complete in {duration:.2f}s")
    
    results = {
        # Core results
        'true_model': true_model,
        'estimated_model': estimated_model,
        'simulation_results': sim_results,
        'context': context,
        # Metadata
        'duration': duration,
        'timestamp': timestamp,
    }
    
    # Add analysis results
    results.update(analysis_results)
    
    return results


def build_simulate_analyze_from_model(
    model: FactorModelData,
    n_periods: int,
    factor_return_samplers: Union[Sampler, List[Sampler]],
    idio_return_sampler: Sampler,
    analyses: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run simulation and analysis from an existing model.
    
    This is useful when you want to reuse the same model with different
    return distributions.
    
    Parameters
    ----------
    model : FactorModelData
        Pre-built model (from FactorModelBuilder)
    n_periods : int
        Number of periods to simulate
    factor_return_samplers : Sampler or List[Sampler]
        Distribution(s) for factor returns
    idio_return_sampler : Sampler
        Distribution for idiosyncratic returns
    analyses : List[str], optional
        Which analyses to run
    rng : np.random.Generator, optional
        Random number generator
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    results : dict
        Same structure as build_simulate_analyze()
    
    Examples
    --------
    >>> # Build model once
    >>> builder = FactorModelBuilder(rng=rng)
    >>> model = builder.build(...)
    >>> 
    >>> # Experiment with different return distributions
    >>> results_normal = build_simulate_analyze_from_model(
    ...     model, 1000,
    ...     factory("normal", loc=0, scale=1),
    ...     factory("normal", loc=0, scale=1)
    ... )
    >>> 
    >>> results_t = build_simulate_analyze_from_model(
    ...     model, 1000,
    ...     factory("student_t", df=5),
    ...     factory("student_t", df=7)
    ... )
    """
    start_time = time.time()
    timestamp = datetime.now()
    
    if analyses is None:
        analyses = ['manifold']
    elif 'all' in analyses:
        analyses = ['manifold', 'eigenvalue', 'eigenvector']
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Step 1: Simulate returns
    if verbose:
        print(f"Simulating {n_periods} periods...")
    
    simulator = FlexibleFlexibleFlexibleFlexibleReturnsSimulator(rng=rng)
    sim_results = simulator.simulate(
        model=model,
        n_periods=n_periods,
        factor_return_samplers=factor_return_samplers,
        idio_return_sampler=idio_return_sampler
    )
    
    if verbose:
        print(f"  ✓ Simulation complete")
    
    # Step 2: Estimate model from returns
    if verbose:
        print(f"Estimating model from returns...")
    
    estimated_model = svd_decomposition(
        sim_results['security_returns'],
        k=model.k,
        center=True
    )
    
    if verbose:
        print(f"  ✓ Model estimated")
    
    # Step 3: Create context
    context = create_simulation_context(
        model=model,
        sim_results=sim_results,
        timestamp=timestamp
    )
    
    # Step 4: Run analyses
    if verbose:
        print(f"Running analyses: {analyses}...")
    
    analysis_results = run_analyses(
        context=context,
        analyses=analyses,
        verbose=verbose
    )
    
    duration = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Complete in {duration:.2f}s")
    
    return {
        'true_model': model,
        'estimated_model': estimated_model,
        'simulation_results': sim_results,
        'context': context,
        'duration': duration,
        'timestamp': timestamp,
        **analysis_results
    }


def create_simulation_context(
    model: FactorModelData,
    sim_results: Dict[str, np.ndarray],
    timestamp: Optional[datetime] = None
):
    """
    Create SimulationContext from model and simulation results.
    
    This bridges the refactored design output to the existing analysis
    framework input.
    
    Parameters
    ----------
    model : FactorModelData
        The true model
    sim_results : dict
        Results from ReturnsSimulator.simulate()
    timestamp : datetime, optional
        When simulation was run
    
    Returns
    -------
    SimulationContext
        Context ready for analysis
    
    Notes
    -----
    This function requires the factor_lab package to be available.
    It imports SimulationContext from factor_lab.analysis.
    """
    try:
        from factor_lab.analysis import SimulationContext
    except ImportError:
        raise ImportError(
            "Cannot create SimulationContext: factor_lab package not found.\n"
            "Make sure factor_lab_manifold_complete is in your Python path."
        )
    
    return SimulationContext(
        model=model,
        security_returns=sim_results['security_returns'],
        factor_returns=sim_results['factor_returns'],
        idio_returns=sim_results['idio_returns'],
        timestamp=timestamp or datetime.now(),
        duration=0.0  # Will be set by caller
    )


def run_analyses(
    context,
    analyses: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run requested analyses on a SimulationContext.
    
    Parameters
    ----------
    context : SimulationContext
        Context containing model and simulation results
    analyses : List[str]
        Which analyses to run:
        - 'manifold': Manifold distances
        - 'eigenvalue': Eigenvalue analysis
        - 'eigenvector': Eigenvector alignment
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    results : dict
        Combined results from all requested analyses
    
    Notes
    -----
    This function requires the factor_lab.analyses package.
    """
    try:
        from factor_lab.analyses import Analyses
    except ImportError:
        raise ImportError(
            "Cannot run analyses: factor_lab.analyses not found.\n"
            "Make sure factor_lab_manifold_complete is in your Python path."
        )
    
    all_results = {}
    
    if 'manifold' in analyses:
        if verbose:
            print("  Running manifold distance analysis...")
        manifold_analysis = Analyses.manifold_distances()
        manifold_results = manifold_analysis.analyze(context)
        all_results.update(manifold_results)
        
        if verbose:
            print(f"    Grassmannian: {manifold_results['dist_grassmannian']:.6f}")
            print(f"    Procrustes: {manifold_results['dist_procrustes']:.6f}")
            print(f"    Chordal: {manifold_results['dist_chordal']:.6f}")
    
    if 'eigenvalue' in analyses:
        if verbose:
            print("  Running eigenvalue analysis...")
        eigen_analysis = Analyses.eigenvalue_analysis(
            k_top=context.k,
            compare_eigenvectors=False
        )
        eigen_results = eigen_analysis.analyze(context)
        all_results.update(eigen_results)
        
        if verbose:
            if 'true_eigenvalues' in eigen_results:
                print(f"    Top eigenvalue: {eigen_results['true_eigenvalues'][0]:.6f}")
    
    if 'eigenvector' in analyses:
        if verbose:
            print("  Running eigenvector alignment analysis...")
        eigvec_analysis = Analyses.eigenvector_comparison(
            k=context.k,
            align_signs=True
        )
        eigvec_results = eigvec_analysis.analyze(context)
        all_results.update(eigvec_results)
        
        if verbose:
            if 'mean_correlation' in eigvec_results:
                print(f"    Mean correlation: {eigvec_results['mean_correlation']:.4f}")
    
    return all_results
