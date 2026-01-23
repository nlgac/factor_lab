"""
Synthetic Data Generation - Clean Design
=========================================

A well-structured framework for generating synthetic factor models with
configurable parameters. Uses dataclasses for type safety, builder pattern
for flexibility, and clean separation of concerns.

Key design principles:
- Immutable configuration objects (frozen dataclasses)
- Type-safe with full type hints
- Separation of concerns (config, generation, analysis, presentation)
- DRY - no code duplication
- Pure functions where possible
- Builder pattern for easy construction
- Easy to test and extend

Author: factor_lab
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from factor_lab import (
    DataSampler,
    DistributionFactory,
    ReturnsSimulator,
    CovarianceValidator,
    FactorModelData
)


# =============================================================================
# CONFIGURATION LAYER - Immutable, Type-Safe
# =============================================================================

@dataclass(frozen=True)
class DistSpec:
    """
    Distribution specification - immutable, type-safe.
    Utilizes the Specification Pattern for flexible config.
    
    Examples
    --------
    >>> DistSpec('normal', mean=1.0, std=0.5)
    >>> DistSpec('constant', value=0.18)
    >>> DistSpec.normal(1.0, 0.5)  # Convenience constructor
    """
    dist_name: str
    params: Dict[str, float] = field(default_factory=dict)
    
    def __init__(self, dist_name: str, **params):
        """
        Allow kwargs for convenient construction.
        A little hacky since frozen, but works fine.
        """
        object.__setattr__(self, 'dist_name', dist_name)
        object.__setattr__(self, 'params', params)
    
    def create_sampler(self, factory: DistributionFactory):
        """Create sampler from factory."""
        return factory.create(self.dist_name, **self.params)
    
    # Convenience constructors
    @classmethod
    def normal(cls, mean: float, std: float) -> 'DistSpec':
        return cls('normal', mean=mean, std=std)
    
    @classmethod
    def constant(cls, value: float) -> 'DistSpec':
        return cls('constant', value=value)
    
    @classmethod
    def student_t(cls, df: float) -> 'DistSpec':
        return cls('student_t', df=df)
    
    @classmethod
    def uniform(cls, low: float, high: float) -> 'DistSpec':
        return cls('uniform', low=low, high=high)


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Complete specification for a synthetic scenario.
    
    Immutable, validated, serializable configuration.
    
    Attributes
    ----------
    name : str
        Scenario name
    n_assets : int
        Number of assets
    n_factors : int
        Number of factors
    betas : List[DistSpec]
        Factor loading specifications (one per factor)
    factor_vols : List[DistSpec]
        Factor volatility specifications (one per factor)
    idio_vol : DistSpec
        Idiosyncratic volatility specification
    description : str
        Optional description
    """
    name: str
    n_assets: int
    n_factors: int
    betas: Tuple[DistSpec, ...]  # Tuple for immutability
    factor_vols: Tuple[DistSpec, ...]
    idio_vol: DistSpec
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.betas) != self.n_factors:
            raise ValueError(f"Need {self.n_factors} beta specs, got {len(self.betas)}")
        if len(self.factor_vols) != self.n_factors:
            raise ValueError(f"Need {self.n_factors} factor_vol specs, got {len(self.factor_vols)}")
    
    def to_dict(self) -> dict:
        """Export as dictionary for serialization."""
        return {
            'name': self.name,
            'n_assets': self.n_assets,
            'n_factors': self.n_factors,
            'betas': [(b.dist_name, b.params) for b in self.betas],
            'factor_vols': [(f.dist_name, f.params) for f in self.factor_vols],
            'idio_vol': (self.idio_vol.dist_name, self.idio_vol.params),
            'description': self.description
        }


# Builder for ergonomic construction
class ScenarioBuilder:
    """
    Fluent builder for ScenarioSpec.
    
    Examples
    --------
    >>> spec = (ScenarioBuilder('Crisis', n_assets=100, n_factors=2)
    ...     .beta(DistSpec.normal(1.0, 0.5))
    ...     .beta(DistSpec.normal(0.0, 1.0))
    ...     .factor_vol(DistSpec.constant(0.18))
    ...     .factor_vol(DistSpec.constant(0.10))
    ...     .idio_vol(DistSpec.constant(0.40))
    ...     .describe("High vol crisis scenario")
    ...     .build())
    """
    
    def __init__(self, name: str, n_assets: int, n_factors: int):
        self.name = name
        self.n_assets = n_assets
        self.n_factors = n_factors
        self._betas: List[DistSpec] = []
        self._factor_vols: List[DistSpec] = []
        self._idio_vol: Optional[DistSpec] = None
        self._description: str = ""
    
    def beta(self, spec: DistSpec) -> 'ScenarioBuilder':
        """Add a beta distribution spec."""
        self._betas.append(spec)
        return self
    
    def factor_vol(self, spec: DistSpec) -> 'ScenarioBuilder':
        """Add a factor vol distribution spec."""
        self._factor_vols.append(spec)
        return self
    
    def idio_vol(self, spec: DistSpec) -> 'ScenarioBuilder':
        """Set idiosyncratic vol distribution spec."""
        self._idio_vol = spec
        return self
    
    def describe(self, description: str) -> 'ScenarioBuilder':
        """Set description."""
        self._description = description
        return self
    
    def build(self) -> ScenarioSpec:
        """Build and validate the specification."""
        if self._idio_vol is None:
            raise ValueError("Must set idio_vol")
        
        return ScenarioSpec(
            name=self.name,
            n_assets=self.n_assets,
            n_factors=self.n_factors,
            betas=tuple(self._betas),
            factor_vols=tuple(self._factor_vols),
            idio_vol=self._idio_vol,
            description=self._description
        )


# =============================================================================
# GENERATION LAYER - Pure Functions
# =============================================================================

def generate_model(
    spec: ScenarioSpec,
    factory: DistributionFactory,
    rng: np.random.Generator
) -> FactorModelData:
    """
    Generate model from specification (pure function).
    
    Parameters
    ----------
    spec : ScenarioSpec
        Scenario specification
    factory : DistributionFactory
        Distribution factory
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    FactorModelData
        Generated model
    """
    sampler = DataSampler(p=spec.n_assets, k=spec.n_factors, rng=rng)
    
    model = sampler.configure(
        beta=[b.create_sampler(factory) for b in spec.betas],
        factor_vol=[f.create_sampler(factory) for f in spec.factor_vols],
        idio_vol=spec.idio_vol.create_sampler(factory)
    ).generate()
    
    return model


def simulate_returns(
    model: FactorModelData,
    rng: np.random.Generator,
    factory: DistributionFactory,
    n_periods: int = 5000,
    innovation: str = 'student_t'
) -> Dict[str, np.ndarray]:
    """
    Simulate returns from model (pure function).
    
    Parameters
    ----------
    model : FactorModelData
        Factor model
    rng : np.random.Generator
        Random number generator
    factory : DistributionFactory
        Distribution factory
    n_periods : int
        Number of periods
    innovation : str
        Innovation distribution ('student_t' or 'normal')
    
    Returns
    -------
    dict
        Simulation results with keys: security_returns, factor_returns, idio_returns
    """
    simulator = ReturnsSimulator(model, rng=rng)
    
    if innovation == 'student_t':
        factor_samplers = [factory.create('student_t', df=5) for _ in range(model.k)]
        idio_samplers = [factory.create('student_t', df=6) for _ in range(model.p)]
    else:  # normal
        factor_samplers = [factory.create('normal', mean=0, std=1) for _ in range(model.k)]
        idio_samplers = [factory.create('normal', mean=0, std=1) for _ in range(model.p)]
    
    return simulator.simulate(
        n_periods=n_periods,
        factor_samplers=factor_samplers,
        idio_samplers=idio_samplers
    )


# =============================================================================
# ANALYSIS LAYER - Extract Statistics
# =============================================================================

@dataclass(frozen=True)
class ModelStats:
    """Statistics about a factor model."""
    n_factors: int
    n_assets: int
    factor_vols: np.ndarray
    mean_idio_vol: float
    beta_ranges: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class SimulationStats:
    """Statistics about simulated returns."""
    mean_return: float
    volatility: float
    annualized_vol: float
    return_range: Tuple[float, float]
    asset_mean_range: Tuple[float, float]
    asset_vol_range: Tuple[float, float]
    mean_correlation: float
    correlation_range: Tuple[float, float]
    frobenius_error: float
    mean_absolute_error: float
    max_absolute_error: float
    quality: str  # 'excellent', 'good', or 'poor'


def compute_model_stats(model: FactorModelData) -> ModelStats:
    """Compute statistics from a model (pure function)."""
    factor_vols = np.sqrt(np.diag(model.F))
    mean_idio_vol = np.sqrt(np.diag(model.D).mean())
    beta_ranges = tuple((model.B[i].min(), model.B[i].max()) for i in range(model.k))
    
    return ModelStats(
        n_factors=model.k,
        n_assets=model.p,
        factor_vols=factor_vols,
        mean_idio_vol=mean_idio_vol,
        beta_ranges=beta_ranges
    )


def compute_simulation_stats(
    returns: np.ndarray,
    model: FactorModelData
) -> SimulationStats:
    """Compute statistics from simulated returns (pure function)."""
    # Basic stats
    mean_return = returns.mean()
    volatility = returns.std()
    annualized_vol = volatility * np.sqrt(252)
    return_range = (returns.min(), returns.max())
    
    # Per-asset stats
    asset_means = returns.mean(axis=0)
    asset_vols = returns.std(axis=0)
    asset_mean_range = (asset_means.min(), asset_means.max())
    asset_vol_range = (asset_vols.min(), asset_vols.max())
    
    # Correlation
    corr = np.corrcoef(returns, rowvar=False)
    n = model.p
    mean_corr = (corr.sum() - n) / (n * (n - 1))
    upper_tri = np.triu_indices_from(corr, k=1)
    corr_range = (corr[upper_tri].min(), corr[upper_tri].max())
    
    # Validation
    validator = CovarianceValidator(model)
    validation = validator.compare(returns)
    
    quality = ('excellent' if validation.frobenius_error < 0.10
               else 'good' if validation.frobenius_error < 0.20
               else 'poor')
    
    return SimulationStats(
        mean_return=mean_return,
        volatility=volatility,
        annualized_vol=annualized_vol,
        return_range=return_range,
        asset_mean_range=asset_mean_range,
        asset_vol_range=asset_vol_range,
        mean_correlation=mean_corr,
        correlation_range=corr_range,
        frobenius_error=validation.frobenius_error,
        mean_absolute_error=validation.mean_absolute_error,
        max_absolute_error=validation.max_absolute_error,
        quality=quality
    )


# =============================================================================
# PRESENTATION LAYER - Format Output
# =============================================================================

def print_scenario_header(spec: ScenarioSpec):
    """Print scenario header."""
    print("\n" + "=" * 70)
    print(f"Scenario: {spec.name}")
    if spec.description:
        print(f"{spec.description}")
    print("=" * 70)


def print_model_stats(stats: ModelStats):
    """Print model statistics."""
    print(f"\nModel Structure:")
    print(f"  Factors: {stats.n_factors}, Assets: {stats.n_assets}")
    print(f"  Factor vols: {stats.factor_vols}")
    print(f"  Mean idio vol: {stats.mean_idio_vol:.4f}")
    for i, (low, high) in enumerate(stats.beta_ranges):
        print(f"  Factor {i} beta range: [{low:.2f}, {high:.2f}]")


def print_simulation_stats(stats: SimulationStats, n_periods: int):
    """Print simulation statistics."""
    print(f"\nSimulation ({n_periods} periods):")
    print(f"  Mean return: {stats.mean_return:.6f}")
    print(f"  Volatility: {stats.volatility:.6f}")
    print(f"  Annualized vol: {stats.annualized_vol:.4f}")
    print(f"  Return range: [{stats.return_range[0]:.6f}, {stats.return_range[1]:.6f}]")
    
    print(f"\nPer-Asset:")
    print(f"  Mean range: [{stats.asset_mean_range[0]:.6f}, {stats.asset_mean_range[1]:.6f}]")
    print(f"  Vol range: [{stats.asset_vol_range[0]:.6f}, {stats.asset_vol_range[1]:.6f}]")
    
    print(f"\nCorrelation:")
    print(f"  Mean: {stats.mean_correlation:.4f}")
    print(f"  Range: [{stats.correlation_range[0]:.4f}, {stats.correlation_range[1]:.4f}]")
    
    print(f"\nValidation:")
    print(f"  Frobenius error: {stats.frobenius_error:.6f}")
    print(f"  Mean absolute error: {stats.mean_absolute_error:.6f}")
    print(f"  Max absolute error: {stats.max_absolute_error:.6f}")
    
    icon = '✓' if stats.quality in ('excellent', 'good') else '⚠'
    print(f"  Quality: {icon} {stats.quality.capitalize()}")


# =============================================================================
# PRE-DEFINED SCENARIOS
# =============================================================================

def crisis_scenario() -> ScenarioSpec:
    """Financial crisis: high volatility, concentrated risk."""
    return (ScenarioBuilder('Crisis', n_assets=100, n_factors=2)
        .beta(DistSpec.normal(1.0, 0.5))
        .beta(DistSpec.normal(0.0, 1.0))
        .factor_vol(DistSpec.constant(0.18))
        .factor_vol(DistSpec.constant(0.10))
        .idio_vol(DistSpec.constant(0.40))
        .describe("High vol, concentrated loadings")
        .build())


def calm_scenario() -> ScenarioSpec:
    """Calm market: low volatility, diversified."""
    return (ScenarioBuilder('Calm', n_assets=100, n_factors=2)
        .beta(DistSpec.normal(0.8, 0.2))
        .beta(DistSpec.normal(0.0, 0.3))
        .factor_vol(DistSpec.constant(0.12))
        .factor_vol(DistSpec.constant(0.08))
        .idio_vol(DistSpec.uniform(0.08, 0.15))
        .describe("Low vol, stable correlations")
        .build())


# =============================================================================
# ORCHESTRATION
# =============================================================================

@dataclass(frozen=True)
class ScenarioResult:
    """Complete result for a scenario."""
    spec: ScenarioSpec
    model: FactorModelData
    model_stats: ModelStats
    simulation_results: Dict[str, np.ndarray]
    simulation_stats: SimulationStats


def run_scenario(
    spec: ScenarioSpec,
    factory: DistributionFactory,
    rng: np.random.Generator,
    n_periods: int = 5000,
    innovation: str = 'student_t',
    verbose: bool = True
) -> ScenarioResult:
    """
    Run complete scenario: generate, simulate, analyze.
    
    Parameters
    ----------
    spec : ScenarioSpec
        Scenario specification
    factory : DistributionFactory
        Distribution factory
    rng : np.random.Generator
        Random number generator
    n_periods : int
        Simulation periods
    innovation : str
        Innovation distribution
    verbose : bool
        Print results
    
    Returns
    -------
    ScenarioResult
        Complete results including spec, model, and statistics
    """
    # Generate
    model = generate_model(spec, factory, rng)
    model_stats = compute_model_stats(model)
    
    # Simulate
    sim_results = simulate_returns(model, rng, factory, n_periods, innovation)
    sim_stats = compute_simulation_stats(sim_results['security_returns'], model)
    
    # Print
    if verbose:
        print_scenario_header(spec)
        print_model_stats(model_stats)
        print_simulation_stats(sim_stats, n_periods)
    
    return ScenarioResult(
        spec=spec,
        model=model,
        model_stats=model_stats,
        simulation_results=sim_results,
        simulation_stats=sim_stats
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run synthetic data generation example."""
    print("=" * 70)
    print("Synthetic Data Generation - Clean Design")
    print("=" * 70)
    
    # Setup
    rng = np.random.default_rng(42)
    factory = DistributionFactory(rng=rng)
    
    # Define scenarios
    scenarios = {
        'crisis': crisis_scenario(),
        'calm': calm_scenario(),
    }
    
    # Run all scenarios
    results = {
        name: run_scenario(spec, factory, rng, n_periods=5000)
        for name, spec in scenarios.items()
    }
    
    print("\n" + "=" * 70)
    print("Generation Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\nAccess results:")
    print("  >>> results['crisis'].spec  # Configuration")
    print("  >>> results['crisis'].model  # Factor model")
    print("  >>> results['crisis'].model_stats  # Model statistics")
    print("  >>> results['crisis'].simulation_stats  # Simulation statistics")
    print("  >>> results['crisis'].spec.to_dict()  # Export config")
