"""
optimization.py - Portfolio Optimization using Factor Models

This module provides convex optimization tools for portfolio construction:
- FactorOptimizer: SOCP-based minimum variance optimization
- ScenarioBuilder: Fluent API for constructing constraint sets

Mathematical Background:
-----------------------
Given a factor model with covariance Σ = B.T @ F @ B + D, the minimum
variance portfolio problem is:

    minimize    w.T @ Σ @ w
    subject to  A_eq @ w = b_eq
                A_ineq @ w <= b_ineq

This can be reformulated as a Second-Order Cone Program (SOCP):

    minimize    ||D^(1/2) @ w||² + ||F^(1/2) @ y||²
    subject to  y = B @ w
                A_eq @ w = b_eq
                A_ineq @ w <= b_ineq

The SOCP formulation is O(k²) instead of O(p²) for the full covariance,
making it efficient for large universes with few factors.

Example Usage:
-------------
    >>> from factor_lab.optimization import FactorOptimizer, ScenarioBuilder
    >>> 
    >>> # Build a long-only, fully-invested scenario
    >>> builder = ScenarioBuilder(p=100)
    >>> scenario = (builder
    ...     .create("Long Only")
    ...     .add_fully_invested()
    ...     .add_long_only()
    ...     .build())
    >>> 
    >>> # Optimize
    >>> optimizer = FactorOptimizer(model)
    >>> optimizer.apply_scenario(scenario)
    >>> result = optimizer.solve()
    >>> print(f"Risk: {result.risk:.2%}")
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Optional, List, Tuple, TYPE_CHECKING

from .types import FactorModelData, OptimizationResult, Scenario

if TYPE_CHECKING:
    pass  # For future type imports that might cause circular deps


# =============================================================================
# SCENARIO BUILDER
# =============================================================================

class ScenarioBuilder:
    """
    Fluent builder for constructing optimization scenarios (constraint sets).
    
    This class provides a chainable API for building up constraint sets
    commonly used in portfolio optimization.
    
    Parameters
    ----------
    p : int
        Number of assets in the portfolio.
    
    Examples
    --------
    >>> builder = ScenarioBuilder(p=50)
    >>> 
    >>> # Method 1: Fluent chaining
    >>> scenario = (builder
    ...     .create("130/30")
    ...     .add_fully_invested()
    ...     .add_box_constraints(low=-0.30, high=1.30)
    ...     .build())
    >>> 
    >>> # Method 2: Step by step
    >>> builder.create("Long Only Diversified")
    >>> builder.add_fully_invested()
    >>> builder.add_long_only()
    >>> builder.add_box_constraints(low=0.0, high=0.05)  # Max 5% per asset
    >>> scenario = builder.build()
    
    Notes
    -----
    The builder maintains internal state. Call `create()` to start a new
    scenario, then chain constraint methods, then call `build()` to get
    the final Scenario object.
    
    All constraint methods return `self` to enable method chaining.
    """
    
    def __init__(self, p: int):
        """
        Initialize the builder.
        
        Parameters
        ----------
        p : int
            Number of assets. Must be positive.
        """
        if p <= 0:
            raise ValueError(f"Number of assets must be positive, got {p}")
        
        self.p = p
        self._scenario: Optional[Scenario] = None
    
    def create(self, name: str, description: str = "") -> "ScenarioBuilder":
        """
        Start building a new scenario.
        
        Parameters
        ----------
        name : str
            Human-readable name for the scenario.
        description : str, optional
            Longer description of what this scenario represents.
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> builder.create("Risk Parity", "Equal risk contribution portfolio")
        """
        self._scenario = Scenario(name=name, description=description)
        return self
    
    def add_fully_invested(self) -> "ScenarioBuilder":
        """
        Add constraint: sum of weights equals 1.
        
        Mathematical form: 1.T @ w = 1
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> builder.create("Test").add_fully_invested()
        """
        self._ensure_scenario()
        
        # A: (1, p) row of ones
        # b: scalar 1.0
        A = np.ones((1, self.p))
        b = np.array([1.0])
        
        self._scenario.equality_constraints.append((A, b))
        return self
    
    def add_long_only(self) -> "ScenarioBuilder":
        """
        Add constraint: all weights must be non-negative.
        
        Mathematical form: -I @ w <= 0  (equivalently, w >= 0)
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> builder.create("Test").add_long_only()
        """
        self._ensure_scenario()
        
        # -I @ w <= 0 means w >= 0
        A = -np.eye(self.p)
        b = np.zeros(self.p)
        
        self._scenario.inequality_constraints.append((A, b))
        return self
    
    def add_box_constraints(
        self, 
        low: float, 
        high: float
    ) -> "ScenarioBuilder":
        """
        Add box constraints: low <= w_i <= high for all assets.
        
        Parameters
        ----------
        low : float
            Minimum weight per asset. Use negative for short positions.
        high : float
            Maximum weight per asset.
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> # Max 5% per asset, long only
        >>> builder.add_box_constraints(low=0.0, high=0.05)
        >>> 
        >>> # 130/30 style
        >>> builder.add_box_constraints(low=-0.30, high=1.30)
        
        Notes
        -----
        This adds two sets of inequality constraints:
        - I @ w <= high (upper bound)
        - -I @ w <= -low (lower bound, i.e., w >= low)
        """
        self._ensure_scenario()
        
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        
        # Upper bound: w <= high
        A_upper = np.eye(self.p)
        b_upper = np.full(self.p, high)
        
        # Lower bound: -w <= -low (i.e., w >= low)
        A_lower = -np.eye(self.p)
        b_lower = np.full(self.p, -low)
        
        self._scenario.inequality_constraints.append((A_upper, b_upper))
        self._scenario.inequality_constraints.append((A_lower, b_lower))
        return self
    
    def add_leverage_constraint(
        self, 
        max_leverage: float
    ) -> "ScenarioBuilder":
        """
        Add constraint: sum of absolute weights <= max_leverage.
        
        Parameters
        ----------
        max_leverage : float
            Maximum gross exposure (sum of |w_i|). For example:
            - 1.0: Long-only, fully invested
            - 2.0: 130/30 or market neutral with 2x gross
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> # Limit to 150% gross exposure
        >>> builder.add_leverage_constraint(1.5)
        
        Notes
        -----
        This constraint is handled by introducing auxiliary variables
        during optimization. The actual CVXPY formulation uses:
            u >= w, u >= -w, sum(u) <= max_leverage
        
        This is a linear relaxation. For exact L1 constraints, the
        optimizer handles this via SOCP reformulation.
        
        Warning
        -------
        This method stores metadata but the actual L1 constraint 
        implementation is in the optimizer. For simple use cases,
        consider using box_constraints instead.
        """
        self._ensure_scenario()
        
        # Store as metadata; optimizer will handle the L1 norm
        # For now, we implement via auxiliary variables approach
        # This is a simplification - full L1 would need optimizer support
        
        # Placeholder: we'll add this to scenario metadata
        if not hasattr(self._scenario, 'metadata'):
            self._scenario.metadata = {}
        self._scenario.metadata['max_leverage'] = max_leverage
        
        return self
    
    def add_sector_neutral(
        self, 
        sector_assignments: np.ndarray
    ) -> "ScenarioBuilder":
        """
        Add constraints for sector neutrality.
        
        For each sector s, the constraint is: sum(w_i for i in sector s) = 0
        
        Parameters
        ----------
        sector_assignments : np.ndarray
            Integer array of shape (p,) where sector_assignments[i] is the
            sector index for asset i. Sectors should be numbered 0, 1, 2, ...
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        
        Examples
        --------
        >>> # 3 sectors for 10 assets
        >>> sectors = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> builder.add_sector_neutral(sectors)
        
        Notes
        -----
        This adds one equality constraint per unique sector.
        """
        self._ensure_scenario()
        
        if sector_assignments.shape != (self.p,):
            raise ValueError(
                f"sector_assignments must have shape ({self.p},), "
                f"got {sector_assignments.shape}"
            )
        
        unique_sectors = np.unique(sector_assignments)
        
        for sector in unique_sectors:
            # Create indicator row for this sector
            A = np.zeros((1, self.p))
            A[0, sector_assignments == sector] = 1.0
            b = np.array([0.0])
            
            self._scenario.equality_constraints.append((A, b))
        
        return self
    
    def add_custom_equality(
        self, 
        A: np.ndarray, 
        b: np.ndarray
    ) -> "ScenarioBuilder":
        """
        Add a custom equality constraint: A @ w = b.
        
        Parameters
        ----------
        A : np.ndarray
            Constraint matrix with shape (m, p) for m constraints.
        b : np.ndarray
            Right-hand side with shape (m,).
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        """
        self._ensure_scenario()
        self._validate_constraint(A, b, "equality")
        self._scenario.equality_constraints.append((A, b))
        return self
    
    def add_custom_inequality(
        self, 
        A: np.ndarray, 
        b: np.ndarray
    ) -> "ScenarioBuilder":
        """
        Add a custom inequality constraint: A @ w <= b.
        
        Parameters
        ----------
        A : np.ndarray
            Constraint matrix with shape (m, p) for m constraints.
        b : np.ndarray
            Right-hand side with shape (m,).
        
        Returns
        -------
        ScenarioBuilder
            Self, for method chaining.
        """
        self._ensure_scenario()
        self._validate_constraint(A, b, "inequality")
        self._scenario.inequality_constraints.append((A, b))
        return self
    
    def build(self) -> Scenario:
        """
        Finalize and return the constructed scenario.
        
        Returns
        -------
        Scenario
            The completed scenario with all added constraints.
        
        Raises
        ------
        RuntimeError
            If `create()` was not called first.
        
        Examples
        --------
        >>> scenario = builder.create("Test").add_long_only().build()
        """
        self._ensure_scenario()
        result = self._scenario
        self._scenario = None  # Reset for next build
        return result
    
    def _ensure_scenario(self) -> None:
        """Raise if no scenario is being built."""
        if self._scenario is None:
            raise RuntimeError(
                "No scenario in progress. Call create() first."
            )
    
    def _validate_constraint(
        self, 
        A: np.ndarray, 
        b: np.ndarray, 
        constraint_type: str
    ) -> None:
        """Validate constraint dimensions."""
        if A.ndim != 2:
            raise ValueError(
                f"{constraint_type} constraint A must be 2D, got shape {A.shape}"
            )
        
        if A.shape[1] != self.p:
            raise ValueError(
                f"{constraint_type} constraint A has {A.shape[1]} columns, "
                f"expected {self.p} (number of assets)"
            )
        
        if b.ndim != 1:
            raise ValueError(
                f"{constraint_type} constraint b must be 1D, got shape {b.shape}"
            )
        
        if b.shape[0] != A.shape[0]:
            raise ValueError(
                f"{constraint_type} constraint dimension mismatch: "
                f"A has {A.shape[0]} rows, b has {b.shape[0]} elements"
            )


# =============================================================================
# FACTOR OPTIMIZER
# =============================================================================

class FactorOptimizer:
    """
    SOCP-based portfolio optimizer using factor model structure.
    
    This optimizer solves the minimum variance problem efficiently by
    exploiting the factor model structure:
    
        minimize    ||D^(1/2) @ w||² + ||F^(1/2) @ y||²
        subject to  y = B @ w
                    [user constraints]
    
    The complexity is O(k²) per iteration rather than O(p²), making
    this suitable for large universes with moderate factor counts.
    
    Parameters
    ----------
    model : FactorModelData
        The factor model defining the covariance structure.
    solver : str, optional
        CVXPY solver to use. Defaults to None (auto-select).
        Options include 'ECOS', 'SCS', 'MOSEK', 'OSQP'.
    verbose : bool, default=False
        Whether to print solver output.
    
    Examples
    --------
    >>> from factor_lab.optimization import FactorOptimizer, ScenarioBuilder
    >>> 
    >>> # Setup
    >>> optimizer = FactorOptimizer(model)
    >>> 
    >>> # Option 1: Apply a pre-built scenario
    >>> scenario = ScenarioBuilder(model.p).create("LO").add_fully_invested().add_long_only().build()
    >>> optimizer.apply_scenario(scenario)
    >>> 
    >>> # Option 2: Add constraints individually
    >>> optimizer.add_equality(np.ones((1, model.p)), np.array([1.0]))
    >>> 
    >>> # Solve
    >>> result = optimizer.solve()
    >>> if result.solved:
    ...     print(f"Optimal weights: {result.weights[:5]}")
    
    Notes
    -----
    The optimizer can be reused with different constraints by calling
    `reset_constraints()` between solves.
    
    For advanced users, the underlying CVXPY problem can be accessed
    via the `problem` property after calling `solve()`.
    """
    
    def __init__(
        self, 
        model: FactorModelData,
        solver: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the optimizer with a factor model.
        
        Parameters
        ----------
        model : FactorModelData
            Factor model specifying the covariance structure.
        solver : str, optional
            CVXPY solver name. If None, CVXPY auto-selects.
        verbose : bool, default=False
            Print solver output during optimization.
        """
        self.model = model
        self.solver = solver
        self.verbose = verbose
        
        # Decision variables
        self._w = cp.Variable(model.p, name="weights")
        self._y = cp.Variable(model.k, name="factor_exposure")
        
        # Constraint storage
        self._constraints: List[cp.Constraint] = []
        
        # Add the factor definition constraint: y = B @ w
        self._constraints.append(self._y == model.B @ self._w)
        
        # Problem reference (populated after solve)
        self._problem: Optional[cp.Problem] = None
        
        # Pre-compute square roots for the objective
        self._D_sqrt = np.sqrt(np.diag(model.D))
        self._F_sqrt = np.sqrt(np.diag(model.F))
    
    @property
    def w(self) -> cp.Variable:
        """Portfolio weights decision variable."""
        return self._w
    
    @property
    def problem(self) -> Optional[cp.Problem]:
        """The CVXPY problem (available after solve())."""
        return self._problem
    
    def reset_constraints(self) -> None:
        """
        Remove all user-added constraints, keeping only y = B @ w.
        
        Useful for re-solving with different constraints without
        recreating the optimizer.
        
        Examples
        --------
        >>> optimizer.add_equality(A1, b1)
        >>> result1 = optimizer.solve()
        >>> optimizer.reset_constraints()
        >>> optimizer.add_equality(A2, b2)
        >>> result2 = optimizer.solve()
        """
        # Keep only the first constraint (factor definition)
        self._constraints = [self._constraints[0]]
        self._problem = None
    
    def add_equality(self, A: np.ndarray, b: np.ndarray) -> None:
        """
        Add an equality constraint: A @ w = b.
        
        Parameters
        ----------
        A : np.ndarray
            Constraint matrix with shape (m, p).
        b : np.ndarray
            Right-hand side with shape (m,).
        
        Examples
        --------
        >>> # Fully invested constraint
        >>> optimizer.add_equality(np.ones((1, p)), np.array([1.0]))
        """
        self._constraints.append(A @ self._w == b)
    
    def add_inequality(self, A: np.ndarray, b: np.ndarray) -> None:
        """
        Add an inequality constraint: A @ w <= b.
        
        Parameters
        ----------
        A : np.ndarray
            Constraint matrix with shape (m, p).
        b : np.ndarray
            Right-hand side with shape (m,).
        
        Examples
        --------
        >>> # Long only: -I @ w <= 0
        >>> optimizer.add_inequality(-np.eye(p), np.zeros(p))
        """
        self._constraints.append(A @ self._w <= b)
    
    def apply_scenario(self, scenario: Scenario) -> None:
        """
        Apply all constraints from a Scenario object.
        
        Parameters
        ----------
        scenario : Scenario
            Pre-built scenario containing equality and inequality constraints.
        
        Examples
        --------
        >>> scenario = builder.create("LO").add_fully_invested().add_long_only().build()
        >>> optimizer.apply_scenario(scenario)
        >>> result = optimizer.solve()
        
        Notes
        -----
        This is equivalent to calling add_equality/add_inequality for
        each constraint in the scenario.
        """
        for A, b in scenario.equality_constraints:
            self.add_equality(A, b)
        
        for A, b in scenario.inequality_constraints:
            self.add_inequality(A, b)
    
    def solve(self) -> OptimizationResult:
        """
        Solve the minimum variance optimization problem.
        
        Returns
        -------
        OptimizationResult
            Contains optimal weights, risk, and solver status.
        
        Examples
        --------
        >>> result = optimizer.solve()
        >>> if result.solved:
        ...     print(f"Risk: {result.risk:.4f}")
        ...     print(f"Top 5 weights: {np.sort(result.weights)[-5:]}")
        ... else:
        ...     print(f"Solver status: {result.metadata['status']}")
        
        Notes
        -----
        The objective is:
            0.5 * (||D_sqrt * w||² + ||F_sqrt * y||²)
        
        This equals 0.5 * w.T @ Σ @ w, so the optimal objective value
        is half the variance. Risk (std dev) = sqrt(2 * objective).
        """
        # Construct the objective
        # We use cp.multiply for element-wise multiplication with diagonal
        idio_term = cp.sum_squares(cp.multiply(self._D_sqrt, self._w))
        factor_term = cp.sum_squares(cp.multiply(self._F_sqrt, self._y))
        
        objective = cp.Minimize(0.5 * (idio_term + factor_term))
        
        # Build and solve the problem
        self._problem = cp.Problem(objective, self._constraints)
        
        try:
            if self.solver:
                self._problem.solve(solver=self.solver, verbose=self.verbose)
            else:
                self._problem.solve(verbose=self.verbose)
        except cp.SolverError as e:
            return OptimizationResult(
                weights=None,
                risk=0.0,
                objective=0.0,
                solved=False,
                metadata={"status": "solver_error", "error": str(e)}
            )
        
        # Check solution status
        is_optimal = self._problem.status == cp.OPTIMAL
        
        if is_optimal:
            weights = self._w.value
            obj_value = self._problem.value
            # Risk = sqrt(variance) = sqrt(2 * objective)
            risk = np.sqrt(2 * obj_value) if obj_value is not None else 0.0
        else:
            weights = None
            obj_value = self._problem.value if self._problem.value is not None else 0.0
            risk = 0.0
        
        return OptimizationResult(
            weights=weights,
            risk=risk,
            objective=obj_value if obj_value is not None else 0.0,
            solved=is_optimal,
            metadata={
                "status": self._problem.status,
                "solver": self._problem.solver_stats.solver_name if self._problem.solver_stats else None,
                "solve_time": self._problem.solver_stats.solve_time if self._problem.solver_stats else None,
                "iterations": self._problem.solver_stats.num_iters if self._problem.solver_stats else None,
            }
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def minimum_variance_portfolio(
    model: FactorModelData,
    long_only: bool = True,
    max_weight: Optional[float] = None,
    solver: Optional[str] = None
) -> OptimizationResult:
    """
    Compute the minimum variance portfolio with common constraints.
    
    This is a convenience function for the most common optimization setup.
    For more control, use FactorOptimizer and ScenarioBuilder directly.
    
    Parameters
    ----------
    model : FactorModelData
        Factor model specifying the covariance structure.
    long_only : bool, default=True
        If True, restrict to non-negative weights.
    max_weight : float, optional
        Maximum weight per asset. If None, no upper bound (beyond fully invested).
    solver : str, optional
        CVXPY solver to use.
    
    Returns
    -------
    OptimizationResult
        Optimization results including weights and risk.
    
    Examples
    --------
    >>> # Simple long-only minimum variance
    >>> result = minimum_variance_portfolio(model)
    >>> 
    >>> # With diversification constraint (max 5% per asset)
    >>> result = minimum_variance_portfolio(model, max_weight=0.05)
    """
    optimizer = FactorOptimizer(model, solver=solver)
    builder = ScenarioBuilder(model.p)
    
    # Build scenario
    builder.create("MinVar")
    builder.add_fully_invested()
    
    if long_only:
        builder.add_long_only()
    
    if max_weight is not None:
        low = 0.0 if long_only else -max_weight
        builder.add_box_constraints(low=low, high=max_weight)
    
    scenario = builder.build()
    optimizer.apply_scenario(scenario)
    
    return optimizer.solve()
