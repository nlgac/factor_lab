"""
Backtest Framework Example
===========================

This example demonstrates how to build a complete backtesting framework
using factor_lab. We'll implement:
1. Rolling factor model estimation
2. Portfolio rebalancing
3. Transaction cost modeling
4. Performance attribution
5. Risk metrics

Author: factor_lab
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from factor_lab import (
    svd_decomposition,
    FactorOptimizer,
    ScenarioBuilder,
    compute_explained_variance
)


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    lookback_days: int = 252  # 1 year for model estimation
    rebalance_days: int = 21  # Monthly rebalancing
    transaction_cost_bps: float = 5.0  # 5 bps per trade
    n_factors: int = 3
    max_weight: float = 0.10  # 10% max per asset
    

@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    transaction_costs: float
    n_rebalances: int


def generate_synthetic_returns(T=1500, p=50, k=3, seed=42):
    """
    Generate synthetic returns with time-varying volatility.
    
    This simulates realistic market conditions with regime changes.
    """
    rng = np.random.default_rng(seed)
    
    # Generate returns with two regimes
    regime_1_periods = T // 2  # Low vol regime
    regime_2_periods = T - regime_1_periods  # High vol regime
    
    # Regime 1: Normal market
    factors_1 = rng.standard_normal((regime_1_periods, k))
    factors_1[:, 0] *= 0.15  # Market vol: 15%
    factors_1[:, 1] *= 0.08  # Size vol: 8%
    factors_1[:, 2] *= 0.05  # Value vol: 5%
    
    # Regime 2: Crisis (higher vol)
    factors_2 = rng.standard_normal((regime_2_periods, k))
    factors_2[:, 0] *= 0.30  # Market vol: 30%
    factors_2[:, 1] *= 0.15  # Size vol: 15%
    factors_2[:, 2] *= 0.10  # Value vol: 10%
    
    factors = np.vstack([factors_1, factors_2])
    
    # Static loadings
    loadings = rng.standard_normal((k, p))
    
    # Idiosyncratic with time-varying vol
    idio = rng.standard_normal((T, p))
    idio[:regime_1_periods] *= 0.05  # Low idio vol
    idio[regime_1_periods:] *= 0.10  # Higher idio vol
    
    returns = factors @ loadings + idio
    
    return returns


class MinimumVarianceStrategy:
    """
    Simple minimum variance strategy with factor models.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.models = []  # Store estimated models
        
    def estimate_model(self, returns_window: np.ndarray):
        """Estimate factor model from returns window."""
        model = svd_decomposition(returns_window, k=self.config.n_factors)
        self.models.append(model)
        return model
    
    def optimize_portfolio(self, model):
        """Optimize portfolio given factor model."""
        optimizer = FactorOptimizer(model)
        builder = ScenarioBuilder(model.p)
        
        scenario = (builder
            .create("MinVar")
            .add_fully_invested()
            .add_long_only()
            .add_box_constraints(low=0.0, high=self.config.max_weight)
            .build())
        
        optimizer.apply_scenario(scenario)
        result = optimizer.solve()
        
        if result.solved:
            return result.weights
        else:
            # Fallback to equal weight
            return np.ones(model.p) / model.p


def run_backtest(returns: np.ndarray, config: BacktestConfig, verbose: bool = True):
    """
    Run a complete backtest of the minimum variance strategy.
    
    Parameters
    ----------
    returns : np.ndarray
        Full returns history (T, p)
    config : BacktestConfig
        Backtest configuration
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Backtest results including portfolio weights, returns, and metrics
    """
    T, p = returns.shape
    strategy = MinimumVarianceStrategy(config)
    
    # Storage
    portfolio_weights = np.zeros((T, p))
    portfolio_returns = np.zeros(T)
    transaction_costs = np.zeros(T)
    rebalance_dates = []
    
    # Initial position: equal weight
    current_weights = np.ones(p) / p
    portfolio_weights[0] = current_weights
    
    if verbose:
        print("=" * 70)
        print("Running Backtest")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Total periods: {T}")
        print(f"  Assets: {p}")
        print(f"  Lookback: {config.lookback_days} days")
        print(f"  Rebalance: Every {config.rebalance_days} days")
        print(f"  Transaction cost: {config.transaction_cost_bps} bps")
        print(f"  Max weight: {config.max_weight:.1%}")
    
    # Main backtest loop
    for t in range(config.lookback_days, T):
        # Check if it's a rebalance date
        days_since_start = t - config.lookback_days
        is_rebalance = (days_since_start % config.rebalance_days == 0)
        
        if is_rebalance:
            # Estimate model using lookback window
            returns_window = returns[t - config.lookback_days:t]
            model = strategy.estimate_model(returns_window)
            
            # Optimize
            new_weights = strategy.optimize_portfolio(model)
            
            # Calculate turnover and costs
            turnover = np.abs(new_weights - current_weights).sum()
            cost = turnover * (config.transaction_cost_bps / 10000)
            
            transaction_costs[t] = cost
            current_weights = new_weights
            rebalance_dates.append(t)
            
            if verbose and len(rebalance_dates) % 10 == 0:
                print(f"  Rebalance {len(rebalance_dates)}: t={t}, "
                      f"turnover={turnover:.2%}, cost={cost:.4%}")
        
        # Record weights
        portfolio_weights[t] = current_weights
        
        # Calculate portfolio return (before costs)
        if t < T:
            portfolio_returns[t] = np.dot(current_weights, returns[t])
    
    # Subtract transaction costs from returns
    net_returns = portfolio_returns - transaction_costs
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        net_returns[config.lookback_days:],
        transaction_costs[config.lookback_days:],
        config
    )
    
    if verbose:
        print(f"\n✓ Backtest complete: {len(rebalance_dates)} rebalances")
    
    return {
        'weights': portfolio_weights,
        'gross_returns': portfolio_returns,
        'net_returns': net_returns,
        'transaction_costs': transaction_costs,
        'rebalance_dates': rebalance_dates,
        'metrics': metrics,
        'models': strategy.models
    }


def calculate_performance_metrics(returns: np.ndarray, 
                                  costs: np.ndarray,
                                  config: BacktestConfig) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    
    # Basic returns
    cumulative_returns = np.cumprod(1 + returns) - 1
    total_return = cumulative_returns[-1]
    
    # Annualized metrics
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Turnover and costs
    total_turnover = (costs > 0).sum()  # Number of rebalances
    total_costs = costs.sum()
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        turnover=total_turnover / n_years,  # Rebalances per year
        transaction_costs=total_costs,
        n_rebalances=int(total_turnover)
    )


def print_performance_report(results: dict):
    """Print formatted performance report."""
    metrics = results['metrics']
    
    print("\n" + "=" * 70)
    print("Performance Report")
    print("=" * 70)
    
    print("\nReturn Metrics:")
    print(f"  Total Return:        {metrics.total_return:>10.2%}")
    print(f"  Annualized Return:   {metrics.annualized_return:>10.2%}")
    print(f"  Annualized Vol:      {metrics.annualized_vol:>10.2%}")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
    
    print("\nRisk Metrics:")
    print(f"  Maximum Drawdown:    {metrics.max_drawdown:>10.2%}")
    
    print("\nTrading Metrics:")
    print(f"  Number of Rebalances:{metrics.n_rebalances:>10}")
    print(f"  Rebalances/Year:     {metrics.turnover:>10.1f}")
    print(f"  Total TxN Costs:     {metrics.transaction_costs:>10.2%}")
    print(f"  Cost per Rebalance:  {metrics.transaction_costs/metrics.n_rebalances:>10.4%}")
    
    # Portfolio characteristics
    weights = results['weights'][-1]
    print("\nFinal Portfolio Characteristics:")
    print(f"  Active Positions:    {np.sum(weights > 0.001):>10}")
    print(f"  Max Weight:          {weights.max():>10.2%}")
    print(f"  Min Weight:          {weights.min():>10.2%}")
    print(f"  Weight Std Dev:      {weights.std():>10.4f}")
    print(f"  Herfindahl Index:    {np.sum(weights**2):>10.4f}")
    
    # Factor model diagnostics
    models = results['models']
    if models:
        last_model = models[-1]
        explained = compute_explained_variance(last_model)
        
        print("\nFactor Model (Latest):")
        print(f"  Explained Variance:  {explained:>10.2%}")
        print(f"  Factor Vols:         {np.sqrt(np.diag(last_model.F))}")


def compare_strategies(returns: np.ndarray):
    """Compare different strategy configurations."""
    print("\n" + "=" * 70)
    print("Strategy Comparison")
    print("=" * 70)
    
    configs = [
        BacktestConfig(rebalance_days=21, max_weight=0.10),   # Monthly, 10% cap
        BacktestConfig(rebalance_days=63, max_weight=0.10),   # Quarterly, 10% cap
        BacktestConfig(rebalance_days=21, max_weight=0.05),   # Monthly, 5% cap
        BacktestConfig(rebalance_days=21, max_weight=0.20),   # Monthly, 20% cap
    ]
    
    config_names = [
        "Monthly-10%",
        "Quarterly-10%",
        "Monthly-5%",
        "Monthly-20%"
    ]
    
    results_list = []
    
    for name, config in zip(config_names, configs):
        print(f"\nRunning: {name}")
        result = run_backtest(returns, config, verbose=False)
        results_list.append((name, result))
    
    # Summary table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    
    print(f"\n{'Strategy':<15} {'Return':<10} {'Vol':<10} {'Sharpe':<10} "
          f"{'MaxDD':<10} {'#Rebal':<10} {'Costs'}")
    print("-" * 80)
    
    for name, result in results_list:
        m = result['metrics']
        print(f"{name:<15} {m.annualized_return:<10.2%} {m.annualized_vol:<10.2%} "
              f"{m.sharpe_ratio:<10.2f} {m.max_drawdown:<10.2%} "
              f"{m.n_rebalances:<10} {m.transaction_costs:.4%}")
    
    print("\nKey Observations:")
    print("  - More frequent rebalancing → Higher costs, potentially better risk control")
    print("  - Tighter position limits → More diversified, slightly lower returns")
    print("  - Transaction costs materially impact net returns")


def main():
    print("=" * 70)
    print("Backtest Framework Example")
    print("=" * 70)
    
    # Generate synthetic data
    print("\nGenerating synthetic market data...")
    returns = generate_synthetic_returns(T=1500, p=50, k=3)
    print(f"Generated {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Data includes regime change at t={returns.shape[0]//2}")
    
    # Run baseline backtest
    config = BacktestConfig(
        lookback_days=252,
        rebalance_days=21,  # Monthly
        transaction_cost_bps=5.0,
        n_factors=3,
        max_weight=0.10
    )
    
    results = run_backtest(returns, config, verbose=True)
    print_performance_report(results)
    
    # Compare strategies
    compare_strategies(returns)
    
    # Analyze factor model evolution
    print("\n" + "=" * 70)
    print("Factor Model Evolution")
    print("=" * 70)
    
    models = results['models']
    explained_vars = [compute_explained_variance(m) for m in models]
    
    print(f"\nEstimated {len(models)} factor models")
    print(f"  Mean explained variance: {np.mean(explained_vars):.2%}")
    print(f"  Std explained variance:  {np.std(explained_vars):.2%}")
    print(f"  Min explained variance:  {np.min(explained_vars):.2%}")
    print(f"  Max explained variance:  {np.max(explained_vars):.2%}")
    
    # Check for regime change detection
    mid_point = len(models) // 2
    early_models = explained_vars[:mid_point]
    late_models = explained_vars[mid_point:]
    
    print(f"\nRegime Analysis:")
    print(f"  Early period (low vol): {np.mean(early_models):.2%} explained")
    print(f"  Late period (high vol): {np.mean(late_models):.2%} explained")
    
    if np.mean(late_models) < np.mean(early_models) - 0.05:
        print("  ⚠ Model fit deteriorates in high vol regime")
    else:
        print("  ✓ Model remains stable across regimes")
    
    print("\n" + "=" * 70)
    print("Backtest Framework Complete")
    print("=" * 70)
    
    print("""
Key Takeaways:

1. ROLLING ESTIMATION:
   - Use sufficient lookback (252+ days recommended)
   - Monitor explained variance for regime changes
   - Consider multiple model specifications

2. REBALANCING:
   - Balance between tracking and transaction costs
   - Monthly (21d) is standard; quarterly for low turnover
   - Consider market conditions for timing

3. TRANSACTION COSTS:
   - 5-10 bps typical for liquid stocks
   - Can meaningfully impact Sharpe ratio
   - Model explicitly, don't ignore

4. POSITION LIMITS:
   - 5-10% max typical for diversification
   - Tighter limits reduce concentration risk
   - May slightly reduce returns

5. RISK MANAGEMENT:
   - Track maximum drawdown continuously
   - Monitor factor exposures
   - Test multiple scenarios
    """)
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\nBacktest results available for further analysis:")
    print("  - results['weights']: Time series of portfolio weights")
    print("  - results['net_returns']: Net returns after costs")
    print("  - results['models']: All estimated factor models")
    print("  - results['metrics']: Performance metrics object")
