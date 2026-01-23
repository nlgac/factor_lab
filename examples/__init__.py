"""
factor_lab Examples Package
============================

This package contains comprehensive, runnable examples demonstrating the
capabilities of factor_lab for quantitative finance research and portfolio
management.

Examples
--------
extract_factors : module
    Basic factor extraction using SVD and PCA with model validation.
simulate_distributions : module
    Multi-distribution simulation demonstrating covariance preservation.
optimize_portfolio : module
    Portfolio optimization with various constraint scenarios.
generate_synthetic : module
    Synthetic data generation for stress testing and backtesting.
backtest_strategy : module
    Complete backtesting framework with rolling estimation.

Quick Start
-----------
Run any example directly from the command line:

    $ cd examples
    $ python extract_factors.py
    $ python simulate_distributions.py
    $ python optimize_portfolio.py
    $ python generate_synthetic.py
    $ python backtest_strategy.py

Or import as modules:

    >>> from examples import extract_factors
    >>> model = extract_factors.main()
    
    >>> from examples.optimize_portfolio import scenario_1_long_only_basic
    >>> from factor_lab import FactorModelData
    >>> # ... create model ...
    >>> result = scenario_1_long_only_basic(model)

Learning Path
-------------
1. extract_factors - Understand factor model basics
2. simulate_distributions - Learn simulation framework
3. optimize_portfolio - Apply to portfolio construction
4. generate_synthetic - Create test scenarios
5. backtest_strategy - Complete production workflow

See Also
--------
README.md : Complete examples documentation
EXAMPLES_SUMMARY.md : Quick reference guide
"""

__version__ = "2.0.0"

# Import submodules for convenient access
# Note: We use lazy imports to avoid circular dependencies
# and to keep the package lightweight

__all__ = [
    "extract_factors",
    "simulate_distributions",
    "optimize_portfolio",
    "generate_synthetic",
    "backtest_strategy",
]


def list_examples():
    """
    List all available examples with descriptions.
    
    Returns
    -------
    dict
        Dictionary mapping example names to their descriptions.
    
    Examples
    --------
    >>> from examples import list_examples
    >>> for name, desc in list_examples().items():
    ...     print(f"{name}: {desc}")
    """
    return {
        "extract_factors": (
            "Basic factor extraction using SVD and PCA. "
            "Demonstrates model validation, explained variance analysis, "
            "and factor loadings examination."
        ),
        "simulate_distributions": (
            "Multi-distribution simulation with Normal, Student's t, and Uniform "
            "innovations. Shows covariance preservation and moment analysis."
        ),
        "optimize_portfolio": (
            "Portfolio optimization across 6 scenarios including long-only, "
            "130/30, sector-neutral, and turnover-constrained portfolios."
        ),
        "generate_synthetic": (
            "Synthetic data generation for 4 market scenarios: crisis, calm, "
            "sector rotation, and random ensemble for Monte Carlo analysis."
        ),
        "backtest_strategy": (
            "Complete backtesting framework with rolling estimation, "
            "transaction costs, and performance attribution."
        ),
    }


def get_example_info(name):
    """
    Get detailed information about a specific example.
    
    Parameters
    ----------
    name : str
        Name of the example (without .py extension).
    
    Returns
    -------
    dict
        Dictionary with keys: 'description', 'features', 'runtime'
    
    Examples
    --------
    >>> from examples import get_example_info
    >>> info = get_example_info('extract_factors')
    >>> print(info['description'])
    """
    examples_info = {
        "extract_factors": {
            "description": "Learn factor model extraction and validation",
            "features": [
                "SVD decomposition (preferred method)",
                "PCA comparison",
                "Automatic factor selection",
                "Explained variance analysis",
                "Model validation",
                "Visualization (optional matplotlib)",
            ],
            "runtime": "~2 seconds",
            "complexity": "Beginner",
        },
        "simulate_distributions": {
            "description": "Master the distribution sampling framework",
            "features": [
                "5 innovation distribution scenarios",
                "Covariance preservation verification",
                "Moment analysis (skewness, kurtosis)",
                "Extreme value comparison",
                "Synthetic model ensemble",
            ],
            "runtime": "~10 seconds",
            "complexity": "Beginner-Intermediate",
        },
        "optimize_portfolio": {
            "description": "Apply factor models to portfolio construction",
            "features": [
                "6 optimization scenarios",
                "Long-only and long-short strategies",
                "Sector and factor constraints",
                "Turnover management",
                "Complete portfolio analytics",
            ],
            "runtime": "~1 second",
            "complexity": "Intermediate",
        },
        "generate_synthetic": {
            "description": "Create realistic synthetic data for testing",
            "features": [
                "4 market scenarios (crisis, calm, rotation, ensemble)",
                "Stress testing framework",
                "Time-varying volatility",
                "Correlation analysis",
                "Monte Carlo preparation",
            ],
            "runtime": "~15 seconds",
            "complexity": "Intermediate",
        },
        "backtest_strategy": {
            "description": "Build production-ready backtesting framework",
            "features": [
                "Rolling factor estimation",
                "Periodic rebalancing logic",
                "Transaction cost modeling",
                "Performance attribution",
                "Strategy comparison",
                "Regime detection",
            ],
            "runtime": "~20 seconds",
            "complexity": "Advanced",
        },
    }
    
    if name not in examples_info:
        available = ", ".join(examples_info.keys())
        raise ValueError(
            f"Unknown example '{name}'. Available: {available}"
        )
    
    return examples_info[name]


def run_example(name, *args, **kwargs):
    """
    Dynamically import and run an example.
    
    Parameters
    ----------
    name : str
        Name of the example to run (without .py extension).
    *args, **kwargs
        Arguments to pass to the example's main() function.
    
    Returns
    -------
    result
        Return value from the example's main() function.
    
    Examples
    --------
    >>> from examples import run_example
    >>> 
    >>> # Run extract_factors example
    >>> model = run_example('extract_factors')
    >>> 
    >>> # Run with custom parameters (if supported)
    >>> results = run_example('backtest_strategy')
    
    Notes
    -----
    This dynamically imports the example module, so the first call may
    be slower as the module is loaded.
    """
    import importlib
    
    valid_examples = list_examples().keys()
    if name not in valid_examples:
        raise ValueError(
            f"Unknown example '{name}'. Valid examples: {', '.join(valid_examples)}"
        )
    
    # Import the module
    module = importlib.import_module(f"examples.{name}")
    
    # Call its main function
    if hasattr(module, "main"):
        return module.main(*args, **kwargs)
    else:
        raise AttributeError(
            f"Example '{name}' does not have a main() function"
        )


def print_examples_menu():
    """
    Print a formatted menu of all available examples.
    
    This is useful for interactive exploration.
    
    Examples
    --------
    >>> from examples import print_examples_menu
    >>> print_examples_menu()
    """
    print("=" * 70)
    print("factor_lab Examples")
    print("=" * 70)
    print("\nAvailable examples:\n")
    
    for i, (name, desc) in enumerate(list_examples().items(), 1):
        info = get_example_info(name)
        print(f"{i}. {name}")
        print(f"   {desc}")
        print(f"   Complexity: {info['complexity']} | Runtime: {info['runtime']}")
        print()
    
    print("Usage:")
    print("  $ python extract_factors.py")
    print("  or")
    print("  >>> from examples import run_example")
    print("  >>> run_example('extract_factors')")
    print("\nFor detailed documentation, see examples/README.md")


# Convenience function for interactive use
def help():
    """
    Display help information about the examples package.
    
    Examples
    --------
    >>> import examples
    >>> examples.help()
    """
    print(__doc__)
    print("\n")
    print_examples_menu()


# Make the functions easily accessible
__all__.extend([
    "list_examples",
    "get_example_info", 
    "run_example",
    "print_examples_menu",
    "help",
])
