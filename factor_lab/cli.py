"""
cli.py - Rich Command Line Interface for Factor Lab

A comprehensive CLI for factor model construction, simulation, and optimization.

Usage:
    factor-lab --help
    factor-lab fit returns.csv --factors 5 --output model.npz
    factor-lab simulate model.npz --periods 1000 --output returns.csv
    factor-lab optimize model.npz --long-only --max-weight 0.05
    factor-lab info model.npz
    factor-lab demo
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich import box
from rich.text import Text
from rich.columns import Columns

# Initialize Typer app and Rich console
app = typer.Typer(
    name="factor-lab",
    help="🧪 Factor Lab: Factor Model Construction, Simulation & Optimization",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Sub-commands
fit_app = typer.Typer(help="Fit factor models from data")
app.add_typer(fit_app, name="fit")


# =============================================================================
# ENUMS FOR CLI OPTIONS
# =============================================================================

class DecompositionMethod(str, Enum):
    """Available decomposition methods."""
    svd = "svd"
    pca = "pca"


class InnovationType(str, Enum):
    """Innovation distribution types."""
    normal = "normal"
    student_t = "student_t"


class OutputFormat(str, Enum):
    """Output file formats."""
    npz = "npz"
    json = "json"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_returns(path: Path) -> np.ndarray:
    """Load returns from CSV file."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)
    
    try:
        # Try numpy first (handles simple CSVs)
        returns = np.loadtxt(path, delimiter=",", skiprows=1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        return returns
    except Exception:
        # Fall back to pandas for complex CSVs
        try:
            import pandas as pd
            df = pd.read_csv(path, index_col=0)
            return df.values
        except ImportError:
            console.print("[red]Error:[/red] Install pandas for complex CSV support: pip install pandas")
            raise typer.Exit(1)


def save_model(model, path: Path, format: OutputFormat = OutputFormat.npz):
    """Save a factor model to disk."""
    if format == OutputFormat.npz:
        np.savez(
            path,
            B=model.B,
            F=model.F,
            D=model.D,
            factor_transform_matrix=model.factor_transform.matrix if model.factor_transform else None,
            factor_transform_type=model.factor_transform.transform_type.name if model.factor_transform else None,
            idio_transform_matrix=model.idio_transform.matrix if model.idio_transform else None,
            idio_transform_type=model.idio_transform.transform_type.name if model.idio_transform else None,
        )
    elif format == OutputFormat.json:
        import json
        data = {
            "B": model.B.tolist(),
            "F": model.F.tolist(),
            "D": model.D.tolist(),
            "k": model.k,
            "p": model.p,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def load_model(path: Path):
    """Load a factor model from disk."""
    from factor_lab import FactorModelData, CovarianceTransform, TransformType
    
    if not path.exists():
        console.print(f"[red]Error:[/red] Model file not found: {path}")
        raise typer.Exit(1)
    
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        
        # Reconstruct transforms if present
        factor_transform = None
        ft_type_val = data.get("factor_transform_type")
        if ft_type_val is not None:
            ft_type_str = str(ft_type_val)
            if ft_type_str not in ('None', 'none', ''):
                ft_matrix = data["factor_transform_matrix"]
                if ft_matrix is not None and hasattr(ft_matrix, 'size') and ft_matrix.size > 0:
                    # Handle case where it's stored as array(None)
                    if ft_matrix.shape != ():
                        ft_type = TransformType[ft_type_str]
                        factor_transform = CovarianceTransform(matrix=ft_matrix, transform_type=ft_type)
        
        idio_transform = None
        it_type_val = data.get("idio_transform_type")
        if it_type_val is not None:
            it_type_str = str(it_type_val)
            if it_type_str not in ('None', 'none', ''):
                it_matrix = data["idio_transform_matrix"]
                if it_matrix is not None and hasattr(it_matrix, 'size') and it_matrix.size > 0:
                    if it_matrix.shape != ():
                        it_type = TransformType[it_type_str]
                        idio_transform = CovarianceTransform(matrix=it_matrix, transform_type=it_type)
        
        return FactorModelData(
            B=data["B"],
            F=data["F"],
            D=data["D"],
            factor_transform=factor_transform,
            idio_transform=idio_transform,
        )
    
    elif path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        return FactorModelData(
            B=np.array(data["B"]),
            F=np.array(data["F"]),
            D=np.array(data["D"]),
        )
    
    else:
        console.print(f"[red]Error:[/red] Unknown model format: {path.suffix}")
        raise typer.Exit(1)


def print_model_summary(model, title: str = "Factor Model Summary"):
    """Print a rich summary of a factor model."""
    from factor_lab import compute_explained_variance
    
    # Create summary table
    table = Table(title=title, box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")
    
    table.add_row("Assets (p)", str(model.p))
    table.add_row("Factors (k)", str(model.k))
    table.add_row("Explained Variance", f"{compute_explained_variance(model):.1%}")
    
    # Factor variances
    factor_vars = np.diag(model.F)
    factor_vols = np.sqrt(factor_vars)
    table.add_row("Factor Volatilities", ", ".join(f"{v:.1%}" for v in factor_vols[:5]) + ("..." if len(factor_vols) > 5 else ""))
    
    # Idio stats
    idio_vars = np.diag(model.D)
    idio_vols = np.sqrt(idio_vars)
    table.add_row("Idio Vol (mean)", f"{np.mean(idio_vols):.1%}")
    table.add_row("Idio Vol (range)", f"{np.min(idio_vols):.1%} - {np.max(idio_vols):.1%}")
    
    # Transforms
    table.add_row("Factor Transform", "Diagonal" if model.factor_transform and model.factor_transform.is_diagonal else "Dense" if model.factor_transform else "None")
    table.add_row("Idio Transform", "Diagonal" if model.idio_transform and model.idio_transform.is_diagonal else "Dense" if model.idio_transform else "None")
    
    console.print(table)


def print_optimization_result(result, model):
    """Print optimization results with rich formatting."""
    if not result.solved:
        console.print(Panel(
            f"[red]Optimization Failed[/red]\n\nStatus: {result.metadata.get('status', 'unknown')}",
            title="❌ Result",
            border_style="red"
        ))
        return
    
    # Summary panel
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="bold green")
    summary.add_row("Status", "✓ Optimal")
    summary.add_row("Portfolio Risk", f"{result.risk:.2%}")
    summary.add_row("Objective Value", f"{result.objective:.6f}")
    if result.metadata.get("solve_time"):
        summary.add_row("Solve Time", f"{result.metadata['solve_time']*1000:.1f} ms")
    
    console.print(Panel(summary, title="📊 Optimization Result", border_style="green"))
    
    # Top holdings table
    weights = result.weights
    n_show = min(10, len(weights))
    
    # Sort by absolute weight
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    
    holdings_table = Table(title="Top Holdings", box=box.SIMPLE)
    holdings_table.add_column("Asset", style="cyan")
    holdings_table.add_column("Weight", justify="right")
    holdings_table.add_column("Bar", justify="left")
    
    max_weight = np.max(np.abs(weights))
    
    for i in range(n_show):
        idx = sorted_idx[i]
        w = weights[idx]
        bar_len = int(20 * abs(w) / max_weight) if max_weight > 0 else 0
        bar_char = "█" if w >= 0 else "░"
        color = "green" if w >= 0 else "red"
        bar = f"[{color}]{bar_char * bar_len}[/{color}]"
        holdings_table.add_row(f"Asset {idx}", f"{w:+.2%}", bar)
    
    if len(weights) > n_show:
        holdings_table.add_row("...", f"({len(weights) - n_show} more)", "")
    
    console.print(holdings_table)
    
    # Statistics
    stats = Table.grid(padding=(0, 3))
    stats.add_column()
    stats.add_column()
    stats.add_column()
    stats.add_column()
    
    long_weight = np.sum(weights[weights > 0])
    short_weight = np.sum(weights[weights < 0])
    n_long = np.sum(weights > 1e-6)
    n_short = np.sum(weights < -1e-6)
    
    stats.add_row(
        f"[dim]Long:[/dim] {long_weight:.1%}",
        f"[dim]Short:[/dim] {short_weight:.1%}",
        f"[dim]# Long:[/dim] {n_long}",
        f"[dim]# Short:[/dim] {n_short}",
    )
    
    console.print(stats)


# =============================================================================
# COMMANDS
# =============================================================================

@app.command()
def fit(
    input_file: Path = typer.Argument(..., help="CSV file with returns data (rows=time, cols=assets)"),
    factors: int = typer.Option(5, "--factors", "-k", help="Number of factors to extract"),
    method: DecompositionMethod = typer.Option(DecompositionMethod.svd, "--method", "-m", help="Decomposition method"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (default: model.npz)"),
    format: OutputFormat = typer.Option(OutputFormat.npz, "--format", "-f", help="Output format"),
    variance_target: Optional[float] = typer.Option(None, "--variance", "-v", help="Auto-select k to explain this variance fraction"),
):
    """
    Fit a factor model from returns data.
    
    Example:
        factor-lab fit returns.csv --factors 5 --output my_model.npz
        factor-lab fit returns.csv --variance 0.90  # Auto-select k
    """
    from factor_lab import svd_decomposition, pca_decomposition, select_k_by_variance
    
    console.print(Panel.fit("🔬 [bold]Factor Model Fitting[/bold]", border_style="blue"))
    
    # Load data
    with console.status("[bold blue]Loading returns data..."):
        returns = load_returns(input_file)
    
    T, p = returns.shape
    console.print(f"  Loaded returns: [cyan]{T}[/cyan] periods × [cyan]{p}[/cyan] assets")
    
    # Auto-select k if variance target specified
    if variance_target is not None:
        with console.status(f"[bold blue]Selecting k for {variance_target:.0%} explained variance..."):
            factors = select_k_by_variance(returns, target_explained=variance_target)
        console.print(f"  Selected [cyan]k={factors}[/cyan] factors")
    
    # Fit model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(f"Fitting {method.value.upper()} model with k={factors}...", total=None)
        
        if method == DecompositionMethod.svd:
            model = svd_decomposition(returns, k=factors)
        else:
            cov = np.cov(returns, rowvar=False)
            B, F = pca_decomposition(cov, k=factors)
            from factor_lab import FactorModelData
            D = np.diag(np.var(returns - returns @ B.T @ np.linalg.pinv(B.T), axis=0))
            model = FactorModelData(B=B, F=F, D=D)
    
    console.print("  [green]✓[/green] Model fitted successfully\n")
    
    # Display summary
    print_model_summary(model, title="Fitted Model")
    
    # Save model
    if output is None:
        output = Path(f"model_{method.value}_k{factors}.{format.value}")
    
    save_model(model, output, format)
    console.print(f"\n  💾 Saved to: [bold]{output}[/bold]")


@app.command()
def simulate(
    model_file: Path = typer.Argument(..., help="Factor model file (.npz or .json)"),
    periods: int = typer.Option(1000, "--periods", "-n", help="Number of time periods to simulate"),
    innovation: InnovationType = typer.Option(InnovationType.normal, "--innovation", "-i", help="Innovation distribution"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility"),
    validate: bool = typer.Option(False, "--validate", help="Validate covariance structure"),
):
    """
    Simulate returns from a fitted factor model.
    
    Example:
        factor-lab simulate model.npz --periods 5000 --output sim_returns.csv
        factor-lab simulate model.npz -n 1000 --innovation student_t --seed 42
    """
    from factor_lab import simulate_returns, CovarianceValidator
    
    console.print(Panel.fit("🎲 [bold]Returns Simulation[/bold]", border_style="blue"))
    
    # Load model
    with console.status("[bold blue]Loading model..."):
        model = load_model(model_file)
    
    console.print(f"  Model: [cyan]{model.k}[/cyan] factors, [cyan]{model.p}[/cyan] assets")
    
    # Simulate
    rng = np.random.default_rng(seed)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(f"Simulating {periods} periods...", total=None)
        returns = simulate_returns(model, n_periods=periods, rng=rng, innovation=innovation.value)
    
    console.print(f"  [green]✓[/green] Generated {returns.shape[0]} × {returns.shape[1]} returns matrix\n")
    
    # Summary statistics
    stats_table = Table(title="Simulation Statistics", box=box.ROUNDED)
    stats_table.add_column("Statistic", style="dim")
    stats_table.add_column("Value", justify="right")
    
    stats_table.add_row("Mean Return", f"{np.mean(returns):.4%}")
    stats_table.add_row("Std Dev (avg)", f"{np.mean(np.std(returns, axis=0)):.2%}")
    stats_table.add_row("Min Return", f"{np.min(returns):.2%}")
    stats_table.add_row("Max Return", f"{np.max(returns):.2%}")
    
    console.print(stats_table)
    
    # Validate if requested
    if validate:
        console.print("\n[bold]Covariance Validation:[/bold]")
        validator = CovarianceValidator(model)
        result = validator.compare(returns)
        
        val_table = Table.grid(padding=(0, 2))
        val_table.add_column(style="dim")
        val_table.add_column()
        val_table.add_row("Frobenius Error:", f"{result.frobenius_error:.4f}")
        val_table.add_row("Mean Abs Error:", f"{result.mean_absolute_error:.6f}")
        val_table.add_row("Explained Var:", f"{result.explained_variance_ratio:.1%}")
        console.print(val_table)
    
    # Save if requested
    if output:
        np.savetxt(output, returns, delimiter=",", header=",".join(f"Asset_{i}" for i in range(model.p)))
        console.print(f"\n  💾 Saved to: [bold]{output}[/bold]")


@app.command()
def optimize(
    model_file: Path = typer.Argument(..., help="Factor model file (.npz or .json)"),
    long_only: bool = typer.Option(True, "--long-only/--allow-short", help="Restrict to long positions"),
    max_weight: Optional[float] = typer.Option(None, "--max-weight", "-w", help="Maximum weight per asset"),
    min_weight: Optional[float] = typer.Option(None, "--min-weight", help="Minimum weight per asset"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save weights to CSV"),
    solver: Optional[str] = typer.Option(None, "--solver", help="CVXPY solver (ECOS, SCS, OSQP, MOSEK)"),
):
    """
    Find the minimum variance portfolio.
    
    Example:
        factor-lab optimize model.npz --long-only --max-weight 0.05
        factor-lab optimize model.npz --allow-short --output weights.csv
    """
    from factor_lab import FactorOptimizer, ScenarioBuilder
    
    console.print(Panel.fit("📈 [bold]Portfolio Optimization[/bold]", border_style="blue"))
    
    # Load model
    with console.status("[bold blue]Loading model..."):
        model = load_model(model_file)
    
    console.print(f"  Model: [cyan]{model.k}[/cyan] factors, [cyan]{model.p}[/cyan] assets")
    
    # Build scenario
    builder = ScenarioBuilder(model.p).create("CLI Optimization")
    builder.add_fully_invested()
    
    constraints_desc = ["Fully invested (Σw = 1)"]
    
    if long_only:
        builder.add_long_only()
        constraints_desc.append("Long only (w ≥ 0)")
    
    if max_weight is not None or min_weight is not None:
        low = min_weight if min_weight is not None else (0.0 if long_only else -1.0)
        high = max_weight if max_weight is not None else 1.0
        builder.add_box_constraints(low=low, high=high)
        constraints_desc.append(f"Box constraints ({low:.0%} ≤ w ≤ {high:.0%})")
    
    scenario = builder.build()
    
    console.print("  Constraints:")
    for c in constraints_desc:
        console.print(f"    • {c}")
    console.print()
    
    # Optimize
    optimizer = FactorOptimizer(model, solver=solver)
    optimizer.apply_scenario(scenario)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("Solving optimization...", total=None)
        result = optimizer.solve()
    
    # Display results
    print_optimization_result(result, model)
    
    # Save if requested
    if output and result.solved:
        np.savetxt(output, result.weights, delimiter=",", header="weight")
        console.print(f"\n  💾 Weights saved to: [bold]{output}[/bold]")


@app.command()
def info(
    model_file: Path = typer.Argument(..., help="Factor model file (.npz or .json)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
):
    """
    Display information about a saved factor model.
    
    Example:
        factor-lab info model.npz
        factor-lab info model.npz --detailed
    """
    console.print(Panel.fit("ℹ️  [bold]Model Information[/bold]", border_style="blue"))
    
    model = load_model(model_file)
    console.print(f"  File: [bold]{model_file}[/bold]\n")
    
    print_model_summary(model)
    
    if detailed:
        console.print("\n[bold]Factor Loadings (B) - First 5 assets:[/bold]")
        loadings_table = Table(box=box.SIMPLE)
        loadings_table.add_column("Factor", style="cyan")
        for i in range(min(5, model.p)):
            loadings_table.add_column(f"Asset {i}", justify="right")
        
        for k in range(model.k):
            row = [f"F{k}"] + [f"{model.B[k, i]:.3f}" for i in range(min(5, model.p))]
            loadings_table.add_row(*row)
        
        console.print(loadings_table)
        
        console.print("\n[bold]Factor Covariance (F):[/bold]")
        f_table = Table(box=box.SIMPLE)
        f_table.add_column("", style="cyan")
        for i in range(model.k):
            f_table.add_column(f"F{i}", justify="right")
        
        for i in range(model.k):
            row = [f"F{i}"] + [f"{model.F[i, j]:.4f}" for j in range(model.k)]
            f_table.add_row(*row)
        
        console.print(f_table)


@app.command()
def generate(
    assets: int = typer.Option(100, "--assets", "-p", help="Number of assets"),
    factors: int = typer.Option(3, "--factors", "-k", help="Number of factors"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: OutputFormat = typer.Option(OutputFormat.npz, "--format", "-f", help="Output format"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    beta_std: float = typer.Option(1.0, "--beta-std", help="Std dev of factor loadings"),
    factor_vol: float = typer.Option(0.20, "--factor-vol", help="Factor volatility"),
    idio_vol: float = typer.Option(0.05, "--idio-vol", help="Idiosyncratic volatility"),
):
    """
    Generate a synthetic factor model for testing.
    
    Example:
        factor-lab generate --assets 50 --factors 3 --output test_model.npz
        factor-lab generate -p 200 -k 5 --factor-vol 0.15 --idio-vol 0.08
    """
    from factor_lab import DistributionFactory, DataSampler
    
    console.print(Panel.fit("🔧 [bold]Synthetic Model Generation[/bold]", border_style="blue"))
    
    rng = np.random.default_rng(seed)
    factory = DistributionFactory(rng=rng)
    
    console.print(f"  Parameters:")
    console.print(f"    • Assets: [cyan]{assets}[/cyan]")
    console.print(f"    • Factors: [cyan]{factors}[/cyan]")
    console.print(f"    • Beta std: [cyan]{beta_std}[/cyan]")
    console.print(f"    • Factor vol: [cyan]{factor_vol:.1%}[/cyan]")
    console.print(f"    • Idio vol: [cyan]{idio_vol:.1%}[/cyan]")
    console.print()
    
    sampler = DataSampler(p=assets, k=factors, rng=rng)
    model = sampler.configure(
        beta=factory.create("normal", mean=0.0, std=beta_std),
        factor_vol=factory.create("constant", value=factor_vol),
        idio_vol=factory.create("constant", value=idio_vol),
    ).generate()
    
    console.print("  [green]✓[/green] Model generated\n")
    print_model_summary(model, title="Generated Model")
    
    if output is None:
        output = Path(f"synthetic_p{assets}_k{factors}.{format.value}")
    
    save_model(model, output, format)
    console.print(f"\n  💾 Saved to: [bold]{output}[/bold]")


@app.command()
def demo():
    """
    Run an interactive demonstration of Factor Lab capabilities.
    
    This walks through the complete workflow:
    1. Generate synthetic data
    2. Fit a factor model
    3. Simulate returns
    4. Optimize a portfolio
    """
    from factor_lab import (
        DistributionFactory, DataSampler, svd_decomposition,
        ReturnsSimulator, FactorOptimizer, ScenarioBuilder,
        CovarianceValidator, compute_explained_variance
    )
    
    console.print(Panel(
        "[bold cyan]Factor Lab Demo[/bold cyan]\n\n"
        "This demo walks through the complete workflow:\n"
        "  1️⃣  Generate synthetic factor model\n"
        "  2️⃣  Simulate historical returns\n"
        "  3️⃣  Extract factors via SVD\n"
        "  4️⃣  Validate covariance structure\n"
        "  5️⃣  Optimize portfolio",
        title="🧪 Welcome to Factor Lab",
        border_style="cyan"
    ))
    
    console.print()
    input("Press Enter to begin...")
    console.print()
    
    # Step 1: Generate model
    console.rule("[bold]Step 1: Generate Synthetic Model[/bold]")
    console.print()
    
    rng = np.random.default_rng(42)
    factory = DistributionFactory(rng=rng)
    
    p_assets, k_factors = 50, 3
    
    sampler = DataSampler(p=p_assets, k=k_factors, rng=rng)
    true_model = sampler.configure(
        beta=[
            factory.create("normal", mean=1.0, std=0.3),   # Market factor
            factory.create("normal", mean=0.0, std=0.5),   # Value factor
            factory.create("normal", mean=0.0, std=0.4),   # Size factor
        ],
        factor_vol=[
            factory.create("constant", value=0.20),  # Market 20%
            factory.create("constant", value=0.10),  # Value 10%
            factory.create("constant", value=0.08),  # Size 8%
        ],
        idio_vol=factory.create("uniform", low=0.03, high=0.08),
    ).generate()
    
    console.print(f"[green]✓[/green] Created model: {k_factors} factors, {p_assets} assets")
    console.print(f"  Factor volatilities: {', '.join(f'{np.sqrt(true_model.F[i,i]):.0%}' for i in range(k_factors))}")
    console.print()
    
    input("Press Enter to continue...")
    console.print()
    
    # Step 2: Simulate returns
    console.rule("[bold]Step 2: Simulate Historical Returns[/bold]")
    console.print()
    
    n_periods = 2000
    simulator = ReturnsSimulator(true_model, rng=rng)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Simulating {n_periods} periods...", total=None)
        
        results = simulator.simulate(
            n_periods=n_periods,
            factor_samplers=[factory.create("student_t", df=5)] * k_factors,  # Fat tails
            idio_samplers=[factory.create("normal", mean=0, std=1)] * p_assets,
        )
    
    returns = results["security_returns"]
    console.print(f"[green]✓[/green] Generated {returns.shape[0]} × {returns.shape[1]} returns matrix")
    console.print(f"  Mean return: {np.mean(returns):.4%}, Std: {np.mean(np.std(returns, axis=0)):.2%}")
    console.print()
    
    input("Press Enter to continue...")
    console.print()
    
    # Step 3: Extract via SVD
    console.rule("[bold]Step 3: Extract Factors via SVD[/bold]")
    console.print()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running SVD decomposition...", total=None)
        extracted_model = svd_decomposition(returns, k=k_factors)
    
    explained = compute_explained_variance(extracted_model)
    console.print(f"[green]✓[/green] Extracted {k_factors}-factor model")
    console.print(f"  Explained variance: {explained:.1%}")
    console.print(f"  Factor vols: {', '.join(f'{np.sqrt(extracted_model.F[i,i]):.1%}' for i in range(k_factors))}")
    console.print()
    
    input("Press Enter to continue...")
    console.print()
    
    # Step 4: Validate
    console.rule("[bold]Step 4: Validate Covariance Structure[/bold]")
    console.print()
    
    validator = CovarianceValidator(extracted_model)
    validation = validator.compare(returns)
    
    val_table = Table(box=box.ROUNDED, show_header=False)
    val_table.add_column(style="dim")
    val_table.add_column(style="bold")
    val_table.add_row("Frobenius Error", f"{validation.frobenius_error:.4f}")
    val_table.add_row("Mean Abs Error", f"{validation.mean_absolute_error:.6f}")
    val_table.add_row("Explained Variance", f"{validation.explained_variance_ratio:.1%}")
    
    console.print(val_table)
    console.print()
    
    input("Press Enter to continue...")
    console.print()
    
    # Step 5: Optimize
    console.rule("[bold]Step 5: Optimize Portfolio[/bold]")
    console.print()
    
    scenario = (ScenarioBuilder(p_assets)
        .create("Long Only Diversified")
        .add_fully_invested()
        .add_long_only()
        .add_box_constraints(low=0.0, high=0.05)
        .build())
    
    optimizer = FactorOptimizer(extracted_model)
    optimizer.apply_scenario(scenario)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Solving...", total=None)
        result = optimizer.solve()
    
    print_optimization_result(result, extracted_model)
    
    console.print()
    console.print(Panel(
        "[bold green]Demo Complete![/bold green]\n\n"
        "You've seen the full Factor Lab workflow. Try it yourself:\n\n"
        "  [cyan]factor-lab generate[/cyan] --assets 100 --factors 5\n"
        "  [cyan]factor-lab fit[/cyan] returns.csv --factors 5\n"
        "  [cyan]factor-lab simulate[/cyan] model.npz --periods 1000\n"
        "  [cyan]factor-lab optimize[/cyan] model.npz --long-only",
        title="🎉 What's Next?",
        border_style="green"
    ))


@app.command()
def version():
    """Show version information."""
    from factor_lab import __version__
    
    console.print(Panel(
        f"[bold cyan]Factor Lab[/bold cyan] v{__version__}\n\n"
        "A Python library for factor model construction,\n"
        "simulation, and portfolio optimization.\n\n"
        "[dim]https://github.com/your-repo/factor-lab[/dim]",
        border_style="cyan"
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
