"""
fl_visualization.py
====================
Dispersion-agnostic visualization harness shared across simulation studies.

The same way ``fl_experiment_setup.run_analyses`` dispatches a list of analyses over
a context, this module dispatches a set of named figure-renderers over a results
DataFrame. A study registers its figures once; any later script renders them with
a single call and never re-implements the save/IO/dispatch plumbing.

    from fl_visualization import register_figure, render_figures, load_results

    register_figure("my_fig", my_render_fn, filename="my_fig.png")
    df = load_results("results.parquet")
    paths = render_figures(df, out_dir, names=["my_fig"], n_show=60)

A renderer has the signature ``fn(df, out_path, **kwargs) -> None`` and is free to
ignore kwargs it does not use, so heterogeneous figures (some needing ``n_show``,
some not) coexist behind one ``render_figures`` call. The registry is process-
global and keyed by name; re-registering a name overwrites it.

This module owns no study-specific column names or titles — those live with the
study (e.g. ``fl_graphics.py`` for the dispersion-bias figures).
"""

from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

__all__ = [
    "register_figure",
    "registered_figures",
    "render_figures",
    "load_results",
    "save_fig",
    "set_theme",
]

# name -> (render_fn, default_filename)
_FIGURE_REGISTRY: dict[str, tuple[Callable, str]] = {}


# ── Registry ────────────────────────────────────────────────────────────────────


def register_figure(
    name: str, fn: Callable, filename: Optional[str] = None
) -> None:
    """Register a figure renderer under ``name``.

    ``fn`` must accept ``(df, out_path, **kwargs)``. ``filename`` is the default
    output basename used by :func:`render_figures` (defaults to ``{name}.png``).
    Re-registering an existing name overwrites it.
    """
    _FIGURE_REGISTRY[name] = (fn, filename or f"{name}.png")


def registered_figures() -> list[str]:
    """Return the names of all currently registered figures, in insertion order."""
    return list(_FIGURE_REGISTRY)


# ── Dispatch ──────────────────────────────────────────────────────────────────


def render_figures(
    df: pd.DataFrame,
    out_dir: Path,
    names: Optional[Sequence[str]] = None,
    **kwargs,
) -> dict[str, Path]:
    """Render registered figures over ``df`` into ``out_dir``.

    ``names`` selects which figures to render (default: all registered). Extra
    ``kwargs`` are forwarded to every renderer; each renderer takes what it needs.
    Returns a mapping ``{name: output_path}`` for the figures rendered.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(names) if names is not None else registered_figures()

    produced: dict[str, Path] = {}
    for name in selected:
        if name not in _FIGURE_REGISTRY:
            raise KeyError(
                f"No figure registered under {name!r}; "
                f"known: {registered_figures()}"
            )
        fn, filename = _FIGURE_REGISTRY[name]
        out_path = out_dir / filename
        fn(df, out_path, **kwargs)
        produced[name] = out_path
    return produced


# ── Shared IO / styling helpers ─────────────────────────────────────────────────


def load_results(path: Path | str) -> pd.DataFrame:
    """Load simulation results from a .parquet or .csv file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def set_theme(style: str = "whitegrid", context: str = "paper") -> None:
    """Apply the shared seaborn theme used across study figures."""
    sns.set_theme(style=style, context=context)


def save_fig(fig, out_path: Path) -> None:
    """Tight-layout, save at 150 dpi, close, and log the basename."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", Path(out_path).name)
