# factor_lab — Factor Models, Return Simulation & Dispersion-Bias Verification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **Build factor models, simulate returns from them, and analyze the geometric
> and spectral relationship between sample and population eigenvectors —
> the numerical infrastructure behind the verification of the multifactor
> dispersion-bias theorems in *"Multifactor Dispersion Bias with Per-Column
> Prevalence."***

---

## 🎯 Overview

`factor_lab` works with the standard linear factor model

```
r = Bᵀ f + ε,        Σ = Bᵀ F B + D
```

where `B` is `(k, p)` loadings, `F` is diagonal factor covariance, and `D` is
diagonal idiosyncratic covariance. The core question it answers numerically:
**as dimension `p` grows, how well does an estimated factor direction line up
with the true one?** — measured by `sin²∠(hⱼ, b̄ⱼ)` and compared against a
closed-form prediction (`floor + weight·rotation`).

### Two tiers

1. **`factor_lab/` — the reusable library.** Problem-agnostic: model
   construction, return simulation, SVD/PCA estimation, and a suite of
   geometric/spectral analyses. Knows nothing about any particular theorem.

2. **Repo-root orchestration — the "engine" + "probes".** Thin layers that drive
   a specific study on top of the library:
   - `fl_experiment_setup.py` — the specs (`ModelSpec`, `DesignSpec`), the
     `Experiment` protocol, model construction, and stateless seams.
   - `fl_experiment_runner.py` — the `n × p` sweep; owns the master-RNG draw
     order (the reproducibility contract).
   - `sim_theorem_partii.py`, `sim_corollary4.py`, `sim_corollary_obs_floor.py`
     — **probes**: theorem-specific analyses + a driver. A new theorem is just a
     new `Experiment` handed to the *same* engine. The filenames are **paper
     labels** ("Part II", "Corollary 4" name results in the paper, not anything
     in the code).
   - `fl_graphics.py` — turns a result DataFrame into convergence figures via
     `plot_all(df, out_dir)`; usable standalone to replot from a saved parquet.

---

## 🚀 Quick Start

```bash
# From the repository root, install in editable mode (pulls in all deps)
pip install -e .

# Run the dispersion-bias probe with built-in defaults
python sim_theorem_partii.py
```

Python ≥ 3.8. Runtime dependencies (declared in `pyproject.toml`): **numpy,
scipy, matplotlib, seaborn, tqdm, pandas, loguru**, plus **pyarrow** for parquet
output.

---

## 🔬 Core Capabilities

- **Flexible model building** — per-factor β samplers, an idiosyncratic-vol
  sampler, explicit factor variances. Inputs are volatilities by default
  (squared into `F`/`D`) or variances via `units="variance"`.
- **Pluggable return distributions** — a universal sampler interface (`normal`,
  `student_t`, `uniform`, `beta`, `exponential`, `gamma`, `constant`); the same
  model can be re-simulated under different return processes (e.g. heavy-tailed
  stress tests). See **[Returns sampling & variance conventions](#-returns-sampling--variance-conventions)**.
- **Memory-efficient spectral analysis** — top-`k` eigenpairs of
  `Σ = BᵀFB + D` via an ARPACK `LinearOperator` in `O(kp)` memory, never forming
  the `p × p` covariance. Sample side uses an `n × n` Gram trick.
- **Geometric distances on manifolds** — Grassmannian, Procrustes, and Chordal
  distances, plus per-factor sine-alignment; an extensible distance registry.
- **Reproducible experiment sweeps** — a deterministic master-RNG draw order, so
  swapping the probe cannot perturb the numbers. Each run returns a tidy `pandas`
  DataFrame; every probe prints a console RMSE table that should shrink as `p`
  grows.
- **JSON-configurable runs** — a run is a *model* + a *design*; both can live in
  one flat file, one nested file, or a split (reusable-model + design) pair.

---

## 📦 Package Structure

```
factor_lab/
├── factor_lab/                  # Reusable, problem-agnostic library
│   ├── model_builder.py         #   FactorModelBuilder (B, F, D construction)
│   ├── distributions.py         #   create_sampler() — the universal sampler interface
│   ├── flexible_simulator.py    #   ReturnsSimulator — Y = BᵀF + Z, with variance logging
│   ├── factor_types.py          #   FactorModelData and core types
│   ├── integration.py           #   build_simulate_analyze() all-in-one pipeline
│   ├── analysis/                #   SimulationContext + analysis protocol
│   └── analyses/                #   manifold / spectral / eigenvector analyses
│
├── fl_experiment_setup.py       # ENGINE · ModelSpec, DesignSpec, Experiment, seams
├── fl_experiment_runner.py      # ENGINE · run_experiment / run_cell (the n×p sweep)
├── fl_graphics.py               # Result DataFrame → convergence figures
│
├── sim_theorem_partii.py        # PROBE · dispersion-bias verification (+ CLI)
├── sim_corollary4.py            # PROBE · subspace-distance corollary (smoke sweep)
├── sim_corollary_obs_floor.py   # PROBE · observable-floor corollary (smoke sweep)
│
├── tests/                       # Engine + probe + library test suite
└── docs/                        # codebase_overview.md, architecture_flowchart.md, …
```

---

## 🎲 Returns sampling & variance conventions

Return distributions are built with `factor_lab.distributions.create_sampler`.
Every sampler field is a dict `{"distribution": <name>, ...params}`:

| `distribution` | Parameters | Notes |
|---|---|---|
| `normal` | `loc` (=0), `scale` (=1) | `N(0, 1)` by default; `scale` is the std. |
| `student_t` | `df` *(required)*, `loc` (=0), `scale` (=1), `standardize` (=True) | Heavy tails. **Standardized to unit variance by default** — see below. |
| `uniform` | `low`, `high` *(required)* | |
| `beta` | `a`, `b` *(required)* | |
| `exponential` | `scale` *(required)* | |
| `gamma` | `shape`, `scale` *(required)* | |
| `constant` | `value` *(required)* | Degenerate (e.g. fixed idio vol). |

Any `callable(n) -> np.ndarray` is also a valid sampler, so scipy distributions
or regime-switching mixtures can be passed directly when driving the engine from
code.

### Student-t is standardized to unit variance

Raw `numpy.standard_t(df)` has variance `df / (df − 2)` (not 1), so a nominal
idiosyncratic or factor **vol would not equal the realized standard deviation**
under heavy tails (e.g. `df = 5` inflates variance by `5/3`). To keep vols
meaning what they say, the `student_t` sampler **standardizes to unit variance by
default**:

```python
from factor_lab.distributions import create_sampler
import numpy as np

rng = np.random.default_rng(0)

s = create_sampler("student_t", rng, df=5)                       # Var ≈ 1.0
raw = create_sampler("student_t", rng, df=5, standardize=False)  # Var ≈ 5/3 (raw)
```

- `standardize=True` (default) **overwrites `scale`** so the distributional
  variance is exactly 1 (the user's `df` is preserved, the shape/tails are
  unchanged). Requires `df > 2` — infinite variance cannot be normalized, so
  `df ≤ 2` raises a `ValueError`.
- `standardize=False` keeps the regular `loc + scale·t` form, with variance
  `scale² · df/(df − 2)`.

Because idiosyncratic draws are scaled by `√D` downstream, a unit-variance idio
sampler makes the realized specific variance exactly `D` — so the dispersion
theorem's `δ² = mean(diag(D))` holds for heavy tails just as it does for normal
returns. (Before standardization, heavy-tailed idio returns silently inflated
the floor by `df/(df − 2)` and broke convergence.)

### Variance logging

After each `simulate()` call, `ReturnsSimulator` logs the realized sample
variance of the factor (per factor), idiosyncratic (mean across assets), and
security return blocks at **DEBUG** level — a quick check that the samplers
delivered the intended spread:

```python
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")     # the example scripts silence logging by default
# ... run an experiment ...
# returns variance (n=63, p=3000): factor=[0.0244 0.0034 0.0033] | idio(mean)=0.1576 | security(mean)=0.2116
```

### Disabling the variance assertion

The standardized `student_t` builder carries an `assert` that the distributional
variance is exactly 1 (a cheap, once-per-build math guard). To strip it — along
with every other `assert` in the codebase — run Python with optimization:

```bash
python -O sim_theorem_partii.py        # -O removes all assert statements
export PYTHONOPTIMIZE=1                 # same effect for the whole session
```

---

## ⚙️ Specifying a model and an experiment

A run is always **a model** (`ModelSpec` — what `(B, F, D)` is) **plus a design**
(`DesignSpec` — the sweep, the return process, and which model to use). Supply
them as Python objects or as JSON; you do **not** edit a probe `.py` to change
them.

### `ModelSpec` — the factor model

| Field | Type | Default | Meaning |
|---|---|---|---|
| `k_factors` | int | `3` | Number of factors `k`. |
| `factor_vols` | list[float] (len `k`) | `[0.16, 0.08, 0.06]` | Per-factor volatilities → squared into diagonal `F`. |
| `beta_samplers` | dict **or** list[dict] | three normals | Per-factor loading (β) distributions; a single dict broadcasts to all `k`. |
| `idio_vol_sampler` | dict | `{"distribution": "constant", "value": 0.4}` | Idiosyncratic-vol distribution → draws squared into diagonal `D`. |
| `units` | str | `"vol"` | `"vol"`: values are volatilities (squared into `F`/`D`). `"variance"`: values are variances (passed straight in). |

### `DesignSpec` — the sweep + return process

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | str \| dict \| None | `None` | A path to a model JSON, an inline model dict, or `None` for defaults. |
| `n_values` | list[int] | `[30, 60, 120]` | Sample sizes (periods) `n` to sweep. |
| `p_values` | list[int] | `[200, 500, 1000, 2000, 5000, 10000]` | Dimensions (assets) `p` to sweep. |
| `n_reps` | int | `300` | Replications per `(n, p)` cell. |
| `random_seed` | int | `20260511` | Master RNG seed (full reproducibility). |
| `factor_return_sampler` | dict \| list[dict] | `normal(0, 1)` | Factor-return distribution(s); single dict broadcasts to `k`. |
| `idio_return_sampler` | dict | `normal(0, 1)` | Idiosyncratic-return distribution. |
| `sampling` | str | `"independent"` | `"independent"`: fresh model + returns per cell. `"nested"`: one superset per replicate at `p_max`, smaller `p` taken as asset subsets (clean monotone-in-`p` curve). |
| `nest_time` | bool | `False` | With `sampling="nested"`, also nest the `n` axis (monotone-in-`n` curve). |

### The three JSON file shapes

```jsonc
// 1) Unified single file — model fields at the top level
{ "k_factors": 3, "factor_vols": [0.16, 0.08, 0.06],
  "n_values": [60], "p_values": [2000], "n_reps": 100 }

// 2) Inline model inside the design
{ "model": { "k_factors": 3, "factor_vols": [0.16, 0.08, 0.06] },
  "n_values": [60], "p_values": [2000] }

// 3) Split: design references a reusable model file
{ "model": "my_model.json", "n_values": [60], "p_values": [2000] }
```

Keys beginning with `_` are treated as comments. Specifying the model both at the
top level *and* via `model` raises an error — pick one form.

---

## ▶️ Running it

### CLI (the dispersion-bias probe)

```bash
python sim_theorem_partii.py                       # built-in defaults
python sim_theorem_partii.py my_spec.json          # single self-contained spec
python sim_theorem_partii.py design.json --model model.json   # split config
python sim_theorem_partii.py my_spec.json --plot-save         # parquet + figures
python sim_theorem_partii.py my_spec.json --out my_results.parquet
```

Output-path precedence: `--out` > `design.output_path` > auto
`results/MM-DD_run_NN/`. The parquet has columns
`n, p, j, sin2_j, rhs, gap, floor, rotation, rho`. The two corollary probes
(`sim_corollary4.py`, `sim_corollary_obs_floor.py`) ship as built-in smoke sweeps
— no flags, console-only RMSE tables.

### Library idiom (standalone sandbox)

Use `factor_lab` as a general factor-model sandbox — no probe, no theorem:

```python
import numpy as np
from factor_lab.integration import build_simulate_analyze
from factor_lab.distributions import create_sampler

rng = np.random.default_rng(42)
mk = lambda name, **p: create_sampler(name, rng, **p)

results = build_simulate_analyze(
    p=1000, k=2,
    beta_samplers=mk("normal", loc=0, scale=1),
    idio_vol_sampler=mk("constant", value=0.4),
    factor_variances=[0.04, 0.01],
    n_periods=120,
    factor_return_samplers=mk("normal", loc=0, scale=1),
    idio_return_sampler=mk("normal", loc=0, scale=1),
    rng=rng,
)
print(results["dist_grassmannian"])   # sample-vs-population subspace distance
```

### Engine idiom (running a study from Python)

```python
from fl_experiment_setup import ModelSpec, DesignSpec
from fl_experiment_runner import run_experiment
from sim_theorem_partii import DispersionBiasExperiment   # study-specific probe

df = run_experiment(
    ModelSpec(),
    DesignSpec(n_values=[60], p_values=[2000], n_reps=100),
    DispersionBiasExperiment(),
)
```

---

## 📈 Reading the results

Every probe verifies an **asymptotic claim**: an observed quantity converges to a
closed-form prediction as `p → ∞`. The diagnostic is always a **gap** (observed −
predicted) whose dispersion should shrink as `p` grows:

- **Dispersion probe** — per-factor `gap` column (`sin² − RHS`) and an RMSE table
  by `(n, p)`. A healthy run: each `sin²ⱼ` curve settles onto its predicted
  floor-plus-rotation value, and **every column of the RMSE table decreases left
  to right** as `p` grows.
- **Corollary probes** — the analogous RMSE table for their own gap. Same read:
  smaller is better, falling with `p`.

The numbers never hit exactly 0 (finite-`n`, finite-rep noise) — the **downward
trend in `p`** is the signal that the theorem holds.

---

## ➕ Adding a new theorem

A new question is **one `Experiment` object** handed to the same
`run_experiment` — no change to the engine. An `Experiment` supplies up to three
hooks:

| Hook | When | Returns | Rule |
|---|---|---|---|
| `setup()` | once, before the sweep | — | Optional one-time setup. Default no-op via `BaseExperiment`. |
| `cell_setup(model, n, p)` | once per `(n, p)` cell | list of analyses | Model-only prep. **Must not draw from the master RNG.** |
| `record(n, p, merged)` | once per replication | list of row dicts | Flatten one replication's merged analysis dict into output rows. |

```python
from fl_experiment_setup import BaseExperiment, register_experiment, ModelSpec, DesignSpec
from fl_experiment_runner import run_experiment

@register_experiment("my_theorem")
class MyTheoremExperiment(BaseExperiment):
    def cell_setup(self, model, n, p):
        return [MyObservedAnalysis(model), MyPredictedAnalysis()]

    def record(self, n, p, merged):
        return [{"n": n, "p": p, "observed": merged["lhs"], "predicted": merged["rhs"]}]

df = run_experiment(ModelSpec(), DesignSpec(p_values=[200, 500, 1000, 2000]), MyTheoremExperiment())
```

**Why this is safe:** the runner owns the master-RNG draw order (build model →
cell setup → per-rep seeds → per-rep returns). Because the probe's hooks never
draw from that RNG, swapping in a different `Experiment` cannot perturb the
numbers — the engine guarantees reproducibility, the probe gives it meaning.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v

python -m pytest tests/ \
    --cov=sim_theorem_partii --cov=fl_experiment_setup --cov=fl_experiment_runner \
    --cov-report=term-missing
```

---

## 📚 Further reading

For the authoritative internals, see the in-repo
[`docs/codebase_overview.md`](docs/codebase_overview.md) and
[`docs/architecture_flowchart.md`](docs/architecture_flowchart.md).
