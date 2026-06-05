# factor_lab: Codebase Overview

*Current as of the three-layer orchestration refactor. For the execution and
layering diagrams see [architecture_flowchart.md](architecture_flowchart.md).
The previous single-script writeup is archived at
[codebase_overview_pre_refactor.md](codebase_overview_pre_refactor.md).*

## Purpose

`factor_lab` is a Python package for building factor models, simulating returns,
and analyzing the geometric and spectral properties of sample vs. population
eigenvectors. It is the infrastructure behind the numerical verification of the
multifactor dispersion bias theorems in *"Multifactor Dispersion Bias with
Per-Column Prevalence."*

The repository is organized in two tiers:

1. **`factor_lab/`** — a reusable, problem-agnostic library: model construction,
   return simulation, SVD estimation, and geometric/spectral analyses.
2. **Repo-root orchestration** — three thin layers that drive a specific study
   (currently the dispersion-bias verification) on top of the library.

## Architecture at a glance

```
fl_experiment_setup.py   ENGINE · setup    ModelSpec · DesignSpec · Experiment ·
                                            build_model · sampler resolution ·
                                            simulate_returns · run_analyses · next_run_dir
fl_experiment_runner.py  ENGINE · runner   run_experiment · run_cell (the n×p sweep;
                                            owns the master-RNG draw order)
sim_theorem_partii.py    dispersion PROBE  SineAlignmentAnalysis · Eq6RHSAnalysis ·
                                            DispersionBiasExperiment · CLI
fl_graphics.py           plotting          three convergence figures from a DataFrame
```

The engine knows nothing about dispersion bias. A new theorem is a new
`Experiment` (three hooks) handed to the same `run_experiment`; no engine change.
See [architecture_flowchart.md](architecture_flowchart.md) for the full control
flow and the master-RNG draw order.

---

# The library: `factor_lab/`

```
factor_lab/
├── factor_types.py        # Core data structure (FactorModelData)
├── fl_types.py            # Additional type definitions
├── distributions.py       # Sampler factory
├── model_builder.py       # Stage 1: model construction
├── flexible_simulator.py  # Stages 2–4: return simulation
├── decomposition.py       # SVD-based factor extraction
├── integration.py         # All-in-one pipeline
├── simulation.py          # Legacy simulator (deprecated)
├── model_io.py            # Save/load models to .npz
├── analysis/
│   └── context.py         # SimulationContext dataclass
├── analyses/
│   ├── manifold.py        # Grassmannian, Procrustes, Chordal distances
│   ├── spectral.py        # Implicit eigenvalue computation via ARPACK
│   ├── eigenvector.py     # Eigenvector alignment analysis
│   └── builder.py         # Analyses factory
└── visualization/
    └── visualization.py   # Visualization utilities
```

## Core data structure — `FactorModelData` (factor_types.py)

The central container passed between components. Represents $r = B^\top f + \varepsilon$:

- `B`: $(k, p)$ factor loadings
- `F`: $(k, k)$ diagonal factor covariance
- `D`: $(p, p)$ diagonal idiosyncratic covariance
- `k`, `p`: derived from `B.shape`
- `implied_covariance()`: $\Sigma = B^\top F B + D$
- `factor_transform`, `idio_transform`: optional transforms (used by `model_io`)

## Library pipeline

- **`distributions.py`** — universal sampler interface: any `callable(n) -> ndarray`. `create_sampler(name, rng, **params)` covers `normal`, `student_t`, `uniform`, `beta`, `exponential`, `gamma`, `constant`. `resolve_samplers` broadcasts one sampler to $k$ or validates a per-factor list.
- **`model_builder.py` — `FactorModelBuilder`** (Stage 1). Per-factor β samplers, an idio-vol sampler, and explicit factor variances. Samples $B$ row by row, samples idio vols and **squares them** to get $D$ (the sampler outputs volatilities; $D$ stores variances), and sets $F = \mathrm{diag}(\text{factor\_variances})$.
- **`flexible_simulator.py` — `ReturnsSimulator`** (Stages 2–4). Stateless. Scales raw factor draws by $\sqrt{F}$, raw idio draws by $\sqrt{D}$, combines via $r = f @ B + \varepsilon$. The same model can be reused with different return distributions.
- **`decomposition.py` — `svd_decomposition`**. Extracts a factor model from returns via SVD; centers, takes top-$k$ right singular vectors as $B$, variances from singular values, residualizes to $D$; sign-normalizes rows of $B$.
- **`analysis/context.py` — `SimulationContext`**. Frozen snapshot of model + returns. Lazy `sample_covariance()` / `pca_decomposition(n)`. Properties `T`, `p`, `k`.
- **`analyses/manifold.py`** — Grassmannian / Procrustes / Chordal distances, plus `compute_sine_alignment` for per-factor $\sin^2\angle(h_j, \bar b_j)$ matched by row order, and an `_EXTRA_DISTANCES` registry (`register_manifold_distance`).
- **`analyses/spectral.py`** — top-$k$ eigenpairs of $\Sigma = B^\top F B + D$ via a `LinearOperator` (ARPACK `eigsh`), $O(kp)$ memory. `compute_true_eigenvalues` is used to get the population directions.
- **`analyses/eigenvector.py`, `analyses/builder.py`, `integration.py`, `model_io.py`, `simulation.py`** — eigenvector alignment, an analyses factory, an all-in-one pipeline, `.npz` save/load, and a deprecated legacy simulator, respectively.

---

# The engine layers (repo root)

The engine is two theorem-agnostic files: a **setup** layer (the data you
configure with, plus the building-block seams) and a **runner** layer (the sweep
that consumes them).

## `fl_experiment_setup.py` — specs, protocol, construction, seams

Theorem-agnostic. Holds the two data specs, the `Experiment` protocol,
`build_model`, and the stateless seams.

**`ModelSpec`** — the factor model: `k_factors`, `factor_vols`,
`beta_samplers`, `idio_vol_sampler`. `factor_vols` and the idio vol are
volatilities (squared into $F$ / $D$).

**`DesignSpec`** — the sweep + return process: `n_values`, `p_values`, `n_reps`,
`random_seed`, `factor_return_sampler`, `idio_return_sampler`, `sampling`,
`output_path`, `plot_mode`, and a `model` field that carries the factor model
(see *Configuration* below).

**`Experiment` (Protocol)** — the only theorem-specific surface (implemented in
the probe, not here). Three hooks, none of which may touch the master RNG:

```python
class Experiment(Protocol):
    def setup(self) -> None: ...                          # optional, once per run
    def cell_setup(self, model, n, p) -> list[Analysis]:  # model-only, per cell
    def record(self, n, p, merged: dict) -> list[dict]:   # flatten one rep
```

**`build_model(model_spec, p, rng)`** — Stage 1: construct one `(B, F, D)` model;
draws β / idio vols from the RNG it's given (`factor_vols` squared into $F$).

**Stateless seams** (the runner reuses these):

| Function | Role |
|---|---|
| `make_one_sampler(spec, rng)` / `make_samplers(spec, rng, k)` | dict → callable; broadcast or per-factor list |
| `simulate_returns(model, n, factor_return_spec, idio_return_spec, k, rep_rng)` | Stages 2–4 → `SimulationContext`; draws **only** from `rep_rng` |
| `run_analyses(context, analyses)` | run each `analyze(context)` and merge result dicts |
| `next_run_dir(base)` | allocate sequential `results/MM-DD_run_NN/` |

## `fl_experiment_runner.py` — the sweep

Consumes the setup layer and owns the master-RNG draw order.

**`run_experiment(model_spec, design_spec, experiment, *, rng=None, progress=True)`**
— the sweep. Seeds the master RNG from `design.random_seed`, calls
`experiment.setup()` once, then loops $n \times p$ via `run_cell` (or the nested
runner when `design.sampling == "nested"`), returning a tidy DataFrame.

**`run_cell(...)`** — drives one cell and **owns the master-RNG draw order**:
(1) `build_model` draws first → (2) `cell_setup` (RNG-free) → (3) per-rep seeds
drawn → (4) each rep gets an independent child generator. Reordering that stream
changes every downstream number.

## `sim_theorem_partii.py` — the dispersion-bias probe

Theorem-specific code only:

- **`SineAlignmentAnalysis`** — observed LHS $\sin^2\angle(h_j, \bar b_j)$ via the $n\times n$ Gram trick ($O(p n^2)$).
- **`Eq6RHSAnalysis`** — predicted RHS: floor + weight·rotation, with $\delta^2$ read from `mean(diag(model.D))`.
- **`DispersionBiasExperiment`** — the `Experiment`: `setup()` registers the diagnostic `dist_sine`; `cell_setup` computes $\bar b_j$ once per cell (ARPACK) and returns `[Sine, Eq6RHS]`; `record` flattens to $k$ rows.
- **`simulate(design)`** — one-call driver: `resolve_model` then `run_experiment(...)`.
- **`main()`** — CLI.

## `fl_graphics.py`

Consumes the result DataFrame and produces the three convergence figures.

---

# Configuration: `ModelSpec` + `DesignSpec`

A run is a **model** plus a **design**. There is no separate "unified" spec
class — a `DesignSpec` carries its model in the `model` field, which accepts:

| `model` value | Meaning | File shape |
|---|---|---|
| `"path.json"` | reference, resolved relative to the design file | split (reusable model) |
| `{ ...fields... }` | inline model object | one file, nested |
| *top-level model fields* | `k_factors` etc. written flat; `from_json` **folds** them into an inline model | one file, flat (the shipped `sim_thmptii_spec.json` shape) |
| absent | `ModelSpec` defaults | — |

`DesignSpec.resolve_model(base_dir)` turns any of these into a concrete
`ModelSpec`. Mixing top-level model fields *and* a `model` reference raises
`ValueError`. JSON is read as UTF-8 so `σ/β/δ` comment fields load on Windows
(cp1252) too; `_`-prefixed keys are comments.

**Default config** (reproduces the original experiment exactly):

- `k_factors`: 3
- `n_values`: $[30, 60, 120]$ — periods (fixed)
- `p_values`: $[200, 500, 1000, 2000, 5000, 10000]$ — dimension (growing)
- `n_reps`: 300 per $(n, p)$ cell · `random_seed`: 20260511
- `factor_vols`: $\sigma = [0.16, 0.08, 0.06]$ (volatilities; variances $\sigma_j^2 = [0.0256, 0.0064, 0.0036]$)
- `beta_samplers`: market-like factor 1 $\beta_1 \sim N(1, 1)$, zero-mean unit factors 2,3 $\beta_j \sim N(0, 1)$ → prevalences $c = [2, 1, 1]$ (factors 2,3 zero-mean ⇒ off-diagonal Gram vanishes, $G_\infty = I_k$)
- `idio_vol_sampler`: constant vol 0.4 → $\delta^2 = 0.16$ after squaring
- return samplers: $N(0,1)$

`factor_vols` and the idio vol are both **volatilities**, squared into the
variance matrices $F$ and $D$ when the model is built. Effective spikes
$d_j = c_j \sigma_j^2 = [0.0512, 0.0064, 0.0036]$ satisfy Assumption 3.

Shipped examples: [sim_thmptii_spec.json](../sim_thmptii_spec.json) (flat single
file), and the [sim_thmptii_model.json](../sim_thmptii_model.json) +
[sim_thmptii_design.json](../sim_thmptii_design.json) split pair.

---

# The verification

## Claim (Theorem, Part ii, diagonal-Gram case)

As $p \to \infty$, for each factor $j$:

$$
\sin^2\angle(h_j, \bar b_j) \;\to\;
\underbrace{\frac{\delta^2}{n\rho_j + \delta^2}}_{\text{floor}}
+
\underbrace{\frac{n\rho_j}{n\rho_j + \delta^2}}_{\text{weight}}
\cdot
\underbrace{\sin^2\angle(\hat w_j, e_j)}_{\text{rotation}}
$$

where $\rho_j$, $\hat w_j$ are the $j$-th eigenpair of $\hat D = C^{1/2}(F^\top F/n)C^{1/2}$.

## LHS — `SineAlignmentAnalysis`

Takes $Y$ as $(p, n)$, forms the $n\times n$ Gram $G = Y^\top Y$, recovers the
top-$k$ left singular vectors $H = Y\,\mathrm{vecs}/s$, and compares to the
population directions $\bar b_j$ (precomputed once per cell) by **row order** —
which is what the theorem predicts. Gram trick: $O(p n^2)$ vs $O(p^2 n)$.

## RHS — `Eq6RHSAnalysis`

1. Empirical prevalences $c_j = \|B[j,:]\|^2/p$.
2. $\hat D = C^{1/2}(F^\top F/n)C^{1/2}$ ($k\times k$); eigendecompose → $\rho_j$, $\hat w_j$.
3. $\delta^2 = \mathrm{mean}(\mathrm{diag}(D))$ — `D` already stores variances, so no extra squaring. Single source of truth shared with the simulation.
4. floor $= \delta^2/(n\rho_j+\delta^2)$, weight $= n\rho_j/(n\rho_j+\delta^2)$, rotation $= 1 - (\hat w_j)_j^2$, RHS $=$ floor $+$ weight·rotation.

Result dict carries `delta2` alongside `rhs`/`floor`/`rotation`/`rhos`.

## Reproducibility invariant

The master-RNG draw order (steps 1, 3, 4 in `run_cell`) is the load-bearing
contract. The probe's hooks (`cell_setup`, the analyses, and `record`) never
draw from it, so swapping in a different `Experiment` cannot perturb the numbers. The per-cell-fresh model (β redrawn
every $(n,p)$) is the conditional-on-$F$ regime the Part-(ii) claim is stated
under; the engine guarantees the draw order, the probe gives it meaning.

## Output

Default: `results/MM-DD_run_NN/` (NN sequential per date), containing
`sim_thmptii.parquet` (`n,p,j,sin2_j,rhs,gap,floor,rotation,rho`) and three
figures; plus a console RMSE table of $(\sin^2 - \text{RHS})$ that should shrink
as $p$ grows. Override the directory with `--out` (CLI) or `output_path` (spec).

---

# CLI usage

```bash
# Built-in defaults
python sim_theorem_partii.py

# Single self-contained file (model folded inline)
python sim_theorem_partii.py sim_thmptii_spec.json

# Design referencing a model, with optional --model override
python sim_theorem_partii.py sim_thmptii_design.json
python sim_theorem_partii.py sim_thmptii_design.json --model sim_thmptii_model.json

# Save parquet + figures, or plot without saving; custom output path
python sim_theorem_partii.py sim_thmptii_spec.json --plot-save
python sim_theorem_partii.py --plot
python sim_theorem_partii.py sim_thmptii_spec.json --out my_results.parquet
```

Output path resolution: `--out` > `design.output_path` > auto `results/MM-DD_run_NN/`.

## Notebook idiom

```python
from fl_experiment_setup import ModelSpec, DesignSpec
from fl_experiment_runner import run_experiment
from sim_theorem_partii import DispersionBiasExperiment

df = run_experiment(ModelSpec(), DesignSpec(n_values=[60], p_values=[2000],
                                            n_reps=100), DispersionBiasExperiment())
```

To fix one model and vary only the return process, build the model once and call
`simulate_returns` (or `run_experiment`) with different `DesignSpec` return
samplers against it.

---

# Testing

`tests/test_sim_theorem_partii.py` + `tests/test_pipeline_e2e.py` — 181 tests,
99% line coverage across `sim_theorem_partii.py`, `fl_experiment_setup.py`, and
`fl_experiment_runner.py` (only `sys.path` bootstrap uncovered). Coverage
includes: spec loading and the unified-fold (incl. the conflict guard and UTF-8
regression), sampler helpers, `build_model` (vol→variance squaring), both
Analysis classes, `_rep_records`, `_next_run_dir`, the setup-layer seams (rep-RNG
isolation, fix-model-vary-returns, disjoint-key merge), nested sampling, the
byte-for-byte equality of the unified and split file shapes, the full
`simulate`/`run_experiment` path, loguru stage logs, graphics smoke tests, and
`main()` CLI dispatch.

```bash
python -m pytest tests/ -v
python -m pytest tests/ \
    --cov=sim_theorem_partii --cov=fl_experiment_setup --cov=fl_experiment_runner --cov-report=term-missing
```

---

# Extending: adding a new theorem

Write a new `Experiment` and hand it to the same engine — no changes to
`fl_experiment_setup` or `fl_experiment_runner`:

```python
class MyTheoremExperiment:
    def setup(self): ...                       # optional one-time registration
    def cell_setup(self, model, n, p):
        return [MyLhsAnalysis(...), MyRhsAnalysis()]
    def record(self, n, p, merged):
        return [ {... one row per factor ...} ]

df = run_experiment(model_spec, design_spec, MyTheoremExperiment())
```

Keep theorem-specific code in the probe script; the package (`factor_lab/`) and
the seams stay general. Genuinely-general helpers can migrate into
`factor_lab/` later if a second probe needs them — not before.

---

# Future ideas (not currently required)

**General Gram case ($G_\infty \neq I_k$).** Would measure rotation against the
population eigenvectors $w_j$ of $M = G_\infty^{1/2}(F^\top F/n)G_\infty^{1/2}$
instead of the standard basis $e_j$. Not load-bearing: any non-diagonal-Gram
model can be whitened/reordered into an equivalent diagonal-Gram model on the
same data, so the current simulation already establishes the substantive math.
If implemented, it would be a *new* `Experiment` (e.g. `GeneralGramExperiment`)
registered alongside the current one — not a replacement — preserving the
existing verification artifacts as a reference.
