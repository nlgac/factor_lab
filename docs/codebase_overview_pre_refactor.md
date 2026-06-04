> # ⚠️ ARCHIVED — pre-refactor (do not rely on)
>
> This document describes the **single-script** design from before the
> orchestration was split into three layers (`fl_orchestration`, `fl_experiment`,
> `sim_theorem_partii`) and before `SimSpec` was folded into `DesignSpec`.
> It is kept only for historical reference.
>
> **Current doc:** [codebase_overview.md](codebase_overview.md) ·
> **Architecture:** [architecture_flowchart.md](architecture_flowchart.md)
>
> Still accurate here: the `factor_lab/` package description and the theorem /
> LHS / RHS math. Out of date: the configuration class, the simulation-loop
> structure, and the "Proposed Refactor" roadmap (now implemented).

---

# factor_lab: Codebase Overview (pre-refactor archive)

## Purpose

`factor_lab` is a Python package for building factor models, simulating returns, and analyzing the geometric and spectral properties of sample vs. population eigenvectors. It is the underlying infrastructure for the numerical verification of the multifactor dispersion bias theorems in *"Multifactor Dispersion Bias with Per-Column Prevalence."*

## Package Structure

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

## Core Data Structure

### `FactorModelData` (factor_types.py)

The central data container passed between all components. Represents the factor model $r = B^\top f + \varepsilon$. Fields:

- `B`: $(k, p)$ factor loadings matrix
- `F`: $(k, k)$ diagonal factor covariance matrix
- `D`: $(p, p)$ diagonal idiosyncratic covariance matrix
- `k`: number of factors (derived from `B.shape[0]`)
- `p`: number of assets (derived from `B.shape[1]`)
- `implied_covariance()`: computes $\Sigma = B^\top F B + D$
- `factor_transform`, `idio_transform`: optional transform matrices (used by `model_io` save/load full)

### `fl_types.py`

Contains additional type definitions used across the package. Not directly instantiated by end users but imported internally for type safety.

## Pipeline Overview

The package implements a four-stage pipeline:

**Stage 1 — Model Construction (model_builder.py)**
Build a static factor model by sampling loadings and idiosyncratic volatilities from specified distributions. Returns a `FactorModelData`. Does not simulate returns.

**Stages 2–4 — Return Simulation (flexible_simulator.py)**
Take an existing `FactorModelData` and simulate returns by sampling factor returns and idiosyncratic returns from specified distributions, then combining via $r = f @ B + \varepsilon$. Stateless — the same model can be reused with different return distributions.

**Estimation (decomposition.py)**
Extract a factor model from a $(T, p)$ returns matrix via SVD. Returns a `FactorModelData` representing the sample estimate. Used inside `SimulationContext.pca_decomposition()`.

**Analysis (analyses/)**
Compare ground truth model to sample estimate using geometric and spectral metrics.

## Component Details

### `distributions.py`

Universal sampler interface: any `callable(n) -> ndarray` works. Built-in distributions via `create_sampler(name, rng, **params)`: `normal`, `student_t`, `uniform`, `beta`, `exponential`, `gamma`, `constant`. `resolve_samplers` handles broadcasting a single sampler to a list of $k$ samplers or validating a per-factor list.

### `model_builder.py` — `FactorModelBuilder`

Implements Stage 1. Takes per-factor beta samplers, an idiosyncratic vol sampler, and explicit factor variances. Samples $B$ row by row (one row per factor), samples idiosyncratic vols, **squares them to get $D$** (i.e. the sampler outputs volatilities; $D$ stores variances), and constructs $F = \mathrm{diag}(\text{factor\_variances})$. Single responsibility: model creation only.

### `flexible_simulator.py` — `ReturnsSimulator`

Implements Stages 2–4. Takes a `FactorModelData` and return distribution samplers. Samples raw factor returns and scales by $\sqrt{F}$ diagonal; samples raw idiosyncratic returns in one vectorized call and scales by $\sqrt{D}$ diagonal; combines via $r = f @ B + \varepsilon$. Stateless and reusable across different return distributions.

### `decomposition.py` — `svd_decomposition`

Extracts a factor model from returns via SVD. Centers the returns matrix, runs full SVD, takes the top-$k$ right singular vectors as $B$, computes factor variances from singular values, and residualizes to get $D$. Sign normalization: rows of $B$ are flipped so their mean is non-negative.

### `analysis/context.py` — `SimulationContext`

Frozen dataclass holding a snapshot of model + returns for analysis. Lazy-cached `sample_covariance()` and `pca_decomposition(n_components)`. Properties: `T` (periods), `p` (assets), `k` (factors).

### `analyses/manifold.py` — `ManifoldDistanceAnalysis`

Three distance metrics between ground truth and estimated factor loadings:

1. **Grassmannian** — L2 norm of principal angles between subspaces, rotation-invariant.
2. **Procrustes** — Frobenius distance after optimal orthogonal alignment, handles sign flips.
3. **Chordal** — raw Frobenius distance between orthonormalized frames, no alignment.

Also contains `compute_sine_alignment` for per-factor $\sin^2\angle(h_j, \bar b_j)$ matched by row order (eigenvalue rank), which is what the main theorem predicts. An `_EXTRA_DISTANCES` registry allows additional scalar metrics to be registered at runtime via `register_manifold_distance`.

### `analyses/spectral.py` — `ImplicitEigenAnalysis`

Computes top-$k$ eigenvalues/eigenvectors of $\Sigma = B^\top F B + D$ using a `LinearOperator` that performs $\Sigma v = B^\top(F(Bv)) + Dv$ without forming the full $p\times p$ matrix. Uses ARPACK (`eigsh`) for iterative solving. Memory: $O(kp)$ instead of $O(p^2)$. Compares true eigenvalues to sample eigenvalues from PCA.

### `analyses/eigenvector.py` — `EigenvectorAlignment`

Compares true eigenvectors of $\Sigma$ (from `compute_true_eigenvalues`) to PCA eigenvectors from sample returns. Metrics: subspace distance (principal angles), Procrustes distance, per-vector canonical correlations. Handles sign ambiguity via dot-product alignment.

### `analyses/builder.py` — `Analyses`

Factory class with static methods for creating the standard analyses: `Analyses.manifold_distances()`, `Analyses.eigenvalue_analysis()`, `Analyses.eigenvector_comparison()`, `Analyses.custom(func)`.

### `integration.py`

All-in-one pipeline function `build_simulate_analyze` that chains model building → return simulation → SVD estimation → analysis. Also exposes `build_simulate_analyze_from_model` for reusing an existing model with different return distributions, and `run_analyses` for running named analyses on an existing `SimulationContext`. Analysis dispatch table maps string names (`'manifold'`, `'eigenvalue'`, `'eigenvector'`) to analysis runners.

### `visualization/visualization.py`

Visualization utilities for the package. Provides plotting helpers used by the broader codebase. Note: the primary plotting layer for the theorem verification simulation lives in `fl_graphics.py` at the repo root rather than inside the package.

### `simulation.py`

Deprecated legacy simulator. Raises `DeprecationWarning` on instantiation and points to `flexible_simulator.ReturnsSimulator`. Uses Cholesky decomposition for normal returns only — no flexible distribution support.

### `model_io.py`

Save and load `FactorModelData` to/from `.npz` files via `save_model` / `load_model` (B, F, D only) and `save_model_full` / `load_model_full` (includes optional `factor_transform` and `idio_transform`).

## Top-Level Files (Repo Root)

- **`sim_theorem_partii.py`** — Main simulation script for verifying the main theorem (see next section).
- **`fl_graphics.py`** — Plotting layer; consumes a DataFrame from simulation and produces three figures: convergence plot (gap vs. $p$), scatter plot (LHS vs. RHS), and components plot (floor and rotation).
- **`sim_thmptii_spec.json`** / **`sim_thmptii_standard_setup.json`** — JSON spec files for the simulation.
- **`tests/test_sim_theorem_partii.py`** — Comprehensive test suite (53 tests, 100% line coverage of `sim_theorem_partii.py`).
- **`perturbation_study.py`**, **`large_sample_perturbation_study.py`** — Earlier perturbation experiments.
- **`proof_walkthrough_k3_cleaned.md`** — Step-by-step numerical walkthrough of the main theorem proof with $k=3$, $p=500$, $n=60$ example.
- **`unified_dispersion_bias_proof_051926_cleaned.md`** — Latest proof document.
- **`defaults.json`**, **`full.json`**, **`micro.json`**, **`toy.json`**, etc. — Various JSON spec files for different simulation configurations ranging from quick smoke tests (`micro`, `toy`) to full runs.

# `sim_theorem_partii.py`: What It Does

## Purpose

Numerically verifies the main theorem (Multifactor Dispersion Bias, Part ii) from *"Multifactor Dispersion Bias with Per-Column Prevalence."* The theorem states that as $p \to \infty$, for each factor $j$:

$$
\sin^2\angle(h_j, \bar b_j) \;\longrightarrow\;
\underbrace{\frac{\delta^2}{n\rho_j + \delta^2}}_{\text{floor}}
+
\underbrace{\frac{n\rho_j}{n\rho_j + \delta^2}}_{\text{weight}}
\cdot
\underbrace{\sin^2\angle(\hat w_j, w_j)}_{\text{rotation}}
$$

The script computes both the LHS (observed $\sin^2$ from simulation) and the RHS (theoretical prediction from the data) across many replications and $p$ values, and checks that the gap between them shrinks to zero as $p$ grows.

> **Note (architecture refresh pending).** The sections below describe the
> original single-script design. The codebase has since been split into three
> layers — `fl_orchestration` (seams), `fl_experiment` (generic engine:
> `ModelSpec`, `DesignSpec`, `Experiment`, `run_experiment`), and
> `sim_theorem_partii` (the dispersion-bias probe). See
> [architecture_flowchart.md](architecture_flowchart.md) for the current shape.
> The configuration notes here remain accurate at the field level.

## Configuration — `ModelSpec` + `DesignSpec`

A run is specified by a **model** (`ModelSpec`: k, factor variances, β and
idio-vol samplers) and a **design** (`DesignSpec`: n/p grids, reps, seed, return
samplers, output). A `DesignSpec` carries its model under the `model` field,
which may be a path reference, an inline object, or — for a single self-contained
file — model fields written at the JSON top level (folded in by
`DesignSpec.from_json`). There is no separate unified spec class. The script can
run three ways:

1. **No arguments** — built-in `ModelSpec()` / `DesignSpec()` defaults, which reproduce the original hardcoded experiment exactly.
2. **Positional JSON spec** — `DesignSpec.from_json(path)` loads one file (unified single-file, inline-model, or model-by-reference); `--model PATH` overrides the model; keys starting with `_` are comments.
3. **CLI overrides** — `--out`, `--plot`, `--plot-save` flags override design-level settings.

JSON files are opened with explicit `encoding="utf-8"` so that comment fields containing Greek letters (σ, β, δ) load correctly on Windows (where the platform default is cp1252).

Key parameters in the default config:

- `k_factors`: 3
- `n_values`: $[30, 60, 120]$ — time periods (fixed, not growing)
- `p_values`: $[200, 500, 1000, 2000, 5000, 10000]$ — ambient dimension (growing)
- `n_reps`: 300 replications per $(n, p)$ cell
- `random_seed`: 20260511
- `factor_variances`: $[0.04, 0.02, 0.01]$ — $\sigma_j^2$, satisfying Assumption 3 (strict decrease of $c_j \sigma_j^2$)
- `beta_samplers`: $N(0, \sqrt{c_j})$ per factor with $c = [1.0, 0.8, 0.6]$, giving diagonal Gram $G_\infty = I_k$
- `idio_vol_sampler`: constant vol $1.0$ (homoskedastic, $\delta^2 = 1$ after squaring)
- `factor_return_sampler` / `idio_return_sampler`: $N(0,1)$, scaled internally by factor variances / idio vols

Effective spikes: $d_j = c_j \sigma_j^2 = [0.040, 0.016, 0.006]$, satisfying Assumption 3.

### Vol-domain knobs (idio side)

The spec's `idio_vol_sampler` outputs **volatilities**; `FactorModelBuilder` squares them internally to populate `D`. There is no separate `idio_variance` knob in the spec — this used to exist but was removed because it duplicated information already in the sampler and could fall out of sync with it. Inside `Eq6RHSAnalysis`, $\delta^2$ is read directly from `mean(diag(model.D))`, guaranteeing the simulation's $D$ and the predicted RHS use the same value.

## Simulation Loop Structure

```
For each n in n_values:
    For each p in p_values:
        - Build a fresh factor model once (B, F, D) via FactorModelBuilder
        - Compute population loading directions b̄_j once via
          compute_true_eigenvalues (ARPACK, runs once per cell)
        - For each of 300 replications (independently seeded):
            - Sample factor returns F and idio returns Z via ReturnsSimulator
            - Compute LHS via SineAlignmentAnalysis
            - Compute RHS via Eq6RHSAnalysis
            - Record k=3 rows per replication
```

Total rows for the default spec: $3 \text{ factors} \times 300 \text{ reps} \times 6 \text{ p-values} \times 3 \text{ n-values} = 16{,}200$ rows.

**Per-cell-fresh model is intentional.** The asymptotic claim in Part (ii) is *conditional on $F$ across fresh $\beta$*, so $B$ is redrawn for every $(n, p)$ cell. Any future refactor that exposes "model" as a savable, reusable artifact must still honor this contract for the verification path; reusing one drawn model across cells would be a different experiment.

## LHS Computation — `SineAlignmentAnalysis`

Takes $Y$ as $(p, n)$. Computes the $n \times n$ Gram $G = Y^\top Y$, eigendecomposes it, recovers the top-$k$ left singular vectors $H$ of $Y$ via $H = Y \cdot \mathrm{vecs} \,/\, \mathrm{singular\_values}$. Computes $\sin^2\angle(h_j, \bar b_j)$ per factor using `compute_sine_alignment` — matching is by row order (eigenvalue rank), not optimal alignment, which is what the main theorem predicts.

The $n \times n$ Gram trick costs $O(p \cdot n^2)$ instead of $O(p^2 \cdot n)$ for the full SVD, which matters at large $p$. Population directions $\bar b_j$ are precomputed once per $(n, p)$ cell and passed in at construction — ARPACK does not run inside the rep loop.

## RHS Computation — `Eq6RHSAnalysis`

Computes the theoretical prediction from the realized factor returns $F$:

1. **Empirical prevalences**: $c_j = \|B[j,:]\|^2 / p$ (mean of squared loadings per factor row).
2. **$\hat D = C^{1/2}\,(F^\top F / n)\,C^{1/2}$**, a $k \times k$ matrix.
3. **Eigendecompose** $\hat D \to \rho_j$ (descending eigenvalues), $\hat w_j$ (eigenvectors).
4. **$\delta^2$** = mean of diagonal of `model.D`. `D` already stores variances (FactorModelBuilder squared the sampled vols), so no additional squaring happens here. This is the single source of truth for $\delta^2$ used in the formula.
5. **Floor** = $\delta^2 / (n\rho_j + \delta^2)$.
6. **Weight** = $n\rho_j / (n\rho_j + \delta^2)$.
7. **Rotation** = $1 - (\hat w_j)_j^2 = \sin^2\angle(\hat w_j, e_j)$ — squaring the diagonal removes sign ambiguity.
8. **RHS** = floor + weight $\cdot$ rotation.

The result dict carries `"delta2"` in addition to `"rhs"`, `"floor"`, `"rotation"`, and `"rhos"`, so the realized $\delta^2$ is inspectable per-cell.

## Output

- **Parquet file** — saved by default to `results/MM-DD_run_NN/sim_thmptii.parquet`, with `NN` auto-allocated sequentially per date (e.g. `results/05-25_run_03/...`). Columns: `n`, `p`, `j`, `sin2_j`, `rhs`, `gap`, `floor`, `rotation`, `rho`.
- **RMSE table** — printed to console: RMSE of $(\sin^2 - \text{RHS})$ by $(n, p, j)$. Should decrease as $p$ grows — this is the numerical confirmation of the theorem.
- **`fig_theorem1_convergence_v2.png`** — gap vs. $p$ (median ± IQR), showing convergence to zero.
- **`fig_theorem1_scatter_v2.png`** — LHS vs. RHS scatter at second-largest $p$, showing tight alignment along the $y = x$ line.
- **`fig_theorem1_components_v2.png`** — floor and rotation terms vs. $p$ as boxplots, showing both are $p$-stable.

Run directories are allocated by a helper that scans existing siblings under `results/`, parses any `MM-DD_run_NN` names matching today's date with regex `^MM-DD_run_(\d+)$`, and picks `max(NN) + 1`. Unrelated directory names are ignored. Both `--out` (CLI) and the optional `output_path` (spec field) override this default.

## CLI Usage

```bash
# Run with built-in defaults (no spec file)
python sim_theorem_partii.py

# Run with custom JSON spec
python sim_theorem_partii.py sim_thmptii_spec.json

# Run and save parquet + plots
python sim_theorem_partii.py sim_thmptii_spec.json --plot-save

# Run, generate plots but skip saving parquet
python sim_theorem_partii.py --plot

# Override output parquet path
python sim_theorem_partii.py sim_thmptii_spec.json --plot-save --out my_results.parquet
```

### Resolution order for the output path

`--out` (CLI) > `spec.output_path` (JSON) > auto-allocated `results/MM-DD_run_NN/sim_thmptii.parquet`.

## Testing

The script ships with a comprehensive pytest suite at `tests/test_sim_theorem_partii.py`:

- **53 tests, 100% line coverage** of `sim_theorem_partii.py`.
- **`ModelSpec` / `DesignSpec`** — defaults reproduce the original experiment; `from_json` round-trips; `_`-prefixed comment keys are dropped; the unified single-file shape folds top-level model fields into an inline model; shipped spec files load; non-ASCII content (σ, β, δ) loads on any platform (regression guard against Windows cp1252).
- **Sampler helpers** — `_make_one_sampler` for normal & constant; `_make_samplers` broadcast vs. list; wrong-length raises.
- **`build_model`** — shapes; $F = \mathrm{diag}(\text{factor\_variances})$; **$D$'s diagonal = vol² (vols squared into variances)**; empirical prevalences converge.
- **`SineAlignmentAnalysis`** — keys, shape/range, perfect-recovery sanity.
- **`Eq6RHSAnalysis`** — $\delta^2$ derived from `model.D`; constant vol $v \Rightarrow \delta^2 = v^2$ (end-to-end vol-to-variance contract); diagonal-$F$ case (rotation = 0, rhs = floor); floor ≤ rhs.
- **`_rep_records`** — $k$-parameterized length; gap = sin² − rhs; $j$ is 1-indexed.
- **`_next_run_dir`** — sequential allocation, 2-digit zero-padding, ignores unrelated names, picks `max(NN) + 1`.
- **`simulate()`** — small-spec schema and row count; reproducible under same seed; different seeds produce different outputs.
- **Graphics** — smoke tests for `plot_convergence`, `plot_scatter`, `plot_components`, `plot_all`.
- **`main()` CLI** — no config → defaults + auto-allocated run dir; positional design spec → `DesignSpec.from_json`; `--model` override; `--out` > `design.output_path` > auto-allocation; `--plot` skips parquet; `--plot-save` writes both.

Run with `python -m pytest tests/test_sim_theorem_partii.py -v` (add `--cov=sim_theorem_partii --cov-report=term-missing` for the coverage report).

## Current Limitations / Known Issues

- Orchestration is coupled inside `sim_theorem_partii.py`: model construction, return sampling, analysis, and plotting are interleaved in one loop, so each stage isn't independently configurable.
- Model and return generation share one JSON file. There is no first-class way to fix the model and vary only the return distribution.
- The script isn't importable into a Jupyter notebook cleanly: `from sim_theorem_partii import …` works with `sys.path` munging, but the public API surface isn't documented or stable.

## Proposed Refactor (Roadmap)

Three changes, sequenced from lowest risk to most speculative. Faithful to the proposals in the original codebase overview.

1. **Decouple stages in the orchestrator.** `sim_theorem_partii.py` interleaves model construction, return sampling, analysis, and plotting in one loop. Refactor so each stage is independently configurable — build a model once, run multiple return-sampling configs against it, dispatch analyses separately. The `factor_lab/` package layer is already split this way; this is the orchestration-layer change.

2. **Split the JSON spec into model + experiment.** *Model spec*: `k_factors`, `beta_samplers`, `idio_vol_sampler`, `factor_variances`. *Experiment spec*: `n_values`, `p_values`, `n_reps`, `random_seed`, both return samplers, `output_path`. The experiment spec references the model spec by path so one model can be reused across many experiments. Alternative worth weighing: keep one schema and support `--config a.json --config b.json` overlay, extending the `defaults.json` + `full.json` pattern already in the repo.

3. **Improve notebook importability without lifting into the package.** Keep `sim_theorem_partii.py` at the repo root — `factor_lab/` stays general-purpose, since dispersion-bias verification is one analysis among many it supports. Make the script itself a clean module: add `__all__`, keep symbol names stable, document the notebook idiom (`from fl_experiment import ModelSpec, DesignSpec, run_experiment` + `from sim_theorem_partii import DispersionBiasExperiment`). Genuinely-general helpers (sampler resolution, run-dir allocator) can move into `factor_lab/utils/` later if a second script needs them — not before.

### Invariants the refactor must preserve

- **Per-cell-fresh model in the verification path.** The verification keeps redrawing β every $(n, p)$ cell — that's the regime the theorem's claim is conditional on. The new flexibility (fix one model, vary returns) is for *other* experiments, not the verification.
- **Reproducibility under seed.** Per-rep seeded RNGs are already deterministic; the refactor must not introduce thread-local state or non-deterministic dispatch that breaks this.
- **`factor_lab/` stays general.** Dispersion-bias-specific code (the two Analysis classes, `DispersionBiasExperiment`) stays in the probe script. Nothing dispersion-bias-shaped should leak into the package.
- **Diagonal-Gram is the canonical case** (see Future Ideas).

## Future Ideas (Not Currently Required)

**General Gram case (Theorem 3, $G_\infty \neq I_k$).** The current `Eq6RHSAnalysis` assumes diagonal Gram, with rotation measured against the standard basis $e_j$. A general-Gram variant would measure rotation against the population eigenvectors $w_j$ of $M = G_\infty^{1/2}\,(F^\top F/n)\,G_\infty^{1/2}$ instead.

We do not consider this load-bearing for the verification project: any non-diagonal-Gram model can be transformed (whitening the loading Gram, then reordering) into an equivalent diagonal-Gram model on the same data, and the diagonal-Gram theorem applies after that reprocessing. So the diagonal-Gram simulation already establishes the substantive math; a general-Gram simulation would only add notational coverage, not new evidence.

If implemented, it would be a *new* analysis class (e.g. `EqXRHSAnalysis`) registered alongside `Eq6RHSAnalysis`, not a replacement — preserving the current verification artifacts as a reference.
