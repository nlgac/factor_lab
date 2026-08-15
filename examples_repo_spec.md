# Spec: isolating the example notebooks into a separate repo

*Drafted 2026-08-15. Scope: what must be untangled from `factor_lab` for the ex-* /
mono example notebooks to live in their own repository.*

## 1. What "the examples" are

Three tiers, with very different dependency closures:

| Tier | Notebooks | Plotting stack | Data generation |
|---|---|---|---|
| **Modern** | ex-6, ex-7, ex-8, ex-9, `mono/ex-2…flplot`, `mono/ex-3…flplot`, `mono/ex-6…mono`, `fl_plot_demo` | `fl_plot` only | in-notebook classes driven by `fl_experiment_setup`/`fl_experiment_runner` (ex-9/mono-flplot: cache-only) |
| **Legacy** | ex-2_heavy_tail_v2 (+violin), ex-3 (+violin), `mono/ex-2…mono`, `mono/ex-3…mono`, ex-2c, ex-1 family | `fl_graphics` + `fl_visualization` (seaborn) | same |
| **Self-contained** | ex-4, ex-5 | none (inline matplotlib) | inline |

**Decision (2026-08-15): the new repo ships the modern tier only** (plus the
self-contained ex-4/ex-5 if wanted — they cost nothing). The legacy tier stays in the
mother repo as provenance; its figures are superseded by the flplot mono remakes, and
dropping it removes the `fl_graphics`/`fl_visualization`/seaborn stack from the closure
entirely.

One consequence to plan for: the flplot ex-2/ex-3 mono notebooks are *composition-only* —
they load `ex2v2_df_{p,n}.parquet` and raise if it's missing, and the sweep-runner code
that regenerates that cache lives only in the legacy ex-2 v2 notebooks that are staying
behind. See work item 2a.

## 2. Dependency closure (audited 2026-08-15)

```
notebooks (modern) ──► fl_plot.py ──► numpy, pandas, matplotlib
                                      (scipy.stats.gaussian_kde, lazy — DiskDensity only)
notebooks (all)    ──► fl_experiment_setup.py ──► factor_lab.distributions   (create_sampler)
                       fl_experiment_runner.py     factor_lab.model_builder  (FactorModelBuilder)
                                                   factor_lab.flexible_simulator (ReturnsSimulator)
                                                   factor_lab.analysis       (SimulationContext)
                                                   factor_lab.factor_types   (FactorModelData)
                                                   + loguru, tqdm
notebooks (legacy) ──► fl_graphics.py ──► fl_visualization.py ──► seaborn, loguru
```

Key facts:

- **`fl_plot.py` is already isolated.** No imports from `fl_graphics`, `fl_visualization`,
  or the `factor_lab` package. It moves as-is and becomes the examples repo's library.
- **The only hard coupling to the mother repo is data generation**: the five
  `factor_lab.*` submodules reached through `fl_experiment_setup`/`fl_experiment_runner`.
- ex-4 and ex-5 import nothing local at all.

## 3. The isolation boundary — three options

**A. Figures-only repo (smallest).** Move `fl_plot.py`, modern notebooks, and the
parquet caches as versioned data. Drop the sweep-runner cells or guard them behind
"requires the mother repo." No `factor_lab` dependency at all.
*Cost:* examples are not reproducible from scratch; caches become load-bearing artifacts.

**B. Self-contained repo (recommended).** Also move `fl_experiment_setup.py`,
`fl_experiment_runner.py`, and vendor the five `factor_lab` submodules (or a trimmed
`examples/_sim/` package exposing `create_sampler`, `FactorModelBuilder`,
`ReturnsSimulator`, `SimulationContext`, `FactorModelData`). Everything regenerates
with one command.
*Cost:* code duplication with the mother repo; needs a sync policy (declare the vendored
copy frozen — examples pin the paper, they don't track the library).

**C. Dependency repo.** Examples repo `pip install`s `factor_lab` from the mother repo
(git URL or private index).
*Cost:* couples the public examples to the private repo's history and packaging health;
weakest isolation, not recommended for a paper-companion repo.

## 4. Work items required regardless of option

1. **Path convention.** Every mono notebook uses the `REPO_ROOT` parent-hop
   (`if not (REPO_ROOT / "fl_plot.py").exists(): REPO_ROOT = REPO_ROOT.parent`) and
   `DATA_DIR = REPO_ROOT / "nb_outputs"`. Replace with a single config cell
   (`DATA_DIR = Path(os.environ.get("FL_DATA", "data"))`) or a tiny `examples_config.py`.
   Legacy mono notebooks hop on `fl_graphics.py` instead — same fix if they move.
2. **Cache policy.** Caches in play: `ex2v2_df_{p,n}` (2–3 MB each), `ex6_df_{p,n}`,
   `ex7_df_{n,p}`, `ex8_df_{p,n}` (+ `.spec.json` fingerprints). ~17 MB total — small
   enough to track in git directly (skip LFS). Required fixes before they become
   shared artifacts:
   - **ex-9 writes `ex8_df_p.parquet` with a shape-only guard** — it will silently
     clobber ex-8's fingerprinted cache. Give ex-9 its own filename or port the
     fingerprint guard.
   - Port the ex-8 spec-fingerprint sidecar pattern to ex-2v2/ex-6/ex-7 (their guards
     are existence/shape-only; stale-cache-serving already bit us once).
   - The big unrelated parquets in `nb_outputs/` (132 MB `rotation_runs_dense…`) do
     **not** move.

   **2a. Port the ex2v2 sweep runner.** With the legacy tier staying behind, the new
   repo needs its own way to regenerate `ex2v2_df_{p,n}.parquet`: extract the
   `MeasuredDecomposition` experiment classes + load-or-run cell from the legacy
   ex-2 v2 notebook into a script (`sim/make_ex2v2_data.py`) or a data-generation
   section in the flplot ex-2 notebook, with a spec-fingerprint sidecar. ex-6/ex-7/
   ex-8/ex-9 already carry their own runners and are unaffected.
3. **Environment manifest.** `pyproject.toml` with: numpy, pandas, matplotlib, scipy,
   pyarrow (parquet), jupyter/nbconvert; plus loguru + tqdm only under option B.
   (No seaborn — that dependency leaves with the legacy tier.) Kernelspec in the
   notebooks says `finance` — normalize to `python3` so fresh clones execute.
4. **Repo hygiene.** The mother `.gitignore` blanket-ignores `*.md`, `*.png`,
   `*.parquet`, `nb_outputs/` — the new repo needs its own written from scratch
   (track caches, ignore `mono/nb_outputs/`-style figure output dirs). Keep the
   established conventions: `SAVE_FIGS = False` default, per-figure `out(..., save=True)`,
   `FIG_FORMATS = ("pdf",)`. Consider lifting the (currently copy-pasted) `out()`
   helper and `FACTOR_HATCHES`/`MONO_THEME` into `fl_plot` so notebooks stop
   duplicating them.
5. **Execution contract.** CI smoke test: `jupyter nbconvert --execute` every notebook
   against the tracked caches (fast, no sweeps) — this is exactly the check we run
   locally after every edit; make it mechanical.
6. **Naming/notation.** If the pending Table-1 notation rename (code symbols → paper
   notation, e.g. the `A_j^{(p,n)}` relabels) is happening, do it **before** the move —
   renames across two repos are twice the work.

## 5. Suggested layout (option B)

```
examples-repo/
├── pyproject.toml            # deps; no install of the mother repo
├── README.md                 # what each notebook demonstrates, how to regenerate
├── fl_plot.py                # moved verbatim (or src/fl_plot/ if packaged)
├── sim/                      # fl_experiment_setup/runner + vendored factor_lab subset
├── data/                     # tracked parquet caches + .spec.json fingerprints
├── notebooks/
│   ├── paper_figures.ipynb   # THE paper figure set (see below)
│   ├── ex-4 … ex-9           # modern tier — exploratory walkthroughs
│   └── mono/                 # flplot mono remakes + ex-6 mono
└── .github/workflows/execute.yml
```

**`paper_figures.ipynb`** — one notebook that renders every figure used in the paper,
in paper order, and nothing else. Composition-only (loads the tracked caches; each
figure is a short `grid(...)` call using the marks + `MONO_THEME`/component themes from
`fl_plot`), and it is the **only** notebook where saving is on — every call carries
`out(..., save=True)`, `suptitle` omitted and `caption=False` for LaTeX captioning, PDF
output into a `figures/` directory the paper's `\includegraphics` points at. The ex-*
and mono notebooks stay as the exploratory record with `SAVE_FIGS = False`; a paper
figure is "promoted" by copying its grid call into `paper_figures.ipynb`. The figure
list is still being decided — the structure doesn't depend on it, and the flplot
notebooks were built precisely so each figure is one self-contained call to move.

```
```

## 6. Open decisions

- ~~Tier cut~~ — **decided: modern-only** (legacy fl_graphics notebooks stay in the
  mother repo).
- Option A vs B (B recommended; A is a legitimate v0 that can grow into B).
- Vendored-sim sync policy: frozen at paper submission vs periodically refreshed.
- Whether `fl_plot` moves as a single module or gets split/packaged (it's ~900 lines,
  single-file is fine for v1).
- Public naming, license, and whether the paper PDF/notation table lives in the repo.
- The `paper_figures.ipynb` figure list — undecided; blocks nothing structural, but
  should be settled before the CI workflow treats its outputs as the canonical
  paper artifacts.
