# legacy/ — archived one-off scripts and configs

Superseded exploratory scripts (mostly April 2026) and their JSON configs,
moved here to keep the repo root focused on the current pipeline. Kept for
reference; not part of the maintained code.

**Current pipeline** (repo root): `fl_orchestration.py`, `fl_experiment.py`,
`sim_theorem_partii.py`, `fl_graphics.py`, `fl_visualization.py`, the
`factor_lab/` package, `tests/`, and `theorem_partii_walkthrough.ipynb`.
See [docs/codebase_overview.md](../docs/codebase_overview.md).

## Contents

- **Perturbation experiments** — `perturbation_study.py`,
  `large_sample_perturbation_study.py`, `perturbation_iter.py` and their configs
  (`perturbation_spec*.json`, `large_sample_study_homo_normal.json`,
  `model_many_observations.json`, `updated_spec_0422.json`).
- **NPZ / misc utilities** — `inspect_npz.py`, `verify_npz_output.py`,
  `distance_compare_plots.py`, `demo.py`, `dispersion_bias_conjecture_test.py`.
- **Old generic spec configs** — `defaults.json`, `full.json`, `micro.json`,
  `toy.json`, `model_spec.json` (an older, different schema unrelated to the
  current `model_spec.example.json`).

These were verified to be imported by nothing in the current pipeline or test
suite before archiving.
