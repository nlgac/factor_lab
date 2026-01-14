Here is the updated Knowledge Transfer document reflecting the addition of SVD, the Loguru refactor, and the documentation fixes.

---

### 📂 Knowledge Transfer: factor_lab Project

Date: January 14, 2026

Status: Beta (Feature Complete)

Current State: Refactored & Enhanced (Loguru + SVD + Hybrid Engine)

#### 1. Architecture Overview

- **Simulation Engine (Hybrid):**
  
  - **Optimization:** Automatically switches between $O(N)$ scalar operations (for diagonal matrices) and $O(N^3)$ Cholesky decomposition (for dense matrices).
  
  - **Caching:** Checks `FactorModelData` for pre-computed square roots (`F_sqrt`, `D_sqrt`) to skip decomposition steps entirely.

- **Math Backend:**
  
  - **SVD (New):** Decomposes raw returns ($T \times N$) directly. Numerically superior to Covariance PCA. Automatically populates simulation cache.
  
  - **PCA:** Supports standard dense (`numpy`) and sparse/subset (`scipy`) methods.
  
  - **Optimization:** SOCP (Second-Order Cone Programming) via `cvxpy`.

- **Logging:**
  
  - Uses **`loguru`** for zero-config, color-coded output.
  
  - **Debug Feature:** `debug_intermediate_quantity_length` argument in `simulate()` logs raw, pre-scaled distribution samples to the console for inspection.

#### 2. Key API Changes

- **`svd_decomposition(returns, k)`**:
  
  - **Input:** Raw returns matrix (Time x Assets).
  
  - **Output:** `FactorModelData` with `F_sqrt` and `D_sqrt` pre-filled.
  
  - **Benefit:** Enables $O(1)$ initialization of `ReturnsSimulator`.

- **`ReturnsSimulator.simulate(...)`**:
  
  - **New Arg:** `debug_intermediate_quantity_length` (int). Controls how many rows of raw data are logged. Requires Logger Level = DEBUG.

- **`DataSampler.configure(...)`**:
  
  - Accepts explicit lists of generators (e.g., `beta=[gen1, gen2]`) or single generators for broadcasting.

#### 3. Current File Structure

- `factor_lab/types.py`: Dataclasses (`FactorModelData` now includes optional `_sqrt` fields).

- `factor_lab/computation.py`: Contains `svd_decomposition`, `pca_decomposition`, and `FactorOptimizer`.

- `factor_lab/simulation.py`: `ReturnsSimulator` (Cache-aware) and `ScenarioBuilder`.

- `factor_lab/samplers.py`: `DistributionFactory` and `DataSampler`.

- `tests/`:
  
  - `test_simulation.py`: Basic mechanics.
  
  - `test_samplers.py`: Broadcasting and validation logic.
  
  - `test_svd.py`: Verifies SVD variance recovery and cache usage.
  
  - `test_advanced.py`: Validates constraints and logging safety.

#### 4. Active Workflow

- **Demo:** `python demo.py` (Runs the full Generative $\to$ SVD $\to$ Optimization pipeline).

- **Tests:** `python -m pytest` (All tests passing).

- **Docs:** `python install_factor_lab_docs.py` (Generates `README.md` and `docs/API_REFERENCE.md`).

### 🚀 Prompt to Resume

Plaintext

```
I am working on the 'factor_lab' Python package.
We have just completed:
1. Implementing 'svd_decomposition' for raw returns analysis.
2. Refactoring logging to use 'loguru' with the 'debug_intermediate_quantity_length' parameter.
3. Updating the 'ReturnsSimulator' to use cached square roots from SVD models.
4. Ensuring full test coverage (SVD recovery, broadcast logic) and documentation.

I have uploaded the current codebase. Please load the context and help me with...
```
