# Factor Lab API Reference

## 1. Computation (factor_lab.computation)

### `svd_decomposition(returns: np.ndarray, k: int) -> FactorModelData`
Decomposes a raw returns matrix into a Factor Model using Singular Value Decomposition.
* **Logic:**
    1. Centers the returns data (Mean = 0).
    2. Performs SVD: $X = U S V^T$.
    3. Extracts top $k$ singular values.
    4. Constructs Loadings ($B$) and Covariances ($F, D$).
* **Benefits:**
    * Numerically more stable than PCA on Covariance matrices.
    * Populates `F_sqrt` and `D_sqrt` in the returned object.

### `pca_decomposition(M, k, method='numpy')`
Standard rank-k decomposition of a covariance matrix.
* **Args:**
    * `method='scipy'`: Uses `scipy.linalg.eigh` with `subset_by_index` for O(N^2 k) efficiency.

## 2. Simulation (factor_lab.simulation)

### `ReturnsSimulator`
#### `__init__(data: FactorModelData)`
* **Optimization:** Checks `data.F_sqrt` / `data.D_sqrt` first. If present, skips expensive Cholesky decomposition.

#### `simulate(..., debug_intermediate_quantity_length=100)`
* **`debug_intermediate_quantity_length`** (int):
    * Number of rows of raw data to log (requires Logger Level DEBUG).

## 3. Data Structures (factor_lab.types)

### `FactorModelData`
* **Fields:** `B`, `F`, `D`
* **Cache Fields:** `F_sqrt`, `D_sqrt` (Optional, populated by SVD)