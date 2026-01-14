# Factor Lab API Reference

## 1. Data Structures (factor_lab.types)

### FactorModelData
Immutable data class representing the structural components of a risk model.
R = f * B + epsilon

| Attribute | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `B` | `np.ndarray` | (k, p) | Factor Loading Matrix (Exposures). |
| `F` | `np.ndarray` | (k, k) | Factor Covariance Matrix. |
| `D` | `np.ndarray` | (p, p) | Idiosyncratic Covariance Matrix. |

## 2. Statistical Sampling (factor_lab.samplers)

### DistributionFactory
Singleton-like orchestrator for creating statistical generators.

#### `create_generator(dist_name: str, **params)`
Creates a frozen generator function that accepts a sample size n.
* **Args:**
    * `dist_name`: Name of distribution (e.g., 'normal', 'student_t').
    * `**params`: Distribution-specific parameters (e.g., mean, df).
* **Returns:** A function `f(n)` returning an array of shape `(n,)`.

### DataSampler
Builder class for generating synthetic `FactorModelData`.

#### `configure(beta, f_vol, d_vol)`
Configures the generators used to populate the matrices.
* **Arguments:**
    * `beta`: Generator(s) for Factor Loadings (B).
    * `f_vol`: Generator(s) for Factor Volatilities (sqrt(F)).
    * `d_vol`: Generator(s) for Idiosyncratic Volatilities (sqrt(D)).
* **Behavior:**
    * **Broadcasting:** Pass a single generator function to apply it to all elements.
    * **Explicit Control:** Pass a `list` of generators to assign specific distributions per element.
        * List length must match `k` (for `beta`, `f_vol`) or `p` (for `d_vol`).

#### `generate() -> FactorModelData`
Executes the sampling logic and returns the populated data object.

## 3. Simulation (factor_lab.simulation)

### ReturnsSimulator
High-performance time-series simulator.

#### `__init__(data: FactorModelData, seed: Optional[int])`
Initializes the simulator. Automatically detects if F and D are diagonal to select the fastest simulation strategy (Scalar vs. Cholesky).

#### `simulate(n: int, f_gens: List, i_gens: List)`
Generates returns for n periods.
* **Logic:**
    1.  Generate raw samples from generators.
    2.  **Standardize** samples to Mean=0, Variance=1.
    3.  **Scale** by model volatility (F and D).
    4.  Combine: Returns = (f_scaled @ B) + epsilon_scaled.

## 4. Computation (factor_lab.computation)

### FactorOptimizer
Convex optimizer for factor models.

#### `solve() -> OptimizationResult`
Solves the Mean-Variance problem using SOCP reformulation.

### pca_decomposition(M, k)
Performs rank-k decomposition of symmetric matrix M.