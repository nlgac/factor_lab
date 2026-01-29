Thank you! It has been a pleasure iterating on this sophisticated quantitative framework with you. We have moved from a simple "script" to a **manifold-aware, sparse-matrix-optimized spectral analysis engine**.

Here is the complete documentation package to ensure this work is preserved, understood, and easily extensible.

### 1. Reflection on the Design

We refactored the codebase with three specific goals in mind, which are now fully realized:

1. **Computational Efficiency (The $O(p)$ vs $O(p^3)$ Problem)**:
   
   - *Challenge:* Computing eigenvectors for large covariance matrices ($\Sigma = B^T F B + D$) is memory-intensive.
   
   - *Solution:* We implemented **Implicit Decomposition** using `scipy.sparse.linalg.LinearOperator`. We never form the $500 \times 500$ matrix. We only define the function $v \mapsto \Sigma v$, reducing memory complexity from Quadratic to Linear.

2. **Geometric Rigor (Manifold Learning)**:
   
   - *Challenge:* Comparing a "True" factor model to an "Estimated" one is ambiguous because factors can be rotated or permuted.
   
   - *Solution:* We implemented distance metrics on two manifolds:
     
     - **Grassmannian ($Gr(k,p)$)**: Measures if the *subspaces* match (Rotation Invariant).
     
     - **Stiefel ($V_{k,p}$)**: Measures if the *frames* match (Rotation Sensitive), using Procrustes Analysis to solve for the optimal alignment.

3. **Data-Driven Configuration**:
   
   - *Challenge:* Hard-coding parameters makes experimentation slow.
   
   - *Solution:* We moved to a **JSON Specification** pattern. The code is now a generic engine that runs whatever experiment the JSON file describes.

---

### 2. Documentation & Cheatsheet (`README_ENHANCED.md`)

Save this file in your root directory. It serves as the primary manual for the project.

Markdown

```
# Factor Lab: Enhanced Spectral Analysis Engine

## 1. Overview
This toolkit simulates high-dimensional factor models ($p \gg k$) and performs "Deep Spectral Analysis" to verify how well estimated factors recover the ground truth. It utilizes **LinearOperators** for memory efficiency and **Manifold Geometry** for rigorous model comparison.

## 2. Quick Start

**1. Install Dependencies**
```bashpip install numpy scipy pandas matplotlib seaborn loguru
```

**2. Configure Your Experiment**

Edit `model_spec.json` (see Cheatsheet below).

**3. Run the Engine**

Bash

```
python build_and_simulate.py model_spec.json
```

**4. Analyze Results**

Output is saved to `.npz` files. Load them in Python:

Python

```
import numpy as np
data = np.load("simulation_student-t.npz")
print(f"Manifold Dist: {data['geodesic_distance']:.4f}")
```

---

## 3. Cheatsheet: JSON Configuration

The `model_spec.json` file controls everything.

| **Section**     | **Key**        | **Description**                   | **Example**                 |
| --------------- | -------------- | --------------------------------- | --------------------------- |
| **Meta**        | `p_assets`     | Number of assets ($p$)            | `500`                       |
|                 | `n_periods`    | Simulation duration ($T$)         | `63`                        |
| **Loadings**    | `distribution` | Distribution for $B$ entries      | `"normal"`                  |
|                 | `transform`    | Post-processing                   | `"gram_schmidt"`            |
| **Covariance**  | `F_diagonal`   | Factor variances (allows strings) | `["0.18^2", "0.05^2"]`      |
|                 | `D_diagonal`   | Idiosyncratic variance (scalar)   | `0.16`                      |
| **Simulations** | `type`         | Innovation distribution           | `"normal"` or `"student_t"` |
|                 | `params`       | Dist specific params              | `{"df_factors": 5}`         |

---

## 4. Cheatsheet: Output Metrics

When you load the results, here is what the metrics mean:

### Spectral Metrics

- **`true_eigenvalues`**: The top $k$ eigenvalues of the *theoretical* covariance $\Sigma = B^T F B + D$. Computed implicitly.

- **`sample_eigenvalues`**: The top $k$ variances explained by the SVD of the *simulated* returns.

### Manifold Metrics (Geometry)

- **`dist_grassmannian`**: Distance between the **Subspaces**.
  
  - *0.0* = Perfect match of the signal space.
  
  - *Invariant* to rotation/sign flips of factors.

- **`dist_stiefel_procrustes`**: Distance between the **Frames** after optimal alignment.
  
  - Solves for the rotation matrix $R$ that makes Sample match Truth.
  
  - Best metric for "Did I find the correct factors?"

- **`dist_stiefel_chordal`**: Raw Euclidean distance between frames.
  
  - High value if factors are flipped (e.g., Factor 1 is negative of Truth).

```
---

### 3. Knowledge Transfer Document (`KNOWLEDGE_TRANSFER.md`)
Save this file to help the next AI assistant (or yourself) pick up exactly where we left off.

```markdown
# Knowledge Transfer: Factor Lab Enhanced

**Date:** January 28, 2026
**Status:** Operational / Optimized
**Context:** High-Dimensional Factor Modeling & Spectral Analysis

## 1. System State
The system is currently capable of:
1.  **Specification:** Parsing complex model specs from JSON.
2.  **Construction:** Building $B, F, D$ matrices with Gram-Schmidt orthogonalization.
3.  **Simulation:** Generating returns with Fat-Tailed (Student-t) innovations.
4.  **Extraction:** Recovering factors via SVD (PCA).
5.  **Analysis:** Computing rigorous distances on Grassmannian/Stiefel manifolds and implicit spectral decomposition.

## 2. Key Files & Responsibilities

| File | Responsibility | Key Class/Function |
| :--- | :--- | :--- |
| **`build_and_simulate.py`** | **Main Driver**. Parses JSON, orchestrates build/sim/analyze loop, saves results. | `AnalysisEngine`, `FactorModelBuilder` |
| **`factor_lab/decomposition.py`** | **Math Core**. Efficient PCA/SVD. | `svd_decomposition`, `pca_decomposition` (LinearOperator support) |
| **`factor_lab/simulation.py`** | **Data Gen**. Simulates $r = B^T f + \epsilon$. | `ReturnsSimulator` |
| **`model_spec.json`** | **Config**. Defines experiment parameters. | N/A |

## 3. Mathematical Implementation Details

### A. Implicit Eigen-Decomposition
We do **not** form the full covariance matrix $\Sigma$ ($p \times p$).
Instead, we solve $\Sigma v = \lambda v$ using `scipy.sparse.linalg.eigsh` with a LinearOperator:
$$\text{matvec}(v) = B^T (F (B v)) + D v$$
* **Location:** `AnalysisEngine._compute_true_eigenvalues` in `build_and_simulate.py`.

### B. Manifold Distances
To compare True $B$ vs Sample $\hat{B}$:
1.  **Grassmannian:** $d_G(B, \hat{B}) = \|\theta\|_2$, where $\theta$ are principal angles.
2.  **Stiefel (Procrustes):** $d_P(B, \hat{B}) = \min_{Q \in O(k)} \| B - \hat{B}Q \|_F$.
    * This handles the "Sign Flip" and "Rotation" ambiguity inherent in SVD.
    * **Location:** `AnalysisEngine._compute_manifold_distances`.

## 4. Current "To-Do" / Future Directions
* **Visualization:** The current script saves data but does not generate PNG plots automatically.
* **Optimization:** `demo.py` contains portfolio optimization logic that could be integrated into the main pipeline.
* **Scaling:** If $p > 10,000$, we may need to switch from `scipy` ARPACK to `cupy` (GPU) for the SVD step.
```

### 4. Final Code Summary

You should now have the following file structure working correctly:

1. `build_and_simulate.py` (The updated script with Manifold logic)

2. `model_spec.json` (The configuration)

3. `factor_lab/decomposition.py` (The efficient solver library)

4. `factor_lab/simulation.py` (The simulator)

5. `README_ENHANCED.md` (Documentation)

6. `KNOWLEDGE_TRANSFER.md` (State persistence)
