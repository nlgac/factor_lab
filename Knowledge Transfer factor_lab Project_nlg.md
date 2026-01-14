Here is a **Knowledge Transfer (KT)** document.

You can save this text and paste it (or the "Prompt to Resume" section) into a new chat session. It contains everything the next AI needs to understand the architecture, design philosophy, and current state of the `factor_lab` project immediately.

---

# 📂 Knowledge Transfer: `factor_lab` Project

Date: January 13, 2026

Status: Alpha / Feature Complete

Architecture: Hybrid Simulation + Convex Optimization

## 1. Project Overview

We have built **`factor_lab`**, a high-performance Python package for financial factor model simulation and optimization. It replaces an older "Prototype" (`hl_factor_optimizer`) and "Production" (`production_factor_model`) attempt with a consolidated, best-practice design.

**Core Capabilities:**

- **Simulation:** Generates synthetic security returns based on factor structure ($R = f B + \epsilon$).

- **Optimization:** Solves Mean-Variance problems using SOCP (Second-Order Cone Programming) to avoid dense covariance matrices.

- **Analysis:** Recovers factors via PCA and validates risk metrics.

## 2. Key Architectural Decisions

The next AI needs to understand *why* the code looks this way:

1. **Hybrid Simulation Strategy:**
   
   - **Logic:** The `ReturnsSimulator` automatically inspects the Factor Covariance ($F$) and Idiosyncratic Covariance ($D$) matrices.
   
   - **Choice A (Diagonal):** If matrices are diagonal, it uses $O(N)$ element-wise scaling (fast).
   
   - **Choice B (Dense):** If matrices are dense (correlated), it falls back to $O(N^3)$ Cholesky decomposition (robust).

2. **Abstract Generators (The "Choice C" Pattern):**
   
   - **Problem:** We needed to separate the "Shape" of a distribution (e.g., Student's t vs. Normal) from its "Scale" (Volatility), which is dictated by the Factor Model.
   
   - **Solution:** `DistributionFactory.create_generator` returns a function `f(n) -> array`.
   
   - **Normalization:** The Simulator enforces `Mean=0, Std=1` on these samples before scaling them by the model's volatility.

3. **Composition over Inheritance:**
   
   - The `samplers` module separates **Registry** (storage), **Validator** (logic), and **Factory** (creation) to avoid a "God Object."

## 3. Current State & Files

The project is defined entirely by two installer scripts.

- **`install_factor_lab.py`**: Generates the package source code.
  
  - `types.py`: Dataclasses (`FactorModelData`, `OptimizationResult`).
  
  - `samplers.py`: `DistributionFactory` and `DataSampler`.
  
  - `simulation.py`: `ReturnsSimulator` and `ScenarioBuilder`.
  
  - `computation.py`: `FactorOptimizer` (CVXPY/Clarabel) and `pca_decomposition`.

- **`install_factor_lab_docs.py`**: Generates `README.md` and `API_REFERENCE.md`.

## 4. Known Issues / Active Context

- **`scikits` Warning:** The user's environment has an old `scikits` package causing `pkg_resources` deprecation warnings.
  
  - *Status:* Solved in `demo.py` via `warnings.filterwarnings`.
  
  - *Long-term fix:* User advised to uninstall/upgrade `scikits`.

---

## 🚀 Prompt to Resume (Copy/Paste this into New Chat)

**System/User Prompt:**

> I am working on a Python package called `factor_lab` for financial factor model simulation and optimization. We have just finished refactoring the architecture to use a "Hybrid Simulation" approach (switching between diagonal scaling and Cholesky based on matrix structure) and "Abstract Generators" for statistical sampling.
> 
> **I have uploaded current codebase as a single installer script in python called install_factor_lab.py.** Please analyze this code to understand the current architecture, types, and logic. Once you have analyzed it, confirm you are ready to continue development.

---

**(Optional) If you want the docs included in the context:**

> **I have also uploaded the documentation generator for context on the API as script called install_factor_lab_docs.py:**
> 
> 
