# Factor Lab Manifold Analysis - Technical Manual

**For Specialist Users**

A comprehensive technical guide to the mathematical foundations, computational methods, interpretation guidelines, and visualization details for factor model analysis using manifold geometry.

---

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
2. [Manifold Distance Metrics](#manifold-distance-metrics)
3. [Eigenvalue Analysis](#eigenvalue-analysis)
4. [Eigenvector Comparison](#eigenvector-comparison)
5. [Computational Methods](#computational-methods)
6. [Interpretation Guidelines](#interpretation-guidelines)
7. [Visualization Reference](#visualization-reference)
8. [Advanced Topics](#advanced-topics)

---

## 1. Mathematical Framework

### 1.1 Factor Model Structure

The fundamental factor model is:

```
r_t = B'f_t + ε_t
```

Where:
- **r_t** ∈ ℝ^p: Security returns at time t
- **B** ∈ ℝ^(k×p): Factor loadings matrix
- **f_t** ∈ ℝ^k: Factor returns at time t (k ≪ p)
- **ε_t** ∈ ℝ^p: Idiosyncratic returns at time t

**Covariance structure:**
```
Σ = B'FB + D
```

Where:
- **Σ** ∈ ℝ^(p×p): Returns covariance matrix
- **F** ∈ ℝ^(k×k): Factor covariance matrix (often diagonal)
- **D** ∈ ℝ^(p×p): Idiosyncratic covariance matrix (often diagonal)

**Key properties:**
- Rank-k approximation: rank(B'FB) = k
- Total variance: tr(Σ) = tr(B'FB) + tr(D)
- Explained variance ratio: tr(B'FB) / tr(Σ)

### 1.2 Manifold Geometry

Factor loadings **B** define a **k-dimensional subspace** in ℝ^p. This subspace lies on the **Grassmann manifold** Gr(k,p), which is the space of all k-dimensional subspaces of ℝ^p.

**Key insight:** Factor models are inherently ambiguous due to rotational invariance:
```
If B is a valid loading matrix, so is RB for any orthogonal R ∈ O(k)
```

This motivates the use of manifold-based distance metrics that are invariant to such transformations.

### 1.3 The Stiefel Manifold

The **Stiefel manifold** St(k,p) is the space of orthonormal k-frames in ℝ^p:
```
St(k,p) = {Q ∈ ℝ^(p×k) : Q'Q = I_k}
```

Any loading matrix B can be projected onto the Stiefel manifold via QR decomposition:
```
B = QR  where Q ∈ St(k,p), R ∈ ℝ^(k×k)
```

**Relationship:**
- Grassmann manifold = Stiefel manifold / O(k)
- Grassmannian distances are rotation-invariant
- Stiefel distances depend on the chosen frame

---

## 2. Manifold Distance Metrics

### 2.1 Grassmannian Distance (Subspace Distance)

**Mathematical Definition:**

The Grassmannian distance between subspaces spanned by B_true and B_est is:

```
d_G(B_true, B_est) = ||θ||_2 = sqrt(Σᵢ θᵢ²)
```

Where **θ = (θ₁, ..., θₖ)** are the **principal angles** between the subspaces.

**Principal Angles:**

Principal angles θᵢ ∈ [0, π/2] are defined recursively as:

```
cos(θᵢ) = max    max    |u'v|
         u∈Uᵢ  v∈Vᵢ
```

Where:
- U₁ = span(B_true), V₁ = span(B_est)
- Uᵢ₊₁ = Uᵢ ∩ {u : u ⊥ uᵢ*}, similarly for Vᵢ₊₁
- uᵢ*, vᵢ* are the optimal vectors achieving the maximum

**Computation via SVD:**

Given orthonormal bases Q_true, Q_est:
```
M = Q_true' Q_est
U, Σ, V' = SVD(M)
σᵢ = cos(θᵢ)
θᵢ = arccos(σᵢ)
```

**Properties:**
- **Range:** d_G ∈ [0, √(kπ²/4)] 
- **Rotation invariant:** d_G(QB, Q'B) = d_G(B, B') for any orthogonal Q
- **Symmetric:** d_G(B₁, B₂) = d_G(B₂, B₁)
- **Triangle inequality:** Satisfies metric axioms
- **Geometric interpretation:** Arc length on Grassmannian geodesic

**Interpretation Guidelines:**

| d_G Value | Subspace Alignment | Interpretation |
|-----------|-------------------|----------------|
| < 0.1 | Excellent (> 99%) | Factors nearly identical |
| 0.1 - 0.3 | Good (> 95%) | Minor rotation/noise |
| 0.3 - 0.5 | Moderate (> 90%) | Significant rotation |
| > 0.5 | Poor (< 90%) | Subspaces diverged |

**When θ₁ is large (> 0.3 rad = 17°):**
- Primary factor directions differ substantially
- May indicate model misspecification or insufficient data
- Check sample size T/p ratio (should be > 2)

**Scale reference:**
- π/2 ≈ 1.571 rad = 90° (orthogonal subspaces)
- π/4 ≈ 0.785 rad = 45° (halfway)
- π/8 ≈ 0.393 rad = 22.5°

### 2.2 Procrustes Distance (Aligned Frame)

**Mathematical Definition:**

The Procrustes distance finds the optimal orthogonal rotation to align frames:

```
d_P(B_true, B_est) = min ||Q_true - Q_est R||_F
                      R∈O(k)
```

Where Q_true, Q_est are orthonormalized versions of B.

**Optimal Rotation (Kabsch Algorithm):**

```
M = Q_est' Q_true
U, Σ, V' = SVD(M)
R* = UV'
d_P = ||Q_true - Q_est R*||_F
```

**Properties:**
- **Range:** d_P ∈ [0, 2√k]
- **Achieves optimal alignment:** Minimizes Frobenius norm after rotation
- **Sign & permutation invariant:** Handles factor ambiguities
- **Not rotation invariant:** Depends on absolute orientations

**Relationship to Grassmannian:**

```
d_P² = 2k - 2Σᵢ cos(θᵢ)
```

For small angles: d_P ≈ ||θ||_2 = d_G

**Interpretation Guidelines:**

| d_P Value | Frame Alignment | Interpretation |
|-----------|----------------|----------------|
| < 0.1 | Excellent | Frames align after optimal rotation |
| 0.1 - 0.5 | Good | Minor misalignment |
| 0.5 - 1.0 | Moderate | Substantial rotation needed |
| > 1.0 | Poor | Frames poorly aligned |

**When d_P ≫ d_G:**
- Frames require significant rotation to align
- Factor ordering may differ
- Sign flips present

**When d_P ≈ d_G:**
- Frames are already well-aligned
- Minimal rotation needed
- Factor interpretations match

### 2.3 Chordal Distance (Raw Frame)

**Mathematical Definition:**

The chordal distance is the direct Frobenius norm:

```
d_C(B_true, B_est) = ||Q_true - Q_est||_F
```

**Properties:**
- **Range:** d_C ∈ [0, 2√k]
- **No alignment:** Measures raw difference
- **Sensitive to everything:** Rotation, sign, permutation
- **Fastest to compute:** No optimization required

**Relationship to Principal Angles:**

```
d_C² = 2k - 2Σᵢ cos²(θᵢ)
```

**Interpretation Guidelines:**

Chordal distance is primarily useful for:
1. **Debugging:** Detecting gross errors
2. **Comparison:** Assessing alignment quality
3. **Initialization:** Starting point for optimization

**Typical pattern:**
```
d_C ≥ d_P ≥ d_G
```

When d_C ≈ d_P ≈ d_G:
- Frames are naturally aligned
- No significant ambiguity
- Good initialization

When d_C ≫ d_P ≈ d_G:
- Significant frame mismatch
- But subspaces align well
- Sign/permutation issues

### 2.4 Comparative Analysis

**Decision Tree:**

```
Use d_G when:
├─ Comparing subspaces (rotation-invariant)
├─ Assessing factor recovery quality
└─ Model validation

Use d_P when:
├─ Aligning specific factor interpretations
├─ Matching factor orderings
└─ Computing optimal rotations

Use d_C when:
├─ Debugging implementations
├─ Quick sanity checks
└─ Assessing raw differences
```

**Theoretical bounds:**

For k factors:
```
d_G ≤ d_P ≤ d_C ≤ 2√k
```

**Asymptotic behavior:**

As T → ∞ (large sample):
```
d_G = O_p(√(k/T))
d_P = O_p(√(k/T))
```

Under mild regularity conditions.

---

## 3. Eigenvalue Analysis

### 3.1 True Eigenvalues via LinearOperator

**Problem:**

Computing eigenvalues of Σ = B'FB + D naively requires O(p²) memory:
```
For p = 10,000: Σ requires 800 MB
```

**Solution:**

Use implicit matrix-vector products without forming Σ:

```python
def matvec(v):
    """Compute Σv = B'(F(Bv)) + Dv"""
    return B.T @ (F @ (B @ v)) + D @ v
```

**Memory complexity:**
- Dense: O(p²) ≈ 100p² bytes
- Sparse: O(kp + p) ≈ 8(kp + p) bytes

**Example savings (p=10,000, k=10):**
- Dense: 800 MB
- Sparse: 8 MB
- **Reduction: 100×**

### 3.2 ARPACK Eigenvalue Solver

We use `scipy.sparse.linalg.eigsh` which wraps ARPACK:

```python
from scipy.sparse.linalg import eigsh, LinearOperator

def compute_eigenvalues(B, F, D, k_top):
    p = B.shape[1]
    
    # Define implicit operator
    def matvec(v):
        return B.T @ (F @ (B @ v)) + np.diag(D) * v
    
    Σ_op = LinearOperator(shape=(p, p), matvec=matvec)
    
    # Compute top k eigenvalues
    evals, evecs = eigsh(Σ_op, k=k_top, which='LA', tol=1e-10)
    
    return evals[::-1], evecs[:, ::-1]  # Descending order
```

**Parameters:**
- **which='LA':** Largest algebraic (eigenvalues)
- **tol=1e-10:** Convergence tolerance
- **maxiter=10000:** Maximum iterations

**Convergence criteria:**

ARPACK uses residual norm:
```
||Σvᵢ - λᵢvᵢ|| / |λᵢ| < tol
```

### 3.3 Eigenvalue Error Metrics

**Absolute error:**
```
εᵢ = λᵢ^true - λᵢ^sample
```

**Relative error:**
```
ρᵢ = (λᵢ^true - λᵢ^sample) / λᵢ^true = εᵢ / λᵢ^true
```

**Root mean squared error:**
```
RMSE = sqrt(1/k Σᵢ εᵢ²)
```

**Mean relative error:**
```
MRE = 1/k Σᵢ |ρᵢ|
```

**Interpretation:**

| RMSE | Quality | Interpretation |
|------|---------|----------------|
| < 0.01σ̄ | Excellent | High-precision estimation |
| 0.01σ̄ - 0.05σ̄ | Good | Minor estimation error |
| 0.05σ̄ - 0.10σ̄ | Moderate | Noticeable error |
| > 0.10σ̄ | Poor | Large estimation error |

Where σ̄ = mean(λᵢ^true) is the mean eigenvalue.

**Asymptotic theory:**

Under regularity conditions:
```
√T(λ̂ᵢ - λᵢ) →_d N(0, 2λᵢ²)
```

For large T, expect:
```
|λ̂ᵢ - λᵢ| ≈ O(λᵢ/√T)
```

### 3.4 Spectrum Visualization

The eigenvalue spectrum plot shows:

**X-axis:** Eigenvalue index (1 to k)
- Ordered from largest to smallest
- Index 1 = dominant eigenvalue
- Later indices = smaller variance components

**Y-axis:** Eigenvalue magnitude (log scale)
- Log scale reveals small eigenvalues
- Exponential decay typical
- Flat spectrum indicates problems

**Visual patterns:**

**Steep decay:**
```
λ₁ ≫ λ₂ ≫ λ₃ ≫ ... ≫ λₖ
```
- Strong factor structure
- Few dominant factors
- Good for dimension reduction

**Gradual decay:**
```
λ₁ > λ₂ > λ₃ > ... > λₖ (similar magnitudes)
```
- Weak factor structure
- Multiple important factors
- May need more factors

**Flat spectrum:**
```
λ₁ ≈ λ₂ ≈ λ₃ ≈ ... ≈ λₖ
```
- Poor factor structure
- Noise dominates
- Model misspecification

---

## 4. Eigenvector Comparison

### 4.1 Canonical Correlations

For eigenvector pairs (u_i^true, u_i^sample), the canonical correlation is:

```
ρᵢ = |u_i^true · u_i^sample|
```

**Properties:**
- Range: ρᵢ ∈ [0, 1]
- Measures directional alignment
- Independent of scaling
- Sign-invariant (absolute value)

**Interpretation:**

| ρᵢ | Alignment | Interpretation |
|----|-----------|----------------|
| > 0.95 | Excellent | Vectors nearly parallel |
| 0.90 - 0.95 | Good | Minor angular difference |
| 0.80 - 0.90 | Moderate | Noticeable misalignment |
| < 0.80 | Poor | Substantial angular error |

**Geometric meaning:**

```
ρᵢ = cos(angle between vectors)
angle = arccos(ρᵢ)
```

Example:
- ρ = 0.95 → angle ≈ 18°
- ρ = 0.90 → angle ≈ 26°
- ρ = 0.80 → angle ≈ 37°

### 4.2 Sign Alignment

Factor models have sign ambiguity:
```
If B is valid, so is -B
```

We resolve this by aligning signs:

```python
for i in range(k):
    if np.dot(u_true[i], u_sample[i]) < 0:
        u_sample[i] *= -1
```

**Effect:**
- Maximizes correlations
- Makes interpretation consistent
- Doesn't change subspace

### 4.3 Eigenvector Loadings Heatmap

The heatmap shows eigenvector components across assets:

**Rows:** Eigenvector index (k factors)
**Columns:** Asset index (p assets)
**Color:** Loading magnitude

**Interpretation:**

**Sparse patterns:**
- Few non-zero elements per vector
- Sector-specific factors
- Interpretable structure

**Dense patterns:**
- Many non-zero elements
- Market-wide factors
- Diversified exposure

**Color patterns:**

**Positive (red):**
- Assets move with factor
- Positive loadings
- Long exposure

**Negative (blue):**
- Assets move against factor
- Negative loadings
- Short exposure

**Near-zero (white):**
- Assets unrelated to factor
- No exposure
- Factor-neutral

**Comparing true vs. sample:**

**Similar patterns:**
- Factor structure recovered
- Loadings match
- Good estimation

**Different patterns:**
- Estimation error
- Rotational ambiguity
- Check alignment

### 4.4 Vector Correlation Plot

Bar chart showing per-vector correlations:

**X-axis:** Eigenvector index
**Y-axis:** Canonical correlation ρᵢ

**Color coding:**
- **Green (ρ > 0.90):** Excellent recovery
- **Orange (0.70 < ρ ≤ 0.90):** Moderate recovery
- **Red (ρ ≤ 0.70):** Poor recovery

**Typical patterns:**

**Decreasing correlations:**
```
ρ₁ > ρ₂ > ρ₃ > ... > ρₖ
```
- Expected pattern
- Later vectors harder to estimate
- Due to smaller eigenvalues

**Uniform correlations:**
```
ρ₁ ≈ ρ₂ ≈ ρ₃ ≈ ... ≈ ρₖ
```
- Consistent estimation
- Balanced factor structure
- Good sample size

**Irregular pattern:**
```
ρ₁ > ρ₃ > ρ₂ > ...
```
- Potential issues
- Check for outliers
- Verify convergence

---

## 5. Computational Methods

### 5.1 SVD-Based Factor Extraction

Given returns matrix R ∈ ℝ^(T×p):

**Step 1: Center returns**
```python
R_centered = R - R.mean(axis=0)
```

**Step 2: Compute SVD**
```python
U, s, Vt = np.linalg.svd(R_centered, full_matrices=False)
```

Where:
- U ∈ ℝ^(T×min(T,p)): Left singular vectors (time)
- s ∈ ℝ^min(T,p): Singular values
- Vt ∈ ℝ^(min(T,p)×p): Right singular vectors (assets)

**Step 3: Extract factors**
```python
k = num_factors
factor_returns = U[:, :k] * s[:k]  # (T, k)
factor_loadings = Vt[:k, :].T       # (p, k)
```

**Step 4: Compute covariances**
```python
F = factor_returns.T @ factor_returns / (T - 1)  # (k, k)
B = factor_loadings.T                             # (k, p)

# Idiosyncratic component
residuals = R_centered - factor_returns @ B
D = np.diag(np.var(residuals, axis=0))            # (p, p)
```

**Explained variance:**
```python
total_var = np.trace(R_centered.T @ R_centered) / (T - 1)
factor_var = np.trace(F)
explained = factor_var / total_var
```

### 5.2 QR Decomposition for Orthonormalization

To project B onto the Stiefel manifold:

```python
Q, R = np.linalg.qr(B.T, mode='economic')
# Q ∈ ℝ^(p×k): Orthonormal basis
# R ∈ ℝ^(k×k): Upper triangular
# B.T = QR
# Q.T @ Q = I_k
```

**Properties:**
- Q spans same subspace as B
- Columns of Q are orthonormal
- Efficient: O(kp²) operations

### 5.3 Principal Angles Computation

```python
def principal_angles(Q1, Q2):
    """
    Compute principal angles between subspaces.
    
    Q1, Q2: Orthonormal bases (p×k)
    Returns: angles (k,) in [0, π/2]
    """
    M = Q1.T @ Q2  # Overlap matrix (k×k)
    
    # SVD of overlap
    _, sigma, _ = np.linalg.svd(M)
    
    # Angles from singular values
    # σᵢ = cos(θᵢ), so θᵢ = arccos(σᵢ)
    sigma = np.clip(sigma, -1, 1)  # Numerical stability
    angles = np.arccos(sigma)
    
    return angles
```

**Numerical considerations:**

- Clip σ to [-1, 1] for stability
- Use QR instead of Gram-Schmidt for better conditioning
- Check for degenerate cases (rank deficiency)

### 5.4 Procrustes Alignment

```python
def procrustes_alignment(Q_target, Q_source):
    """
    Find optimal rotation R: Q_target ≈ Q_source @ R
    
    Returns: R (k×k), aligned Q_source @ R
    """
    # Cross-covariance
    M = Q_source.T @ Q_target  # (k×k)
    
    # SVD
    U, _, Vt = np.linalg.svd(M)
    
    # Optimal rotation (Kabsch)
    R = U @ Vt
    
    # Apply rotation
    Q_aligned = Q_source @ R
    
    # Distance after alignment
    dist = np.linalg.norm(Q_target - Q_aligned, 'fro')
    
    return R, Q_aligned, dist
```

**Kabsch algorithm:**

Solves: min_R ||A - BR||_F subject to R ∈ O(k)

Solution: R = UV' where UΣV' = B'A

### 5.5 Implicit Eigenvalue Computation

**Memory-efficient matvec:**

```python
def make_matvec(B, F, D_diag):
    """
    Create matvec function for Σ = B'FB + D
    
    B: (k, p) loadings
    F: (k, k) factor covariance
    D_diag: (p,) diagonal of D
    
    Returns: function v ↦ Σv
    """
    def matvec(v):
        # Step 1: Bv (k, p) × (p,) → (k,)
        Bv = B @ v
        
        # Step 2: F(Bv) (k, k) × (k,) → (k,)
        FBv = F @ Bv
        
        # Step 3: B'(FBv) (k, p)' × (k,) → (p,)
        BtFBv = B.T @ FBv
        
        # Step 4: Add D term
        Dv = D_diag * v
        
        return BtFBv + Dv
    
    return matvec
```

**Operation counts:**
- Matrix-vector: 2kp + k² + p = O(kp)
- Dense matrix-vector: 2p² = O(p²)
- Speedup: ~p/k for p ≫ k

**Memory footprint:**
- Store B: 8kp bytes
- Store F: 8k² bytes  
- Store D_diag: 8p bytes
- Total: O(kp) bytes vs. O(p²) for dense

---

## 6. Interpretation Guidelines

### 6.1 Sample Size Requirements

**Minimum requirements:**

| Ratio T/p | Quality | Recommendation |
|-----------|---------|----------------|
| < 1 | Poor | Increase sample size |
| 1 - 2 | Marginal | Use regularization |
| 2 - 5 | Good | Standard methods work |
| > 5 | Excellent | High reliability |

**Factor-specific:**

For k-factor model:
```
T ≥ 2(k + p)  (bare minimum)
T ≥ 5(k + p)  (recommended)
```

**Rule of thumb:**
```
T should be at least 2× the number of estimated parameters
Parameters = kp + k(k+1)/2 + p ≈ kp + p
```

### 6.2 Convergence Assessment

**Eigenvalue solver convergence:**

Check residual norms:
```python
for i in range(k):
    residual = Σ @ evecs[:, i] - evals[i] * evecs[:, i]
    norm = np.linalg.norm(residual) / abs(evals[i])
    print(f"λ_{i+1}: residual = {norm:.2e}")
```

Target: < 1e-10

**Statistical convergence:**

Bootstrap to assess stability:
```python
n_boot = 1000
boot_distances = []

for _ in range(n_boot):
    # Resample returns
    idx = np.random.choice(T, T, replace=True)
    R_boot = returns[idx]
    
    # Estimate model
    model_boot = svd_decomposition(R_boot, k)
    
    # Compute distance
    dist = grassmannian_distance(model.B, model_boot.B)
    boot_distances.append(dist)

# Check stability
std = np.std(boot_distances)
print(f"Bootstrap std: {std:.4f}")
```

Target: CV < 0.1 (coefficient of variation)

### 6.3 Model Validation

**Explained variance check:**

```
R² = tr(B'FB) / tr(Σ)
```

Target:
- R² > 0.70 for 3-5 factor model
- R² > 0.80 for 10+ factor model

**Residual analysis:**

```python
residuals = returns - factors @ B
```

Check:
1. **Mean:** Should be ≈ 0
2. **Autocorrelation:** Should be small
3. **Heteroskedasticity:** Constant variance over time

**Factor interpretation:**

Examine loading patterns:
```python
top_loadings = np.argsort(np.abs(B[0, :]))[-10:]
print("Factor 1 top 10 assets:", top_loadings)
```

Factors should have economic interpretation:
- Market factor: All positive loadings
- Size factor: Small vs. large cap
- Sector factors: Industry-specific

### 6.4 Common Issues and Diagnostics

**Issue 1: Large Grassmannian distance**

Symptoms:
- d_G > 0.5
- Poor subspace recovery

Potential causes:
1. Insufficient sample size (T/p < 2)
2. Model misspecification (wrong k)
3. Heavy-tailed returns (fat tails)
4. Time-varying parameters

Diagnostics:
```python
# Check T/p ratio
print(f"T/p ratio: {T/p:.2f}")

# Try different k
for k_try in range(1, 10):
    model = svd_decomposition(returns, k_try)
    explained = np.trace(model.F) / total_variance
    print(f"k={k_try}: R²={explained:.3f}")

# Check kurtosis
kurt = scipy.stats.kurtosis(returns.flatten())
print(f"Excess kurtosis: {kurt:.2f}")
```

**Issue 2: Eigenvalues don't match**

Symptoms:
- Large eigenvalue RMSE
- Relative errors > 10%

Potential causes:
1. Sampling variability
2. Non-Gaussianity
3. Model misspecification

Diagnostics:
```python
# Compare true vs. sample eigenvalue ratios
ratio_true = evals_true[0] / evals_true[-1]
ratio_sample = evals_sample[0] / evals_sample[-1]
print(f"Eigenvalue spread: true={ratio_true:.1f}, sample={ratio_sample:.1f}")

# Check if later eigenvalues converged
for i, (t, s) in enumerate(zip(evals_true, evals_sample)):
    rel_err = abs(t - s) / t
    print(f"λ_{i+1}: rel_error={rel_err:.1%}")
```

**Issue 3: Poor eigenvector correlations**

Symptoms:
- ρᵢ < 0.80 for some factors
- Irregular correlation pattern

Potential causes:
1. Rotational ambiguity not resolved
2. Factor ordering mismatch
3. Nearly equal eigenvalues (degeneracy)

Diagnostics:
```python
# Check eigenvalue gaps
gaps = np.diff(evals_true)
print("Eigenvalue gaps:", gaps)

# If gap is small, eigenvectors poorly determined
degenerate = gaps < 0.01 * evals_true[0]
print("Degenerate factors:", np.where(degenerate)[0])

# Try alternative alignment
R, _, _ = procrustes_alignment(Q_true, Q_sample)
Q_aligned = Q_sample @ R
corr_aligned = np.abs(np.diag(Q_true.T @ Q_aligned))
print("Correlations after Procrustes:", corr_aligned)
```

---

## 7. Visualization Reference

### 7.1 Dashboard Layout (9-Panel)

```
┌─────────────┬─────────────┬─────────────┐
│ Eigenvalue  │ Principal   │  Manifold   │
│  Spectrum   │   Angles    │  Distances  │
│   (log)     │   (bars)    │   (bars)    │
├─────────────┼─────────────┼─────────────┤
│ Eigenvalue  │ Eigenvector │  Relative   │
│   Errors    │ Correlations│   Errors    │
│  (filled)   │   (bars)    │  (filled)   │
├─────────────┼─────────────┼─────────────┤
│  Summary    │    True     │   Sample    │
│ Statistics  │ Eigenvector │ Eigenvector │
│   (text)    │   Heatmap   │   Heatmap   │
└─────────────┴─────────────┴─────────────┘
```

### 7.2 Panel Descriptions

**Panel 1: Eigenvalue Spectrum (Top-Left)**

**Purpose:** Compare true vs. sample eigenvalue magnitudes

**X-axis:** Factor index (1 to k)
**Y-axis:** Eigenvalue (log scale)
**Lines:** 
- Blue: True eigenvalues
- Orange: Sample eigenvalues

**Interpretation:**
- Parallel lines → Good agreement
- Divergence → Estimation error
- Slope → Factor importance decay

**Panel 2: Principal Angles (Top-Center)**

**Purpose:** Show subspace angles

**X-axis:** Angle index
**Y-axis:** Angle in radians
**Bars:** Colored by magnitude (viridis palette)
**Labels:** Degrees shown on bars

**Interpretation:**
- Small angles (< 0.3 rad) → Good
- Increasing angles → Later factors worse
- Angle > π/4 → Poor recovery

**Panel 3: Manifold Distances (Top-Right)**

**Purpose:** Compare distance metrics

**Bars:**
1. Grassmannian (green)
2. Procrustes (blue)
3. Chordal (red)

**Height:** Distance value
**Labels:** Exact values shown

**Interpretation:**
- Green < 0.3 → Good subspace recovery
- Red ≈ Green → Well-aligned frames
- Red ≫ Green → Need rotation

**Panel 4: Eigenvalue Errors (Middle-Left)**

**Purpose:** Show signed errors over index

**X-axis:** Factor index
**Y-axis:** Error (λ_true - λ_sample)
**Fill:** Between zero and error line

**Interpretation:**
- Positive → Overestimation
- Negative → Underestimation
- Magnitude → Absolute error

**Panel 5: Eigenvector Correlations (Middle-Center)**

**Purpose:** Per-vector alignment quality

**X-axis:** Vector index
**Y-axis:** Canonical correlation
**Colors:**
- Green (> 0.90): Excellent
- Orange (0.70-0.90): Moderate
- Red (< 0.70): Poor

**Interpretation:**
- All green → Excellent recovery
- Decreasing trend → Expected
- Red bars → Problem factors

**Panel 6: Relative Errors (Middle-Right)**

**Purpose:** Show percentage errors

**X-axis:** Factor index
**Y-axis:** Relative error (%)
**Fill:** Between zero and error line

**Interpretation:**
- Small (< 5%) → Good
- Large (> 20%) → Poor
- Pattern → Systematic bias

**Panel 7: Summary Statistics (Bottom-Left)**

**Purpose:** Key metrics at a glance

**Content:**
```
Grassmannian Distance: 0.1234
Procrustes Distance:   0.0234
Eigenvalue RMSE:       0.0012
Mean Eigvec Corr:      0.9543

Assessment:
✓ Excellent subspace recovery
✓ Good eigenvector alignment
```

**Panel 8: True Eigenvector Heatmap (Bottom-Center)**

**Purpose:** Visualize true loading structure

**Rows:** k factors
**Columns:** p assets (or subset)
**Color:** 
- Red: Positive loadings
- Blue: Negative loadings
- White: Near-zero

**Panel 9: Sample Eigenvector Heatmap (Bottom-Right)**

**Purpose:** Visualize estimated loading structure

**Layout:** Same as Panel 8

**Comparison:**
- Similar patterns → Good recovery
- Different patterns → Estimation issues

### 7.3 Interactive Features (Plotly)

The HTML dashboard adds:

**Zoom:** 
- Box zoom on any panel
- Reset with double-click

**Hover:**
- Shows exact values
- Identifies factor/asset indices

**Pan:**
- Click and drag to explore

**Export:**
- Save as PNG from browser
- Preserve current zoom/pan

**Responsive:**
- Adjusts to window size
- Mobile-friendly

### 7.4 Color Schemes

**Default palette:**
- Blue: True/reference values
- Orange: Sample/estimated values
- Green: Good performance
- Red: Poor performance

**Heatmap:**
- Diverging: RdBu_r (red-white-blue reversed)
- Center: White (zero loading)
- Extremes: -max to +max (symmetric)

**Accessibility:**
- Colorblind-safe palettes
- High contrast
- Pattern-based distinction

---

## 8. Advanced Topics

### 8.1 Concentration of Measure

In high dimensions (p large), random projections concentrate:

**Johnson-Lindenstrauss lemma:**

For any ε ∈ (0, 1), if:
```
T ≥ C ε⁻² log(p)
```

Then random k-dimensional projection preserves distances within (1±ε).

**Implications for factor models:**

Sample size requirements grow logarithmically:
```
T = O(k log(p))  (sufficient for concentration)
```

But factor estimation requires:
```
T = Ω(kp)  (for consistent estimation)
```

**Marchenko-Pastur distribution:**

For large p, k, eigenvalues of sample covariance follow:
```
λ_edge = (1 ± √(k/T))²
```

This gives bounds on eigenvalue estimation error.

### 8.2 Random Matrix Theory

**Spectral statistics:**

For k/p → γ and T/p → c as p → ∞:

Top eigenvalue limit:
```
λ₁ / p → (1 + √γ)² · c
```

**Tracy-Widom distribution:**

Largest eigenvalue fluctuations:
```
T^(2/3)(λ₁ - μ) / σ →_d TW₂
```

Where TW₂ is Tracy-Widom distribution of order 2.

**Spiked model:**

If true eigenvalues exceed threshold:
```
λᵢ > (1 + √γ)²
```

Then PCA consistently recovers factors.

### 8.3 Riemannian Optimization

Grassmann manifold has natural Riemannian structure:

**Tangent space at [U]:**
```
T_U Gr(k,p) = {UΩ + U_⊥Σ : Ω skew-symmetric}
```

**Geodesic distance:**
```
d([U], [V]) = ||θ||₂
```

Where θ are principal angles.

**Gradient descent on Grassmannian:**

To minimize f([U]) over Gr(k,p):

1. Compute Euclidean gradient: ∇f(U)
2. Project to tangent space: grad = (I - UU')∇f(U)
3. Retract to manifold: U_new = qr((U - α·grad).T).T

This framework enables optimization over factor loadings directly on the manifold.

### 8.4 Robust Estimation

For heavy-tailed returns (kurtosis > 5):

**Robust PCA alternatives:**

1. **Spherical PCA:** Normalize each return vector to unit length
2. **Huber PCA:** Use robust M-estimators
3. **Elliptical PCA:** Account for elliptical distributions

**Diagnostic:**
```python
kurt = scipy.stats.kurtosis(returns.flatten())

if kurt > 5:
    print("Heavy tails detected")
    print("Consider robust methods")
```

**Implementation:**
```python
# Spherical PCA
returns_norm = returns / np.linalg.norm(returns, axis=1, keepdims=True)
model_robust = svd_decomposition(returns_norm, k)
```

### 8.5 Time-Varying Parameters

Factor models often assume stationarity, but parameters may vary:

**Rolling window estimation:**
```python
window = 252  # 1 year of daily returns
distances = []

for t in range(window, T):
    R_window = returns[t-window:t]
    model_t = svd_decomposition(R_window, k)
    dist = grassmannian_distance(model_true.B, model_t.B)
    distances.append(dist)

# Plot time series of distances
plt.plot(distances)
plt.xlabel('Time')
plt.ylabel('Grassmannian Distance')
```

**Interpretation:**
- Stable distance → Stationary parameters
- Trending distance → Time variation
- Spikes → Regime changes

### 8.6 Bootstrap Inference

Assess estimation uncertainty via bootstrap:

```python
def bootstrap_manifold(returns, model_true, k, n_boot=1000):
    T, p = returns.shape
    distances = []
    
    for _ in range(n_boot):
        # Resample with replacement
        idx = np.random.choice(T, T, replace=True)
        R_boot = returns[idx]
        
        # Estimate model
        model_boot = svd_decomposition(R_boot, k)
        
        # Compute distance
        dist = grassmannian_distance(model_true.B, model_boot.B)
        distances.append(dist)
    
    # Statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    ci_lower = np.percentile(distances, 2.5)
    ci_upper = np.percentile(distances, 97.5)
    
    return {
        'mean': mean_dist,
        'std': std_dist,
        'ci': (ci_lower, ci_upper),
        'distances': distances
    }

# Usage
boot_results = bootstrap_manifold(returns, model_true, k)
print(f"Mean distance: {boot_results['mean']:.4f} ± {boot_results['std']:.4f}")
print(f"95% CI: [{boot_results['ci'][0]:.4f}, {boot_results['ci'][1]:.4f}]")
```

---

## References

### Key Papers

1. **Manifold geometry:**
   - Edelman, A., Arias, T. A., & Smith, S. T. (1998). "The geometry of algorithms with orthogonality constraints." *SIAM Journal on Matrix Analysis and Applications*, 20(2), 303-353.

2. **Factor models:**
   - Bai, J., & Ng, S. (2002). "Determining the number of factors in approximate factor models." *Econometrica*, 70(1), 191-221.

3. **Random matrix theory:**
   - Johnstone, I. M. (2001). "On the distribution of the largest eigenvalue in principal components analysis." *Annals of Statistics*, 29(2), 295-327.

4. **Procrustes analysis:**
   - Schönemann, P. H. (1966). "A generalized solution of the orthogonal Procrustes problem." *Psychometrika*, 31(1), 1-10.

5. **Grassmannian distances:**
   - Ye, K., & Lim, L. H. (2016). "Schubert varieties and distances between subspaces of different dimensions." *SIAM Journal on Matrix Analysis and Applications*, 37(3), 1176-1197.

### Software References

1. **NumPy/SciPy:**
   - Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature*, 585, 357-362.

2. **ARPACK:**
   - Lehoucq, R. B., Sorensen, D. C., & Yang, C. (1998). *ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods*.

3. **Scikit-learn (PCA reference):**
   - Pedregosa, F., et al. (2011). "Scikit-learn: Machine learning in Python." *JMLR*, 12, 2825-2830.

---

## Appendix: Mathematical Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| p | Number of assets | scalar |
| k | Number of factors | scalar |
| T | Number of time periods | scalar |
| **r_t** | Returns at time t | ℝ^p |
| **f_t** | Factor returns at time t | ℝ^k |
| **ε_t** | Idiosyncratic returns | ℝ^p |
| **B** | Factor loadings matrix | ℝ^(k×p) |
| **F** | Factor covariance | ℝ^(k×k) |
| **D** | Idiosyncratic covariance | ℝ^(p×p) |
| **Σ** | Returns covariance | ℝ^(p×p) |
| **Q** | Orthonormal frame | St(k,p) |
| **θ** | Principal angles | ℝ^k |
| **λ** | Eigenvalue | ℝ |
| **u** | Eigenvector | ℝ^p |
| d_G | Grassmannian distance | ℝ₊ |
| d_P | Procrustes distance | ℝ₊ |
| d_C | Chordal distance | ℝ₊ |

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-30  
**For Questions:** See README.md or API.md
