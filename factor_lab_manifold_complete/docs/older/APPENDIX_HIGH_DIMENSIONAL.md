# Technical Appendix: High-Dimensional Effects and Concentration of Measure

**Asymptotic Analysis in the p >> T Regime**

A deep dive into how the high-dimensional regime (p >> T) affects factor model estimation, manifold distances, and statistical inference, with emphasis on concentration phenomena, eigenvalue biases, and practical corrections.

---

## Table of Contents

1. [The High-Dimensional Regime](#1-the-high-dimensional-regime)
2. [Concentration of Measure Phenomena](#2-concentration-of-measure-phenomena)
3. [Marchenko-Pastur Distribution](#3-marchenko-pastur-distribution)
4. [Eigenvalue Bias in High Dimensions](#4-eigenvalue-bias-in-high-dimensions)
5. [Spiked Covariance Model](#5-spiked-covariance-model)
6. [Phase Transitions](#6-phase-transitions)
7. [Bias in Manifold Distances](#7-bias-in-manifold-distances)
8. [Practical Corrections](#8-practical-corrections)
9. [Regularization Methods](#9-regularization-methods)
10. [Recommendations](#10-recommendations)

---

## 1. The High-Dimensional Regime

### 1.1 Classical vs. High-Dimensional Statistics

**Classical regime:**
```
p fixed, T → ∞
```
- Sample covariance Ŝ → Σ (population covariance)
- Central limit theorem applies
- Standard asymptotics work

**High-dimensional regime:**
```
p/T → c ∈ (0, ∞) as p, T → ∞
```
- Sample covariance Ŝ ≠ Σ even as T → ∞
- New limiting distributions
- Random matrix theory applies

**Ultra-high-dimensional regime:**
```
p >> T (or p/T → ∞)
```
- Sample covariance singular (rank at most T)
- Extreme bias in eigenvalues
- Regularization essential

### 1.2 Why High Dimensions Are Problematic

**Problem 1: Sample covariance is rank-deficient**

When p > T:
```
rank(Ŝ) ≤ min(T-1, p) = T-1 < p
```

This means:
- At most T-1 non-zero eigenvalues
- p - T + 1 eigenvalues exactly zero
- Cannot estimate full covariance structure

**Problem 2: Eigenvalue spread**

Sample eigenvalues are more dispersed than true eigenvalues:
```
λ̂_max / λ̂_min >> λ_max / λ_min
```

**Problem 3: Eigenvector inconsistency**

For finite p/T, sample eigenvectors are biased:
```
angle(û_i, u_i) = O(√(p/T))
```

Does not vanish unless T/p → ∞.

**Problem 4: Concentration phenomena**

High-dimensional geometry has counterintuitive properties:
- Most volume near surface of sphere
- Random vectors approximately orthogonal
- Distances concentrate near mean

### 1.3 Financial Applications

**Typical scenarios:**

| Application | p | T | p/T | Regime |
|-------------|---|---|-----|--------|
| Large-cap stocks | 500 | 252 | 2.0 | Moderate |
| Small-cap universe | 2000 | 252 | 7.9 | High |
| Daily factors | 500 | 63 | 7.9 | High |
| Intraday factors | 1000 | 30 | 33.3 | Ultra-high |
| Cross-section study | 5000 | 120 | 41.7 | Ultra-high |

**Common in practice:**
- p/T ≥ 2: Noticeable bias
- p/T ≥ 5: Severe bias
- p/T ≥ 10: Requires regularization

---

## 2. Concentration of Measure Phenomena

### 2.1 Concentration on the Sphere

**Key result:** In high dimensions, most volume of sphere concentrates near equator.

**Distance concentration:**

For random vectors x, y uniformly on S^(p-1):
```
E[||x - y||²] = 2
Var[||x - y||²] = O(1/p)
```

As p → ∞:
```
||x - y||² →_p 2 (concentration)
```

**Implication:** All pairwise distances become similar in high dimensions.

### 2.2 Orthogonality in High Dimensions

For random vectors x, y ~ N(0, I_p):
```
Corr(x, y) = x·y / (||x|| ||y||)
```

**Distribution:**
```
√p · Corr(x, y) →_d N(0, 1)
```

So:
```
|Corr(x, y)| = O_p(1/√p) → 0
```

**Implication:** Random vectors become approximately orthogonal.

**For factor loadings:**

Sample factor loadings b̂_i behave like random vectors when T is small:
```
|b̂_i · b̂_j| = O_p(√(p/T))
```

If p >> T:
```
Spurious orthogonality in estimated factors
```

### 2.3 Curse of Dimensionality

**Volume of hypercube:**

Fraction of volume within distance ε of boundary:
```
1 - (1 - 2ε)^p
```

For ε = 0.1, p = 100:
```
≈ 1 - 10^(-9) (essentially all volume at boundary!)
```

**Volume of hypersphere:**

Ratio of p-ball to p-cube volume:
```
V_p(1) / (2)^p = π^(p/2) / (2^p · Γ(p/2 + 1))
```

Decays exponentially in p.

**Implication:** High-dimensional data lives in a thin shell.

### 2.4 Johnson-Lindenstrauss Lemma

**Statement:**

For any set of n points in ℝ^p, there exists a projection onto ℝ^k with k = O(log(n)/ε²) such that all pairwise distances are preserved within (1 ± ε).

**Implication for factor models:**

If k factors explain data, then:
```
T = O(k log(p) / ε²)
```

samples suffice to preserve factor structure within ε.

**But:** This is for distance preservation, not parameter estimation.

**For consistent estimation:**
```
T = Ω(kp)  (much larger!)
```

### 2.5 Concentration in Sample Covariance

For returns R ∈ ℝ^(T×p) with iid rows ~ N(0, Σ):

**Limiting spectral distribution (LSD):**

As p, T → ∞ with p/T → c:
```
Empirical eigenvalue distribution → Marchenko-Pastur
```

Even when Σ = I (no structure), sample eigenvalues spread out!

**Largest sample eigenvalue:**

When Σ = I_p:
```
λ̂_max → (1 + √c)²
```

For c = 5 (p = 5T):
```
λ̂_max ≈ 9.47  (vs. true λ = 1)
```

**Smallest non-zero eigenvalue:**
```
λ̂_min → (1 - √c)²  if c < 1
λ̂_min → 0           if c ≥ 1
```

**Conclusion:** Sample eigenvalues are systematically biased!

---

## 3. Marchenko-Pastur Distribution

### 3.1 Mathematical Framework

**Setup:**

- Returns: X ∈ ℝ^(T×p) with iid entries
- True covariance: Σ = I_p (null model)
- Sample covariance: Ŝ = (1/T) X'X

**Limiting spectral distribution:**

As p, T → ∞ with p/T → c ∈ (0, ∞), the empirical distribution of eigenvalues of Ŝ converges to the **Marchenko-Pastur distribution** with density:

```
f_MP(λ) = (1/(2πcλ)) √((λ_+ - λ)(λ - λ_-))
```

for λ ∈ [λ_-, λ_+], where:

```
λ_- = (1 - √c)²
λ_+ = (1 + √c)²
```

**Support:**
- If c < 1: λ ∈ [(1-√c)², (1+√c)²]
- If c = 1: λ ∈ [0, 4]
- If c > 1: λ ∈ [0, (1+√c)²] with point mass at 0

### 3.2 Eigenvalue Spread

**For null case (Σ = I):**

| c = p/T | λ_- | λ_+ | Spread λ_+/λ_- |
|---------|-----|-----|----------------|
| 0.5 | 0.172 | 2.914 | 16.9 |
| 1.0 | 0.000 | 4.000 | ∞ |
| 2.0 | — | 5.828 | ∞ |
| 5.0 | — | 9.472 | ∞ |
| 10.0 | — | 14.314 | ∞ |

**Observations:**

1. **Overestimation:** λ_+ > 1 always (true eigenvalue is 1)
2. **Underestimation:** λ_- < 1 for c < 1
3. **Singularity:** For c ≥ 1, smallest eigenvalues exactly zero
4. **Massive spread:** Even with no structure, huge condition number

### 3.3 Tracy-Widom Distribution

**Largest eigenvalue fluctuations:**

The largest sample eigenvalue λ̂_max has limiting distribution:
```
T^(2/3) (λ̂_max - μ_c) / σ_c →_d TW_2
```

Where:
- μ_c = (1 + √c)² (Marchenko-Pastur edge)
- σ_c = (1 + √c) · (1 + √c)^(1/3)
- TW_2 is Tracy-Widom distribution of order 2

**Tracy-Widom distribution:**
- Highly left-skewed (long left tail)
- Mode ≈ -1.77
- Mean ≈ -1.21
- Var ≈ 1.61

**Practical implication:**

Largest sample eigenvalue fluctuates at rate T^(-2/3), much slower than classical T^(-1/2).

### 3.4 General Covariance Case

When Σ ≠ I, the limiting distribution is a deformed Marchenko-Pastur:

**Free convolution:**

If Σ = diag(σ₁², ..., σ_p²), the LSD is the free convolution:
```
ν_Ŝ = ν_Σ ⊞ ν_MP
```

**Effect:**

True eigenvalues are "smeared out" by MP distribution.

**Bulk eigenvalues:**

Those corresponding to σᵢ² ≈ 1 remain in MP bulk.

**Spike eigenvalues:**

Large true eigenvalues "pop out" of bulk (see §5).

---

## 4. Eigenvalue Bias in High Dimensions

### 4.1 Bias in Largest Eigenvalue

**Null case (Σ = I):**

```
E[λ̂_max] ≈ (1 + √c)²
Bias = (1 + √c)² - 1 = 2√c + c
```

**Examples:**

| p/T | True λ | E[λ̂_max] | Bias | % Bias |
|-----|--------|----------|------|--------|
| 0.5 | 1 | 2.91 | 1.91 | 191% |
| 1.0 | 1 | 4.00 | 3.00 | 300% |
| 2.0 | 1 | 5.83 | 4.83 | 483% |
| 5.0 | 1 | 9.47 | 8.47 | 847% |
| 10.0 | 1 | 14.31 | 13.31 | 1331% |

**Massive upward bias!**

### 4.2 Bias in Bulk Eigenvalues

For true eigenvalue λ = 1 (in bulk):

**Approximate bias formula:**
```
E[λ̂ - λ] ≈ 2c  for c small
```

**Variance:**
```
Var(λ̂) ≈ 2c²  for c small
```

**Mean squared error:**
```
MSE(λ̂) ≈ 4c² + 2c²  = 6c²
```

### 4.3 Eigenvector Alignment Bias

**Davis-Kahan sin Θ theorem:**

For eigenpair (λ, u) and sample (λ̂, û):
```
||sin(Θ)|| ≤ ||Ŝ - Σ||_F / min_i |λ̂_i - λ_j|  (j ≠ i)
```

Where Θ is angle between u and û.

**In high dimensions:**
```
||Ŝ - Σ||_F = O_p(√(p/T))
```

If eigenvalue gap δ is O(1):
```
||sin(Θ)|| = O_p(√(p/T))
```

**Conclusion:**
```
angle(û, u) = O_p(√(p/T))
```

**For p/T = 5:**
```
Expected angle ≈ √5 ≈ 2.24 radians ≈ 128°
```

This is nearly orthogonal! (π/2 ≈ 1.57 rad = 90°)

### 4.4 Factor Loading Bias

For factor model B'f + ε with k factors:

**Sample loading bias:**
```
E[||B̂ - B||_F²] = O(kp/T)
```

**Relative bias:**
```
E[||B̂ - B||_F²] / ||B||_F² = O(k/T)
```

If k/T is not small, severe bias.

**Example:** k = 5, T = 63, p = 500

```
Relative MSE ≈ 5/63 ≈ 8%  (RMS error ≈ 28%)
```

Plus additional error from p/T = 500/63 ≈ 7.9.

### 4.5 Systematic Patterns

**Pattern 1: Larger eigenvalues more biased**

Largest eigenvalues have largest absolute bias.

**Pattern 2: Bias increases with c = p/T**

Linear or quadratic in c.

**Pattern 3: Small gaps exacerbate bias**

When λᵢ ≈ λⱼ, eigenvectors unstable.

**Pattern 4: Rotational ambiguity**

High-dimensional data has approximate symmetries.

---

## 5. Spiked Covariance Model

### 5.1 Model Specification

**Spiked model:**

Population covariance:
```
Σ = diag(ℓ₁, ..., ℓₖ, 1, ..., 1)
```

Where:
- ℓ₁ ≥ ... ≥ ℓₖ > 1: Spike eigenvalues
- 1, ..., 1: Bulk eigenvalues (p - k of them)

**Interpretation:**

- k true factors
- p - k idiosyncratic components

**Limiting regime:**
```
k fixed, p, T → ∞, p/T → c
```

### 5.2 Baik-Ben Arous-Péché (BBP) Transition

**Critical threshold:**

Define:
```
ρ(c) = 1 + √c  (phase transition threshold)
```

**Three cases:**

**Case 1: Subcritical (ℓᵢ ≤ ρ(c))**

Spike eigenvalue **invisible** in sample:
```
λ̂ᵢ merges into MP bulk
λ̂ᵢ → (1 + √c)² ≠ ℓᵢ
```

Cannot detect factor!

**Case 2: Critical (ℓᵢ = ρ(c))**

Spike eigenvalue at edge:
```
λ̂ᵢ → (1 + √c)²
```

Barely detectable.

**Case 3: Supercritical (ℓᵢ > ρ(c))**

Spike eigenvalue **visible**, pops out of bulk:
```
λ̂ᵢ → ℓᵢ(1 + c/ℓᵢ)  (biased upward!)
```

Factor detectable.

### 5.3 Quantitative Results

**Supercritical limit (ℓ > 1 + √c):**

```
λ̂ → ℓ(1 + c/ℓ) = ℓ + c
```

**Bias:**
```
Bias(λ̂) = c (independent of ℓ for large ℓ)
```

**Relative bias:**
```
Relative bias = c/ℓ
```

**Eigenvector consistency:**

For ℓ > 1 + √c:
```
û · u →_p 1  (consistent!)
```

But:
```
angle(û, u) = O_p(√(c/(ℓ - 1 - √c)))
```

**Convergence rate depends on distance from threshold.**

### 5.4 Phase Diagram

```
                  λ̂ behavior
    
    Supercritical │
    (Detectable)  │ λ̂ → ℓ + c
                  │ û → u
                  │
    ─────────────┼───────────  ℓ = 1 + √c (critical)
                  │
    Subcritical   │ λ̂ merges into bulk
    (Hidden)      │ û random
                  │
                  └─────────────→ ℓ (true eigenvalue)
```

**Critical line:** ℓ_critical = 1 + √c

**Examples:**

| c = p/T | ρ(c) | Interpretation |
|---------|------|----------------|
| 0.5 | 1.71 | Need ℓ > 1.71 to detect |
| 1.0 | 2.00 | Need ℓ > 2.00 to detect |
| 2.0 | 2.41 | Need ℓ > 2.41 to detect |
| 5.0 | 3.24 | Need ℓ > 3.24 to detect |
| 10.0 | 4.16 | Need ℓ > 4.16 to detect |

**Implication:**

Weak factors (ℓ slightly above 1) are invisible when p/T is large!

### 5.5 Multiple Spikes

For k spikes ℓ₁ ≥ ... ≥ ℓₖ:

**Each spike has its own phase transition.**

**Case 1:** All supercritical
```
ℓᵢ > 1 + √c  for all i
```
All factors detectable.

**Case 2:** Some supercritical
```
ℓ₁, ..., ℓₘ > 1 + √c
ℓₘ₊₁, ..., ℓₖ ≤ 1 + √c
```
Only first m factors detectable. Others hidden.

**Case 3:** All subcritical
```
ℓᵢ ≤ 1 + √c  for all i
```
No factors detectable from eigenvalues alone.

**Practical scenario:**

If p = 500, T = 63, c ≈ 7.9:
```
ρ(7.9) ≈ 3.81
```

Need factors explaining > 381% of idiosyncratic variance to detect!

---

## 6. Phase Transitions

### 6.1 Detection vs. Estimation

**Detection:** Can we determine k (number of factors)?

**Estimation:** Given k, how well can we estimate B?

**Key insight:** These have different thresholds!

### 6.2 Detectability Threshold

From BBP theory:
```
Detectable if ℓᵢ > 1 + √c
```

**Example:** p = 1000, T = 100, c = 10

```
ρ(10) = 1 + √10 ≈ 4.16
```

Factor must explain 316% more variance than idiosyncratic!

**Variance ratio interpretation:**

If σ²_factor = 0.10 and σ²_idio = 0.01:
```
ℓ = σ²_factor / σ²_idio = 10 > 4.16 ✓ (detectable)
```

But if σ²_factor = 0.03:
```
ℓ = 0.03 / 0.01 = 3 < 4.16 ✗ (not detectable)
```

### 6.3 Estimation Threshold

Even if detectable, estimation quality depends on signal strength.

**Eigenvector convergence rate:**
```
angle(û, u) = O_p(√(c/(ℓ - 1 - √c)²))
```

**For good estimation:**

Need:
```
ℓ - 1 - √c = Ω(√c)
```

So:
```
ℓ ≥ 1 + 2√c  (estimation threshold, stricter!)
```

**Comparison:**

| c | Detection ℓ > | Estimation ℓ > |
|---|---------------|----------------|
| 1 | 2.00 | 3.00 |
| 2 | 2.41 | 3.83 |
| 5 | 3.24 | 5.47 |
| 10 | 4.16 | 7.32 |

**Conclusion:** Detection easier than accurate estimation.

### 6.4 Information-Theoretic Limits

**Minimax rate for subspace estimation:**

Over all estimators B̂, worst-case over models with signal strength ≥ ℓ:
```
inf_B̂ sup_B E[d²(span(B), span(B̂))] ≍ (kp/T) · (1/ℓ²)
```

**Implications:**

1. Rate kp/T fundamental (cannot improve)
2. Stronger signal (larger ℓ) helps quadratically
3. More factors k increases difficulty linearly
4. More assets p increases difficulty linearly

**For p >> T:** Rate is kp/T >> k, very slow.

### 6.5 Concentration vs. Phase Transition

**Two phenomena:**

1. **Concentration:** Geometry of high dimensions
2. **Phase transition:** Signal vs. noise in random matrices

**Interaction:**

In high dimensions:
- Small signals buried by concentration
- Phase transition determines visibility
- Even visible signals have bias

**Combined effect:**

For factor model with p >> T:
```
- Detection threshold: ℓ > 1 + √c (BBP)
- Concentration: Distances approximately equal
- Bias: λ̂ ≈ ℓ + c (upward bias)
- Eigenvector error: angle = O(√(c/ℓ²))
```

All effects compound!

---

## 7. Bias in Manifold Distances

### 7.1 Grassmannian Distance Bias

**Population distance:**
```
d_G(B_true, B_true) = 0
```

**Sample distance:**
```
d_G(B_true, B̂_sample) = ||θ||₂
```

Where θ are principal angles between true and estimated subspaces.

**In high dimensions:**

Each angle θᵢ has expected value:
```
E[θᵢ²] ≈ kp/T · (1/ℓᵢ²)
```

So:
```
E[d_G²] ≈ (kp/T) · Σᵢ (1/ℓᵢ²)
```

**For equal eigenvalues ℓ:**
```
E[d_G²] ≈ (kp/T) · (k/ℓ²) = k²p/(Tℓ²)
E[d_G] ≈ k√(p/(Tℓ²)) = k√p/(ℓ√T)
```

**Example:** k = 3, p = 500, T = 63, ℓ = 5

```
E[d_G] ≈ 3√500/(5√63) ≈ 2.67
```

This is large! (Recall good < 0.3)

### 7.2 Upward Bias

**Key point:** Grassmannian distance is biased upward.

```
E[d_G(B_true, B̂)] > d_G(B_true, B_population) = 0
```

**Reason:** Estimation error always increases distance.

**Asymptotic distribution:**

For large T but p/T → c:
```
√T · d_G →_d some distribution (not normal!)
```

**Convergence rate:** O(√(p/T))

**For p >> T:** Very slow convergence.

### 7.3 Procrustes Distance Bias

Procrustes distance has similar behavior:
```
E[d_P²] ≈ (kp/T) · f(ℓ)
```

Where f(ℓ) depends on signal strength.

**Relationship:**
```
d_P² = 2k - 2Σᵢ cos(θᵢ)
     ≈ 2k - 2k + Σᵢ θᵢ²
     ≈ Σᵢ θᵢ²
     ≈ d_G²
```

For small angles (good estimation).

**For large angles (poor estimation):**
```
d_P > d_G  (can be substantial)
```

### 7.4 Bootstrap Bias Estimation

To assess bias in practice:

```python
def estimate_manifold_bias(returns, B_true, k, n_boot=1000):
    """
    Estimate bias in Grassmannian distance via bootstrap.
    
    Returns empirical distribution of d_G under null (B_true = B_population).
    """
    T, p = returns.shape
    distances = []
    
    for _ in range(n_boot):
        # Resample returns
        idx = np.random.choice(T, T, replace=True)
        R_boot = returns[idx]
        
        # Estimate factors
        B_boot = svd_decomposition(R_boot, k)
        
        # Compute distance
        d = grassmannian_distance(B_true, B_boot)
        distances.append(d)
    
    # Empirical bias
    bias = np.mean(distances)
    std = np.std(distances)
    
    return {
        'bias': bias,
        'std': std,
        'distribution': distances
    }
```

**Use:**

If observed d_G = 0.5 but bootstrap bias = 0.4:
```
Adjusted distance ≈ 0.5 - 0.4 = 0.1 (much better!)
```

### 7.5 Theoretical Bias Correction

**Ledoit-Wolf correction for eigenvalues:**

```
λ̂_corrected = (λ̂_sample - γ) / (1 - γ)
```

Where γ = p/T.

**For Grassmannian distance:**

Approximate correction:
```
d_G,corrected² ≈ d_G,observed² - k²p/(Tℓ̄²)
```

Where ℓ̄ is average signal strength.

**Problem:** ℓ̄ unknown in practice.

**Practical approach:**

Use cross-validation or out-of-sample validation.

---

## 8. Practical Corrections

### 8.1 Eigenvalue Shrinkage

**Ledoit-Wolf shrinkage:**

Optimal linear combination:
```
Ŝ_shrink = (1 - α)Ŝ + αI
```

Where α chosen to minimize MSE:
```
α* = Σᵢ Var(λ̂ᵢ) / Σᵢ (λ̂ᵢ - 1)²
```

**Ledoit-Wolf estimator:**

```python
def ledoit_wolf_shrinkage(returns):
    """
    Ledoit-Wolf shrinkage estimator.
    
    Reference: Ledoit & Wolf (2004)
    """
    T, p = returns.shape
    
    # Sample covariance
    S = np.cov(returns, rowvar=False)
    
    # Shrinkage target (identity)
    target = np.eye(p)
    
    # Optimal shrinkage intensity
    # (simplified formula)
    numerator = 0
    denominator = 0
    
    for t in range(T):
        r = returns[t]
        numerator += np.sum((np.outer(r, r) - S)**2)
        
    numerator /= T
    
    # Frobenius norm squared
    denominator = np.sum((S - target)**2)
    
    alpha = min(1, numerator / denominator)
    
    # Shrunk covariance
    S_shrink = (1 - alpha) * S + alpha * target
    
    return S_shrink, alpha
```

**Effect:** Reduces eigenvalue spread, shrinks towards target.

### 8.2 Nonlinear Shrinkage

**Ledoit-Wolf nonlinear shrinkage:**

Instead of linear combination, apply nonlinear transformation to eigenvalues:
```
λᵢ → φ(λᵢ)
```

Where φ chosen optimally.

**Optimal transformation:**

Under rotation-equivariant loss:
```
φ(λ) = conditional E[true eigenvalue | observed λ]
```

**Oracle approximating shrinkage (OAS):**

```python
def oracle_approximating_shrinkage(returns):
    """
    Oracle Approximating Shrinkage (Chen et al. 2010)
    
    Improves over Ledoit-Wolf.
    """
    T, p = returns.shape
    
    # Sample covariance eigenvalues
    S = np.cov(returns, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(S)
    
    # OAS shrinkage
    trace_S2 = np.sum(eigvals**2)
    trace_S = np.sum(eigvals)
    
    # OAS intensity
    rho = min(1, ((1 - 2/p) * trace_S2 + trace_S**2) / 
                  ((T + 1 - 2/p) * (trace_S2 - trace_S**2 / p)))
    
    # Shrink eigenvalues
    target = trace_S / p
    eigvals_shrunk = (1 - rho) * eigvals + rho * target
    
    # Reconstruct
    S_shrunk = eigvecs @ np.diag(eigvals_shrunk) @ eigvecs.T
    
    return S_shrunk, rho
```

### 8.3 Clipping/Filtering

**Remove small eigenvalues:**

```python
def clip_eigenvalues(S, threshold=None, n_keep=None):
    """
    Clip small eigenvalues to reduce noise.
    
    Parameters
    ----------
    S : array (p, p)
        Covariance matrix
    threshold : float, optional
        Eigenvalue threshold (keep if λ > threshold)
    n_keep : int, optional
        Number of eigenvalues to keep
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    if n_keep is not None:
        eigvals[n_keep:] = 0
    elif threshold is not None:
        eigvals[eigvals < threshold] = 0
    
    S_filtered = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return S_filtered
```

**Marchenko-Pastur threshold:**

Keep eigenvalues above:
```
λ_threshold = (1 + √(p/T))²
```

### 8.4 Cross-Validation

**Rolling window validation:**

```python
def cross_validate_factors(returns, k_values, window=252):
    """
    Cross-validate number of factors.
    
    Use rolling window to assess out-of-sample performance.
    """
    T, p = returns.shape
    errors = {k: [] for k in k_values}
    
    for t in range(window, T):
        # Training window
        R_train = returns[t-window:t]
        
        # Test observation
        r_test = returns[t]
        
        for k in k_values:
            # Estimate factors
            model = svd_decomposition(R_train, k)
            
            # Predict
            f_test = estimate_factors(r_test, model.B)
            r_pred = model.B.T @ f_test
            
            # Error
            error = np.mean((r_test - r_pred)**2)
            errors[k].append(error)
    
    # Choose k with smallest error
    mean_errors = {k: np.mean(err) for k, err in errors.items()}
    optimal_k = min(mean_errors, key=mean_errors.get)
    
    return optimal_k, mean_errors
```

### 8.5 Robust PCA

For heavy-tailed distributions (common in finance):

**Huber PCA:**

Replace least squares with Huber loss:
```
min Σᵢ ρ(||xᵢ - Bf̂ᵢ||₂)
```

Where:
```
ρ(r) = { r²/2        if |r| ≤ δ
       { δ|r| - δ²/2  if |r| > δ
```

**Spherical PCA:**

Normalize to unit sphere:
```python
def spherical_pca(returns, k):
    """
    Spherical PCA: Normalize observations before PCA.
    
    Reduces influence of outliers.
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(returns, axis=1, keepdims=True)
    returns_normalized = returns / norms
    
    # Standard PCA on normalized data
    model = svd_decomposition(returns_normalized, k)
    
    return model
```

---

## 9. Regularization Methods

### 9.1 Ridge Regularization

Add penalty to factor loadings:
```
min ||R - FB'||²_F + λ||B||²_F
```

**Solution:**
```
B̂_ridge = (F̂'F̂ + λI)^(-1) F̂'R
```

**Effect:** Shrinks loadings towards zero, reduces overfitting.

**Choosing λ:** Cross-validation or GCV.

### 9.2 Lasso/Sparse PCA

Encourage sparsity:
```
min ||R - FB'||²_F + λ||B||₁
```

**Benefit:** Interpretable factors with few non-zero loadings.

**Implementation:** Coordinate descent or proximal gradient.

### 9.3 Nuclear Norm Minimization

Encourage low-rank structure:
```
min ||R - M||²_F + λ||M||_*
```

Where ||M||_* = Σᵢ σᵢ(M) is nuclear norm (sum of singular values).

**Solution via soft thresholding:**
```
M̂ = U diag(max(σᵢ - λ, 0)) V'
```

Where USV' = SVD(R).

### 9.4 Factor Model Regularization

**Generalized approach:**

```
min_{B,F} ||R - FB'||²_F + λ_B Ω_B(B) + λ_F Ω_F(F)
```

Where:
- Ω_B(B): Regularization on loadings (e.g., ℓ₁, ℓ₂, sparsity)
- Ω_F(F): Regularization on factors (e.g., smoothness, AR structure)

**Alternating optimization:**

1. Fix F, optimize B
2. Fix B, optimize F
3. Repeat until convergence

### 9.5 Random Matrix Denoising

**Optimal Shrinkage (Donoho et al.):**

For asymptotically optimal denoising:
```
σᵢ → φ*(σᵢ)
```

Where φ* computed from Marchenko-Pastur distribution.

**Procedure:**

1. Compute sample eigenvalues σ₁, ..., σ_p
2. Estimate noise level σ_noise from bulk
3. Apply optimal shrinker φ* to each σᵢ
4. Reconstruct: Σ̂_denoised = Σᵢ φ*(σᵢ) vᵢvᵢ'

---

## 10. Recommendations

### 10.1 When p/T ≤ 1 (Classical Regime)

**Bias:** Moderate, decreases as T increases.

**Recommendation:**
- Standard methods usually work
- Eigenvalue bias small but present
- Eigenvector estimation reasonable

**Checks:**
- Bootstrap confidence intervals
- Check eigenvalue gaps
- Validate out-of-sample

### 10.2 When 1 < p/T ≤ 5 (Moderate High-Dimensional)

**Bias:** Substantial, requires attention.

**Recommendation:**
- Use shrinkage estimators (Ledoit-Wolf)
- Be aware of eigenvalue bias (+c bias)
- Eigenvectors less reliable
- Increase sample size if possible

**Checks:**
- Apply bias correction
- Use robust methods
- Cross-validate k (number of factors)

**Manifold distances:**
- Expect upward bias
- Use bootstrap to estimate bias
- Interpret conservatively

### 10.3 When 5 < p/T ≤ 10 (High-Dimensional)

**Bias:** Severe, regularization essential.

**Recommendation:**
- **Always use regularization** (ridge, nuclear norm)
- Apply strong shrinkage
- Consider dimension reduction first
- Factor detection difficult

**BBP threshold:**
- Check if factors above ℓ > 1 + √c threshold
- Weak factors invisible
- May need domain knowledge

**Manifold distances:**
- Large upward bias expected
- Use corrected distances
- Focus on relative comparisons

### 10.4 When p/T > 10 (Ultra-High-Dimensional)

**Bias:** Extreme, standard methods fail.

**Recommendation:**
- **Regularization mandatory**
- Use domain knowledge for factor selection
- Consider alternative methods:
  * Pre-screening (reduce p)
  * Grouping assets (reduce p)
  * Hierarchical models
  * Bayesian priors

**Factor detection:**
- Eigenvalue-based methods unreliable
- Use economic theory
- Impose sparsity
- Cross-sectional information

**Manifold distances:**
- Bias dominates signal
- Not reliable for absolute assessment
- May still work for relative comparisons
- Bootstrap essential

### 10.5 Sample Size Guidelines

**For reliable factor analysis:**

| p | Minimum T | Recommended T |
|---|-----------|---------------|
| 100 | 200 | 500 |
| 500 | 1000 | 2500 |
| 1000 | 2000 | 5000 |
| 5000 | 10000 | 25000 |

**Rule of thumb:**
```
T_recommended ≥ 5p
```

**For detection only:**
```
T_minimum ≥ 2p
```

**For accurate estimation:**
```
T_recommended ≥ 10p (conservative)
```

### 10.6 Diagnostic Checklist

Before trusting results with p/T > 2:

- [ ] **Check BBP threshold:** Are factors above ℓ > 1 + √c?
- [ ] **Estimate bias:** Bootstrap Grassmannian distance distribution
- [ ] **Apply shrinkage:** Use Ledoit-Wolf or stronger
- [ ] **Cross-validate:** Out-of-sample validation
- [ ] **Compare methods:** Try multiple regularization approaches
- [ ] **Check stability:** Small perturbations shouldn't change results dramatically
- [ ] **Domain knowledge:** Do factors make economic sense?
- [ ] **Visualize:** Inspect loading patterns for interpretability

### 10.7 When to Be Concerned

**Red flags:**

1. **p/T > 5** and no regularization used
2. **Eigenvalue spread** > 100× in null regions
3. **Factor interpretations** change with small data changes
4. **Grassmannian distance** > 0.5 without bias correction
5. **Eigenvector correlations** < 0.8 for supposedly strong factors
6. **Out-of-sample performance** much worse than in-sample

**Action:**
- Increase sample size
- Apply stronger regularization
- Reduce dimension (p)
- Use domain constraints

### 10.8 Reporting Standards

When reporting results with high p/T:

**Essential:**
- Report p, T, and p/T ratio explicitly
- State whether regularization was used
- Provide bootstrap confidence intervals
- Show out-of-sample validation

**Recommended:**
- Compare multiple methods
- Report bias-corrected distances
- Show eigenvalue spectrum plot
- Discuss BBP threshold

**Interpretive:**
- Acknowledge bias
- Discuss limitations
- Provide robustness checks
- Connect to economic theory

---

## Mathematical Appendix

### A. Marchenko-Pastur Density

For c = p/T, the Marchenko-Pastur density is:

```
f_MP(λ; c) = (1/(2πc)) · √((λ_+ - λ)(λ - λ_-)) / λ

λ_- = (1 - √c)²
λ_+ = (1 + √c)²

Support: [λ_-, λ_+] if c < 1
         [0, λ_+] with point mass at 0 if c ≥ 1
```

### B. Tracy-Widom Distribution

The Tracy-Widom distribution TW₂ has CDF:

```
F_TW₂(s) = exp(-∫_s^∞ (x - s)q²(x)dx)
```

Where q(x) solves the Painlevé II equation:
```
q''(x) = xq(x) + 2q³(x)
```

Numerical values:
- Mode: -1.77
- Mean: -1.21
- Median: -1.27
- Variance: 1.61

### C. BBP Phase Transition Formula

For spike eigenvalue ℓ:

**Subcritical (ℓ ≤ 1 + √c):**
```
λ̂ → (1 + √c)²
û random (no signal)
```

**Supercritical (ℓ > 1 + √c):**
```
λ̂ → ℓ + c/ℓ
û · u → √(1 - (1 + √c)²/ℓ²)
```

**Critical (ℓ = 1 + √c):**
```
T^(2/3)(λ̂ - (1 + √c)²) →_d TW₂
```

### D. Concentration Inequalities

**Dvoretzky-Kiefer-Wolfowitz:**

For empirical distribution of eigenvalues:
```
P(sup_λ |F̂_p(λ) - F_MP(λ)| > ε) ≤ 2e^(-2pε²)
```

**Matrix Bernstein:**

For sum of random matrices:
```
P(||Σᵢ Xᵢ|| > t) ≤ 2n exp(-t²/(2σ² + 2Rt/3))
```

---

## References

### Key Papers

1. **Marchenko, V. A., & Pastur, L. A. (1967).** "Distribution of eigenvalues for some sets of random matrices." *Matematicheskii Sbornik*, 114(4), 507-536.

2. **Baik, J., Ben Arous, G., & Péché, S. (2005).** "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices." *Annals of Probability*, 33(5), 1643-1697.

3. **Johnstone, I. M. (2001).** "On the distribution of the largest eigenvalue in principal components analysis." *Annals of Statistics*, 29(2), 295-327.

4. **Ledoit, O., & Wolf, M. (2004).** "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.

5. **Paul, D. (2007).** "Asymptotics of sample eigenstructure for a large dimensional spiked covariance model." *Statistica Sinica*, 17(4), 1617-1642.

6. **Bai, J., & Ng, S. (2002).** "Determining the number of factors in approximate factor models." *Econometrica*, 70(1), 191-221.

7. **Fan, J., Liao, Y., & Mincheva, M. (2013).** "Large covariance estimation by thresholding principal orthogonal complements." *Journal of the Royal Statistical Society B*, 75(4), 603-680.

8. **Donoho, D. L., Gavish, M., & Johnstone, I. M. (2018).** "Optimal shrinkage of eigenvalues in the spiked covariance model." *Annals of Statistics*, 46(4), 1742-1778.

### Books

1. **Bai, Z., & Silverstein, J. W. (2010).** *Spectral Analysis of Large Dimensional Random Matrices*. Springer.

2. **Anderson, G. W., Guionnet, A., & Zeitouni, O. (2010).** *An Introduction to Random Matrices*. Cambridge University Press.

3. **Mehta, M. L. (2004).** *Random Matrices*. Academic Press.

---

## Summary

### Key Takeaways

1. **High dimensions change everything:** When p >> T, standard asymptotics fail.

2. **BBP threshold is fundamental:** Factors must be strong (ℓ > 1 + √c) to be detectable.

3. **Eigenvalues are biased:** Upward bias of approximately c in high dimensions.

4. **Eigenvectors are unreliable:** Angles scale as √(p/T), not 1/√T.

5. **Manifold distances biased upward:** Bootstrap to estimate bias.

6. **Regularization is not optional:** Essential when p/T > 2.

7. **Sample size requirements strict:** Need T ≥ 5p for reliable estimation.

8. **Weak factors invisible:** Small factors below BBP threshold undetectable.

9. **Phase transitions exist:** Abrupt change in detectability at critical threshold.

10. **Concentration phenomena:** High-dimensional geometry has counterintuitive properties.

### Practical Implications

**For factor lab manifold analysis:**

- Grassmannian distances **biased upward** when p/T large
- Bootstrap to assess bias magnitude
- Use bias-corrected distances for inference
- Interpret threshold tables conservatively
- Increase sample size if possible
- Apply regularization when p/T > 2
- Check BBP threshold for factor detectability
- Use cross-validation
- Report p/T ratio always

**Bottom line:** High dimensions require careful statistical treatment. Standard methods can be misleading without proper corrections.

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-30  
**Companion to:** TECHNICAL_MANUAL.md
