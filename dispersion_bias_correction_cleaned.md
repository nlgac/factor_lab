# Dispersion Bias: The James-Stein Correction

*Companion to `Proof_Theorem_3.1_prime_v3.md`. Develops the bias correction implied by
Theorem 3.1$'$, gives it a tractable closed form under Assumptions 2.5$'$ and 2.6$'$,
and illustrates it with a $k=2$ simulation. Simulation code: `bias_correction_demo.py`.*

---

## 1. What Is the Bias

The bias is a systematic underestimate of the equal-weight portfolio's true factor exposure, caused by the sample factor directions rotating away from the population ones.

The equal-weight portfolio is $z = e/\sqrt{p}$. In the population, its squared exposure to
the factor subspace is $|\Pi_B z|^2 = \sum_i c_i^2$, where $c_i = \mu_\infty(\beta_i)/\alpha_i$ is the normalised mean loading of factor $i$.
In the sample, we estimate this exposure using $H$ (the top-$k$ left singular vectors
of $Y/\sqrt{n}$), giving $|\Pi_H z|^2$. The Corollary to Theorem 3.1$'$ gives the gap:

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\longrightarrow\; \sum_{i=1}^k (1 - \psi_{\infty,i}^2)\,c_i^2 \;>\; 0
\quad \text{a.s.}
$$

Each term $(1 - \psi_{\infty,i}^2)c_i^2$ is the contribution of factor $i$ to the bias.
The source is Part (ii) of Theorem 3.1$'$: $h_i$ is not $b_i$, but a rotated version
satisfying $\langle h_i, b_i \rangle \to \psi_{\infty,i} < 1$. Because $h_i$ has drifted toward the noise subspace, the sample projection systematically undershoots the population projection. The shortfall grows with the noise-to-signal ratio and shrinks as the factor becomes stronger.

The practitioner consequence is that sample factor models understate how much of the equal-weight portfolio's variance is systematic, and correspondingly overstate the
idiosyncratic ("dispersion") component.

---

## 2. The James-Stein Correction in Vector Form

### 2.1 Structure of the bias

Under Assumptions 2.5$'$ and 2.6$'$, Part (ii) gives a precise decomposition of every
sample direction $h_i$ relative to its population counterpart $b_i$:

$$
h_i \;=\; \psi_{\infty,i}\, b_i \;+\; \Pi_{\mathcal{B}}^\perp h_i,
$$

where the two terms are orthogonal. Equivalently, $\Pi_{\mathcal{B}} h_i = \psi_{\infty,i} b_i$:
the $\mathcal{B}$-component of $h_i$ is correct in direction but scaled down by $\psi_{\infty,i}$, while the $\mathcal{B}^\perp$-component is pure noise.

The population projection decomposes as:

$$
\Pi_B z \;=\; \underbrace{\sum_i \frac{h_i^\top z}{\psi_{\infty,i}} h_i}_{\text{in } \mathcal{H}}
\;+\; \underbrace{\sum_i (b_i^\top z)\,\Pi_{\mathcal{B}}^\perp h_i}_{\text{in }
\mathcal{B} \cap \mathcal{H}^\perp}.
$$

The first term is fully observable and correctable. The second term lies in $\mathcal{B} \cap \mathcal{H}^\perp$ — the part of the true factor subspace the sample subspace missed entirely. This component is irreducible.

### 2.2 Why the irreducible component cannot be recovered

To estimate $\Pi_{\mathcal{B}}^\perp h_i = h_i - \psi_{\infty,i} b_i$ we need $b_i$, which requires estimating $p$ loading parameters from $n$ observations. Under the $p \to \infty$, $n$ fixed asymptotics, no estimator of $b_i$ as a direction in $\mathbb{R}^p$ can improve on $h_i$: the natural OLS estimate $\hat\beta_j := Y\chi_{p,j}/|\hat X_j|$ satisfies $\hat b_j := \hat\beta_j/|\hat\beta_j| \to h_j$ asymptotically (their inner product $\hat b_j^\top h_j \to 1$), so it carries exactly the same angular information as $h_j$ and no more. The angular error $\arccos(\psi_{\infty,i})$ is a fundamental lower bound imposed by the signal-to-noise ratio, not an artifact of the estimation procedure.

### 2.3 The tractable James-Stein correction

The correctable part gives the estimator

$$
\hat\Pi_B^{\mathrm{JS}} z \;:=\; H\hat D_\psi^{-1} H^\top z
\;=\; \sum_{i=1}^k \frac{h_i^\top z}{\hat\psi_i}\, h_i,
$$

where $\hat D_\psi = \operatorname{diag}(\hat\psi_1, \ldots, \hat\psi_k)$. Compared to the naive sample projection $\Pi_H z = H H^\top z$, the only change is replacing the identity weight matrix with $\hat D_\psi^{-1}$: each coordinate is inflated by $1/\hat\psi_i$ to undo the shrinkage. This estimator lives in the sample subspace $\mathcal{H}$.

The squared norm is now consistent:

$$
|\hat\Pi_B^{\mathrm{JS}} z|^2 \;=\; \sum_i \frac{(h_i^\top z)^2}{\hat\psi_i^2}
\;\longrightarrow\; \sum_i c_i^2 \;=\; |\Pi_B z|^2 \quad \text{a.s.}
$$

since $(h_i^\top z)^2/\psi_{\infty,i}^2 \to (b_i^\top z)^2 = c_i^2$ by Part (ii).
As a vector, $\hat\Pi_B^{\mathrm{JS}} z - \Pi_B z$ has residual norm $\sqrt{\sum_i (1-\psi_{\infty,i}^2)c_i^2}$ pointing into $\mathcal{B} \cap \mathcal{H}^\perp$
— the same quantity that measures the bias in the Corollary, now appearing as an
irreducible directional error rather than a scalar gap.

### 2.4 Estimating the shrinkage factors

From Lemma A.2$'$ Part 2, $s_{p,i}^2/p \to \alpha_i^2|X_i|^2 + \delta^2$ (where $s_{p,i}$ are singular values of $Y$). Defining $\hat\lambda_i = s_{p,i}^2/p$:

$$
\hat\psi_i \;=\; \sqrt{\max\!\left(0,\; 1 - \frac{\hat\delta^2}{\hat\lambda_i}\right)}
\;=\; \sqrt{\max\!\left(0,\; 1 - \frac{\hat\delta^2 \cdot p}{s_{p,i}^2}\right)},
$$

where the noise-variance estimate is

$$
\hat\delta^2 \;=\; \frac{\|(I - HH^\top)Y\|_F^2}{(p-k)\,n}.
$$

This is a closed-form expression in the singular values and the residual variance — no
additional model fitting required. $\hat\psi_i \to 1$ when the singular value dominates the noise floor (strong factor) and $\hat\psi_i \to 0$ when it barely clears it (weak factor, large correction, high variance). A floor at some $\tau > 0$ is advisable in practice to prevent amplifying estimation noise for weak factors.

### 2.5 Relationship to the Ledoit-Wolf literature

The operator $H\hat D_\psi^{-1} H^\top$ is the factor-subspace analog of the Ledoit-Wolf nonlinear shrinkage estimator for covariance matrices. Ledoit-Wolf applies an Oracle function to each empirical eigenvalue to correct for eigenvalue bias in large-dimensional covariance estimation; here, $1/\hat\psi_i$ is the analogous Oracle correction applied to each eigenvector's inner product with the probe vector $z$. The difference is that covariance shrinkage corrects a quadratic form in the full spectrum, while this correction targets a single direction $z$ and operates through the factor subspace only.

---

## 3. Illustration: $k = 2$

### 3.1 Model setup

$$
Y = B F^\top + Z, \qquad Y \in \mathbb{R}^{p \times n},\quad k = 2.
$$

**Loading matrix** (two-block, satisfying Assumptions 2.5$'$ and 2.2$'$):

$$
\beta_1 = (\underbrace{3, \ldots, 3}_{p/2},\; \underbrace{1, \ldots, 1}_{p/2}),
\qquad
\beta_2 = (\underbrace{-1, \ldots, -1}_{p/2},\; \underbrace{3, \ldots, 3}_{p/2}).
$$

Orthogonality: $\beta_1^\top\beta_2 = \frac{p}{2}(3)(-1) + \frac{p}{2}(1)(3) = 0$. $\checkmark$

Loading scales: $\alpha_1 = \alpha_2 = \sqrt{(9+1)/2} = \sqrt{5}$.

Cross-sectional means: $\mu_\infty(\beta_1) = 2$, $\mu_\infty(\beta_2) = 1$.

Equal-weight coefficients: $c_1 = 2/\sqrt{5}$, $c_2 = 1/\sqrt{5}$.

Population projection: $|\Pi_B z|^2 \to c_1^2 + c_2^2 = 4/5 + 1/5 = 1$.

**Simulation parameters**: $n = 60$, $\delta = 1.0$,
$\sigma_1 = 0.10$ (factor 1 return std), $\sigma_2 = 0.05$ (factor 2 return std).

### 3.2 Theoretical predictions

The asymptotic shrinkage factors ($|X_i|^2 \approx n\sigma_i^2$):

$$
\psi_{\infty,i} \;=\; \sqrt{\frac{\alpha_i^2 n \sigma_i^2}{\alpha_i^2 n \sigma_i^2 + \delta^2}}
$$

| Factor | $\alpha_i^2 n \sigma_i^2$          | $\psi_{\infty,i}$          | $c_i^2$       | Bias contribution $(1-\psi_i^2)c_i^2$ |
|:------:|:----------------------------------:|:--------------------------:|:-------------:|:-------------------------------------:|
| 1      | $5 \times 60 \times 0.01 = 3.00$   | $\sqrt{3/4} \approx 0.866$ | $4/5 = 0.800$ | $0.25 \times 0.800 = 0.200$           |
| 2      | $5 \times 60 \times 0.0025 = 0.75$ | $\sqrt{3/7} \approx 0.655$ | $1/5 = 0.200$ | $0.571 \times 0.200 = 0.114$          |

Total asymptotic bias: $0.200 + 0.114 = 0.314$.

Asymptotic limits: $\lim |\Pi_H z|^2 = 0.686$, $\lim |\hat\Pi_B^{\mathrm{JS}} z|^2 = 1.000$.

Factor 2 is weaker ($\psi_2 \approx 0.655$ vs $\psi_1 \approx 0.866$), so each unit of its equal-weight exposure is more severely distorted. Despite carrying only $1/5$ of the total exposure, it accounts for $0.114/0.314 = 36\%$ of the bias.

### 3.3 Simulation results

Monte Carlo: $M = 400$ draws at each $p$-slice. Reported as mean $\pm$ std.

| $p$               | $\|\Pi_H z\|^2$   | $\|\hat\Pi_B^{\mathrm{JS}} z\|^2$ | $\hat\psi_1$ | $\hat\psi_2$ | Bias (raw) | Bias (JS) |
| -----------------:|:-----------------:|:---------------------------------:|:------------:|:------------:|:----------:|:---------:|
| 50                | $0.536 \pm 0.119$ | $0.648 \pm 0.137$                 | 0.911        | 0.878        | 0.464      | 0.352     |
| 100               | $0.591 \pm 0.086$ | $0.744 \pm 0.100$                 | 0.894        | 0.830        | 0.409      | 0.256     |
| 200               | $0.616 \pm 0.070$ | $0.806 \pm 0.080$                 | 0.883        | 0.778        | 0.384      | 0.194     |
| 500               | $0.655 \pm 0.050$ | $0.894 \pm 0.047$                 | 0.875        | 0.721        | 0.346      | 0.106     |
| 1000              | $0.663 \pm 0.042$ | $0.933 \pm 0.030$                 | 0.870        | 0.692        | 0.337      | 0.068     |
| 2000              | $0.672 \pm 0.041$ | $0.956 \pm 0.021$                 | 0.869        | 0.682        | 0.328      | 0.044     |
| 5000              | $0.675 \pm 0.038$ | $0.971 \pm 0.013$                 | 0.870        | 0.667        | 0.325      | 0.029     |
| $\infty$ (theory) | $0.686$           | $1.000$                           | $0.866$      | $0.655$      | $0.314$    | $0$       |

The sample projection is essentially flat as $p$ grows — it is a consistent estimator of the wrong quantity. The JS-corrected projection converges toward 1 as the $\hat\psi_i$ estimates converge to their limits. The slow convergence of $\hat\psi_2$ (0.878 at $p=50$ vs the limit 0.655) reflects the well-known upward bias of sample eigenvalues at small $p$: the Gram matrix inflates the weaker factor's eigenvalue, making the signal look stronger than it is and causing the correction to under-inflate.
![Squared projection onto factor subspace vs p](chart1_projection_convergence.svg)

![Estimated shrinkage factors converging to theory](chart2_psi_convergence.svg)

---

## 4. Demonstration That the Correction Works: MSE

Showing that the corrected number is larger than the biased one is not a  emonstration
— it could be overshooting, or the remaining gap could be non-vanishing.

The proper measure is mean squared error relative to the truth.

$$
\mathrm{MSE}(\hat\theta) \;=\; \underbrace{(\mathbb{E}[\hat\theta] - \theta^*)^2}_{\mathrm{bias}^2}
\;+\; \underbrace{\mathrm{Var}(\hat\theta)}_{\mathrm{variance}}
$$

### 4.1 Decomposition

| $p$  | bias² (raw) | var (raw) | MSE (raw) | bias² (JS) | var (JS) | MSE (JS) | Ratio    |
| ----:|:-----------:|:---------:|:---------:|:----------:|:--------:|:--------:|:--------:|
| 50   | 0.2152      | 0.0141    | 0.229     | 0.1239     | 0.0187   | 0.143    | **1.6×** |
| 100  | 0.1675      | 0.0074    | 0.175     | 0.0656     | 0.0099   | 0.076    | **2.3×** |
| 200  | 0.1473      | 0.0049    | 0.152     | 0.0377     | 0.0064   | 0.044    | **3.4×** |
| 500  | 0.1194      | 0.0025    | 0.122     | 0.01134    | 0.00220  | 0.0135   | **9.0×** |
| 1000 | 0.1138      | 0.0018    | 0.116     | 0.00456    | 0.00091  | 0.0055   | **21×**  |
| 2000 | 0.1076      | 0.0017    | 0.109     | 0.00195    | 0.00044  | 0.0024   | **46×**  |
| 5000 | 0.1057      | 0.0014    | 0.107     | 0.00083    | 0.00016  | 0.0010   | **108×** |

![MSE decomposition: bias-squared plus variance for sample and JS estimators](chart3_mse_decomposition.svg)

### 4.2 Reading the table

**The sample estimator** is dominated by squared bias ($\approx 0.314^2 \approx 0.099$) at every $p$. Its MSE does not shrink with $p$ because the bias is asymptotic — it is converging to the wrong limit. The variance contribution is small and also not shrinking fast. The sample estimator is consistent for $|\Pi_B z|^2 - \text{bias}$, not for
$|\Pi_B z|^2$.

**The JS correction** has both bias and variance falling as $p$ grows. The bias falls because $\hat\psi_i \to \psi_{\infty,i}$ (Lemma A.2$'$), so the correction factors converge to the right values. The variance falls because the stabilised $\hat\psi_i$ introduce less amplification noise. At $p = 5000$ the MSE ratio is $108:1$.

![MSE ratio: JS correction MSE advantage by p-slice](chart4_mse_ratio.svg)

**The small-$p$ region** shows one honest cost. At $p = 50$ the JS correction has higher variance than the sample ($0.019$ vs $0.014$), because $1/\hat\psi_i^2$ amplifies estimation noise when the eigenvalues are inflated and $\hat\psi_i$ is itself poorly estimated. The bias reduction still wins — MSE is $1.6\times$ lower — but individual draws from the JS estimator can be worse than the sample estimator. This is the same tradeoff as in the original James-Stein result: the correction dominates in expectation (lower MSE), and the dominance strengthens in the regime where $p$ is large relative to $n$.

### 4.3 Summary

The correction works in the precise sense that $\mathrm{MSE}(\hat\Pi_B^{\mathrm{JS}} z) < \mathrm{MSE}(\Pi_H z)$ at every $p$ tested, and the ratio grows without bound as $p \to \infty$. The residual bias of the JS estimator at finite $p$ is due entirely to the finite-sample bias in $\hat\psi_i$ (eigenvalue inflation), not to any flaw in the correction formula itself. As $p$ grows the eigenvalue bias vanishes (Lemma A.2$'$) and the MSE of the corrected estimator shrinks to zero, while the sample MSE stays near $0.10$.

---

## 5. Reproducing the Simulation

```bash
cd factor_lab
python bias_correction_demo.py
```

The script is self-contained (numpy + pandas only). It builds the two-block orthogonal
loading matrix, simulates $M = 400$ draws of $Y = BF^\top + Z$ at each $p$-slice, and
reports sample projection, JS-corrected projection, estimated shrinkage factors, and
MSE components. Output is also saved to `bias_correction_demo_results.csv`.

Key lines in `bias_correction_demo.py`:

```python
# Shrinkage-factor estimate (from Lemma A.2' Part 2)
psi_hat = np.sqrt(np.maximum(0.0, 1.0 - delta2_hat * p / S**2))

# JS-corrected squared projection
js_proj = float(np.sum(coords**2 / np.maximum(psi_hat**2, 1e-8)))
```

where `S` are the top-$k$ singular values of $Y$ and `delta2_hat` is the residual
noise-variance estimate $\|(I - HH^\top)Y\|_F^2 / ((p-k)n)$.

---

*End of document.*
