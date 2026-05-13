# Dispersion Bias: The James-Stein Correction

*Companion to `unified_dispersion_bias_proof_050626.md`. Develops the bias correction
implied by Theorem 3.1$'$ (unified), gives it a tractable closed form under Assumptions
2.5$'$ and 2.6$'$ (§1–5), and extends to general $G_\infty$ (§6). A placeholder for the
further extension to a $k$-frame probe is reserved in §7. Simulation code:
`bias_correction_demo.py`.*

---

## 1. What Is the Bias

*This section and §2–5 assume Assumptions 2.5$'$ (orthogonal loading columns, $G_\infty = I_k$) and 2.6$'$ (orthogonal factor returns). Section 6 removes both restrictions.*

The bias is a systematic underestimate of the equal-weight portfolio's true factor exposure,
caused by the sample factor directions rotating away from the population ones.

The equal-weight portfolio is $z = e/\sqrt{p}$. Under Assumptions 2.5$'$ and 2.6$'$, its
squared exposure to the factor subspace is $|\Pi_B z|^2 = \sum_i c_i^2$, where
$c_i = \langle \bar{b}_i, z\rangle \to \mu_\infty(\beta_i)/\alpha_i$ is the normalised mean
loading of factor $i$. In the sample, we estimate this exposure using $H$ (the top-$k$ left
singular vectors of $Y/\sqrt{n}$), giving $|\Pi_H z|^2$. The Corollary to Theorem 3.1$'$
gives the gap:

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\longrightarrow\; \sum_{i=1}^k (1 - \psi_{\infty,i}^2)\,c_i^2 \;>\; 0
\quad \text{a.s.}
$$

Each term $(1 - \psi_{\infty,i}^2)c_i^2$ is the contribution of factor $i$ to the bias. The
source is Part (ii) of Theorem 3.1$'$: $h_i$ is not $\bar{b}_i$, but a rotated version
satisfying $\langle h_i, \bar{b}_i \rangle \to \psi_{\infty,i} < 1$. Because $h_i$ has drifted
toward the noise subspace, the sample projection systematically undershoots the population
projection. The shortfall grows with the noise-to-signal ratio and shrinks as the factor
becomes stronger.

The practitioner consequence is that sample factor models understate how much of the
equal-weight portfolio's variance is systematic, and correspondingly overstate the
idiosyncratic ("dispersion") component.

---

## 2. The James-Stein Correction in Vector Form

### 2.1 Structure of the bias

Under Assumptions 2.5$'$ and 2.6$'$, Part (ii) gives a precise decomposition of every
sample direction $h_i$ relative to its population counterpart $\bar{b}_i$:

$$
h_i \;=\; \psi_{\infty,i}\, \bar{b}_i \;+\; \Pi_{\mathcal{B}}^\perp h_i,
$$

where the two terms are orthogonal. Equivalently, $\Pi_{\mathcal{B}} h_i = \psi_{\infty,i} \bar{b}_i$:
the $\mathcal{B}$-component of $h_i$ is correct in direction but scaled down by $\psi_{\infty,i}$,
while the $\mathcal{B}^\perp$-component is pure noise.

The population projection decomposes as:

$$
\Pi_B z \;=\; \underbrace{\sum_i \frac{h_i^\top z}{\psi_{\infty,i}} h_i}_{\text{in } \mathcal{H}}
\;+\; \underbrace{\sum_i ({\bar{b}_i}^\top z)\,\Pi_{\mathcal{B}}^\perp h_i}_{\text{in }
\mathcal{B} \cap \mathcal{H}^\perp}.
$$

The first term is fully observable and correctable. The second term lies in
$\mathcal{B} \cap \mathcal{H}^\perp$ — the part of the true factor subspace the sample
subspace missed entirely. This component is irreducible.

### 2.2 Why the irreducible component cannot be recovered

To estimate $\Pi_{\mathcal{B}}^\perp h_i = h_i - \psi_{\infty,i} \bar{b}_i$ we need $\bar{b}_i$,
which requires estimating $p$ loading parameters from $n$ observations. Under the
$p \to \infty$, $n$ fixed asymptotics, no estimator of $\bar{b}_i$ as a direction in
$\mathbb{R}^p$ can improve on $h_i$: the natural OLS estimate
$\hat\beta_j := Y\chi_{p,j}/|\hat X_j|$ satisfies $\hat b_j := \hat\beta_j/|\hat\beta_j| \to h_j$
asymptotically (their inner product $\hat b_j^\top h_j \to 1$), so it carries exactly the
same angular information as $h_j$ and no more. The angular error $\arccos(\psi_{\infty,i})$
is a fundamental lower bound imposed by the signal-to-noise ratio, not an artifact of the
estimation procedure.

### 2.3 The tractable James-Stein correction

The correctable part gives the estimator

$$
\hat\Pi_B^{\mathrm{JS}} z \;:=\; H\hat D_\psi^{-1} H^\top z
\;=\; \sum_{i=1}^k \frac{h_i^\top z}{\hat\psi_i}\, h_i,
$$

where $\hat D_\psi = \operatorname{diag}(\hat\psi_1, \ldots, \hat\psi_k)$. Compared to the
naive sample projection $\Pi_H z = H H^\top z$, the only change is replacing the identity
weight matrix with $\hat D_\psi^{-1}$: each coordinate is inflated by $1/\hat\psi_i$ to
undo the shrinkage. This estimator lives in the sample subspace $\mathcal{H}$.

The squared norm is consistent:

$$
|\hat\Pi_B^{\mathrm{JS}} z|^2 \;=\; \sum_i \frac{(h_i^\top z)^2}{\hat\psi_i^2}
\;\longrightarrow\; \sum_i c_i^2 \;=\; |\Pi_B z|^2 \quad \text{a.s.}
$$

since $(h_i^\top z)^2/\psi_{\infty,i}^2 \to (\bar{b}_i^\top z)^2 = c_i^2$ by Part (ii).
As a vector, $\hat\Pi_B^{\mathrm{JS}} z - \Pi_B z$ has residual norm
$\sqrt{\sum_i (1-\psi_{\infty,i}^2)c_i^2}$ pointing into $\mathcal{B} \cap \mathcal{H}^\perp$
— the same quantity that measures the bias in the Corollary, now appearing as an
irreducible directional error rather than a scalar gap.

### 2.4 Estimating the shrinkage factors

From Lemma A.2$'$ Part 2, $s_{p,i}^2/p \to \alpha_i^2|X_i|^2 + \delta^2$ (where $s_{p,i}$
are singular values of $Y$). Defining $\hat\lambda_i = s_{p,i}^2/p$:

$$
\hat\psi_i \;=\; \sqrt{\max\!\left(0,\; 1 - \frac{\hat\delta^2}{\hat\lambda_i}\right)}
\;=\; \sqrt{\max\!\left(0,\; 1 - \frac{\hat\delta^2 \cdot p}{s_{p,i}^2}\right)},
$$

where the noise-variance estimate is

$$
\hat\delta^2 \;=\; \frac{\|(I - HH^\top)Y\|_F^2}{(p-k)\,n}.
$$

This is a closed-form expression in the singular values and the residual variance — no
additional model fitting required. $\hat\psi_i \to 1$ when the singular value dominates the
noise floor (strong factor) and $\hat\psi_i \to 0$ when it barely clears it (weak factor,
large correction, high variance). A floor at some $\tau > 0$ is advisable in practice to
prevent amplifying estimation noise for weak factors.

### 2.5 Relationship to the Ledoit-Wolf literature

The operator $H\hat D_\psi^{-1} H^\top$ is the factor-subspace analog of the Ledoit-Wolf
nonlinear shrinkage estimator for covariance matrices. Ledoit-Wolf applies an oracle
function to each empirical eigenvalue to correct for eigenvalue bias in large-dimensional
covariance estimation; here, $1/\hat\psi_i$ is the analogous oracle correction applied to
each eigenvector's inner product with the probe vector $z$. The difference is that
covariance shrinkage corrects a quadratic form in the full spectrum, while this correction
targets a single direction $z$ and operates through the factor subspace only.

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

Factor 2 is weaker ($\psi_2 \approx 0.655$ vs $\psi_1 \approx 0.866$), so each unit of its
equal-weight exposure is more severely distorted. Despite carrying only $1/5$ of the total
exposure, it accounts for $0.114/0.314 = 36\%$ of the bias.

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

The sample projection is essentially flat as $p$ grows — it is a consistent estimator of the
wrong quantity. The JS-corrected projection converges toward 1 as the $\hat\psi_i$ estimates
converge to their limits. The slow convergence of $\hat\psi_2$ (0.878 at $p=50$ vs the
limit 0.655) reflects the well-known upward bias of sample eigenvalues at small $p$: the
Gram matrix inflates the weaker factor's eigenvalue, making the signal look stronger than
it is and causing the correction to under-inflate.

![Squared projection onto factor subspace vs p](chart1_projection_convergence.svg)

![Estimated shrinkage factors converging to theory](chart2_psi_convergence.svg)

---

## 4. Demonstration That the Correction Works: MSE

Showing that the corrected number is larger than the biased one is not a demonstration
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

**The sample estimator** is dominated by squared bias ($\approx 0.314^2 \approx 0.099$) at
every $p$. Its MSE does not shrink with $p$ because the bias is asymptotic — it is
converging to the wrong limit. The variance contribution is small and also not shrinking
fast. The sample estimator is consistent for $|\Pi_B z|^2 - \text{bias}$, not for
$|\Pi_B z|^2$.

**The JS correction** has both bias and variance falling as $p$ grows. The bias falls
because $\hat\psi_i \to \psi_{\infty,i}$ (Lemma A.2$'$), so the correction factors converge
to the right values. The variance falls because the stabilised $\hat\psi_i$ introduce less
amplification noise. At $p = 5000$ the MSE ratio is $108:1$.

![MSE ratio: JS correction MSE advantage by p-slice](chart4_mse_ratio.svg)

**The small-$p$ region** shows one honest cost. At $p = 50$ the JS correction has higher
variance than the sample ($0.019$ vs $0.014$), because $1/\hat\psi_i^2$ amplifies estimation
noise when the eigenvalues are inflated and $\hat\psi_i$ is itself poorly estimated. The
bias reduction still wins — MSE is $1.6\times$ lower — but individual draws from the JS
estimator can be worse than the sample estimator. This is the same tradeoff as in the
original James-Stein result: the correction dominates in expectation (lower MSE), and the
dominance strengthens in the regime where $p$ is large relative to $n$.

### 4.3 Summary

The correction works in the precise sense that
$\mathrm{MSE}(\hat\Pi_B^{\mathrm{JS}} z) < \mathrm{MSE}(\Pi_H z)$ at every $p$ tested, and
the ratio grows without bound as $p \to \infty$. The residual bias of the JS estimator at
finite $p$ is due entirely to the finite-sample bias in $\hat\psi_i$ (eigenvalue inflation),
not to any flaw in the correction formula itself. As $p$ grows the eigenvalue bias vanishes
(Lemma A.2$'$) and the MSE of the corrected estimator shrinks to zero, while the sample MSE
stays near $0.10$.

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

Section 6 below extends the theory to general $G_\infty$; an extended version of
`bias_correction_demo.py` for the non-orthogonal case is outlined in §6.8.

---

## 6. Extension 1: General $G_\infty$

*This section drops Assumptions 2.5$'$ (orthogonal loading columns) and 2.6$'$ (orthogonal
factor returns). Loading columns may now point in correlated directions ($G_\infty \ne I_k$),
and factor returns may be mutually correlated. All other assumptions from the companion
proof (prevalence, noise, spectral separation) continue to hold. The main finding is that
the JSE correction formula $H\hat D_\psi^{-1} H^\top z$ is unchanged — it works for every
$G_\infty$ without any modification. What changes is the interpretation of the per-factor
bias terms and the objects that parametrise them.*

### 6.1 Setting and new objects

When $G_\infty \ne I_k$, the unit loading columns $\bar{b}_1, \ldots, \bar{b}_k$ are not
orthonormal, so $\tilde{B} = [\bar{b}_1, \ldots, \bar{b}_k]$ does not serve as its own
orthonormal basis. Three new objects replace the role played by $\tilde{B}$, $\hat{D}$,
and $c_j$ in §1–5.

**The orthonormal basis $U$.** Write $\tilde{B} = U(p)\,\Sigma(p)\,V(p)^\top$ for the thin
SVD of the unit loading matrix. The columns of $U(p) \in \mathbb{R}^{p \times k}$ are
orthonormal and span $\mathcal{B} = \mathrm{col}(B)$, so $\Pi_B = U U^\top$. As
$p \to \infty$, $\Sigma(p) \to G_\infty^{1/2}$ and $V(p) \to Q$, where $G_\infty = Q\Lambda_G Q^\top$
is the spectral decomposition of the limiting Gram matrix (with eigenvalues
$g_1 \ge \cdots \ge g_k > 0$). When $G_\infty = I_k$, $U = \tilde{B}$ exactly.

**The rotated factor covariance $\hat{M}$.** Define

$$
\hat{M} \;=\; \Lambda_G^{1/2}(Q^\top \hat{D}\, Q)\Lambda_G^{1/2},
\qquad M \;=\; \Lambda_G^{1/2}(Q^\top D\, Q)\Lambda_G^{1/2},
$$

where $\hat{D} = C^{1/2}(F^\top F / n)C^{1/2}$ is the prevalence-weighted sample factor
covariance and $D$ its population counterpart. When $G_\infty = I_k$, we have $Q = I_k$,
$\Lambda_G = I_k$, and $\hat{M} = \hat{D}$.

**Eigenvectors of $\hat{M}$ and $M$.** Let $\hat{w}_j$ (resp.\ $w_j$) be the $j$-th
eigenvector of $\hat{M}$ (resp.\ $M$) with eigenvalue $\rho_j$ (resp.\ $m_j$). When
$G_\infty = I_k$ and factor returns are orthogonal, $\hat{M} = \hat{D}$ is diagonal and
$\hat{w}_j = e_j$ (standard basis vectors), recovering the NG case.

**The population loading direction.** The $j$-th population loading direction satisfies
$\bar{b}_j = U w_j$: it is the direction in $\mathcal{B}$ whose coordinates in the
$U$-basis are the $j$-th eigenvector of $M$.

**The subspace coordinate vector of the probe.** For a deterministic probe $z$ with
$\|z\| \le 1$, define

$$
u \;:=\; U^\top z \;\in\; \mathbb{R}^k.
$$

This is the coordinate vector of $\Pi_B z$ in the orthonormal $U$-basis, and it satisfies
$|\Pi_B z|^2 = \|u\|^2$. When $G_\infty = I_k$, $u = \tilde{B}^\top z$ and $u_j = c_j$
for the equal-weight portfolio (recovering the NG coordinates). For general $G_\infty$,
$u = \Sigma^{-1}V^\top(\tilde{B}^\top z)$ encodes the probe's projection onto $\mathcal{B}$
in the orthonormal basis, accounting for the non-orthogonality of the loading columns.

*Note on $|\Pi_B z|^2$ for general $G_\infty$.* In §1, the formula
$|\Pi_B z|^2 = \sum_j c_j^2 = \|\tilde{B}^\top z\|^2$ uses $G_\infty = I_k$ implicitly.
For general $G_\infty$, $|\Pi_B z|^2 = \|u\|^2 = (\tilde{B}^\top z)^\top G_\infty^{-1}(\tilde{B}^\top z)$.
Only when $G_\infty = I_k$ does this reduce to $\sum_j c_j^2$.

### 6.2 The structural lemma

The following result is equation (52) from §8.3 of the unified proof. It holds for all
$G_\infty$ and is the key to everything that follows.

**Lemma** (In-subspace limit, general $G_\infty$). *Under Assumptions 1–3 of the unified
proof, for each $j \in \{1, \ldots, k\}$:*

$$
U(p)^\top h_j \;\xrightarrow{\;\mathrm{a.s.}\;}\; \psi_{\infty,j}\,\hat{w}_j,
\qquad \psi_{\infty,j} = \sqrt{\frac{n\rho_j}{n\rho_j + \delta^2}}.
$$

*The $\mathcal{B}$-component of $h_j$ converges to the direction $U\hat{w}_j$ — the $j$-th
sample principal loading direction — scaled down by the signal-to-noise factor $\psi_{\infty,j}$.*

**Remark.** When $G_\infty = I_k$ and factor returns are orthogonal, $\hat{w}_j = e_j$,
$\rho_j = c_j\sigma_j^2\|X_j\|^2/n$, and $U^\top h_j \to \psi_{\infty,j} e_j$ means
$\langle h_j, \bar{b}_j\rangle \to \psi_{\infty,j}$, recovering Part (ii) of Theorem 3.1$'$.

**Corollary.** *For any deterministic probe $z$ with $\|z\| \le 1$, and with
$u = U^\top z$:*

$$
h_j^\top z \;\xrightarrow{\;\mathrm{a.s.}\;}\; \psi_{\infty,j}\,\hat{w}_j^\top u.
$$

*Proof.* Decompose $z = \Pi_B z + \Pi_B^\perp z$. Then
$h_j^\top z = (U^\top h_j)^\top(U^\top z) + h_j^\top(\Pi_B^\perp z)$.
The first term converges to $\psi_{\infty,j}\hat{w}_j^\top u$ by the lemma.
The second term converges to zero almost surely by Part (i) of Theorem 3.1$'$
(subspace alignment), since $\|\Pi_B^\perp z\| \le 1$. $\square$

Assembling across all $j$, with $\hat{W} = [\hat{w}_1, \ldots, \hat{w}_k] \in O(k)$
(columns are the eigenvectors of $\hat{M}$):

$$
H^\top z \;\xrightarrow{\;\mathrm{a.s.}\;}\; \Psi_\infty(\hat{W}^\top u),
\qquad \Psi_\infty = \mathrm{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k}).
$$

The $j$-th component of the limit, $\psi_{\infty,j}\hat{w}_j^\top u$, is the shrinkage factor
$\psi_{\infty,j}$ times the probe's exposure to the $j$-th principal loading direction.

### 6.3 The general bias formula

**Theorem.** *Under Assumptions 1–3 (no restrictions on $G_\infty$ or factor return
covariance), for any deterministic probe $z$ with $\|z\| \le 1$:*

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\xrightarrow{\;\mathrm{a.s.}\;}\;
\sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2}\,(\hat{w}_j^\top u)^2 \;\ge\; 0.
$$

*Proof.* Since $\hat{W} \in O(k)$, the set $\{\hat{w}_j\}$ is an orthonormal basis of
$\mathbb{R}^k$, so $|\Pi_B z|^2 = \|u\|^2 = \sum_j(\hat{w}_j^\top u)^2$. The sample
projection satisfies

$$
|\Pi_H z|^2 = \|H^\top z\|^2 \;\xrightarrow{\;\mathrm{a.s.}\;}\;
\|\Psi_\infty\hat{W}^\top u\|^2 = \sum_j \psi_{\infty,j}^2\,(\hat{w}_j^\top u)^2.
$$

Subtracting gives the result, with $1 - \psi_{\infty,j}^2 = \delta^2/(n\rho_j + \delta^2)$.
$\square$

**Reduction to the NG case.** When $G_\infty = I_k$ and factor returns are orthogonal,
$\hat{w}_j = e_j$ and $\hat{w}_j^\top u = u_j = c_j$. The formula reduces to
$\sum_j(1-\psi_{\infty,j}^2)c_j^2$, as in §1. $\checkmark$

**Interpretation.** Each factor $j$ contributes $({\delta^2}/{(n\rho_j+\delta^2)})$ times
its *principal loading exposure* $(\hat{w}_j^\top u)^2$. The principal loading exposure is
the squared inner product of the probe's $\mathcal{B}$-coordinates $u$ with $\hat{w}_j$,
the direction in $\mathbb{R}^k$ that factor $j$ occupies in the rotated-reweighted
eigenbasis of $\hat{M}$. When loading columns are correlated ($G_\infty \ne I_k$), this
direction mixes the original factor indices: a probe concentrated on factor $j$'s nominal
loading column $\bar{b}_j$ will have its energy spread across several $\hat{w}_l$ directions,
and the bias attribution changes accordingly.

### 6.4 Invariance of the JSE correction

**Theorem.** *For all $G_\infty$ and all deterministic probes $z$ with $\|z\| \le 1$:*

$$
|\hat\Pi_B^{\mathrm{JS}} z|^2 \;=\; \sum_{j=1}^k \frac{(h_j^\top z)^2}{\hat\psi_j^2}
\;\xrightarrow{\;\mathrm{a.s.}\;}\; |\Pi_B z|^2.
$$

*The formula $\hat\Pi_B^{\mathrm{JS}} z = H\hat D_\psi^{-1} H^\top z$ is a consistent
estimator of $|\Pi_B z|^2$ regardless of the loading geometry.*

*Proof.* By the corollary of §6.2, $(h_j^\top z)^2 \to \psi_{\infty,j}^2(\hat{w}_j^\top u)^2$.
Dividing by $\hat\psi_j^2 \to \psi_{\infty,j}^2$:

$$
\sum_j \frac{(h_j^\top z)^2}{\hat\psi_j^2} \;\longrightarrow\;
\sum_j (\hat{w}_j^\top u)^2 \;=\; \|\hat{W}^\top u\|^2 \;=\; \|u\|^2 \;=\; |\Pi_B z|^2,
$$

where the final equality uses $\hat{W} \in O(k)$. $\square$

**The orthogonality of $\hat{W}$ is the key.** When $G_\infty = I_k$, $\hat{W} = I_k$ and
the argument reduces to dividing $\psi_{\infty,j}^2 c_j^2$ by $\psi_{\infty,j}^2$. For
general $G_\infty$, $\hat{W}$ rotates $u$ into the eigenbasis of $\hat{M}$, but because
$\hat{W}$ is orthogonal this rotation preserves $\|u\|^2$. Inverting $\psi_{\infty,j}$ then
undoes the shrinkage in each rotated coordinate, and the sum reconstructs the total squared
norm. The correction works without knowing how $u$ is distributed across the $\hat{w}_j$
directions.

**$G_\infty$-free estimation.** The estimator $\hat\psi_j = \sqrt{\max(0, 1 - \hat\delta^2 p / s_{p,j}^2)}$
from §2.4 is consistent for $\psi_{\infty,j}$ for all $G_\infty$: it depends only on the
singular values $s_{p,j}$ of $Y$ and the residual noise estimate $\hat\delta^2$, neither
of which requires knowledge of $G_\infty$. The singular values of $Y$ automatically reflect
whatever loading geometry is present — a larger $g = G_{\infty,12}$ changes the spike
structure of $YY^\top$, and the estimator tracks this through the observed $s_{p,j}$.

### 6.5 Decomposition of $h_j$ in the general case

In §2.1, under the NG assumptions, $h_j$ decomposed cleanly as
$h_j = \psi_{\infty,j}\bar{b}_j + \Pi_\mathcal{B}^\perp h_j$.
The general-case decomposition replaces $\bar{b}_j$ with $U\hat{w}_j$:

$$
h_j \;=\; \underbrace{\psi_{\infty,j}\,U\hat{w}_j}_{\text{in-subspace component}}
\;+\; \underbrace{\Pi_\mathcal{B}^\perp h_j}_{\text{out-of-subspace component, noise}}.
$$

The two components are orthogonal. The in-subspace component $\psi_{\infty,j}U\hat{w}_j$
points toward the $j$-th sample principal loading direction, not toward the $j$-th
population loading direction $\bar{b}_j = Uw_j$. The discrepancy $U\hat{w}_j \ne \bar{b}_j$
is the in-subspace rotation $\sin^2\angle(\hat{w}_j, w_j)$ from Part (iii) of the unified
theorem — a finite-$n$ effect that vanishes as $n \to \infty$.

**Comparison with the NG case.** Under Assumptions 2.5$'$ and 2.6$'$: $\hat{w}_j = w_j = e_j$
and $U = \tilde{B}$, so $U\hat{w}_j = \tilde{B}\,e_j = \bar{b}_j$. The sample and
population directions coincide in the $\mathcal{B}$-component, and the decomposition reduces
to the NG form in §2.1.

For the probe:
$h_j^\top z = \psi_{\infty,j}(U\hat{w}_j)^\top z + (\Pi_\mathcal{B}^\perp h_j)^\top z = \psi_{\infty,j}\hat{w}_j^\top u + o(1)$ a.s., matching the corollary of §6.2.

### 6.6 The irreducible component revisited

The argument of §2.2 extends without change. The OLS estimator
$\hat{b}_j = Y\chi_{p,j}/|\hat{X}_j|$ satisfies $\hat{b}_j^\top h_j \to 1$ for all
$G_\infty$ (since the OLS direction converges to $h_j$ in all cases), so no
estimator based on $Y$ carries more angular information than $h_j$. The bound
$\arccos(\psi_{\infty,j})$ on the angle between the sample subspace and the true
loading direction remains fundamental.

The irreducible component of the projection error lives in $\mathcal{B} \cap \mathcal{H}^\perp$
and has squared norm

$$
\left\|\Pi_B z - \hat\Pi_B^{\mathrm{JS}} z\right\|^2 \;\xrightarrow{\;\mathrm{a.s.}\;}\;
\sum_j (1 - \psi_{\infty,j}^2)\,(\hat{w}_j^\top u)^2.
$$

This equals the total bias from §6.3 — the JSE correction eliminates the scalar gap in
$|\Pi_B z|^2$ but cannot eliminate the directional error in the underlying vector, exactly
as in the NG case.

### 6.7 What changes in practice

The table below summarises the differences between the NG formulas (§1–5) and the general
case (§6). The correction formula itself is the final row — it is the only entry that does
not change.

| Quantity                                | NG case ($G_\infty = I_k$, orth.\ returns)   | General $G_\infty$                                                  |
|:--------------------------------------- |:--------------------------------------------:|:-------------------------------------------------------------------:|
| Population target $\vert\Pi_B z\vert^2$ | $\sum_j c_j^2$                               | $\|u\|^2 = (\tilde{B}^\top z)^\top G_\infty^{-1}(\tilde{B}^\top z)$ |
| Factor covariance matrix                | $\hat{D} = \mathrm{diag}(c_j\hat\sigma_j^2)$ | $\hat{M} = \Lambda_G^{1/2}(Q^\top\hat{D}Q)\Lambda_G^{1/2}$          |
| Signal strength $\rho_j$                | $j$-th eigenvalue of $\hat{D}$               | $j$-th eigenvalue of $\hat{M}$                                      |
| Factor direction $\hat{w}_j$            | $e_j$ (standard basis)                       | $j$-th eigenvector of $\hat{M}$                                     |
| Per-factor exposure                     | $c_j^2 = (\bar{b}_j^\top z)^2$               | $(\hat{w}_j^\top u)^2$                                              |
| Bias formula                            | $\sum_j(1-\psi_{\infty,j}^2)c_j^2$           | $\sum_j(1-\psi_{\infty,j}^2)(\hat{w}_j^\top u)^2$                   |
| Correction formula                      | $H\hat D_\psi^{-1} H^\top z$                 | $H\hat D_\psi^{-1} H^\top z$ (unchanged)                            |
| $\hat\psi_j$ estimator                  | $\sqrt{\max(0,1-\hat\delta^2 p/s_{p,j}^2)}$  | same (unchanged)                                                    |

Two practical remarks follow.

*Remark 1 (Bias attribution requires $G_\infty$).* To attribute the total bias to individual
factors using $(\hat{w}_j^\top u)^2$, one needs both $G_\infty$ (to form $\hat{M}$ and
obtain $\hat{w}_j$) and the probe's $U$-basis coordinates $u = U^\top z$ (which require
the loading structure). The total bias and its correction are $G_\infty$-free, but the
per-factor decomposition is not.

*Remark 2 (Bias magnitude shifts across factors).* Correlation between loading columns
concentrates signal in the first principal loading direction and diffuses it in weaker ones.
As $g = G_{\infty,12}$ increases from 0: $\rho_1$ increases (factor 1 strengthens, its
floor shrinks), $\rho_2$ decreases (factor 2 weakens, its floor grows). The total bias
changes because both the floors $\delta^2/(n\rho_j + \delta^2)$ and the exposures
$(\hat{w}_j^\top u)^2$ are affected.

### 6.8 Illustration: $k = 2$, non-orthogonal loadings

We compare two loading geometries with identical return parameters ($c_1 = c_2 = 5$,
$\sigma_1 = 0.10$, $\sigma_2 = 0.05$, $n = 60$, $\delta = 1.0$) but different Gram matrices.

**Case A (NG, §3):** $G_\infty = I_2$. Loading columns are orthogonal; $\hat{M} = \hat{D}$
and $\hat{w}_j = e_j$.

**Case B (general):** $G_\infty = \begin{pmatrix}1 & \tfrac{1}{2}\\[2pt]\tfrac{1}{2} & 1\end{pmatrix}$,
a positive definite matrix with off-diagonal entry $g = 1/2$. Loading columns are
correlated at $45°$ in the sense that $\langle\bar{b}_1,\bar{b}_2\rangle \to 1/2$.
The spectral decomposition is $G_\infty = Q\Lambda_G Q^\top$ with
$\Lambda_G = \mathrm{diag}(3/2,\, 1/2)$ and
$Q = \frac{1}{\sqrt{2}}\begin{pmatrix}1&-1\\1&1\end{pmatrix}$.

**Computing $\hat{M}$ for Case B.** With $\hat{D} \approx \mathrm{diag}(0.050, 0.0125)$
(population values at the given parameters):

$$
Q^\top\hat{D}Q \;=\; \tfrac{1}{2}\begin{pmatrix}0.0625 & 0.0375\\0.0375 & 0.0625\end{pmatrix},
$$

$$
\hat{M} \;=\; \Lambda_G^{1/2}(Q^\top\hat{D}Q)\Lambda_G^{1/2}
\;=\; \begin{pmatrix}0.04688 & {-0.01624}\\{-0.01624} & 0.01563\end{pmatrix}.
$$

**Eigenstructure of $\hat{M}$.** The eigenvalues of $\hat{M}$ are $\rho_1 \approx 0.0538$
and $\rho_2 \approx 0.00872$, with eigenvectors

$$
\hat{w}_1 \approx \begin{pmatrix}-0.920\\0.392\end{pmatrix}, \qquad
\hat{w}_2 \approx \begin{pmatrix}-0.392\\-0.920\end{pmatrix}.
$$

Factor 1's principal direction $\hat{w}_1$ mixes the two original factor indices (primary
weight on index 1, secondary weight on index 2); the factors are no longer decoupled in
the loading eigenbasis.

**Shrinkage and floor comparison.**

|                                 | Case A ($G_\infty = I_2$) | Case B ($G_\infty$ as above) |
|:------------------------------- |:-------------------------:|:----------------------------:|
| $\rho_1$                        | 0.0500                    | 0.0538                       |
| $\rho_2$                        | 0.0125                    | 0.00872                      |
| $\psi_{\infty,1}$               | 0.866                     | 0.874                        |
| $\psi_{\infty,2}$               | 0.655                     | 0.586                        |
| Floor$_1 = 1-\psi_{\infty,1}^2$ | 0.250                     | 0.237                        |
| Floor$_2 = 1-\psi_{\infty,2}^2$ | 0.571                     | 0.657                        |

The correlated loading geometry strengthens the dominant factor (larger $\rho_1$, smaller
floor) and weakens the subdominant one (smaller $\rho_2$, larger floor by $15\%$). The
JSE correction for factor 2 must now divide by $\hat\psi_2^2 \approx 0.344$ rather than
$0.429$ — a stronger inflation by factor $1/0.344 \approx 2.9$ vs $1/0.429 \approx 2.3$.

**The bias formula.** The total bias is $\sum_j(1-\psi_{\infty,j}^2)(\hat{w}_j^\top u)^2$.
The per-factor exposures $(\hat{w}_j^\top u)^2$ depend on $u = U^\top z$, which requires
specifying the loading structure beyond $G_\infty$ alone. In particular:

- If $u$ is concentrated in the direction of $\hat{w}_1$ (probe aligned with factor 1's
  principal loading), factor 1 dominates the bias and its correction. Case B has a smaller
  factor-1 floor than Case A, so the total bias is slightly smaller.
- If $u$ is concentrated in the direction of $\hat{w}_2$, factor 2's larger floor in
  Case B makes the total bias significantly larger than in Case A.
- For a probe with equal exposure to both directions ($\hat{w}_j^\top u$ equal for $j=1,2$),
  the change in total bias between Cases A and B is a weighted average of the floor changes.

**The correction formula is the same.** In both cases, $\hat\Pi_B^{\mathrm{JS}} z = H\hat D_\psi^{-1} H^\top z$
with $\hat\psi_j = \sqrt{\max(0, 1 - \hat\delta^2 p/s_{p,j}^2)}$. The singular values
$s_{p,j}$ of $Y$ in Case B reflect the correlated loading geometry and will consistently
estimate the Case B values of $\psi_{\infty,j}$. No knowledge of $G_\infty$ is required
to run the correction.

**Extending the simulation.** To reproduce the Case B scenario numerically, the
`bias_correction_demo.py` script requires a loading matrix with $G_\infty \ne I_2$.
A concrete construction: let $p/3$ assets load on $(\beta_{1j}, \beta_{2j}) = (a, 0)$,
$p/3$ load on $(0, a)$, and $p/3$ load on $(a/\sqrt{2}, a/\sqrt{2})$. Computing
$\langle\bar{b}_1,\bar{b}_2\rangle$ gives an off-diagonal $G_\infty$ entry of $1/3$ in
this example; adjusting the proportions controls $g$. The formula for $\hat\psi_j$ in
the script is unchanged — only the loading matrix passed to the simulation changes.

---

## 7. Extension 2: $k$-Frame Probe

*This section will develop the JSE correction for a probe frame
$W \in \mathbb{R}^{p \times k_W}$ (a matrix of $k_W$ orthonormal probe vectors) in place
of the single probe vector $z$. The frame-level bias, frame JSE correction, and their
relationship to the Frobenius deficit from Corollary 5 of the unified proof will be
established. The general-$G_\infty$ case of §6 will be incorporated from the outset.*

*[Content to follow.]*

---

*End of document.*
