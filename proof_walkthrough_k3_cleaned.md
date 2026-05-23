# Proof Walkthrough: Theorem Part (ii), Equation (5)
### A step-by-step illustration with a k=3, p=500, n=60 example

*This document follows Appendix B.3 of "Multifactor Dispersion Bias with Per-Column Prevalence: A Unified Treatment" step by step. At each step the general argument from the paper is stated first, then the concrete numbers from our example are substituted in so that the algebra is visible.*

---

## 0. The theorem we are proving

**Theorem Part (ii).** Under Assumptions 1–3, for each $j \in \{1,\ldots,k\}$, almost surely as $p \to \infty$:

$$
\sin^2\angle(h_j,\,\bar{b}_j)
\;\xrightarrow{a.s.}\;
\underbrace{\dfrac{\delta^2}{n\rho_j + \delta^2}}_{\text{out-of-subspace floor}}
\;+\;
\underbrace{\dfrac{n\rho_j}{n\rho_j + \delta^2}\,\sin^2\angle(\hat{w}_j,\,w_j)}_{\text{in-subspace rotation}}
\tag{5}
$$

The left side is the squared sine of the angle between the $j$-th sample principal component $h_j$ and the $j$-th population loading direction $\bar{b}_j$. It is the fundamental measure of how wrong PCA is about factor $j$.

The right side says the error has exactly two independent sources:

1. **The floor** $\delta^2/(n\rho_j + \delta^2) = 1/(1+\mathrm{SNR}_j)$: the fraction of $h_j$ that lies *outside* the population subspace $\mathcal{B}$ entirely — pure noise leakage. It depends only on the signal-to-noise ratio and is irreducible at fixed $n$.

2. **The in-subspace rotation** $\frac{n\rho_j}{n\rho_j+\delta^2}\sin^2\angle(\hat{w}_j,w_j)$: even the part of $h_j$ that *does* land inside $\mathcal{B}$ is rotated away from $\bar{b}_j$ by a finite-$n$ misalignment of sample and population factor-covariance eigenvectors. This term vanishes as $n\to\infty$.

---

## 1. The example

### 1.1 Parameters

We use the diagonal-Gram special case ($G_\infty = I_k$), which simplifies the general formula while preserving both bias terms.

| Symbol | Meaning | Value |
|--------|---------|-------|
| $k$ | number of factors | 3 |
| $p$ | number of assets (growing) | 500 (illustration) |
| $n$ | number of time periods (fixed) | 60 |
| $\delta^2$ | idiosyncratic noise variance | 1.0 |
| $c_1, c_2, c_3$ | prevalences $\lim \Vert \beta_j\Vert^2/p$ | 1.0, 0.8, 0.6 |
| $\sigma_1^2, \sigma_2^2, \sigma_3^2$ | factor return variances | 0.04, 0.02, 0.01 |
| $d_j = c_j\sigma_j^2$ | population spikes | 0.040, 0.016, 0.006 |
| $\mathrm{SNR}_j = nd_j/\delta^2$ | signal-to-noise ratios | 2.40, 0.96, 0.36 |

**Assumption 3 check:** $d_1 = 0.040 > d_2 = 0.016 > d_3 = 0.006$ ✓

![Model parameters](walkthrough_figs/fig_w01_model_setup.png)

The SNR gradient across factors is steep. The graph shows *population* SNR $= nd_j/\delta^2$ (2.40, 0.96, 0.36). In the script the *realized* SNR $= n\rho_j/\delta^2$ is slightly different (2.74, 0.94, 0.41) because $\rho_j$ is the eigenvalue of the finite-$n$ matrix $\hat{D}$, not the population limit $d_j$. Both quantities converge as $n\to\infty$; throughout the rest of this document "SNR" refers to the realized value $n\rho_j/\delta^2$ used in equation (5).

### 1.2 Data model

$$
Y = BF^\top + Z, \quad Y \in \mathbb{R}^{p \times n}
$$

- $B \in \mathbb{R}^{p \times k}$: loading matrix with column $j$ drawn i.i.d. $\mathcal{N}(0, c_j)$, so $\Vert \beta_j\Vert^2/p \approx c_j$ for large $p$ and $G(p) = b(p)^\top b(p) \approx I_k$ (Assumptions 1–2).
- $F \in \mathbb{R}^{n \times k}$: factor returns, column $j$ drawn i.i.d. $\mathcal{N}(0, \sigma_j^2)$.
- $Z \in \mathbb{R}^{p \times n}$: idiosyncratic noise, entries i.i.d. $\mathcal{N}(0, \delta^2)$.

### 1.3 Key matrices

Because $G_\infty = I_k$ we have $Q = I_k$, $\Lambda_G = I_k$, $\Gamma_B = C = \mathrm{diag}(c_1,c_2,c_3)$, and:

$$
\hat{M} = \hat{D} = C^{1/2}\!\left(\frac{F^\top F}{n}\right)C^{1/2}, \qquad
M = D = C^{1/2}\Sigma_F C^{1/2} = \mathrm{diag}(c_j\sigma_j^2)
$$

The population eigenvectors of $M$ are simply $w_j = e_j$ (standard basis vectors), so the in-subspace rotation reduces to $\sin^2\angle(\hat{w}_j, e_j) = 1 - (\hat{w}_j)_j^2$.

**Realized $\hat{D}$ eigenvalues** (from our seed):

| $j$ | $\rho_j$ | $d_j$ (pop. limit) | $\mathrm{SNR}_j = n\rho_j/\delta^2$ |
|-----|----------|-------------------|--------------------------------------|
| 1   | 0.04560  | 0.040             | 2.736 |
| 2   | 0.01561  | 0.016             | 0.937 |
| 3   | 0.00679  | 0.006             | 0.407 |

---

## 2. Step B.3.1 — Parallel/Perpendicular decomposition

**General argument.** Since $\bar{b}_j \in \mathcal{B} = \mathrm{col}(B)$, we can split $h_j$ into its component inside and outside $\mathcal{B}$:

$$
h_j = \underbrace{\Pi_B h_j}_{h_j^\parallel} + \underbrace{(I - \Pi_B)h_j}_{h_j^\perp}
$$

The angle-decomposition identity then gives:

$$
\sin^2\angle(h_j, \bar{b}_j)
= \underbrace{\Vert h_j^\perp\Vert^2}_{\text{out-of-subspace}}
+ \underbrace{\Vert h_j^\parallel\Vert^2 \sin^2\!\angle\!\left(\frac{h_j^\parallel}{\Vert h_j^\parallel\Vert }, \bar{b}_j\right)}_{\text{in-subspace}} \tag{7}
$$

This is the geometric heart of the proof. The total misalignment is the sum of:
- how much of $h_j$ escapes $\mathcal{B}$ entirely ($\Vert h_j^\perp\Vert^2$, which becomes the floor), and
- how badly the in-$\mathcal{B}$ part of $h_j$ points toward $\bar{b}_j$ (which becomes the rotation term).

**In our example** (at $p = 500$):

| $j$ | $\Vert h_j^\perp\Vert^2$ (obs., $p=500$) | floor (predicted, $p\to\infty$) | gap |
|-----|------------------------------------|---------------------------------|-----|
| 1   | 0.3051                             | 0.2677                          | +0.037 |
| 2   | 0.5580                             | 0.5164                          | +0.042 |
| 3   | 0.7949                             | 0.7106                          | +0.084 |

The observed perpendicular norms sit above the floor at $p=500$, as expected: the floor is the $p\to\infty$ limit and the gap closes as $p$ grows. For factor 3, with realized SNR $= n\rho_3/\delta^2 \approx 0.41$, already more than 79% of $h_3$'s squared norm lies outside the signal subspace at $p=500$, converging down to the floor of 71% as $p\to\infty$.

![Angle decomposition](walkthrough_figs/fig_w07_angle_decomp.png)

*Each panel shows $h_j$ (colored arrow) and $\bar{b}_j$ (gray arrow) as unit vectors. The dashed horizontal line marks the out-of-subspace component $\Vert h_j^\perp\Vert $. The arc labels the total angle. As SNR falls from left to right, both the total angle and the out-of-subspace fraction grow.*

---

## 3. Step B.3.2 — Expansion of $W^{(p)}$ and its limit

**General argument.** The key analytical move is to replace the $p \times p$ matrix $YY^\top/n$ (whose spectrum diverges as $p \to \infty$) with its $n \times n$ dual:

$$
W^{(p)} = \frac{Y^\top Y}{np} \in \mathbb{R}^{n \times n}
$$

The two share the same nonzero eigenvalues up to the factor $p$, and eigenvectors are related via $h_j = Y\chi_{p,j}/(\sqrt{n}\, s_{p,j})$, so all of the PCA geometry of $h_j$ is recoverable from $W^{(p)}$.

Substituting $Y = BF^\top + Z$:

$$
W^{(p)} =
\underbrace{\frac{F(B^\top B/p)F^\top}{n}}_{\text{(A) signal}}
+
\underbrace{\frac{FB^\top Z + Z^\top BF^\top}{np}}_{\text{(B) cross}}
+
\underbrace{\frac{Z^\top Z}{np}}_{\text{(C) noise}}
\tag{8}
$$

- **Term (A):** $B^\top B/p = \Gamma_p \to \Gamma_B = C$ (Assumptions 1–2). So $(A) \to F C F^\top/n$.
- **Term (B):** Vanishes a.s. by Corollary 1.1 (noise–signal cross-terms are $o(1)$ because each unit loading column $b_j(p)$ is deterministic and Lemma 1 controls $\Vert b_j^\top Z\Vert $).
- **Term (C):** $\to (\delta^2/n)I_n$ a.s. by Lemma 4 (law of large numbers on the noise Gram).

Therefore:

$$
W^{(p)} \xrightarrow{a.s.} W_\infty := \frac{F\,\Gamma_B\,F^\top}{n} + \frac{\delta^2}{n}I_n
\tag{9}
$$

**In our example.** With $\Gamma_B = C = \mathrm{diag}(1.0, 0.8, 0.6)$:

$$
W_\infty = \frac{F C F^\top}{60} + \frac{1.0}{60}\,I_{60}
\quad \in \mathbb{R}^{60 \times 60}
$$

At $p = 500$ the operator-norm error is:

$$
\Vert W^{(500)} - W_\infty\Vert_\mathrm{op} = 0.0146
$$

This decays at rate $\approx p^{-1/2}$:

![W(p) convergence](walkthrough_figs/fig_w02_Wp_convergence.png)

*The operator norm error follows the $p^{-1/2}$ reference line closely, confirming the a.s. convergence with the expected rate.*

---

## 4. Step B.3.3 — Eigenstructure of $W_\infty$ (Lemma 7)

**General argument.** Write $\Gamma_B = PP^\top$ with $P = C^{1/2}Q\Lambda_G^{1/2}$ (in our case $P = C^{1/2}$). Then $W_\infty - (\delta^2/n)I_n = (FP)(FP)^\top/n$ is a rank-$k$ matrix. By the AB/BA identity its nonzero eigenvalues equal those of

$$
\frac{(FP)^\top(FP)}{n} = P^\top\!\left(\frac{F^\top F}{n}\right)P = \hat{M}
$$

So the top-$k$ eigenvalues of $W_\infty$ are $\tau_j = \rho_j + \delta^2/n$, where $\rho_j$ are the eigenvalues of $\hat{M}$. The remaining $n-k$ eigenvalues all equal $\delta^2/n$ (the noise floor). The top-$k$ eigenvectors are:

$$
v_j = \frac{F\,C^{1/2}\hat{w}_j}{\sqrt{n\rho_j}}
\quad \text{(formula (10) with $G_\infty = I_k$)}
$$

**In our example.**

| $j$ | $\rho_j$ | $\tau_j = \rho_j + \delta^2/n$ | Lemma 7 prediction | Actual $\lambda_j(W_\infty)$ | Match |
|-----|----------|--------------------------------|--------------------|-------------------------------|-------|
| 1   | 0.04560  | 0.06226 | 0.06226 | 0.06226 | ✓ |
| 2   | 0.01561  | 0.03228 | 0.03228 | 0.03228 | ✓ |
| 3   | 0.00679  | 0.02345 | 0.02345 | 0.02345 | ✓ |

The remaining 57 eigenvalues of $W_\infty$ all equal $\delta^2/n = 1/60 \approx 0.01667$.

The formula (10) eigenvectors $v_j$ match the directly computed eigenvectors of $W_\infty$ to machine precision ($|\cos\angle(v_j, \text{evec}_j)| = 1.000000000$).

![Eigenspectrum of W_inf](walkthrough_figs/fig_w03_eigenspectrum.png)

*The three signal eigenvalues (blue dots) sit clearly above the noise floor of 57 eigenvalues all at $\delta^2/n \approx 0.0167$ (gray). The gap between $\tau_3$ and the noise floor equals $\rho_3 = 0.00679$ — this is the spectral gap that Assumption 3 guarantees and that Kato's eigenprojection continuity theorem requires.*

---

## 5. Step B.3.4 — Spectral convergence

**General argument.** Since $W^{(p)} \to W_\infty$ in operator norm on the fixed $n \times n$ space, and the top-$k$ eigenvalues of $W_\infty$ are simple (Assumption 3) and separated from the noise level by gap $\rho_k > 0$, Kato's eigenprojection continuity theorem (§II.1) gives:

$$
\frac{s_{p,j}^2}{p} \to \tau_j \quad \text{and} \quad \chi_{p,j} \to v_j \quad \text{a.s., up to sign.} \tag{11}
$$

Here $s_{p,j}$ are the top-$k$ singular values of $Y/\sqrt{n}$ and $\chi_{p,j}$ are the corresponding right singular vectors — the eigenvectors of $W^{(p)}$.

The critical point is that this convergence holds for the **full sequence** in $p$, not just along subsequences. This is what the $\Gamma_B$-coordinate framework (introduced in B.3.5) ensures by eliminating the rotational ambiguity of the SVD basis.

**In our example** (at $p = 500$):

| $j$ | $s_{p,j}^2/p$ | $\tau_j$ | $\vert\cos\angle(\chi_{p,j}, v_j)\vert$ |
|-----|---------------|----------|---------------------------------|
| 1   | 0.06289       | 0.06226  | 0.963 |
| 2   | 0.03742       | 0.03228  | 0.899 |
| 3   | 0.02895       | 0.02345  | 0.703 |

The cosines are furthest from 1 for factor 3, which has the smallest spectral gap ($\rho_3 = 0.00679$). With a small gap, $W^{(p)}$ needs to be much closer to $W_\infty$ before the eigenvectors are stable — which requires larger $p$.

![Eigenvector alignment vs p](walkthrough_figs/fig_w04_eigvec_alignment.png)

*All three factors converge to $|\cos\angle| = 1$, but factor 3 (coral, lowest SNR) requires substantially larger $p$ to approach the limit. The vertical dotted line marks $p = 500$, our illustration point.*

---

## 6. Step B.3.5 — The $\Gamma_B$-coordinate framework

**General argument.** Define the loading map $\Phi_p : \mathbb{R}^k \to \mathcal{B}$ by

$$
\Phi_p(x) = \frac{Bx}{\sqrt{p}}
$$

so that $\Phi_p^\top \Phi_p = B^\top B/p = \Gamma_p \to \Gamma_B$. Any vector in $\mathcal{B}$ can be written as $\Phi_p(x)$ for a unique $x \in \mathbb{R}^k$, and the $\Gamma_B$-inner product $x^\top \Gamma_B y$ on $\mathbb{R}^k$ corresponds to the Euclidean inner product in $\mathbb{R}^p$. This coordinate system is the key device that lets us track convergence without ever referring to the SVD of $b(p)$.

**Population direction.** The population loading direction satisfies $\bar{b}_j = \Phi_p(a_j^{(p)})$ where $a_j^{(p)} \to a_j^\infty$ a.s. In the $G_\infty = I_k$ case:

$$
a_j^\infty = C^{-1/2}e_j = \frac{e_j}{\sqrt{c_j}}
$$

**Sample direction.** The in-subspace component of $h_j$ satisfies $\Pi_B h_j = \Phi_p(g_j^{(p)}) + o(1)$ a.s. where, using $\chi_{p,j} \to v_j$ from B.3.4 and $s_{p,j}^2/p \to \tau_j$:

$$
g_j^{(p)} \to g_j^\infty := \frac{F^\top v_j}{\sqrt{n\rho_j + \delta^2}}
$$

Substituting formula (10) for $v_j$ and the key identity $(*)$ from Lemma 7 gives the explicit form:

$$
g_j^\infty = \sqrt{\frac{n\rho_j}{n\rho_j + \delta^2}} \cdot C^{-1/2}\hat{w}_j \tag{12}
$$

**In our example.** The $\Gamma_B$ coordinates $g_j^\infty$ and $a_j^\infty$ (vectors in $\mathbb{R}^3$) are:

| Coordinate | $j=1$ | $j=2$ | $j=3$ |
|---|---|---|---|
| $g_j^\infty$ | $(-0.852,\,-0.059,\,+0.085)$ | $(-0.053,\,+0.759,\,-0.184)$ | $(-0.033,\,-0.127,\,-0.678)$ |
| $a_j^\infty$ | $(1.000,\,0,\,0)$ | $(0,\,1.118,\,0)$ | $(0,\,0,\,1.291)$ |

Notice that $g_j^\infty$ is *nearly* parallel to $a_j^\infty$ for each $j$ — the dominant component of $g_1^\infty$ is in the first coordinate, of $g_2^\infty$ in the second, etc. The small off-diagonal entries of $g_j^\infty$ are the in-$\mathcal{B}$ rotation from the fact that $\hat{D}$ is not exactly diagonal (correlated factor returns at finite $n$).

The scaling factor $\sqrt{n\rho_j/(n\rho_j+\delta^2)}$ in equation (12) is exactly $\sqrt{1-\text{floor}_j}$: the in-subspace component of $h_j$ is already "shrunk" relative to the target, with the shrinkage determined by the SNR.

---

## 7. Step B.3.6 — Floor and in-subspace angle via $\Gamma_B$ inner products

With $g_j^{(p)} \to g_j^\infty$, $a_j^{(p)} \to a_j^\infty$, and $\Gamma_p \to \Gamma_B$, the lengths and inner products in $\mathbb{R}^p$ converge to $\Gamma_B$-inner products in $\mathbb{R}^k$.

### 7.1 The floor (out-of-subspace norm)

$$
\Vert \Pi_B h_j\Vert^2 = (g_j^{(p)})^\top \Gamma_p\, g_j^{(p)} + o(1)
\;\longrightarrow\; (g_j^\infty)^\top \Gamma_B\, g_j^\infty
$$

Substituting (12) and using $\Lambda_G^{-1/2}Q^\top C^{-1/2} \cdot C^{1/2}Q\Lambda_G Q^\top C^{1/2} \cdot C^{-1/2}Q\Lambda_G^{-1/2} = I_k$ (in our case this is just $C^{-1/2} C\, C^{-1/2} = I$):

$$
(g_j^\infty)^\top \Gamma_B\, g_j^\infty
= \frac{n\rho_j}{n\rho_j+\delta^2}\,\hat{w}_j^\top\hat{w}_j
= \frac{n\rho_j}{n\rho_j+\delta^2}
$$

Since $\Vert h_j\Vert  = 1$:

$$
\Vert h_j^\perp\Vert^2 = 1 - \Vert \Pi_B h_j\Vert^2 \;\longrightarrow\; \frac{\delta^2}{n\rho_j+\delta^2} \tag{13}
$$

This is the **floor** term of equation (5).

### 7.2 The in-subspace inner product

$$
\langle \Pi_B h_j, \bar{b}_j\rangle = (g_j^{(p)})^\top\Gamma_p\, a_j^{(p)} + o(1)
\;\longrightarrow\; (g_j^\infty)^\top \Gamma_B\, a_j^\infty
$$

The same simplification gives:

$$
(g_j^\infty)^\top \Gamma_B\, a_j^\infty
= \sqrt{\frac{n\rho_j}{n\rho_j+\delta^2}}\;\hat{w}_j^\top w_j
$$

(In our $G_\infty = I_k$ case: $w_j = e_j$, so $\hat{w}_j^\top w_j = (\hat{w}_j)_j$, the $j$-th diagonal entry of $\hat{W}$.)

### 7.3 The in-subspace angle

Combining the norm and inner product:

$$
\sin^2\!\angle\!\left(\frac{\Pi_B h_j}{\Vert \Pi_B h_j\Vert }, \bar{b}_j\right)
= 1 - \frac{\langle\Pi_B h_j,\bar{b}_j\rangle^2}{\Vert \Pi_B h_j\Vert^2}
\;\longrightarrow\;
1 - \frac{(n\rho_j/(n\rho_j+\delta^2))\,(\hat{w}_j^\top w_j)^2}{n\rho_j/(n\rho_j+\delta^2)}
= \sin^2\!\angle(\hat{w}_j, w_j)
$$

**In our example.** Checking the $\Gamma_B$ inner products numerically:

| $j$ | $(g_j^\infty)^\top\Gamma_B g_j^\infty$ | $n\rho_j/(n\rho_j+\delta^2)$ | Match |
|-----|----------------------------------------|-------------------------------|-------|
| 1   | 0.73232 | 0.73232 | ✓ |
| 2   | 0.48361 | 0.48361 | ✓ |
| 3   | 0.28941 | 0.28941 | ✓ |

The floors derived from the $\Gamma_B$ framework:

| $j$ | Floor $= \delta^2/(n\rho_j+\delta^2)$ | $\sin^2\angle(\hat{w}_j, e_j)$ = rotation |
|-----|-----------------------------------------|-------------------------------------------|
| 1   | 0.2677 | 0.0097 |
| 2   | 0.5164 | 0.0480 |
| 3   | 0.7106 | 0.0481 |

Note that the rotation is small for all three factors because with $G_\infty = I_k$ and $n=60$ i.i.d. factor draws, $\hat{D}$ is close to diagonal and $\hat{w}_j \approx e_j$. The floor is the dominant bias, especially for factors 2 and 3.

---

## 8. Step B.3.7 — Assembly

Substituting (13) and the in-subspace angle from B.3.6 into decomposition (7):

$$
\sin^2\angle(h_j, \bar{b}_j)
= \Vert h_j^\perp\Vert^2 + \Vert \Pi_B h_j\Vert^2\,\sin^2\!\angle\!\left(\frac{\Pi_B h_j}{\Vert \Pi_B h_j\Vert }, \bar{b}_j\right)
$$

$$
\longrightarrow\; \frac{\delta^2}{n\rho_j+\delta^2} + \frac{n\rho_j}{n\rho_j+\delta^2}\,\sin^2\angle(\hat{w}_j, w_j)
\quad \text{a.s.}
$$

This is equation (5). All limits hold for the **full sequence** in $p$ — no subsequence extraction is needed, because the $\Gamma_B$-coordinate framework operates directly on $\Gamma_p \to \Gamma_B$ and bypasses the rotational ambiguity of the SVD basis $U(p)$.

**In our example.** The assembled RHS and the observed LHS at $p=500$:

| $j$ | Floor | Weight $\times$ Rotation | RHS (predicted) | LHS (observed, $p=500$) | Gap |
|-----|-------|--------------------------|-----------------|--------------------------|-----|
| 1   | 0.2677 | $0.7323 \times 0.0097 = 0.0071$ | **0.2748** | 0.3099 | +0.035 |
| 2   | 0.5164 | $0.4836 \times 0.0480 = 0.0232$ | **0.5396** | 0.5697 | +0.030 |
| 3   | 0.7106 | $0.2894 \times 0.0481 = 0.0139$ | **0.7245** | 0.7975 | +0.073 |

The gaps are positive and largest for factor 3 (lowest SNR, slowest convergence in $p$). They shrink to zero as $p \to \infty$.

![Floor and rotation assembly](walkthrough_figs/fig_w05_floor_rotation.png)

*The stacked bars show the floor (coral) and weighted rotation (purple) contributions. The floor dominates for factors 2 and 3. The rotation contribution is small because $G_\infty = I_k$ forces $w_j = e_j$ so only finite-$n$ fluctuations in $\hat{D}$ create in-subspace misalignment.*

---

## 9. Verification: LHS converges to RHS as $p \to \infty$

The theorem's claim is an asymptotic result. The gap at any finite $p$ is real noise from the finite-sample approximations at steps B.3.2 and B.3.4. As $p$ grows the operator norm $\Vert W^{(p)} - W_\infty\Vert_\mathrm{op} \to 0$ and the eigenvector cosines $|\cos\angle(\chi_{p,j}, v_j)| \to 1$, both pulling the gap to zero.

![LHS vs RHS convergence](walkthrough_figs/fig_w06_lhs_vs_rhs.png)

*For each factor, the observed LHS (colored circles) approaches the predicted RHS (gray squares) as $p$ increases. Factor 3 converges most slowly, consistent with its smaller spectral gap and the weaker eigenvector alignment seen in B.3.4.*

![Gap vs p](walkthrough_figs/fig_w08_gap_vs_p.png)

*The absolute gap $|\mathrm{LHS} - \mathrm{RHS}|$ decays roughly as $p^{-1/2}$ (dashed reference line), matching the rate of $\Vert W^{(p)} - W_\infty\Vert_\mathrm{op}$ from B.3.2.*

---

## 10. Corollary 4: Grassmannian distance

When measuring the error at the level of the whole subspace rather than per eigenvector, the in-subspace rotation drops out entirely. Corollary 4 states:

$$
d_{\mathrm{Gr}}^2(\mathrm{col}(H), \mathcal{B})
= \sum_{j=1}^k \Vert h_j^\perp\Vert^2
\;\longrightarrow\;
\sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2}
$$

**In our example.**

$$
\sum_j \frac{\delta^2}{n\rho_j+\delta^2} = 0.2677 + 0.5164 + 0.7106 = 1.4947
$$

Observed at $p=500$: $\sum_j \Vert h_j^\perp\Vert^2 \approx 1.658$. Gap: +0.163, converging to zero as $p \to \infty$.

The practical implication: if you only care about whether the *subspace* $\mathrm{col}(H)$ aligns with $\mathcal{B}$ (as in PCA-based factor estimation), the total error is $\sum_j 1/(1+\mathrm{SNR}_j)$ — determined entirely by the signal-to-noise ratios, with no rotation contribution.

---

## 11. Summary of the proof logic

| Step | What it does | Key tool |
|------|-------------|----------|
| B.3.1 | Splits $\sin^2\angle$ into floor + in-subspace terms | Pythagoras in $\mathbb{R}^p$ |
| B.3.2 | Replaces $p\times p$ problem with $n\times n$ limit $W_\infty$ | LLN on noise; loading Gram convergence |
| B.3.3 | Reads off eigenvalues and eigenvectors of $W_\infty$ | AB/BA identity |
| B.3.4 | Transfers convergence from $W_\infty$ to $W^{(p)}$ eigenvectors | Kato eigenprojection continuity |
| B.3.5 | Expresses everything in $\Gamma_B$ coordinates | $\Phi_p$ loading map; full-sequence convergence |
| B.3.6 | Evaluates the two inner products as $\Gamma_B$ bilinear forms | Algebraic cancellation |
| B.3.7 | Assembles the two terms of equation (5) | Substitution into (7) |

The architectural insight is B.3.5: by working in the loading map's coordinate system rather than the SVD basis of $b(p)$, the proof avoids the rotational ambiguity that previously required subsequence extraction when $G_\infty = I_k$ made singular values of $b(p)$ coalesce.

---

*Generated from `proof_walkthrough_k3.py` and `proof_walkthrough_figures.py`. Seed: 20260522. Parameters: k=3, p=500, n=60, c=[1.0,0.8,0.6], σ²=[0.04,0.02,0.01], δ²=1.0.*
