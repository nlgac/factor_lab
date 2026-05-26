# Multifactor Eigenvector Alignment: Part (ii) — Statement and Proof

> **Notational conventions.** Superscript $(p)$ denotes a quantity that depends on $p$ and converges as $p\to\infty$; no superscript denotes the limit. Hat $\hat{\cdot}$ denotes a finite-$n$ sample quantity converging as $n\to\infty$; no hat denotes the population limit. Subscripts $B$ and $b$ identify the matrix whose columns form the Gram.

---

## Data Model

The $p \times n$ data matrix is

$$
Y = BF + Z,
$$

where $B \in \mathbb{R}^{p \times k}$ is a deterministic loading matrix; $F \in \mathbb{R}^{k \times n}$ is the factor-return matrix, conditioned on throughout; and $Z \in \mathbb{R}^{p \times n}$ is a noise matrix whose entries are mean-zero, mutually independent within each column, with common variance $\delta^2 > 0$ and uniformly bounded $(2+\eta)$-th moments for some $\eta > 0$. The asymptotic regime is $p \to \infty$ with $n$ and $k$ fixed.

*(Convention note: $F \in \mathbb{R}^{k \times n}$ throughout this document, so the factor Gram is $FF^\top/n \in \mathbb{R}^{k\times k}$. The dual matrix $W^{(p)} = Y^\top Y/(np) \in \mathbb{R}^{n\times n}$ is analysed via $F^\top G_B F/n$. In the proof walkthrough document the convention is $F \in \mathbb{R}^{n\times k}$; both give the same Gram $G^{(n)}_F$.)*

---

## Notation

**Sample eigenvectors.** Let $h_1, \ldots, h_k \in \mathbb{R}^p$ be the orthonormal eigenvectors of $YY^\top/n$ corresponding to its $k$ largest eigenvalues, in decreasing order.

**Population loading directions.** Let $b_j \in \mathbb{R}^p$ be the $j$-th eigenvector (by decreasing eigenvalue) of the population signal covariance

$$
\Sigma^{(p)}_0 = \frac{B\Sigma_F B^\top}{p},
$$

where $\Sigma_F := \lim_{n\to\infty} FF^\top/n$ is the population factor-return covariance.

**The matrices $\hat{M}$ and $M$.** The loading Gram $B^\top B/p \in \mathbb{R}^{k\times k}$ is assumed to converge (Assumption 1 below) to a positive definite limit $G_B$. Let $G_B^{1/2}$ denote the symmetric positive definite square root of $G_B$. Define

$$
\hat{M} = G_B^{1/2}\,\frac{FF^\top}{n}\,G_B^{1/2} \in \mathbb{R}^{k\times k}, \tag{1}
$$

and set

$$
M := \lim_{n\to\infty} \hat{M} = G_B^{1/2}\,\Sigma_F\,G_B^{1/2}. \tag{2}
$$

Both $\hat{M}$ and $M$ are symmetric positive definite. Write $\hat\lambda_1 > \cdots > \hat\lambda_k > 0$ for the eigenvalues of $\hat{M}$, with corresponding orthonormal eigenvectors $\hat{w}_1, \ldots, \hat{w}_k$; write $\lambda_1 > \cdots > \lambda_k > 0$ and $w_1, \ldots, w_k$ for the eigenvalues and orthonormal eigenvectors of $M$. The *signal-to-noise ratio* of factor $j$ is $\widehat{\mathrm{SNR}}_j := n\hat\lambda_j/\delta^2$.

---

## Assumptions

**Assumption 1** (Gram convergence). *As $p \to \infty$,*

$$
\frac{B^\top B}{p} \longrightarrow G_B \quad \text{for some positive definite } G_B \in \mathbb{R}^{k \times k}.
$$

**Assumption 2** (Spectral separation). *$M$ has $k$ strictly ordered positive eigenvalues $\lambda_1 > \lambda_2 > \cdots > \lambda_k > 0$.*

The *regular-event hypothesis* is that $\hat{M}$ also has $k$ distinct positive eigenvalues. Under any law of $F$ with density absolutely continuous on $\mathbb{R}^{k \times n}$, this holds almost surely, because the discriminant of the characteristic polynomial of $\hat{M}$ is a non-trivial polynomial in the entries of $FF^\top/n$, hence nonzero a.s.

---

## Theorem 1: Per-Direction Alignment (Part (ii))

**Theorem 1.** *Under Assumptions 1–2, the regular-event hypothesis, and the stated noise conditions, conditional on $F$ and almost surely as $p \to \infty$: for each $j \in \{1, \ldots, k\}$,*

$$
\sin^2\angle(h_j,\, b_j)
\;\xrightarrow{a.s.}\;
\underbrace{\dfrac{\delta^2}{n\hat\lambda_j + \delta^2}}_{\text{out-of-subspace floor}}
\;+\;
\underbrace{\dfrac{n\hat\lambda_j}{n\hat\lambda_j + \delta^2}\,\sin^2\angle(\hat{w}_j,\, w_j)}_{\text{in-subspace rotation}}.
\tag{3}
$$

**Remark (Structure of the two terms).** The two weights $\delta^2/(n\hat\lambda_j+\delta^2)$ and $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$ sum to one and equal $1/(1+\widehat{\mathrm{SNR}}_j)$ and $\widehat{\mathrm{SNR}}_j/(1+\widehat{\mathrm{SNR}}_j)$, respectively.

The floor $\delta^2/(n\hat\lambda_j + \delta^2)$ is irreducible: it is determined entirely by $\widehat{\mathrm{SNR}}_j$ and cannot be reduced by increasing $p$. It vanishes only as $n \to \infty$ or $\delta^2 \to 0$.

The rotation $\sin^2\angle(\hat{w}_j, w_j)$ measures how far the finite-sample eigenvectors of $\hat{M}$ have rotated from those of its population limit $M$. Since $FF^\top/n \to \Sigma_F$ as $n \to \infty$, we have $\hat{M} \to M$, so this term vanishes in the large-$n$ limit. In the noiseless limit $\delta^2 \to 0$, however, the rotation term survives with weight 1.

**Remark (Special cases).** When $G_b := \lim_{p\to\infty} b(p)^\top b(p) = I_k$ (orthonormal limiting unit-loading columns), $G_B = C := \mathrm{diag}(c_1,\ldots,c_k)$ where $c_j = \lim\Vert \beta_j\Vert^2/p$, giving $\hat{M} = C^{1/2}(FF^\top/n)C^{1/2}$ and $M = C^{1/2}\Sigma_F C^{1/2}$.

When additionally $F$ has orthogonal rows, $FF^\top/n$ is diagonal, so $\hat{M}$ is diagonal, $\hat{w}_j = w_j = e_j$, the rotation term vanishes, and the formula reduces to $\sin^2\angle(h_j, b_j) \to \delta^2/(n\hat\lambda_j + \delta^2)$.

When $k = 1$, the rotation term is trivially zero and the formula gives $\sin^2\angle(h, b) \to \delta^2/(c\Vert X\Vert^2 + \delta^2)$, recovering the GPS2022 result.

---

## Proof of Theorem 1

The proof proceeds in seven steps. We first collect the three auxiliary results the argument depends on.

### Auxiliary Results

#### Lemma 1 (Noise–signal orthogonality) — cited

*For any deterministic sequence $\eta_p \in \mathbb{R}^p$ with $\Vert \eta_p\Vert  \le C$, and each noise column $Z_{\cdot\ell}$,*

$$
\frac{\eta_p^\top Z_{\cdot\ell}}{\sqrt{p}} \xrightarrow{a.s.} 0.
$$

*Proof sketch.* The summands $(\eta_p)_i (Z_{\cdot\ell})_i$ are mean-zero and independent with uniformly bounded $(2+\eta)$-th moments. Borel–Cantelli applied to fourth moments gives $\sum_p \mathbf{E}[(\eta_p^\top Z_{\cdot\ell})^4]/p^2 < \infty$, which implies the a.s. limit. Applied column-by-column to the normalized loading vectors $\beta_j/\Vert \beta_j\Vert $, this gives:

$$
\frac{\Vert \beta_j^\top Z\Vert_F^2}{p} \xrightarrow{a.s.} 0, \tag{L1}
$$

which kills the cross-term in Step 2.

#### Lemma 4 (Noise Gram concentration) — cited

*As $p \to \infty$,*

$$
\frac{Z^\top Z}{p} \xrightarrow{a.s.} \delta^2 I_n \quad \text{in spectral norm.}
$$

*Proof sketch.* Since $n$ is fixed, it suffices to show entrywise convergence of the $n\times n$ matrix. Each diagonal entry is $\Vert Z_{\cdot\ell}\Vert^2/p \to \delta^2$ by the strong law of large numbers, and each off-diagonal entry $Z_{\cdot\ell}^\top Z_{\cdot m}/p \to 0$ by Lemma 1 (with $\eta_p = Z_{\cdot m}/\Vert Z_{\cdot m}\Vert $, or directly by the same Borel–Cantelli argument). Entrywise convergence on a fixed-size matrix implies spectral convergence.

#### Lemma 7 (Eigenstructure of $W$) — stated and proved

Define the $n\times n$ limit matrix

$$
W := \frac{F^\top G_B F}{n} + \frac{\delta^2}{n}I_n. \tag{4}
$$

**Lemma 7.** *$W$ has:*

*(a) Top-$k$ eigenvalues $\hat\lambda_j + \delta^2/n$, for $j = 1,\ldots,k$, where $\hat\lambda_1 > \cdots > \hat\lambda_k > 0$ are the eigenvalues of $\hat{M}$.*

*(b) Remaining $n - k$ eigenvalues all equal to $\delta^2/n$.*

*(c) Top-$k$ eigenvectors*

$$
v_j = \frac{F^\top G_B^{1/2}\hat{w}_j}{\sqrt{n\hat\lambda_j}}, \tag{5}
$$

*where $\hat{w}_j$ is the $j$-th eigenvector of $\hat{M}$.*

**Proof.** Write

$$
W - \frac{\delta^2}{n}I_n = \frac{F^\top G_B F}{n} = \frac{(G_B^{1/2}F)^\top (G_B^{1/2}F)}{n}.
$$

Let $A = G_B^{1/2}F \in \mathbb{R}^{k\times n}$. Then $W - (\delta^2/n)I_n = A^\top A/n$, which has rank at most $k$. The nonzero eigenvalues of $A^\top A/n$ equal those of $AA^\top/n$ by the **AB/BA identity**: if $A^\top A u = \mu u$ then $AA^\top(Au) = A(A^\top A u) = \mu(Au)$, so $Au$ is an eigenvector of $AA^\top$ with the same eigenvalue. But

$$
\frac{AA^\top}{n} = \frac{G_B^{1/2} FF^\top G_B^{1/2}}{n} = G_B^{1/2}\,\frac{FF^\top}{n}\,G_B^{1/2} = \hat{M},
$$

whose eigenvalues are $\hat\lambda_1 > \cdots > \hat\lambda_k$. This proves (a). Since $A^\top A/n$ is $n\times n$ with rank $k$, the remaining $n-k$ eigenvalues are zero, giving eigenvalues of $W$ equal to $\delta^2/n$. This proves (b).

For (c): given $\hat{M}\hat{w}_j = \hat\lambda_j\hat{w}_j$, set $v_j = A^\top\hat{w}_j/\sqrt{n\hat\lambda_j} = F^\top G_B^{1/2}\hat{w}_j/\sqrt{n\hat\lambda_j}$. Then

$$
\frac{A^\top A}{n}\, v_j = \frac{A^\top(A\hat{w}_j/\sqrt{n\hat\lambda_j})}{n} = \frac{A^\top(G_B^{1/2}FF^\top/n \cdot G_B^{1/2}\hat{w}_j)/\sqrt{\hat\lambda_j}}{\sqrt{n}}.
$$

Using $AA^\top\hat{w}_j = n\hat{M}\hat{w}_j = n\hat\lambda_j\hat{w}_j$:

$$
\frac{A^\top A}{n}\,v_j = \frac{A^\top(\hat\lambda_j\hat{w}_j)}{\sqrt{n\hat\lambda_j}} = \hat\lambda_j \cdot \frac{A^\top\hat{w}_j}{\sqrt{n\hat\lambda_j}} = \hat\lambda_j v_j,
$$

confirming $v_j$ is an eigenvector of $A^\top A/n$ with eigenvalue $\hat\lambda_j$, hence an eigenvector of $W$ with eigenvalue $\hat\lambda_j + \delta^2/n$. Unit-norm check: $\Vert v_j\Vert^2 = \hat{w}_j^\top G_B^{1/2}(FF^\top/n)G_B^{1/2}\hat{w}_j/(n\hat\lambda_j) = \hat{w}_j^\top\hat{M}\hat{w}_j/\hat\lambda_j = \hat\lambda_j/\hat\lambda_j = 1$. $\square$

**Key identity.** Left-multiplying $\hat{M}\hat{w}_j = \hat\lambda_j\hat{w}_j$ by $G_B^{-1/2}$:

$$
\frac{FF^\top}{n}\, G_B^{1/2}\hat{w}_j = \hat\lambda_j\, G_B^{-1/2}\hat{w}_j. \tag{$*$}
$$

This is used in Step 5 to compute the limit of $g_j^{(p)}$ explicitly.

---

> **k=3 example — Lemma 7 verification** *(parameters: $k=3$, $p=500$, $n=60$, $G_B = C = \mathrm{diag}(1.0,\,0.8,\,0.6)$, $\delta^2 = 1.0$, seed 20260522)*
>
> | $j$ | $\hat\lambda_j$ | $\hat\lambda_j + \delta^2/n$ | Predicted $\lambda_j(W)$ | Actual $\lambda_j(W)$ |
> |:---:|:---------------:|:-----------------------------:|:------------------------:|:---------------------:|
> | 1 | 0.04560 | 0.06226 | 0.06226 | 0.06226 ✓ |
> | 2 | 0.01561 | 0.03228 | 0.03228 | 0.03228 ✓ |
> | 3 | 0.00679 | 0.02345 | 0.02345 | 0.02345 ✓ |
>
> The remaining 57 eigenvalues of $W$ all equal $\delta^2/n = 1/60 \approx 0.0167$.
> The spectral gap above the noise floor equals $\hat\lambda_3 = 0.00679$ — the smallest gap, and the reason factor 3's eigenvectors converge most slowly in $p$.

---

### Step 1 — Parallel/Perpendicular Decomposition

**Intuition.** We want $\sin^2\angle(h_j, b_j)$. Since $b_j \in \mathcal{B} := \mathrm{col}(B)$ and $h_j$ is a unit vector in $\mathbb{R}^p$, split $h_j$ into its component inside and outside the signal subspace:

$$
h_j = \underbrace{\Pi_B h_j}_{h_j^\parallel} + \underbrace{(I-\Pi_B)h_j}_{h_j^\perp}.
$$

Because $b_j \in \mathcal{B}$ and $h_j^\perp \perp \mathcal{B}$, the Pythagorean identity gives:

$$
\sin^2\angle(h_j, b_j) = \Vert h_j^\perp\Vert^2 + \Vert h_j^\parallel\Vert^2\,\sin^2\angle\!\left(\frac{h_j^\parallel}{\Vert h_j^\parallel\Vert },\, b_j\right). \tag{6}
$$

The proof evaluates each term on the right as $p \to \infty$. The first becomes the floor; the second becomes the weighted rotation.

---

> **k=3 example — Step 1** *(at $p=500$)*
>
> | $j$ | $\Vert h_j^\perp\Vert^2$ observed | Floor $= \delta^2/(n\hat\lambda_j+\delta^2)$ | Gap |
> |:---:|:---------------------------:|:--------------------------------------------:|:---:|
> | 1 | 0.3051 | 0.2677 | +0.037 |
> | 2 | 0.5580 | 0.5164 | +0.042 |
> | 3 | 0.7949 | 0.7106 | +0.084 |
>
> At $p=500$, factor 3 already has 79% of its squared norm outside $\mathcal{B}$, converging down to 71% as $p\to\infty$. The gaps are largest for factor 3 (smallest $\widehat{\mathrm{SNR}}_3 = 0.41$) and shrink at rate $O(p^{-1/2})$.

---

### Step 2 — The $n\times n$ Dual Matrix and Its Limit $W$

**Intuition.** The natural matrix to analyze is $YY^\top/n \in \mathbb{R}^{p\times p}$, but its spectrum diverges as $p\to\infty$. The key is to pass to the $n\times n$ **dual matrix**

$$
W^{(p)} = \frac{Y^\top Y}{np} \in \mathbb{R}^{n\times n},
$$

which shares the same nonzero eigenvalues (scaled by $p$) and whose eigenvectors $v^{(p)}_j$ are related to the $h_j$ by $h_j = Yv^{(p)}_j/(\sqrt{n}\,s^{(p)}_j)$, where $s^{(p)}_j$ is the $j$-th singular value of $Y/\sqrt{n}$. Crucially, $W^{(p)}$ is $n\times n$ regardless of $p$: the law of large numbers acts entrywise as $p\to\infty$ and drives it to a fixed limit.

**Limit computation.** Substituting $Y = BF + Z$:

$$
W^{(p)} = \underbrace{\frac{F(B^\top B/p)F^\top}{n}}_{\text{signal} \;\to\; F^\top G_B F/n} + \underbrace{\frac{FB^\top Z + Z^\top BF^\top}{np}}_{\text{cross} \;\to\; 0 \text{ (Lemma 1)}} + \underbrace{\frac{Z^\top Z}{np}}_{\text{noise} \;\to\; (\delta^2/n)I_n \text{ (Lemma 4)}}. \tag{7}
$$

*(Note: the cross term has entries of the form $F_{\cdot j}^\top(B_{\cdot j}^\top Z)/\sqrt{np}$; Lemma 1 applied to each normalized loading column makes this vanish a.s.)* The signal term converges because $B^\top B/p \to G_B$ by Assumption 1. Therefore:

$$
W^{(p)} \xrightarrow{a.s.} W := \frac{F^\top G_B F}{n} + \frac{\delta^2}{n}I_n \tag{8}
$$

in spectral norm on the fixed $n\times n$ space.

---

> **k=3 example — Step 2** *(with $G_B = C = \mathrm{diag}(1.0,\,0.8,\,0.6)$ and $n=60$)*
>
> 

$$
W = \frac{F^\top C F}{60} + \frac{1.0}{60}\,I_{60} \;\in\; \mathbb{R}^{60\times 60}.
$$

>
> Observed operator-norm error at $p=500$: $\Vert W^{(500)} - W\Vert_\mathrm{op} = 0.0146$, decaying at rate $\approx p^{-1/2}$, consistent with the CLT-rate at which $B^\top B/p \to G_B$.

---

### Step 3 — Eigenstructure of $W$

**Intuition.** $W$ is a rank-$k$ perturbation of $(\delta^2/n)I_n$: the signal term $F^\top G_B F/n$ has rank at most $k$. So $W$ has exactly $k$ "signal" eigenvalues above the noise level $\delta^2/n$ and $n-k$ eigenvalues exactly equal to $\delta^2/n$.

Lemma 7 (proved above) gives these explicitly:
- **Signal eigenvalues:** $\hat\lambda_j + \delta^2/n$, where $\hat\lambda_j$ are the eigenvalues of $\hat{M}$.
- **Signal eigenvectors:** $v_j = F^\top G_B^{1/2}\hat{w}_j/\sqrt{n\hat\lambda_j}$.
- **Noise eigenspace:** $\mathrm{col}(F^\top G_B^{1/2})^\perp$ (the $(n-k)$-dimensional space orthogonal to all signal directions).

The eigenvectors $v_j$ have an illuminating interpretation: $G_B^{1/2}\hat{w}_j$ is the $j$-th "prevalence-weighted factor direction" in $\mathbb{R}^k$; $F^\top$ maps it into the time-series space $\mathbb{R}^n$; and $1/\sqrt{n\hat\lambda_j}$ normalizes the result. The eigenvectors of $W$ are time-series projections of the factor-covariance eigenvectors, weighted by loading prevalence.

*(The k=3 verification table for this step appears in the Lemma 7 callout above.)*

---

### Step 4 — Spectral Convergence

**Intuition.** We know $W^{(p)} \to W$ in operator norm (Step 2) and we know the eigenvalues and eigenvectors of $W$ explicitly (Step 3). The top-$k$ eigenvalues of $W$ are simple and separated from the noise level $\delta^2/n$ by gap $\hat\lambda_k > 0$ (Assumption 2 ensures $\hat\lambda_k > 0$ a.s. under the regular-event hypothesis). A classical perturbation theorem transfers this to the sequence $W^{(p)}$.

**Theorem (Kato 1995, §II.1).** *If $A_p \to A$ in operator norm and $\tau$ is a simple eigenvalue of $A$ separated from the rest of its spectrum by gap $\gamma > 0$, then the corresponding eigenvalue and eigenvector (up to sign) of $A_p$ converge to those of $A$.*

Applying this to each of the $k$ signal eigenvalues of $W$:

$$
\frac{(s^{(p)}_j)^2}{p} \to \hat\lambda_j + \frac{\delta^2}{n} \quad \text{and} \quad v^{(p)}_j \to v_j \quad \text{a.s., up to sign,} \tag{9}
$$

for the **full sequence** in $p$. The convergence is not merely along subsequences, because the $G_B$-coordinate framework introduced in Step 5 eliminates the rotational ambiguity that would otherwise require subsequence extraction when loading columns are not well-separated.

---

> **k=3 example — Step 4** *(at $p=500$)*
>
> | $j$ | $(s^{(p)}_j)^2/p$ at $p=500$ | $\hat\lambda_j + \delta^2/n$ | $|\cos\angle(v^{(p)}_j, v_j)|$ |
> |:---:|:------------------------------:|:-----------------------------:|:-------------------------------:|
> | 1 | 0.06289 | 0.06226 | 0.963 |
> | 2 | 0.03742 | 0.03228 | 0.899 |
> | 3 | 0.02895 | 0.02345 | 0.703 |
>
> Factor 3's eigenvector cosine is only 0.703 at $p=500$ because the spectral gap $\hat\lambda_3 = 0.00679$ is small — Kato's perturbation bound degrades proportionally to $\Vert W^{(p)}-W\Vert_\mathrm{op}/\hat\lambda_3$. Both the eigenvalue and the eigenvector converge, but slowly; larger $p$ is needed before $v^{(p)}_3$ is close to its limit.

---

### Step 5 — The $G_B$-Coordinate Framework

**Purpose.** To compute the two terms on the right of (6), we need to express both $b_j$ (the population target) and $h_j^\parallel$ (the in-subspace component of $h_j$) in a common coordinate system that converges cleanly for the full sequence in $p$. The SVD basis of $B$ would create rotational ambiguity when loading-column singular values are close; the **loading map** avoids this entirely.

**The loading map.** Define $\Phi_p : \mathbb{R}^k \to \mathcal{B}$ by

$$
\Phi_p(x) = \frac{Bx}{\sqrt{p}}.
$$

Its Gram satisfies $\Phi_p^\top\Phi_p = B^\top B/p \to G_B$. Any vector in $\mathcal{B}$ writes uniquely as $\Phi_p(x)$ for some $x \in \mathbb{R}^k$, and Euclidean inner products in $\mathbb{R}^p$ correspond to $G_B$-inner products in $\mathbb{R}^k$: $\langle \Phi_p(x), \Phi_p(y)\rangle \to x^\top G_B y$.

**Population direction in $\Phi_p$ coordinates.** The eigenvalue equation $\Sigma^{(p)}_0 b_j = \lambda_j b_j$ translates to a $k\times k$ equation for the coordinate $a_j^{(p)}$ with $b_j = \Phi_p(a_j^{(p)})$. As $p\to\infty$ this converges to

$$
a_j := G_B^{-1/2} w_j, \tag{10}
$$

where $w_j$ is the $j$-th eigenvector of $M = G_B^{1/2}\Sigma_F G_B^{1/2}$. When $G_B = C$ (diagonal), $a_j = C^{-1/2}e_j$: the population direction for factor $j$ concentrates in the $j$-th coordinate, rescaled by $\sqrt{c_j}$.

**Sample direction in $\Phi_p$ coordinates.** The in-subspace component of $h_j$ satisfies $\Pi_B h_j = \Phi_p(g_j^{(p)}) + o(1)$ a.s., where

$$
g_j^{(p)} = \frac{F v^{(p)}_j}{\sqrt{(s^{(p)}_j)^2/p}}.
$$

Using $v^{(p)}_j \to v_j$ and $(s^{(p)}_j)^2/p \to \hat\lambda_j + \delta^2/n$ from Step 4, then substituting the formula (5) for $v_j$ and the key identity $(*)$:

$$
g_j^{(p)} \;\longrightarrow\; g_j := \sqrt{\frac{n\hat\lambda_j}{n\hat\lambda_j + \delta^2}}\cdot G_B^{-1/2}\hat{w}_j. \tag{11}
$$

*(Derivation: $F v_j = FF^\top G_B^{1/2}\hat{w}_j / (n\sqrt{\hat\lambda_j}) \cdot n^{1/2}$... more carefully: substituting $v_j = F^\top G_B^{1/2}\hat{w}_j/\sqrt{n\hat\lambda_j}$ gives $Fv_j = FF^\top G_B^{1/2}\hat{w}_j/(n\sqrt{\hat\lambda_j}) \cdot \sqrt{n} = (FF^\top/n)G_B^{1/2}\hat{w}_j/\sqrt{\hat\lambda_j}$; by $(*)$ this equals $\hat\lambda_j G_B^{-1/2}\hat{w}_j/\sqrt{\hat\lambda_j} = \sqrt{\hat\lambda_j}G_B^{-1/2}\hat{w}_j$; dividing by $\sqrt{\hat\lambda_j + \delta^2/n}$ gives equation (11).)*

The scale factor $\sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)} \in (0,1)$ is the attenuation of the in-subspace component due to noise: $h_j^\parallel$ is shrunk relative to $b_j$, with the shrinkage controlled entirely by $\widehat{\mathrm{SNR}}_j$.

---

> **k=3 example — Step 5** *(diagonal-Gram case $G_B = C$, so $G_B^{-1/2} = C^{-1/2}$)*
>
> | Coordinate | $j=1$ | $j=2$ | $j=3$ |
> |:----------:|:-----:|:-----:|:-----:|
> | $g_j$ | $(-0.852,\,-0.059,\,+0.085)$ | $(-0.053,\,+0.759,\,-0.184)$ | $(-0.033,\,-0.127,\,-0.678)$ |
> | $a_j = C^{-1/2}e_j$ | $(1.000,\,0,\,0)$ | $(0,\,1.118,\,0)$ | $(0,\,0,\,1.291)$ |
>
> Each $g_j$ is nearly parallel to $a_j$: the dominant component of $g_j$ is in the $j$-th coordinate. The small off-diagonal entries arise from the finite-$n$ correlation structure of $\hat{M}$ (its off-diagonal elements are nonzero because the 60 factor returns are not exactly orthogonal). The magnitude of $g_j$ is $\sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)} < 1$: the in-subspace component is attenuated by the noise.

---

### Step 6 — Floor and In-Subspace Angle via $G_B$ Inner Products

With $g_j^{(p)} \to g_j$, $a_j^{(p)} \to a_j$, and $B^\top B/p \to G_B$, lengths and inner products in $\mathbb{R}^p$ converge to $G_B$-inner products in $\mathbb{R}^k$. The key algebraic fact is that $G_B^{-1/2} G_B G_B^{-1/2} = I_k$, which causes the $G_B$ factors to cancel cleanly.

**The floor.** $\Vert h_j^\parallel\Vert^2 = \Vert \Pi_B h_j\Vert^2 \to g_j^\top G_B g_j$. Substituting (11):

$$
g_j^\top G_B\, g_j = \frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}\,(G_B^{-1/2}\hat{w}_j)^\top G_B (G_B^{-1/2}\hat{w}_j) = \frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}\,\hat{w}_j^\top\hat{w}_j = \frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}.
$$

Since $\Vert h_j\Vert  = 1$:

$$
\Vert h_j^\perp\Vert^2 = 1 - \Vert h_j^\parallel\Vert^2 \;\longrightarrow\; \frac{\delta^2}{n\hat\lambda_j + \delta^2}. \tag{12}
$$

**The in-subspace inner product.** $\langle h_j^\parallel, b_j\rangle \to g_j^\top G_B\, a_j$. Substituting (10) and (11):

$$
g_j^\top G_B\, a_j = \sqrt{\frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}}\,(G_B^{-1/2}\hat{w}_j)^\top G_B (G_B^{-1/2} w_j) = \sqrt{\frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}}\,\hat{w}_j^\top w_j.
$$

**The in-subspace angle.** Dividing the squared inner product by $\Vert h_j^\parallel\Vert^2$:

$$
\sin^2\!\angle\!\left(\frac{h_j^\parallel}{\Vert h_j^\parallel\Vert },\, b_j\right) \;\longrightarrow\; 1 - \frac{(n\hat\lambda_j/(n\hat\lambda_j+\delta^2))\,(\hat{w}_j^\top w_j)^2}{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)} = 1 - (\hat{w}_j^\top w_j)^2 = \sin^2\!\angle(\hat{w}_j,\, w_j). \tag{13}
$$

The shrinkage factor cancels exactly. The in-subspace angle converges to the angle between the sample and population eigenvectors of $\hat{M}$ and $M$ — independent of the noise level.

---

> **k=3 example — Step 6**
>
> **Norm verification** ($G_B$-inner products equal the SNR weights):
>
> | $j$ | $g_j^\top G_B g_j$ | $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$ | Match |
> |:---:|:------------------:|:------------------------------------------:|:-----:|
> | 1 | 0.73232 | 0.73232 | ✓ |
> | 2 | 0.48361 | 0.48361 | ✓ |
> | 3 | 0.28941 | 0.28941 | ✓ |
>
> **Floor and rotation for each factor:**
>
> | $j$ | $\widehat{\mathrm{SNR}}_j$ | Floor $= \delta^2/(n\hat\lambda_j+\delta^2)$ | Rotation $= \sin^2\angle(\hat{w}_j,\, e_j)$ |
> |:---:|:--------------------------:|:--------------------------------------------:|:--------------------------------------------:|
> | 1 | 2.736 | 0.2677 | 0.0097 |
> | 2 | 0.937 | 0.5164 | 0.0480 |
> | 3 | 0.407 | 0.7106 | 0.0481 |
>
> The rotation is small for all three factors because with $G_b = I_k$ and 60 i.i.d. factor draws, $\hat{M}$ is nearly diagonal ($\hat{w}_j \approx e_j$). The floor dominates: for factor 3, a realized $\widehat{\mathrm{SNR}}_3 = 0.41$ means 71% of $h_3$'s squared norm is permanently outside $\mathcal{B}$.

---

### Step 7 — Assembly

Substituting (12) and (13) back into the decomposition (6):

$$
\sin^2\angle(h_j, b_j)
= \Vert h_j^\perp\Vert^2 + \Vert h_j^\parallel\Vert^2\,\sin^2\!\angle\!\left(\frac{h_j^\parallel}{\Vert h_j^\parallel\Vert }, b_j\right)
$$

$$
\xrightarrow{a.s.}\; \frac{\delta^2}{n\hat\lambda_j+\delta^2} \;+\; \frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}\,\sin^2\angle(\hat{w}_j,\, w_j). \tag{3}
$$

All limits (Steps 2–6) hold for the **full sequence** in $p$, not just along subsequences. This is because the $G_B$-coordinate framework works directly with $B^\top B/p \to G_B$, bypassing the rotational ambiguity of the SVD basis $U(p)$ of $B$. $\square$

---

> **k=3 example — Step 7** (assembled formula)
>
> | $j$ | Floor | $\Vert h_j^\parallel\Vert^2 \times$ Rotation | RHS predicted | LHS observed ($p=500$) | Gap |
> |:---:|:-----:|:--------------------------------------:|:-------------:|:----------------------:|:---:|
> | 1 | 0.2677 | $0.7323 \times 0.0097 = 0.0071$ | **0.2748** | 0.3099 | +0.035 |
> | 2 | 0.5164 | $0.4836 \times 0.0480 = 0.0232$ | **0.5396** | 0.5697 | +0.030 |
> | 3 | 0.7106 | $0.2894 \times 0.0481 = 0.0139$ | **0.7245** | 0.7975 | +0.073 |
>
> The gaps are positive (the limit is approached from above at finite $p$) and largest for factor 3, which has the smallest spectral gap and the slowest convergence in $p$. All gaps decay at rate $O(p^{-1/2})$, matching the rate of $\Vert W^{(p)} - W\Vert_\mathrm{op}$.

---

## Corollary 4: Grassmannian Distance

Summing equation (3) over $j = 1, \ldots, k$ and using the fact that the rotation terms cancel in the Grassmannian metric (the per-direction in-subspace angles do not contribute to the subspace-level distance):

$$
d_{\mathrm{Gr}}^2(\mathrm{col}(H),\, \mathcal{B}) = \sum_{j=1}^k \Vert h_j^\perp\Vert^2 \;\xrightarrow{a.s.}\; \sum_{j=1}^k \frac{\delta^2}{n\hat\lambda_j + \delta^2}. \tag{14}
$$

The Grassmannian squared distance is determined entirely by the $k$ signal-to-noise ratios. The rotation terms $\sin^2\angle(\hat{w}_j, w_j)$, which appear in the per-direction formula (3), drop out completely here: if you care only about whether the *subspace* $\mathrm{col}(H)$ aligns with $\mathcal{B}$, the total misalignment is $\sum_j 1/(1+\widehat{\mathrm{SNR}}_j)$.

---

## Worked Example: $k=3$, $p=500$, $n=60$

This section consolidates all the inline callouts into a single self-contained numerical illustration of Theorem 1.

### Parameters and Setup

We use the diagonal-Gram special case ($G_b = I_k$), so $G_B = C = \mathrm{diag}(c_1, c_2, c_3)$ and $M = C^{1/2}\Sigma_F C^{1/2}$ is diagonal with population eigenvectors $w_j = e_j$.

| Symbol | Meaning | Value |
|:------:|:--------|:-----:|
| $k$, $p$, $n$ | factors, assets, periods | 3, 500, 60 |
| $\delta^2$ | noise variance | 1.0 |
| $c_1, c_2, c_3$ | prevalences $\lim\Vert \beta_j\Vert^2/p$ | 1.0, 0.8, 0.6 |
| $\sigma_1^2, \sigma_2^2, \sigma_3^2$ | factor return variances | 0.04, 0.02, 0.01 |
| $\lambda_j = c_j\sigma_j^2$ | population spikes | 0.040, 0.016, 0.006 |
| $\mathrm{SNR}_j = n\lambda_j/\delta^2$ | population SNRs | 2.40, 0.96, 0.36 |

Data model: $Y = BF + Z$ with $B \in \mathbb{R}^{500\times3}$ (columns drawn i.i.d. $\mathcal{N}(0, c_j)$), $F \in \mathbb{R}^{3\times60}$ (rows drawn i.i.d. $\mathcal{N}(0, \sigma_j^2)$), $Z \in \mathbb{R}^{500\times60}$ (entries i.i.d. $\mathcal{N}(0,1)$). Seed: 20260522.

### Computing $\hat{M}$ and Its Eigensystem

$$
\hat{M} = C^{1/2}\,\frac{FF^\top}{60}\,C^{1/2} \;\in\; \mathbb{R}^{3\times3}.
$$

The realized eigenvalues $\hat\lambda_j$ differ from the population limits $\lambda_j$ due to sampling noise in $F$:

| $j$ | $\hat\lambda_j$ | $\lambda_j$ (population) | $\widehat{\mathrm{SNR}}_j = n\hat\lambda_j/\delta^2$ |
|:---:|:---------------:|:------------------------:|:----------------------------------------------------:|
| 1 | 0.04560 | 0.040 | 2.736 |
| 2 | 0.01561 | 0.016 | 0.937 |
| 3 | 0.00679 | 0.006 | 0.407 |

Since $G_b = I_k$, the population eigenvectors are $w_j = e_j$. The realized eigenvectors $\hat{w}_j$ deviate slightly from $e_j$ due to the finite-$n$ off-diagonal structure of $\hat{M}$, producing small rotation terms $\sin^2\angle(\hat{w}_j, e_j) = 1 - (\hat{w}_j)_j^2$.

### The Limit Matrix $W$

$$
W = \frac{F^\top C F}{60} + \frac{1}{60}I_{60} \;\in\; \mathbb{R}^{60\times60}.
$$

By Lemma 7, its spectrum consists of three signal eigenvalues $\hat\lambda_j + 1/60$ and 57 noise eigenvalues at $1/60 \approx 0.0167$:

| $j$ | $\hat\lambda_j + \delta^2/n$ |
|:---:|:----------------------------:|
| 1 | 0.06226 |
| 2 | 0.03228 |
| 3 | 0.02345 |

At $p=500$: $\Vert W^{(500)} - W\Vert_\mathrm{op} = 0.0146$, consistent with the $O(p^{-1/2})$ convergence rate.

### Floor and Rotation for Each Factor

| $j$ | $\widehat{\mathrm{SNR}}_j$ | Floor | $\sin^2\angle(\hat{w}_j,\,e_j)$ | Weight $\frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}$ | Weighted rotation |
|:---:|:--------------------------:|:-----:|:--------------------------------:|:-------------------------------------------------------:|:-----------------:|
| 1 | 2.736 | 0.2677 | 0.0097 | 0.7323 | 0.0071 |
| 2 | 0.937 | 0.5164 | 0.0480 | 0.4836 | 0.0232 |
| 3 | 0.407 | 0.7106 | 0.0481 | 0.2894 | 0.0139 |

### Assembly and Comparison

| $j$ | Floor + Weighted Rotation | RHS (predicted) | LHS (observed, $p=500$) | Relative gap |
|:---:|:-------------------------:|:---------------:|:------------------------:|:------------:|
| 1 | $0.2677 + 0.0071$ | **0.2748** | 0.3099 | +12.8% |
| 2 | $0.5164 + 0.0232$ | **0.5396** | 0.5697 | +5.6% |
| 3 | $0.7106 + 0.0139$ | **0.7245** | 0.7975 | +10.1% |

The observed LHS exceeds the predicted RHS at $p=500$; the gap closes at rate $O(p^{-1/2})$ as $p$ increases.

### Grassmannian Distance

$$
\sum_{j=1}^3 \frac{\delta^2}{n\hat\lambda_j + \delta^2} = 0.2677 + 0.5164 + 0.7106 = 1.4947.
$$

Observed at $p=500$: $\sum_j \Vert h_j^\perp\Vert^2 \approx 1.658$. The gap of $+0.163$ contracts to zero as $p\to\infty$. Note that the rotation contributions (0.0071 + 0.0232 + 0.0139 = 0.0442 total) do not appear in the Grassmannian sum — they cancel precisely in the subspace metric.

### Convergence as $p$ Grows

For each factor, $\sin^2\angle(h_j, b_j)$ converges to the predicted RHS as $p$ increases, driven by the rate at which $\Vert W^{(p)} - W\Vert_\mathrm{op} \to 0$. Factor 3 converges most slowly because its spectral gap $\hat\lambda_3 = 0.00679$ is smallest, making Kato's perturbation bound weakest for $v^{(p)}_3$. By $p \approx 2000$, all three factors are within a few percent of their limits.

The rate is $O(p^{-1/2})$ throughout — the same rate as $\Vert W^{(p)} - W\Vert_\mathrm{op}$, which is controlled by the CLT rate of $B^\top B/p \to G_B$.

---

## Summary of Proof Logic

| Step | What it establishes | Key tool |
|:----:|:--------------------|:--------:|
| Lemma 7 | Exact eigenstructure of the limit $W$ | AB/BA identity |
| Step 1 | Splits $\sin^2\angle$ into floor + in-subspace terms | Pythagorean identity in $\mathbb{R}^p$ |
| Step 2 | Replaces $p\times p$ problem with $n\times n$ limit $W$ | LLN on noise (Lemmas 1, 4) |
| Step 3 | Reads eigenvalues and eigenvectors of $W$ off explicitly | Lemma 7 |
| Step 4 | Transfers convergence to finite-$p$ eigenvectors $v^{(p)}_j$ | Kato continuity |
| Step 5 | Expresses $b_j$ and $h_j^\parallel$ in $G_B$ coordinates | Loading map $\Phi_p$; key identity $(*)$ |
| Step 6 | Evaluates floor and in-subspace angle as $G_B$ bilinear forms | Algebraic cancellation |
| Step 7 | Assembles equation (3) | Substitution into (6) |

The architectural insight is Step 5: by working in the loading map's coordinate system rather than the SVD basis of $B$, the proof avoids the rotational ambiguity that would otherwise require subsequence extraction when loading-column singular values cluster (e.g., when $G_b = I_k$ makes them coincide in the limit).

---

*Numerical example generated from `proof_walkthrough_k3.py` and `proof_walkthrough_figures.py`. Seed: 20260522. Parameters: $k=3$, $p=500$, $n=60$, $c = [1.0, 0.8, 0.6]$, $\sigma^2 = [0.04, 0.02, 0.01]$, $\delta^2 = 1.0$.*
