# Proof of Theorem 3.1$'$: $k$-Factor Generalization of the Dispersion Bias

*Final version. Uses a coordinate-free projection argument for Part (i), achieving a shorter proof and dropping Assumption 2.5$'$ from Part (i)'s hypotheses. Proof of Part (ii) uses Lemma A.2$'$ and Assumptions 2.5$'$–2.6$'$. All lemmas proved in full.*

------------------------------------------------------------------------

## 0. The Central Geometric Idea

Before any notation, here is the essence of the proof. *(All symbols are defined precisely in §2–3; this section is a guide to the key ideas only.)*

The $k$-factor model says every column of $Y$ is a sum of a vector in the population factor subspace $\mathcal{B} = \operatorname{col}(B) \subset \mathbb{R}^p$ and an idiosyncratic noise term. The top-$k$ left singular vectors of $Y$ — collected in $H$ — are therefore "pulled toward" $\mathcal{B}$ by the signal. Part (i) makes this precise: $H^\top v$ is asymptotically the same as $H^\top \Pi_B v$ for any fixed bounded probe vector $v$, meaning $H$ only "sees" $v$ through its $\mathcal{B}$-component.

**Why?** Project the SVD identity onto $\mathcal{B}^\perp$. The signal term $\Pi_B^\perp B = 0$ (trivially — columns of $B$ lie in $\mathcal{B}$), so the entire signal vanishes and only noise remains. That noise is a bounded random linear functional of $Z$, which vanishes in the $p \to \infty$ limit by a law-of-large-numbers argument.

Part (ii) goes further: it identifies *which* direction of $\mathcal{B}$ each $h_i$ aligns with, and by how much. This requires knowing the individual right singular vectors (Davis–Kahan) and orthogonality of the factor structure (Assumptions 2.5$'$, 2.6$'$).

**The signal-to-noise picture for $\psi_{\infty,i}$.** Think of detecting the $i$-th factor direction $b_i$ in the noisy data matrix. The signal energy in direction $i$ is $\alpha_i^2 |X_i|^2$ (loading scale squared times factor return energy), accumulated over $n$ observations. The noise energy in the same direction is $\delta^2$ per observation, giving total noise $\delta^2$ (since $n$ is fixed). The cosine of the angle between $h_i$ and $b_i$ is then exactly the signal-to-total ratio:

$$
\psi_{\infty,i} \;=\; \frac{\alpha_i |X_i|}{\sqrt{\alpha_i^2 |X_i|^2 + \delta^2}}
\;=\; \sqrt{\frac{\text{signal energy}}{\text{signal} + \text{noise energy}}}.
$$

When signal dominates, $\psi_{\infty,i} \to 1$ (perfect alignment). When noise dominates, $\psi_{\infty,i} \to 0$ (no information). In the high-dimensional limit studied here, neither extreme holds, and $\psi_{\infty,i}$ is strictly between 0 and 1 — the source of the dispersion bias.

------------------------------------------------------------------------

## 1. Statement of Theorem 3.1$'$

**Theorem 3.1$'$.** Under Assumptions 2.1$'$–2.4$'$ (see §3), almost surely as $p \to \infty$, for any deterministic $v \in \mathbb{R}^p$ with $|v| \le 1$:

$$
\textbf{(Part i)} \qquad H^\top v \;-\; H^\top \Pi_B v \;\longrightarrow\; 0.
$$

Under the additional Assumptions 2.5$'$–2.6$'$:

$$
\textbf{(Part ii)} \qquad H^\top \tilde B \;\longrightarrow\; \operatorname{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k}) \quad \text{a.s.,}
$$

where $\psi_{\infty,i} := \alpha_i |X_i| / \sqrt{\alpha_i^2 |X_i|^2 + \delta^2} \in (0,1)$ a.s.

*(All notation — $H$ (top-$k$ left singular vectors of $Y/\sqrt{n}$), $\Pi_B$ (orthogonal projector onto $\mathcal{B} = \operatorname{col}(B)$), $\tilde B$ (orthonormal basis for $\mathcal{B}$), $\alpha_i$ (asymptotic loading scale), $X_i$ ($i$-th factor return vector), $\delta$ (noise standard deviation), $k$ (number of factors) — is introduced formally in §2 and §3.)*

**What each part says.**

*Part (i)* is a subspace alignment statement. The sample frame $H$ can only "see" the probe vector $v$ through $v$'s projection onto the population factor subspace $\mathcal{B}$; the idiosyncratic component $\Pi_B^\perp v$ is asymptotically invisible to $H$. Crucially, Part (i) holds for any bounded $v$ and requires no orthogonality assumption on $B$. Setting $v = z = e/\sqrt{p}$ recovers the headline dispersion bias statement.

*Part (ii)* is a per-factor calibration. Each sample direction $h_i$ converges (in inner product with $b_i$) to the shrinkage factor $\psi_{\infty,i} \in (0,1)$, and all cross-alignments $\langle h_i, b_j \rangle$ vanish for $i \ne j$. This requires the stronger structural assumptions 2.5$'$ and 2.6$'$, which ensure the population factor directions are cleanly separated in the spectrum of the Gram matrix.

**Assumption accounting.** Part (i) uses 2.1$'$–2.4$'$ only; Assumption 2.5$'$ (orthogonal loadings) is *not* needed. Part (ii) additionally requires 2.5$'$ and 2.6$'$. The Corollary (§8) uses all six. See §11 for a full table.

------------------------------------------------------------------------

## 2. Setup and Notation

### 2.1 The $k$-Factor Model

The data matrix satisfies

$$
Y \;=\; B F^\top + Z, \tag{$8'$}
$$

where $Y \in \mathbb{R}^{p \times n}$ (returns: $p$ securities, $n$ time periods), $B \in \mathbb{R}^{p \times k}$ (population factor loading matrix, columns $\beta_1, \ldots, \beta_k \in \mathbb{R}^p$), $F \in \mathbb{R}^{n \times k}$ (factor return matrix, columns $X_1, \ldots, X_k \in \mathbb{R}^n$), and $Z \in \mathbb{R}^{p \times n}$ (idiosyncratic noise). Throughout, $p \to \infty$ with $n, k$ fixed and $k < n$.

We write $|\cdot|$ for the Euclidean norm and $\langle x, y \rangle_p := x^\top y$ for $x, y \in \mathbb{R}^p$ (subscript $p$ flags the ambient space, to distinguish from time-series inner products in $\mathbb{R}^n$). When $\langle x_p, y_p \rangle_p$ converges a.s. as $p \to \infty$, we write the limit $\langle x, y \rangle_\infty$.

### 2.2 Population Subspace and Projectors

The **population factor subspace** is $\mathcal{B} := \operatorname{col}(B) \subset \mathbb{R}^p$, with $\dim \mathcal{B} = k$ a.s.

The **orthogonal projector onto the column space of $B$** is

$$
\Pi_B \;:=\; \tilde B \tilde B^\top \;=\; B(B^\top B)^{-1} B^\top,
$$

where $\tilde B$ is any orthonormal basis for $\mathcal{B}$; the projector $\Pi_B$ does not depend on the choice of basis. The **idiosyncratic projector** is $\Pi_B^\perp := I_p - \Pi_B$, the orthogonal projector onto the **orthogonal complement** $\mathcal{B}^\perp := \{x \in \mathbb{R}^p : x^\top y = 0 \text{ for all } y \in \mathcal{B}\}$.

Under Assumption 2.5$'$ (§3), the natural orthonormal basis is $\tilde B = B \cdot \operatorname{diag}(|\beta_j|^{-1})$, with columns $b_j := \beta_j / |\beta_j|$. The sample projector is $\Pi_H := H H^\top$, where $H$ is defined below.

### 2.3 SVD and the Fundamental Identity

The thin SVD of $Y/\sqrt{n}$ retains only the top $k$ singular triplets. Precisely, write the full SVD as $Y/\sqrt{n} = U_{\mathrm{full}} \Sigma_{\mathrm{full}} V_{\mathrm{full}}^\top$ where $U_{\mathrm{full}} \in \mathbb{R}^{p \times p}$ and $V_{\mathrm{full}} \in \mathbb{R}^{n \times n}$ are orthogonal and $\Sigma_{\mathrm{full}} \in \mathbb{R}^{p \times n}$ is rectangular diagonal. Define:

- $H := U_{\mathrm{full}}[:,{:}k] \in \mathbb{R}^{p \times k}$: the top-$k$ **left singular vectors** (columns $h_1, \ldots, h_k$), satisfying $H^\top H = I_k$.
- $\mathcal{X}_p := V_{\mathrm{full}}[:,{:}k] \in \mathbb{R}^{n \times k}$: the top-$k$ **right singular vectors** (columns $\chi_{p,1}, \ldots, \chi_{p,k}$), satisfying $\mathcal{X}_p^\top \mathcal{X}_p = I_k$.
- $S_p := \operatorname{diag}(s_{p,1}, \ldots, s_{p,k})$: the top-$k$ **singular values** in decreasing order.

The subscript $p$ on $\mathcal{X}_p$, $\chi_{p,i}$, and $s_{p,i}$ indicates these are sample quantities varying with the growing dimension; the second subscript $i$ indexes the factor.

The relation $H S_p = Y \mathcal{X}_p / \sqrt{n}$ is an exact equality. To see why: right-multiplying the full SVD by $\mathcal{X}_p$ gives $V_{\mathrm{full}}^\top \mathcal{X}_p = \bigl(\begin{smallmatrix}I_k \\ 0\end{smallmatrix}\bigr)$ (by orthogonality of $V_{\mathrm{full}}$), and then

$$
\Sigma_{\mathrm{full}} \begin{pmatrix} I_k \\ 0 \end{pmatrix}
\;=\; \begin{pmatrix} S_p \\ 0 \end{pmatrix},
\qquad
U_{\mathrm{full}} \begin{pmatrix} S_p \\ 0 \end{pmatrix}
\;=\; H S_p.
$$

The tail right singular vectors are orthogonal to $\mathcal{X}_p$, so they vanish — no approximation is involved. Substituting $(8')$:

$$
H S_p \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt{n}} \;+\; \frac{Z \mathcal{X}_p}{\sqrt{n}}. \tag{$33'$}
$$

This is an exact $p \times k$ matrix identity, valid for every finite $p$.

### 2.4 The Dispersionless Vector

The **dispersionless** (equal-weight) vector is $z := e/\sqrt{p} \in \mathbb{R}^p$, where $e := (1, \ldots, 1)^\top \in \mathbb{R}^p$ is the all-ones vector. It satisfies $|z| = 1$ for all $p$, and $\langle b_j, z \rangle_p = e^\top \beta_j / (|\beta_j| \sqrt{p})$ measures factor $j$'s average loading per unit loading-norm, scaled by $1/\sqrt{p}$.

### 2.5 Sign Convention

Each $h_i$ is normalised so that $\mu_p(h_i) := e^\top h_i / p \ge 0$. This mirrors the paper's convention and ensures $h_i$ aligns with the "positive loading" direction of the market. The sign of $\chi_{p,i}$ is then determined by the SVD relation $(33')$.

------------------------------------------------------------------------

## 3. Assumptions

**Assumption 2.1$'$ (Moments and independence).** The entries $Z_{il}$ satisfy $\mathbb{E}[Z_{il}] = 0$, $\operatorname{Var}(Z_{il}) = \delta^2 > 0$, and $\operatorname{Cov}(Z_{il}, Z_{i'l'}) = 0$ for $(i,l) \ne (i',l')$. Factor returns satisfy $\mathbb{E}[X_{jt}] = 0$ and $\operatorname{Var}(X_{jt}) = \sigma_j^2 > 0$, and the distribution of $X_j = (X_{j1},\ldots,X_{jn})^\top$ in $\mathbb{R}^n$ is **atomless** (absolutely continuous with respect to Lebesgue measure on $\mathbb{R}^n$). The columns of $Z$ are independent of $F$, and each $X_j \ne 0$ a.s.

**Assumption 2.2$'$ (Loading regularity).** For each $j = 1,\ldots,k$, the following limits hold a.s.:

$$
\mu_p(\beta_j) \;:=\; \frac{e^\top \beta_j}{p} \;\to\; \mu_\infty(\beta_j) \in (0, \infty),
\qquad
d_p(\beta_j) \;:=\; \frac{|\beta_j - \mu_p(\beta_j) e|}{|e|\,\mu_p(\beta_j)} \;\to\; d_\infty(\beta_j) \in [0, \infty).
$$

Setting $\alpha_j := \mu_\infty(\beta_j)\sqrt{1 + d_\infty^2(\beta_j)}$, this gives $|\beta_j|^2/p \to \alpha_j^2 \in (0,\infty)$ a.s. Intuitively, $\alpha_j$ is the asymptotic "loading scale" of factor $j$ — capturing both the average loading magnitude $\mu_\infty(\beta_j)$ and the cross-sectional spread $d_\infty(\beta_j)$.

**What $\alpha_j$ measures.** The loading scale $\alpha_j$ is the root-mean-square (RMS) loading of factor $j$ across all $p$ securities in the large-$p$ limit. To see why, write the squared norm of the $j$-th loading column:

$$
\frac{|\beta_j|^2}{p} \;=\; \frac{1}{p}\sum_{i=1}^p \beta_{ij}^2 \;\to\; \alpha_j^2.
$$

This is the second moment of the cross-sectional loading distribution. The definition $\alpha_j^2 = \mu_\infty^2(1 + d_\infty^2)$ decomposes it as mean-squared plus variance:

$$
\alpha_j^2 \;=\; \underbrace{\mu_\infty(\beta_j)^2}_{\text{mean}^2}
+ \underbrace{\mu_\infty(\beta_j)^2\, d_\infty(\beta_j)^2}_{\text{variance}},
$$

since $d_\infty$ is the coefficient of variation, so $\mu_\infty^2 d_\infty^2$ is the cross-sectional variance of the loadings.

Concretely: if every security has the same loading $\beta$ on factor $j$, then $\mu_\infty = \beta$, $d_\infty = 0$, and $\alpha_j = \beta$. If loadings are heterogeneous, $d_\infty > 0$ and $\alpha_j > \mu_\infty$, since the RMS exceeds the mean when there is spread.

**Role in $\psi_{\infty,j}$.** The formula

$$
\psi_{\infty,j} \;=\; \frac{\alpha_j |X_j|}{\sqrt{\alpha_j^2 |X_j|^2 + \delta^2}}
$$

can be read as: signal amplitude $= \alpha_j |X_j|$ (RMS loading times total factor return), noise amplitude $= \delta$, and $\psi_{\infty,j}$ is the cosine of the angle between the sample direction $h_j$ and the population direction $b_j$. When $\alpha_j$ is large (strong, broadly-held factor) the signal dominates and $\psi_{\infty,j} \approx 1$; when $\alpha_j$ is small (weak or concentrated factor) noise pulls $h_j$ away from $b_j$ and $\psi_{\infty,j} \ll 1$.

**Assumption 2.3$'$ (Noise structure).** The entries $\{Z_{il}\}_{i,l}$ are pairwise independent with uniformly bounded fourth moments: $\sup_{i,l} \mathbb{E}[Z_{il}^4] \le M < \infty$.

**Assumption 2.4$'$ (Dimension ordering).** $k < n$.

**Assumption 2.5$'$ (Orthogonal loadings, needed for Part (ii)).** The columns of $B$ are mutually orthogonal: $\beta_m^\top \beta_j = |\beta_j|^2 \delta_{mj}$ for all $m, j$, where $\delta_{mj}$ is the Kronecker delta ($1$ if $m = j$, $0$ otherwise). Equivalently, $b_m^\top b_j = \delta_{mj}$ and $\tilde B^\top \tilde B = I_k$.

*Note: Part (i) does not require Assumption 2.5$'$. The assumption is listed here for completeness; it enters only in Lemma A.2$'$ and Part (ii).*

**Assumption 2.6$'$ (Orthogonal factor returns, needed for Part (ii)).** The columns of $F$ are mutually orthogonal in $\mathbb{R}^n$: $X_i^\top X_j = |X_i|^2 \delta_{ij}$.

**Modeling note.** Together, 2.5$'$ and 2.6$'$ mean the population return covariance $\Sigma := \operatorname{Var}(y_t) \in \mathbb{R}^{p \times p}$ (where $y_t$ is any column of $Y$) takes the form

$$
\Sigma \;=\; B \Sigma_F B^\top + \delta^2 I_p,
$$

where $\Sigma_F := F^\top F / n$ is the sample factor-return covariance, and both $B^\top B$ and $\Sigma_F$ are diagonal. This is the **diagonalised spiked covariance** with $k$ spikes — the natural $k$-factor extension of the paper's $k = 1$ setting (where orthogonality is automatic since $B$ and $F$ are both single-column). Both assumptions are restrictive in practice but are the conditions under which $\psi_{\infty,i}$ takes the explicit closed form above.

------------------------------------------------------------------------

## 4. Auxiliary Lemmas

### Lemma A.1 (Paper's Lemma A.1)

**Lemma A.1.** *Let $\eta \in \mathbb{R}^p$ be deterministic with $c < |\eta|^2/p < C$ for constants $0 < c < C$ and all $p$ large. Under Assumption 2.3$'$, for each $l \in \{1, \ldots, n\}$:*

$$
\frac{(\eta^\top Z)_l}{|\eta|\sqrt{p}} \;\to\; 0 \quad \text{a.s. as } p \to \infty.
$$

*Proof.* This is the paper's Lemma A.1 (Goldberg, Papanicolaou, Shkolnik 2022, p.\~542). It follows from Chebyshev's inequality and the Borel–Cantelli lemma, using pairwise independence and bounded fourth moments. $\square$

**Why Lemma A.1 does not directly apply to bounded $\eta_p$.** Lemma A.1 requires $|\eta|^2/p \asymp 1$, i.e., $\eta$ grows like $\sqrt{p}$. In Part (i), the noise is controlled via $\eta_p = \Pi_B^\perp v$ with $|\eta_p| \le 1$ (a bounded, not growing, vector). This requires the companion lemma below.

------------------------------------------------------------------------

### Lemma A.1$''$ (Bounded-$\eta$ Companion)

**Lemma A.1$''$.** *Let $\eta_p \in \mathbb{R}^p$ be a deterministic sequence with $|\eta_p| \le C$ uniformly in $p$. Under Assumption 2.3$'$ (with independence of entries $\{Z_{it}\}_i$ within each column), for each fixed $l \in \{1, \ldots, n\}$:*

$$
\frac{(\eta_p^\top Z)_l}{\sqrt{p}} \;\to\; 0 \quad \text{a.s. as } p \to \infty.
$$

*More precisely, $(\eta_p^\top Z)_l = o(p^{1/2 - \epsilon})$ a.s. for any fixed $\epsilon \in (0, 1/4)$.*

*Proof.* Fix $l$ and write $W_p := (\eta_p^\top Z)_l = \sum_{i=1}^p a_i Z_{il}$ where $a_i := (\eta_p)_i$. We bound the fourth moment of $W_p$ and apply Markov's inequality and Borel–Cantelli.

**Step 1: Fourth moment bound.** Under independence of $\{Z_{il}\}_i$ and mean zero, the expansion of $\mathbb{E}[W_p^4]$ has two nonzero types of terms:

$$
\mathbb{E}[W_p^4] \;=\; \sum_{i=1}^p a_i^4 \,\mathbb{E}[Z_{il}^4]
\;+\; 3 \sum_{i \ne j} a_i^2 a_j^2 \,\mathbb{E}[Z_{il}^2]\,\mathbb{E}[Z_{jl}^2].
$$

(All terms involving three or more distinct indices vanish by mean-zero and independence; the factor 3 counts the three pairings of four indices into two equal pairs.) Bounding each piece:

$$
\sum_i a_i^4 \;\le\; \Bigl(\max_i |a_i|\Bigr)^2 \sum_i a_i^2
\;\le\; |\eta_p|^2 \cdot |\eta_p|^2 \;=\; |\eta_p|^4 \;\le\; C^4,
$$

using $\max_i |a_i| \le \sqrt{\sum_i a_i^2} = |\eta_p|$ and $|\eta_p| \le C$. For the pair sum: $\sum_{i \ne j} a_i^2 a_j^2 \le \bigl(\sum_i a_i^2\bigr)^2 = |\eta_p|^4 \le C^4$. With $\mathbb{E}[Z_{il}^4] \le M$ and $\mathbb{E}[Z_{il}^2] = \delta^2$:

$$
\mathbb{E}[W_p^4] \;\le\; C^4 M + 3 C^4 \delta^4 \;=:\; K \;<\; \infty,
$$

*uniformly in $p$.*

**Step 2: Borel–Cantelli.** By the Markov inequality applied to the fourth moment:

$$
\Pr\!\bigl(|W_p| > p^{1/2 - \epsilon}\bigr) \;\le\; \frac{\mathbb{E}[W_p^4]}{p^{2 - 4\epsilon}}
\;\le\; \frac{K}{p^{2 - 4\epsilon}}.
$$

For $\epsilon < 1/4$, the exponent $2 - 4\epsilon > 1$, so $\sum_{p=1}^\infty K/p^{2-4\epsilon} < \infty$. By the Borel–Cantelli lemma, $|W_p|/p^{1/2-\epsilon} \to 0$ a.s.

**Step 3: Conclusion.** Choosing any fixed $\epsilon_0 \in (0, 1/4)$ (e.g., $\epsilon_0 = 1/8$): $W_p = o(p^{1/2 - \epsilon_0})$ a.s., so

$$
\frac{|W_p|}{\sqrt{p}} \;=\; \frac{|W_p|}{p^{1/2 - \epsilon_0}} \cdot p^{-\epsilon_0}
\;\to\; 0 \cdot 0 \;=\; 0 \quad \text{a.s.} \quad \square
$$

**Note on pairwise vs. full independence.** The fourth moment expansion above uses full independence of $\{Z_{il}\}_i$ to ensure that terms involving three or four distinct indices vanish. Under pairwise independence (Assumption 2.3$'$), products of two distinct random variables factor (since $(Z_{il}, Z_{jl})$ independent implies $\mathbb{E}[f(Z_{il}) g(Z_{jl})] = \mathbb{E}[f(Z_{il})]\mathbb{E}[g(Z_{jl})]$), but products of three or more distinct variables need not. In practice, $Z$'s entries are typically independent within a column (the paper assumes so implicitly), so the above holds. If only pairwise independence is available, Lemma A.1$''$ can be recovered at a slightly slower rate via a more involved Riesz–Thorin interpolation argument; we proceed under independence here.

------------------------------------------------------------------------

### Lemma A.2$'$ (Spectral Convergence)

**Lemma A.2$'$.** *Under Assumptions 2.1$'$–2.5$'$, the following hold almost surely as $p \to \infty$.*

1. **Operator convergence.** The $n \times n$ Gram matrix $R_p := Y^\top Y/(np)$ converges in operator norm:

$$
R_p \;\to\; A_\infty \;:=\; \frac{F M_\infty F^\top}{n} + \frac{\delta^2}{n} I_n,
$$

where $M_\infty := \operatorname{diag}(\alpha_1^2, \ldots, \alpha_k^2) \in \mathbb{R}^{k \times k}$.

2. **Singular value convergence.** The normalised squared singular values satisfy $s_{p,i}^2/p \to \lambda_i$ a.s., where $\lambda_i > \delta^2/n > 0$. Under Assumption 2.6$'$, the explicit formula is:

$$
\lambda_i \;=\; \frac{\alpha_i^2 |X_i|^2 + \delta^2}{n}.
$$

3. **Right singular vector convergence (requires Assumption 2.6$'$).** The right singular vectors converge to exact eigenvectors of $A_\infty$:

$$
\chi_{p,i} \;\to\; \xi_i \;:=\; \frac{X_i}{|X_i|} \quad \text{a.s.}
$$

The eigenvalues $\lambda_1 > \cdots > \lambda_k$ of $A_\infty$ are distinct a.s.

4. **Denominator bound.** $\sqrt{p}/s_{p,i} \to 1/\sqrt{\lambda_i} \in (0, \infty)$ a.s.

**Proof.**

**Part 1.** Expand $Y^\top Y = FB^\top BF^\top + FB^\top Z + Z^\top BF^\top + Z^\top Z$ and divide by $np$, obtaining $R_p = T_1 + T_2 + T_3$:

*Signal term $T_1 = F(B^\top B/p)F^\top/n$.* By Assumptions 2.5$'$ and 2.2$'$, $B^\top B/p = \operatorname{diag}(|\beta_j|^2/p) \to M_\infty = \operatorname{diag}(\alpha_j^2)$ a.s. Since $F \in \mathbb{R}^{n \times k}$ is fixed (finite $n$ and $k$), multiplication is continuous in operator norm, so $T_1 \to FM_\infty F^\top/n$ a.s. in operator norm.

*Cross terms $T_2$.* The $(i, t)$ entry of $FB^\top Z/(np)$ equals

$$
\frac{1}{np} \sum_{j=1}^k (F)_{ij}\,\beta_j^\top Z_{\cdot t} \;=\; \sum_{j=1}^k (F)_{ij} \cdot \frac{\beta_j^\top Z_{\cdot t}}{|\beta_j|\sqrt{p}} \cdot \frac{|\beta_j|}{n\sqrt{p}}.
$$

By Lemma A.1 applied to $\eta = \beta_j$ (with $|\beta_j|^2/p \to \alpha_j^2 > 0$), the middle factor $\beta_j^\top Z_{\cdot t}/(|\beta_j|\sqrt{p}) \to 0$ a.s. The last factor $|\beta_j|/(n\sqrt{p}) \to \alpha_j/n$ a.s. (bounded). Since $(F)_{ij}$ is fixed, each entry of $T_2$ converges to zero a.s. As $T_2$ has fixed dimension $n \times n$, entrywise convergence implies operator-norm convergence. By symmetry, $T_3 \to 0$ a.s. in operator norm.

*Noise term $T_3 = Z^\top Z/(np)$.* The $(t, s)$ entry is $\sum_{i=1}^p Z_{it}Z_{is}/(np)$. For $t = s$: this equals $\sum_i Z_{it}^2/(np) \to \delta^2/n$ a.s. by the SLLN (the summands are pairwise uncorrelated under Assumption 2.3$'$, and bounded 4th moments ensure Chebyshev + Borel–Cantelli). For $t \ne s$: $\sum_i Z_{it}Z_{is}/(np)$ has mean zero (by Assumption 2.1$'$) and variance $\delta^4/n^2p \to 0$, so it converges to 0 a.s. Since $Z^\top Z/(np)$ has fixed dimension $n \times n$, entrywise convergence gives $Z^\top Z/(np) \to (\delta^2/n)I_n$ a.s. in operator norm.

Combining the three terms: $R_p \to FM_\infty F^\top/n + (\delta^2/n)I_n =: A_\infty$ a.s. in operator norm. $\square_{\mathrm{Part}\,1}$

**Part 2.** The matrix $A_\infty$ has rank at most $k$ plus the noise floor $(\delta^2/n)I_n$. By Weyl's inequality (eigenvalues are 1-Lipschitz in operator norm): if $\|R_p - A_\infty\|_{\mathrm{op}} \to 0$ a.s. (where $\|\cdot\|_{\mathrm{op}}$ denotes the operator norm, equal to the largest singular value), then the $i$-th eigenvalue of $R_p$ converges to the $i$-th eigenvalue of $A_\infty$ a.s. The $i$-th eigenvalue of $R_p$ equals $s_{p,i}^2/(np)$ (since $R_p = Y^\top Y/(np)$), so $s_{p,i}^2/(np) \to \lambda_i/n$, i.e., $s_{p,i}^2/p \to \lambda_i$, a.s.

Under Assumption 2.6$'$, the signal matrix $FM_\infty F^\top = \sum_j \alpha_j^2 X_j X_j^\top$ is a sum of $k$ mutually orthogonal rank-1 matrices (orthogonal because $X_i^\top X_j = 0$ for $i \ne j$). Its nonzero eigenvalues are $\alpha_j^2 |X_j|^2$ with eigenvectors $X_j/|X_j|$, and the eigenvalues of $A_\infty$ follow as stated. $\square_{\mathrm{Part}\,2}$

**Part 3.** Under Assumption 2.6$'$, the $k$ eigenvalues $\alpha_i^2|X_i|^2$ of the signal matrix are distinct a.s.: the event $\{\alpha_i^2|X_i|^2 = \alpha_j^2|X_j|^2\}$ has probability zero for each $i \ne j$. To see this, condition on $X_j$: since $|X_i|^2$ is atomless (the map $x \mapsto |x|^2$ pushes the absolutely continuous distribution of $X_i$ in $\mathbb{R}^n$ forward to an atomless distribution on $\mathbb{R}_{\ge 0}$, by Assumption 2.1$'$), we have $P\!\left(|X_i|^2 = \tfrac{\alpha_j^2}{\alpha_i^2}|X_j|^2 \,\Big|\, X_j\right) = 0$ a.s., and the unconditional probability is therefore also zero. (The independence $X_i \perp X_j$ used here is part of Assumption 2.6$'$.) Hence the eigenvalues $\lambda_1 > \cdots > \lambda_k$ of $A_\infty$ are Hence the eigenvalues $\lambda_1 > \cdots > \lambda_k$ of $A_\infty$ are distinct a.s., and the **Davis–Kahan $\sin\theta$ theorem** applies: if two symmetric matrices have small operator-norm difference and the target eigenvalue is well-separated, then the corresponding eigenvectors are close. Precisely (Bhatia, *Matrix Analysis*, Theorem VII.3.1), for the $i$-th eigenvector:

$$
\sin\angle(\chi_{p,i},\, \xi_i) \;\le\; \frac{\|R_p - A_\infty\|_{\mathrm{op}}}{g_i},
$$

where $g_i := \min_{j \ne i}|\lambda_i - \lambda_j| > 0$ a.s. is the **spectral gap** at position $i$. Since $\|R_p - A_\infty\|_{\mathrm{op}} \to 0$ a.s. (Part 1) and $g_i > 0$ a.s., the right side tends to 0 a.s., so $\chi_{p,i} \to \xi_i = X_i/|X_i|$ a.s. (up to sign; the sign is fixed by the convention in §2.5). $\square_{\mathrm{Part}\,3}$

**Part 4.** From Part 2, $s_{p,i}/\sqrt{p} \to \sqrt{\lambda_i}$ a.s., so $\sqrt{p}/s_{p,i} \to 1/\sqrt{\lambda_i} \in (0,\infty)$ a.s. $\square_{\mathrm{Part}\,4}$

**Geometric picture for Part 1.** The matrix $R_p = Y^\top Y/(np)$ is the $n \times n$ Gram matrix of the (scaled) columns of $Y$. As $p$ grows, the signal contribution $F(B^\top B/p)F^\top/n$ converges to a fixed rank-$k$ operator $FM_\infty F^\top/n$, while the noise $Z^\top Z/(np)$ becomes a perfect isotropic floor $(\delta^2/n)I_n$. The top-$k$ eigenvectors of $R_p$ therefore converge to those of the signal operator, which under Assumption 2.6$'$ are exactly the normalised factor return directions $X_i/|X_i|$.

------------------------------------------------------------------------

## 5. Proof of Part (i)

*Uses only Assumptions 2.1$'$–2.4$'$. Does not use orthogonality of $B$'s columns.*

**Geometric setup.** For any deterministic $v \in \mathbb{R}^p$ with $|v| \le 1$, decompose

$$
v \;=\; \underbrace{\Pi_B v}_{\text{in } \mathcal{B}} \;+\; \underbrace{\eta_p}_{\text{in } \mathcal{B}^\perp}
\;=\; \Pi_B v \;+\; \Pi_B^\perp v,
$$

with $|\eta_p| = |\Pi_B^\perp v| \le |v| \le 1$ (projectors are non-expansive). We want to show $H^\top v - H^\top \Pi_B v = H^\top \eta_p \to 0$ a.s. Note: $H^\top \eta_p$ measures the components of $H$ in the idiosyncratic direction $\eta_p$; the claim is these vanish.

**Step 1: Project the SVD identity.** Apply $\Pi_B^\perp$ to $(33')$:

$$
\Pi_B^\perp H \cdot S_p \;=\; \frac{\Pi_B^\perp B F^\top \mathcal{X}_p}{\sqrt{n}} \;+\; \frac{\Pi_B^\perp Z \mathcal{X}_p}{\sqrt{n}}.
$$

The first term on the right vanishes identically: $\Pi_B^\perp B = 0$ because each column of $B$ lies in $\mathcal{B}$ by definition, so projecting onto $\mathcal{B}^\perp$ kills it. This is the key step — the entire signal evaporates, regardless of the correlation structure of $B$'s columns. What remains is:

$$
\Pi_B^\perp H \cdot S_p \;=\; \frac{\Pi_B^\perp Z \mathcal{X}_p}{\sqrt{n}}. \tag{$\star$}
$$

**Step 2: Extract the scalar identity.** Right-multiply $(\star)$ by $S_p^{-1}$ (well-defined a.s. since $s_{p,i} > 0$ a.s. by Lemma A.2$'$ Part 2):

$$
\Pi_B^\perp H \;=\; \frac{\Pi_B^\perp Z \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1}.
$$

Now left-multiply by $v^\top$. Since $\Pi_B^\perp$ is symmetric and $\Pi_B^\perp v = \eta_p$:

$$
v^\top \Pi_B^\perp H \;=\; \frac{\eta_p^\top Z \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1}.
$$

Recognising the left side: $v^\top \Pi_B^\perp H = v^\top H - v^\top \Pi_B H$, whose transpose is exactly $H^\top v - H^\top \Pi_B v$. So, transposing both sides:

$$
H^\top v - H^\top \Pi_B v \;=\; S_p^{-1} \cdot \frac{\mathcal{X}_p^\top Z^\top \eta_p}{\sqrt{n}}. \tag{$\dagger$}
$$

**Step 3: Bound the right side.** The $i$-th entry of $(\dagger)$ is

$$
\left[H^\top v - H^\top \Pi_B v\right]_i \;=\; \frac{\chi_{p,i}^\top Z^\top \eta_p}{s_{p,i}\sqrt{n}}
\;=\; \frac{1}{s_{p,i}\sqrt{n}} \sum_{l=1}^n (\chi_{p,i})_l \cdot (\eta_p^\top Z_{\cdot l}).
$$

For each $l$: $|(\chi_{p,i})_l| \le |\chi_{p,i}| = 1$ (since $\chi_{p,i}$ is a unit vector). By Lemma A.1$''$ applied to the bounded vector $\eta_p$ (recall $|\eta_p| \le 1 \le C$), we have $|\eta_p^\top Z_{\cdot l}| = o(\sqrt{p})$ a.s. Summing over the $n$ (fixed) terms:

$$
\left|\chi_{p,i}^\top Z^\top \eta_p\right| \;\le\; \sum_{l=1}^n |(\chi_{p,i})_l| \cdot |\eta_p^\top Z_{\cdot l}|
\;\le\; n \cdot \max_l |\eta_p^\top Z_{\cdot l}| \;=\; o(\sqrt{p}) \quad \text{a.s.}
$$

By Lemma A.2$'$ Part 4, $s_{p,i} \asymp \sqrt{p}$ a.s. (since $s_{p,i}/\sqrt{p} \to \sqrt{\lambda_i} > 0$), so $1/s_{p,i} = O(1/\sqrt{p})$ a.s. Therefore:

$$
\left|\left[H^\top v - H^\top \Pi_B v\right]_i\right| \;=\; \frac{o(\sqrt{p})}{s_{p,i}\sqrt{n}}
\;=\; \frac{o(\sqrt{p})}{O(\sqrt{p})} \;=\; o(1) \quad \text{a.s.}
$$

**Step 4: Conclusion.** This holds for each $i = 1, \ldots, k$ simultaneously (a finite intersection of probability-one events). Hence $H^\top v - H^\top \Pi_B v \to 0$ a.s. $\square$

**Specialising to $v = z$.** Setting $v = z = e/\sqrt{p}$ (which satisfies $|z| = 1$) gives $H^\top z - H^\top \Pi_B z \to 0$ a.s. — the dispersion-bias-specific statement. The proof required nothing about $z$ beyond $|z| = 1$.

**Why Assumption 2.5$'$ was not needed.** The key algebraic step was $\Pi_B^\perp B = 0$, which holds for any $B$ whose columns span $\mathcal{B}$ — orthogonal or not. An earlier proof (following the paper's $k=1$ argument directly) dotted the SVD identity with each basis vector $b_m$, which required $b_m^\top B$ to collapse to a scaled standard basis row under Assumption 2.5$'$. The projection argument bypasses this entirely.

------------------------------------------------------------------------

## 6. Proof of Part (ii)

*Adds Assumptions 2.5$'$ and 2.6$'$. Uses Lemma A.2$'$ Parts 2–4.*

**Strategy.** Project the SVD identity *onto* $\mathcal{B}$ (rather than onto $\mathcal{B}^\perp$), left-multiply by $\tilde B^\top$, and take limits using Lemma A.2$'$.

**Step 1: Project onto $\mathcal{B}$.** Apply $\Pi_B$ to $(33')$:

$$
\Pi_B H S_p \;=\; \frac{\Pi_B B F^\top \mathcal{X}_p}{\sqrt{n}} \;+\; \frac{\Pi_B Z \mathcal{X}_p}{\sqrt{n}}.
$$

Now $\Pi_B B = B$ (since columns of $B$ lie in $\mathcal{B}$). Right-multiply by $S_p^{-1}$:

$$
\Pi_B H \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt{n}} S_p^{-1} \;+\; \underbrace{\frac{\Pi_B Z \mathcal{X}_p}{\sqrt{n}} S_p^{-1}}_{\text{noise term} =:\, \mathcal{E}_p}. \tag{$\star\star$}
$$

**Step 2: Noise term vanishes.** The $(i,j)$ entry of $\mathcal{E}_p$ is $b_i^\top Z \chi_{p,j}/(s_{p,j}\sqrt{n})$. Using $b_i^\top Z \chi_{p,j} = \sum_{l=1}^n (\chi_{p,j})_l (b_i^\top Z_{\cdot l})$, and applying Lemma A.1$''$ to the unit vector $\eta_p = b_i$ (which has $|b_i| = 1 \le C$): $|b_i^\top Z_{\cdot l}| = o(\sqrt{p})$ a.s. By the same argument as Part (i) Step 3, the full entry is $o(\sqrt{p})/s_{p,j} = o(1)$ a.s. All $k^2$ entries of $\mathcal{E}_p$ vanish a.s., so $\mathcal{E}_p \to 0$ a.s.

**Step 3: Left-multiply by $\tilde B^\top$.** Since $\tilde B^\top \Pi_B = \tilde B^\top$ (the identity $\Pi_B \tilde B = \tilde B$ holds because $\tilde B \subset \mathcal{B}$, and transposing gives $\tilde B^\top \Pi_B = \tilde B^\top$), left-multiplying $(\star\star)$ by $\tilde B^\top$:

$$
\tilde B^\top H \;=\; \underbrace{\tilde B^\top B}_{\text{(a)}} \cdot \underbrace{\frac{F^\top \mathcal{X}_p}{\sqrt{n}}}_{\text{(b)}} \cdot \underbrace{S_p^{-1}}_{\text{(c)}} \;+\; o(1).
$$

**Step 4: Evaluate each factor.** We now take limits in each of (a), (b), (c).

*(a) $\tilde B^\top B$:* Under Assumption 2.5$'$, $b_i^\top \beta_j = (|\beta_i|/|\beta_i|)\delta_{ij}|\beta_j| = |\beta_j|\delta_{ij}$. So

$$
\tilde B^\top B \;=\; \operatorname{diag}(|\beta_1|, \ldots, |\beta_k|).
$$

Rescaling: $\tilde B^\top B = \sqrt{p} \cdot \operatorname{diag}(|\beta_j|/\sqrt{p}) \to \sqrt{p} \cdot \operatorname{diag}(\alpha_j)$ a.s.

*(b) $F^\top \mathcal{X}_p / \sqrt{n}$:* The $(i,j)$ entry is $X_i^\top \chi_{p,j}/\sqrt{n}$. By Lemma A.2$'$ Part 3, $\chi_{p,j} \to X_j/|X_j|$ a.s. By Assumption 2.6$'$, $X_i^\top X_j/|X_j| = |X_j|\delta_{ij}$. Hence

$$
\frac{X_i^\top \chi_{p,j}}{\sqrt{n}} \;\to\; \frac{X_i^\top X_j}{|X_j|\sqrt{n}} \;=\; \frac{|X_j|\delta_{ij}}{\sqrt{n}}.
$$

So $F^\top \mathcal{X}_p / \sqrt{n} \to \operatorname{diag}(|X_j|/\sqrt{n})$ a.s.

*(c) $S_p^{-1}$:* By Lemma A.2$'$ Part 4, $\sqrt{p}/s_{p,j} \to 1/\sqrt{\lambda_j}$ a.s. So $S_p^{-1} = (1/\sqrt{p}) \cdot \operatorname{diag}(\sqrt{p}/s_{p,j}) \to (1/\sqrt{p}) \cdot \operatorname{diag}(1/\sqrt{\lambda_j})$ a.s.

**Step 5: Assemble the limit.** Combining the three factors and the $\sqrt{p}$ from (a):

$$
(\tilde B^\top H)_{ij} \;\to\; \alpha_i \cdot \frac{|X_j|\delta_{ij}}{\sqrt{n}} \cdot \frac{1}{\sqrt{\lambda_j}} \;=\; \frac{\alpha_j |X_j|}{\sqrt{n\lambda_j}} \cdot \delta_{ij}.
$$

Using $n\lambda_j = \alpha_j^2|X_j|^2 + \delta^2$ (Lemma A.2$'$ Part 2):

$$
\frac{\alpha_j |X_j|}{\sqrt{n\lambda_j}} \;=\; \frac{\alpha_j |X_j|}{\sqrt{\alpha_j^2|X_j|^2 + \delta^2}} \;=:\; \psi_{\infty,j} \;\in\; (0,1).
$$

The positivity $\psi_{\infty,j} > 0$ holds a.s. since $\alpha_j > 0$, $|X_j| > 0$ a.s. (Assumptions 2.1$'$, 2.2$'$). The upper bound $\psi_{\infty,j} < 1$ holds because $\delta > 0$. Therefore $\tilde B^\top H \to \operatorname{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k})$ a.s.

Transposing: $H^\top \tilde B \to \operatorname{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k})$ a.s. $\square$

**Sign and labeling convention.** Davis–Kahan gives $\chi_{p,i} \to \pm X_i/|X_i|$; the sign is fixed by the sign convention $\mu_p(h_i) \ge 0$. From the SVD identity, left-multiplying by $z^\top$: $\mu_p(h_i) s_{p,i} \approx \sum_j |\beta_j|\langle b_j, z\rangle_p X_j^\top\chi_{p,i}/\sqrt{n}$. As $p \to \infty$, the diagonal term $j = i$ dominates and converges to a positive quantity $\psi_{\infty,i}c_i\sqrt{\lambda_i p}$ (where $c_i := \langle b_i, z\rangle_\infty > 0$ is the asymptotic factor-$i$ loading on the equal-weight portfolio, defined precisely in §8). Since $\mu_p(h_i) \ge 0$ and $s_{p,i} > 0$, we must have $X_i^\top\chi_{p,i} > 0$ for large $p$, confirming $\chi_{p,i} \to +X_i/|X_i|$.

For labeling: the $i$-th sample direction $h_i$ is matched to the $i$-th population direction $b_i$ via the singular value ordering. The singular values $s_{p,1} \ge \cdots \ge s_{p,k}$ converge to $\sqrt{\lambda_1 p} \ge \cdots \ge \sqrt{\lambda_k p}$, and the $\lambda_i$ are ordered by $\alpha_i^2|X_i|^2$. This naturally pairs the $i$-th sample singular vector with the population factor of $i$-th largest signal strength.

------------------------------------------------------------------------

## 7. Worked Example: $k = 2$

Let $k = 2$, so all matrices are $2 \times 2$ or $2$-vectors. The identity $(\dagger)$ from Part (i) reads:

$$
H^\top v - H^\top \Pi_B v
\;=\;
\begin{pmatrix} 1/s_{p,1} & 0 \\ 0 & 1/s_{p,2} \end{pmatrix}
\begin{pmatrix} \chi_{p,1}^\top \\ \chi_{p,2}^\top \end{pmatrix}
Z^\top \eta_p \cdot \frac{1}{\sqrt{n}}.
$$

Setting $v = z$:

$$
\begin{pmatrix} \langle h_1, z\rangle_p \\ \langle h_2, z\rangle_p \end{pmatrix}
-
\begin{pmatrix} \langle h_1, b_1\rangle_p\langle b_1, z\rangle_p + \langle h_1, b_2\rangle_p\langle b_2, z\rangle_p \\
                \langle h_2, b_1\rangle_p\langle b_1, z\rangle_p + \langle h_2, b_2\rangle_p\langle b_2, z\rangle_p \end{pmatrix}
\;\to\; 0.
$$

This says $h_1$ and $h_2$ jointly "see" $z$ only through its projection onto the 2-dimensional factor subspace. Part (ii) additionally constrains the cross-terms: under Assumptions 2.5$'$ and 2.6$'$, $\langle h_1, b_2\rangle_\infty = 0$ and $\langle h_2, b_1\rangle_\infty = 0$, so the matrix $H^\top\tilde B$ is asymptotically diagonal:

$$
H^\top \tilde B \;\to\; \begin{pmatrix} \psi_{\infty,1} & 0 \\ 0 & \psi_{\infty,2} \end{pmatrix}.
$$

**Concrete numbers.** Take $\alpha_1 = \alpha_2 = 1$, $|X_1|^2 = |X_2|^2 = n = 50$, $\delta^2 = 1$. Then $n\lambda_i = 50 + 1 = 51$, so $\psi_{\infty,i}^2 = 50/51 \approx 0.980$. Each sample direction recovers $98\%$ of its population direction's variance.

Now take $n = 5$ (small $n$, the regime where dispersion bias matters): $n\lambda_i = 5 + 1 = 6$, $\psi_{\infty,i}^2 = 5/6 \approx 0.833$. The sample frame recovers only $83\%$ of each direction. With $c_1 = c_2 = c$ (say, equal exposures):

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\approx\; 2 \cdot (1 - 5/6) \cdot c^2 \;=\; \frac{c^2}{3}.
$$

A third of the factor exposure is "lost" to noise — this is the dispersion bias.

------------------------------------------------------------------------

## 8. Corollary: The Grassmannian Dispersion Bias

**Corollary.** *Under Assumptions 2.1$'$–2.6$'$, almost surely as $p \to \infty$:*

$$
|\Pi_B z|^2 \;\to\; \sum_{i=1}^k c_i^2, \qquad |\Pi_H z|^2 \;\to\; \sum_{i=1}^k \psi_{\infty,i}^2 c_i^2,
$$

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\to\; \sum_{i=1}^k (1 - \psi_{\infty,i}^2)\,c_i^2 \;>\; 0 \quad \text{a.s.,}
$$

where $c_i := \mu_\infty(\beta_i)/\alpha_i = 1/\sqrt{1 + d_\infty^2(\beta_i)} \in (0, 1]$.

**Proof.**

**Step 1: Limit of $\langle b_i, z \rangle_p$.** Write

$$
\langle b_i, z \rangle_p \;=\; \frac{e^\top \beta_i / p}{|\beta_i|/\sqrt{p}}
\;\to\; \frac{\mu_\infty(\beta_i)}{\alpha_i} \;=:\; c_i \;>\; 0 \quad \text{a.s.}
$$

using Assumption 2.2$'$. Note $c_i = 1/\sqrt{1 + d_\infty^2(\beta_i)} \in (0,1]$. Intuitively, $c_i$ measures the "average loading direction" of factor $i$: if all loadings point the same way ($d_\infty = 0$), then $c_i = 1$ (maximum exposure to the equal-weight portfolio); if loadings are dispersed ($d_\infty > 0$), then $c_i < 1$.

**Step 2: Limit of $|\Pi_B z|^2$.** Since $\Pi_B = \tilde B \tilde B^\top$ and $\tilde B^\top \tilde B = I_k$:

$$
|\Pi_B z|^2 \;=\; |\tilde B^\top z|^2 \;=\; \sum_{i=1}^k \langle b_i, z\rangle_p^2 \;\to\; \sum_{i=1}^k c_i^2 \quad \text{a.s.}
$$

**Step 3: Limit of $|\Pi_H z|^2$.** Since $\Pi_H = HH^\top$:

$$
|\Pi_H z|^2 \;=\; |H^\top z|^2 \;=\; \sum_{i=1}^k \langle h_i, z\rangle_p^2.
$$

By Part (i), $H^\top z \to H^\top \Pi_B z$ a.s. By Part (ii), $H^\top \tilde B \to \operatorname{diag}(\psi_{\infty,i})$ a.s. Therefore:

$$
[H^\top \Pi_B z]_i \;=\; \sum_{j=1}^k (H^\top \tilde B)_{ij}\,\langle b_j, z\rangle_p
\;\to\; \psi_{\infty,i}\, c_i.
$$

So $H^\top z \to (\psi_{\infty,1}c_1, \ldots, \psi_{\infty,k}c_k)^\top$ a.s., and $|\Pi_H z|^2 \to \sum_i \psi_{\infty,i}^2 c_i^2$ a.s.

**Step 4: Positivity of the gap.** Each summand in the deficit:

$$
(1 - \psi_{\infty,i}^2)\,c_i^2 \;=\; \frac{\delta^2}{\alpha_i^2|X_i|^2 + \delta^2}\,c_i^2 \;>\; 0 \quad \text{a.s.,}
$$

since $\delta > 0$, $c_i > 0$. The deficit is strictly positive a.s. $\square$

**Interpretation.** The equal-weight portfolio $z$ has exposure $\sum_i c_i^2$ to the population factor subspace. The sample frame $H$ can only capture a fraction $\psi_{\infty,i}^2 < 1$ of each factor's contribution — the remainder $\delta^2/(\alpha_i^2|X_i|^2 + \delta^2)$ is lost to noise. This per-factor bias:

- *Increases* with noise level $\delta^2$: more noise, less reliable sample directions.
- *Decreases* with signal strength $\alpha_i^2|X_i|^2$: stronger factor, better recovery.
- *Is independent across factors* (under 2.5$'$, 2.6$'$): each factor contributes its own deficit term, which simply sum.

This is the $k$-factor generalization of the dispersion bias in the paper's Theorem 3.1.

------------------------------------------------------------------------

## 9. Reduction to $k = 1$

Setting $k = 1$ collapses every matrix to a scalar or vector:

- $H = h \in \mathbb{R}^p$, $\tilde B = b$, $S_p = s_p$, $\mathcal{X}_p = \chi_p \in \mathbb{R}^n$.
- $\Pi_B = bb^\top$ (rank-1 projector).
- $\Pi_H = hh^\top$.

Part (i) at $v = z$ becomes $\langle h, z\rangle - \langle h, b\rangle\langle b, z\rangle \to 0$ a.s., which is the first half of the paper's equation (13).

Part (ii) becomes $\langle h, b\rangle_\infty = \alpha|X|/\sqrt{\alpha^2|X|^2 + \delta^2} = \psi_\infty$, matching the paper's formula (using $\nu_X^2 = \alpha^2|X|^2/n$: the paper writes $\psi_\infty = \sqrt{n\nu_X^2/(n\nu_X^2 + \delta^2)}$ and $n\nu_X^2 = \alpha^2|X|^2$).

The proof collapses to the paper's argument line-by-line: $(33')$ is the paper's (33); the projection onto $\mathcal{B}^\perp = b^\perp$ in Part (i) is the paper's Lemma A.1 argument; Davis–Kahan in Part 3 of Lemma A.2$'$ is trivial when $k = 1$ (there is a unique maximizer of a rank-1 operator). What is genuinely new at $k > 1$ is: (a) the projection argument for Part (i) (no analog was needed at $k = 1$), (b) Assumption 2.5$'$ (nothing to be orthogonal to when $k = 1$), and (c) Davis–Kahan for $k > 1$ eigenvalue gaps.

------------------------------------------------------------------------

## 10. What Changed Relative to the First Draft, and Why

### 10.1 The Earlier Approach

The first draft (and the paper's $k = 1$ argument) proceeded by dotting the SVD identity $(33')$ with each population basis vector $b_m$ and with $z$, obtaining $k+1$ scalar equations:

$$
\langle h_i, b_m\rangle_p s_{p,i} \;=\; |\beta_m|\frac{X_m^\top\chi_{p,i}}{\sqrt{n}} + \frac{b_m^\top Z\chi_{p,i}}{\sqrt{n}},
\tag{$34'$}
$$

$$
\langle h_i, z\rangle_p s_{p,i} \;=\; \sum_j\langle b_j, z\rangle_p |\beta_j|\frac{X_j^\top\chi_{p,i}}{\sqrt{n}} + \frac{z^\top Z\chi_{p,i}}{\sqrt{n}}.
\tag{$35'$}
$$

Substituting $(34')$ into $(35')$, the signal sum became $\sum_j \langle b_j, z\rangle_p \langle h_i, b_j\rangle_p s_{p,i} = [H^\top \Pi_B z]_i s_{p,i}$ (using Assumption 2.5$'$ to make $b_j^\top B = |\beta_j|e_j^\top$, where $e_j \in \mathbb{R}^k$ is the $j$-th standard basis vector), and the residual was $k + 1$ noise terms, each controlled by Lemma A.1.

### 10.2 Why the New Approach Is Better

**Advantage 1: Part (i) drops Assumption 2.5$'$.** The old argument needed $b_m^\top B$ to collapse via orthogonality. The new argument needs only $\Pi_B^\perp B = 0$ — true for any $B$ whose columns span $\mathcal{B}$, orthogonal or not. This is not a technicality: the numerical experiment showed the matrix residual decays at rate $p^{-1/2}$ even with deliberately correlated raw loadings, and now the proof explains why.

**Advantage 2: Part (i) holds for any bounded $v$, not just $z$.** The old argument used specific properties of $z$ (via $z^\top\beta_j = |\beta_j|\langle b_j, z\rangle_p$). The new argument uses only $|\eta_p| = |\Pi_B^\perp v| \le |v| \le 1$.

**Advantage 3: Part (i) is shorter.** The proof is roughly half a page versus several pages. The mathematical depth is unchanged; the algebraic overhead was eliminated.

**Advantage 4: Cleaner separation.** Part (i) is coordinate-free (no basis needed for $\mathcal{B}$). Part (ii) genuinely needs the per-direction structure — Davis–Kahan is irreducible. The old approach blurred this distinction by using both $b_m$ and $z$ in the same calculation.

**Why Part (ii) still needs 2.5$'$ and 2.6$'$.** Part (ii) asks: which sample direction corresponds to which population direction, and with what shrinkage? To answer this, we need the individual limit singular vectors $\xi_i = X_i/|X_i|$ (from Davis–Kahan via 2.6$'$), and we need the matrix $\tilde B^\top B$ to be diagonal (from 2.5$'$). Removing either assumption makes the limit of $H^\top\tilde B$ a more complex function of the eigenvectors of $FM_\infty F^\top$, rather than a clean diagonal.

------------------------------------------------------------------------

## 11. Assumption Accounting

| Assumption | Content                                                        | First used in                                                                            |
|:---------- |:-------------------------------------------------------------- |:---------------------------------------------------------------------------------------- |
| 2.1$'$     | Moments, $F \perp Z$, $X_j \ne 0$ a.s., atomless distributions | Lemma A.1 (cross terms), Lemma A.2$'$ Parts 1, 3                                         |
| 2.2$'$     | $\|\beta_j\|^2/p \to \alpha_j^2$, loading regularity           | Lemma A.1 (hypothesis), Part (ii) Step 4(a), Corollary Step 1                            |
| 2.3$'$     | Pairwise independence, bounded 4th moments                     | Lemma A.1, Lemma A.1$''$                                                                 |
| 2.4$'$     | $k < n$                                                        | Dimensional feasibility throughout                                                       |
| 2.5$'$     | Orthogonal loading columns                                     | Lemma A.2$'$ Part 1 ($M_\infty$ diagonal), Part (ii) Step 3 ($\tilde B^\top B$ diagonal) |
| 2.6$'$     | Orthogonal factor returns                                      | Lemma A.2$'$ Parts 2–3 (explicit $\lambda_i$, Davis–Kahan target), Part (ii) Step 4(b)   |

**Part (i)** uses only 2.1$'$–2.4$'$. **Part (ii)** additionally requires 2.5$'$ and 2.6$'$. The **Corollary** uses all six.

**Notes on relaxations.**

- *Assumption 2.5$'$ in Part (i).* The projection argument shows 2.5$'$ is unnecessary for Part (i): the key step $\Pi_B^\perp B = 0$ holds for any $B$ whose columns span $\mathcal{B}$, orthogonal or not. Removing 2.5$'$ from Part (i) is the principal structural gain over the earlier dot-product approach.

- *Assumption 2.5$'$ in Part (ii).* It is used in two places: (a) to conclude $B^\top B/p \to M_\infty = \operatorname{diag}(\alpha_j^2)$ in Lemma A.2$'$ Part 1 (without orthogonality, $B^\top B/p$ converges to a full $k \times k$ matrix, not a diagonal one), and (b) to obtain the diagonal form $\tilde B^\top B = \operatorname{diag}(|\beta_j|)$ in Part (ii) Step 3. Removing 2.5$'$ from Part (ii) would change $\psi_{\infty,i}$ from a scalar shrinkage factor to a more complex expression involving the eigenvectors of $FM_\infty^{\mathrm{full}} F^\top$, losing the clean closed form.

- *Assumption 2.6$'$ in Part (ii).* Used to establish that the signal matrix $FM_\infty F^\top$ has rank-1 components pointing in the directions $X_j/|X_j|$, and that these directions are distinct (spectral gap). Without 2.6$'$, Davis–Kahan still
