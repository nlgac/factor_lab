# A final reflection

The projection argument is genuinely better than the dot-product-and-substitute approach: shorter, more general (Part (i) drops Assumption 2.5$'$), and more transparent (it makes geometrically vivid what the result *is*). Before committing to it, let me ask whether there's a still-better argument I'm missing.

**Could the proof be even shorter?** The key inequality is $|H^\top \eta| \le |S_p^{-1}|_{\text{op}} \cdot |Z^\top \eta|/\sqrt{n}$. The first factor is bounded by $|S_p^{-1}| \asymp 1/\sqrt{p}$ (Lemma A.2$'$). The second is the norm of an $n$-vector each of whose entries has variance $\le \delta^2$, so $|Z^\top \eta| = O_P(\sqrt{n}) = O_P(1)$ since $n$ is fixed. Multiplying: $|H^\top \eta| = O_P(1/\sqrt{p}) \to 0$. Upgrading to a.s. via Borel–Cantelli with bounded fourth moments gives a.s. convergence at rate $p^{-1/2 + \epsilon}$.

That's the entire proof in three sentences. I don't see how to make it shorter without losing readability.

**Is there a still more general statement?** Yes, actually. The proof never uses that $z$ is the equal-weight vector — only that $|z| \le 1$ (a fixed bounded vector). So Part (i) holds for any deterministic bounded $v \in \mathbb{R}^p$:

$$
H^\top v - H^\top \Pi_B v \;\to\; 0 \quad \text{a.s.}
$$

This is a strictly stronger result. The "dispersion bias" interpretation enters only through the *specific* choice $v = z = e/\sqrt{p}$, which makes $\langle b_i, z\rangle_\infty$ have the dispersion-related limit $c_i$.

I'll state this generalization explicitly in the proof — it's free, and it clarifies what's structural versus what's specific to the dispersion application.

**Could Part (ii) be shortened further?** I sketched it above. The argument reduces $\tilde B^\top H S_p$ to a matrix limit via Lemma A.2$'$, and Assumption 2.6$'$ collapses $F^\top \mathcal{X}_p$ to a diagonal. I don't see a cleaner version. The Davis–Kahan invocation is irreducible — without it, individual right singular vectors don't converge, and the per-direction statement of Part (ii) doesn't make sense.

I'm satisfied this is the proof to write. Let me also fix the markdown rendering issue you flagged — I'll avoid blockquotes containing display math.

---

# Proof of Theorem 3.1$'$: $k$-Factor Generalization of the Dispersion Bias

*Final consolidated version. Uses the projection-onto-$\mathcal{B}^\perp$ argument for Part (i), achieving a substantially shorter proof and dropping Assumption 2.5$'$ from Part (i)'s hypotheses.*

## Statement

Under Assumptions 2.1$'$–2.4$'$, almost surely as $p \to \infty$:

$$
\textbf{(Part i)} \qquad H^\top v \;-\; H^\top \Pi_B v \;\longrightarrow\; 0 \quad \text{for any deterministic } v \in \mathbb{R}^p \text{ with } |v| \le 1.
$$

Under the additional Assumptions 2.5$'$–2.6$'$:

$$
\textbf{(Part ii)} \qquad H^\top \tilde B \;\longrightarrow\; \operatorname{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k}),
$$

where each $\psi_{\infty,i} = \alpha_i |X_i| / \sqrt{\alpha_i^2 |X_i|^2 + \delta^2} \in (0, 1)$ a.s.

**Two notes on what this says.** First, Part (i) holds for *any* deterministic bounded vector $v$, not just the dispersionless $z = e/\sqrt{p}$. The matrix factorization is a structural statement about how $H$ relates to the population subspace $\mathcal{B}$, independent of which probe vector we choose. Second, **Part (i) does not require orthogonality of $B$'s columns** (Assumption 2.5$'$). It holds for arbitrary loading correlation structures — confirming what our numerics suggested and what an earlier proof had hidden behind unnecessary algebra.

## Setup

The model is $Y = B F^\top + Z$, dimensions $p \times n$, with $B \in \mathbb{R}^{p\times k}$, $F \in \mathbb{R}^{n\times k}$, $Z \in \mathbb{R}^{p\times n}$. Asymptotics: $p \to \infty$, $n$ and $k$ fixed.

Let $\mathcal{B} := \mathrm{span}(B) \subset \mathbb{R}^p$, with orthogonal projector $\Pi_B$ and complement projector $\Pi_B^\perp = I - \Pi_B$. Let $\tilde B$ denote any orthonormal basis for $\mathcal{B}$ (under Assumption 2.5$'$, the natural choice is $\tilde B = B \cdot \operatorname{diag}(|\beta_j|^{-1})$, but the choice doesn't affect $\Pi_B$).

The thin SVD of $Y/\sqrt{n}$ gives $H \in \mathbb{R}^{p\times k}$ (orthonormal top-$k$ left singular vectors), $\mathcal{X}_p \in \mathbb{R}^{n\times k}$ (orthonormal right singular vectors), and $S_p = \operatorname{diag}(s_{p,1}, \ldots, s_{p,k})$ in decreasing order. The fundamental identity is

$$
H S_p \;=\; \frac{Y \mathcal{X}_p}{\sqrt{n}} \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt{n}} \;+\; \frac{Z \mathcal{X}_p}{\sqrt{n}}, \tag{$33'$}
$$

an exact $p \times k$ matrix identity for every finite $p$.

## Assumptions

For Part (i):

- **2.1$'$.** Entries of $Z$ are mean-zero with variance $\delta^2 > 0$, factor returns $X_j$ are mean-zero, $F$ and $Z$ are independent, each $X_j \ne 0$ a.s.
- **2.2$'$.** For each $j$, $\mu_p(\beta_j) \to \mu_\infty(\beta_j) \in (0, \infty)$ and $d_p(\beta_j) \to d_\infty(\beta_j) \in [0, \infty)$, giving $|\beta_j|^2/p \to \alpha_j^2 = \mu_\infty^2(\beta_j)(1+d_\infty^2(\beta_j))$.
- **2.3$'$.** Entries of $Z$ are pairwise independent with uniformly bounded fourth moments.
- **2.4$'$.** $k < n$.

For Part (ii), additionally:

- **2.5$'$.** $B$'s columns are orthogonal: $\beta_m^\top \beta_j = |\beta_j|^2 \delta_{mj}$.
- **2.6$'$.** $F$'s columns are orthogonal: $X_i^\top X_j = |X_i|^2 \delta_{ij}$.

## Lemmas

**Lemma A.1 (paper's Lemma A.1).** For deterministic $\eta \in \mathbb{R}^p$ with $|\eta|^2/p \in (c, C)$ for $p$ large, and under Assumption 2.3$'$, $(\eta^\top Z)_l / (|\eta|\sqrt{p}) \to 0$ a.s. for each $l$.

---

**Lemma A.1$''$ (bounded-$\eta$ companion).** *Let $\eta_p \in \mathbb{R}^p$ be a deterministic sequence with $|\eta_p| \le C$ uniformly in $p$. Under Assumption 2.3$'$, for each fixed $l \in \{1, \ldots, n\}$:*

$$
\frac{(\eta_p^\top Z)_l}{\sqrt{p}} \;\to\; 0 \quad \text{a.s. as } p \to \infty.
$$

*More precisely, $(\eta_p^\top Z)_l = o(p^{1/2 - \epsilon})$ a.s. for any fixed $\epsilon \in (0, 1/4)$.*

**Why Lemma A.1 does not apply here.** The paper's Lemma A.1 handles vectors $\eta$ whose norm grows like $\sqrt{p}$: it requires $|\eta|^2/p \asymp 1$ and normalizes by $|\eta|\sqrt{p} \asymp p$. In Part (i), the noise vector is $\eta_p = \Pi_B^\perp v$ with $|v| \le 1$, so $|\eta_p| \le 1$: *bounded*, not growing. Mechanically, one could divide by $|\eta_p|\sqrt{p}$ and get convergence trivially (since $|\eta_p| \le C$ while $\sqrt{p} \to \infty$). But what the proof actually needs is that $(\eta_p^\top Z)_l$ itself is $o(\sqrt{p})$ — a statement about the size of the numerator, not the ratio. Lemma A.1$''$ provides exactly this.

**Intuition.** Write $W_p := (\eta_p^\top Z)_l = \sum_{i=1}^p (\eta_p)_i\, Z_{il}$. This is a weighted sum of $p$ mean-zero random variables. The key observation is that although there are $p$ terms, the total "weight budget" is bounded: $\sum_{i=1}^p (\eta_p)_i^2 = |\eta_p|^2 \le C^2$, fixed independently of $p$. So as $p$ grows, each individual coefficient $|(\eta_p)_i|$ must become small on average — the weight is spread over more and more terms. The variance of $W_p$ is therefore $\operatorname{Var}(W_p) = \sum_i (\eta_p)_i^2 \delta^2 = |\eta_p|^2 \delta^2 \le C^2\delta^2$, bounded for all $p$. A sum that stays bounded in $L^2$ while having $p$ terms is $o(\sqrt{p})$ — compare a sample mean $\bar{Z}_{\cdot l} = (1/p)\sum_i Z_{il}$ which has variance $\delta^2/p \to 0$; here the situation is similar with a different weighting. The a.s. conclusion requires upgrading from $L^2$ to a.s. via the fourth moment and Borel–Cantelli.

**Proof.** Fix $l$ and write $W_p := (\eta_p^\top Z)_l = \sum_{i=1}^p a_i Z_{il}$, where $a_i := (\eta_p)_i$. We have $\sum_i a_i^2 = |\eta_p|^2 \le C^2$.

**Step 1: Compute the fourth moment.** Expand $W_p^4 = \bigl(\sum_i a_i Z_{il}\bigr)^4 = \sum_{i_1, i_2, i_3, i_4} a_{i_1}a_{i_2}a_{i_3}a_{i_4} Z_{i_1 l}Z_{i_2 l}Z_{i_3 l}Z_{i_4 l}$.

Taking expectations and using that the $Z_{il}$ are mean-zero and (fully) independent within each column, a term $\mathbb{E}[Z_{i_1 l}Z_{i_2 l}Z_{i_3 l}Z_{i_4 l}]$ is nonzero only when no index appears exactly once (otherwise factoring out that mean-zero variable gives zero). The surviving patterns are:

- *All four indices equal* ($i_1 = i_2 = i_3 = i_4 = i$): contributes $\sum_i a_i^4\, \mathbb{E}[Z_{il}^4]$.
- *Exactly two distinct values, each appearing twice* (three such pairings of $\{1,2,3,4\}$: $\{\{1,2\},\{3,4\}\}$, $\{\{1,3\},\{2,4\}\}$, $\{\{1,4\},\{2,3\}\}$): contributes $3 \sum_{i \ne j} a_i^2 a_j^2\, \mathbb{E}[Z_{il}^2]\,\mathbb{E}[Z_{jl}^2]$.

Therefore:

$$
\mathbb{E}[W_p^4] \;=\; \sum_i a_i^4\, \mathbb{E}[Z_{il}^4] \;+\; 3 \sum_{i \ne j} a_i^2 a_j^2\, \mathbb{E}[Z_{il}^2]\,\mathbb{E}[Z_{jl}^2].
$$

Now bound each piece using $\mathbb{E}[Z_{il}^4] \le M$ (Assumption 2.3$'$) and $\mathbb{E}[Z_{il}^2] = \delta^2$:

- Diagonal: $\displaystyle\sum_i a_i^4 \le \Bigl(\max_i |a_i|\Bigr)^2 \sum_i a_i^2 \le |\eta_p|^2 \cdot |\eta_p|^2 = |\eta_p|^4 \le C^4$,
  
  where we used $\max_i |a_i| \le \sqrt{\sum_i a_i^2} = |\eta_p|$.

- Off-diagonal: $\displaystyle\sum_{i \ne j} a_i^2 a_j^2 \le \Bigl(\sum_i a_i^2\Bigr)^2 = |\eta_p|^4 \le C^4$.

Combining:

$$
\mathbb{E}[W_p^4] \;\le\; C^4 M + 3 C^4 \delta^4 \;=:\; K \;<\; \infty,
$$

*uniformly in $p$*. This is the crucial conclusion of Step 1: even though $W_p$ is a sum of $p$ terms, its fourth moment does not grow with $p$. The bounded-norm condition $|\eta_p| \le C$ is what prevents blowup — it ensures the coefficient vector cannot concentrate all its weight on a single large entry as $p$ grows.

**Step 2: Markov's inequality.** Applying the fourth-moment Markov inequality with threshold $t = p^{1/2 - \epsilon}$:

$$
\Pr\!\bigl(|W_p| > p^{1/2-\epsilon}\bigr) \;\le\; \frac{\mathbb{E}[W_p^4]}{(p^{1/2-\epsilon})^4} \;=\; \frac{\mathbb{E}[W_p^4]}{p^{2-4\epsilon}} \;\le\; \frac{K}{p^{2-4\epsilon}}.
$$

**Step 3: Borel–Cantelli.** The series $\sum_{p=1}^\infty K/p^{2-4\epsilon}$ is a $p$-series with exponent $2-4\epsilon$, which converges if and only if $2 - 4\epsilon > 1$, i.e., $\epsilon < 1/4$. For any such $\epsilon$, the sum is finite and the first Borel–Cantelli lemma gives:

$$
\Pr\!\Bigl(|W_p| > p^{1/2-\epsilon} \text{ for infinitely many } p\Bigr) \;=\; 0,
$$

so $|W_p|/p^{1/2-\epsilon} \to 0$ a.s.

*Why $\epsilon < 1/4$ and not $\epsilon < 1/2$:* Using a second-moment (Chebyshev) bound instead gives $\Pr(|W_p| > p^{1/2-\epsilon}) \le \mathbb{E}[W_p^2]/p^{1-2\epsilon} \le C^2\delta^2/p^{1-2\epsilon}$, summable only when $1 - 2\epsilon > 1$, i.e., $\epsilon < 0$ — useless. The fourth moment is essential for getting a summable series; it yields exponent $2 - 4\epsilon > 1$ when $\epsilon < 1/4$.

**Step 4: Conclusion.** Choose any $\epsilon_0 \in (0, 1/4)$, e.g., $\epsilon_0 = 1/8$. Then $W_p = o(p^{1/2-\epsilon_0})$ a.s., so:

$$
\frac{|W_p|}{\sqrt{p}} \;=\; \frac{|W_p|}{p^{1/2-\epsilon_0}} \cdot p^{-\epsilon_0} \;\to\; 0 \cdot 0 \;=\; 0 \quad \text{a.s.} \qquad \square
$$

**Example.** To see the lemma concretely, consider a one-factor ($k=1$) model with equal loadings $\beta_j = 1$ for all $j$, so the population factor direction is $b = e/\sqrt{p}$. Fix a security $j$ and take probe vector $v = e_j$ (the $j$-th standard basis vector). Its projection onto the factor subspace is $\Pi_B v = \langle b, e_j\rangle\, b = (1/\sqrt{p})(e/\sqrt{p}) = e/p$, so the idiosyncratic residual is:

$$
\eta_p \;=\; \Pi_B^\perp e_j \;=\; e_j \;-\; \frac{e}{p}.
$$

This satisfies $|\eta_p|^2 = 1 - 1/p \le 1$, so $|\eta_p| \le 1$ for all $p$. The key quantity is:

$$
W_p \;=\; \eta_p^\top Z_{\cdot l} \;=\; Z_{jl} \;-\; \frac{1}{p}\sum_{i=1}^p Z_{il} \;=\; Z_{jl} \;-\; \bar{Z}_{\cdot l},
$$

where $\bar{Z}_{\cdot l}$ is the cross-sectional mean of idiosyncratic returns at time $l$. As $p \to \infty$, $\bar{Z}_{\cdot l} \to 0$ a.s. by the strong law of large numbers (mean-zero, bounded variance, pairwise independence), so $W_p \to Z_{jl}$ a.s. — a bounded limit. In particular $W_p = O(1)$ a.s., hence trivially $W_p/\sqrt{p} \to 0$ a.s., confirming the lemma.

The interpretation is clean: $\eta_p$ is security $j$'s "idiosyncratic direction" — the part of $e_j$ orthogonal to the market. The inner product $W_p$ measures how much of column $l$ of $Z$ points in that direction, which is exactly security $j$'s own noise $Z_{jl}$ minus a negligible market-average correction. The lemma says this quantity, though nonzero, is $o(\sqrt{p})$ — small compared to the signal strength $s_{p,i} \asymp \sqrt{p}$, which is why noise cannot overwhelm signal as $p \to \infty$.

---

**Lemma A.2$'$ (spectral convergence).** Under Assumptions 2.1$'$–2.5$'$:

1. $R_p := Y^\top Y/(np) \to A_\infty := F M_\infty F^\top / n + (\delta^2/n) I_n$ in operator norm a.s., with $M_\infty = \operatorname{diag}(\alpha_1^2, \ldots, \alpha_k^2)$.
2. $s_{p,i}^2/p \to \lambda_i$ a.s. *Under 2.6$'$,* $\lambda_i = (\alpha_i^2 |X_i|^2 + \delta^2)/n$.
3. *Under 2.6$'$:* $\chi_{p,i} \to X_i/|X_i|$ a.s., and the $\lambda_i$ are distinct a.s.
4. $\sqrt{p} / s_{p,i} \to 1/\sqrt{\lambda_i} < \infty$ a.s.

*Proof sketch.* Expand $Y^\top Y / (np)$ into signal, cross, and noise terms. Signal: $F(B^\top B/p)F^\top / n \to F M_\infty F^\top / n$ a.s. Cross terms: vanish a.s. by Lemma A.1 applied entrywise. Noise: $Z^\top Z/(np) \to (\delta^2/n) I_n$ a.s. by SLLN. Together gives Part 1. Weyl's inequality gives Part 2. Under 2.6$'$, $F M_\infty F^\top = \sum_j \alpha_j^2 X_j X_j^\top$ has orthogonal rank-1 summands with eigenvectors $X_j/|X_j|$ and distinct eigenvalues $\alpha_j^2|X_j|^2$ a.s.; Davis–Kahan with the spectral gap gives Part 3. Part 4 is immediate. $\square$

## Proof of Part (i)

The argument is short; let me state it inline.

Decompose $v = \Pi_B v + \eta_p$ where $\eta_p := \Pi_B^\perp v$, with $|\eta_p| \le |v| \le 1$ (projectors are non-expansive).

Apply $\Pi_B^\perp$ to both sides of the fundamental identity $(33')$. The signal term vanishes because $\Pi_B^\perp B = 0$ (each column of $B$ lies in $\mathcal{B}$). What remains is

$$
\Pi_B^\perp H \cdot S_p \;=\; \frac{\Pi_B^\perp Z \mathcal{X}_p}{\sqrt{n}}.
$$

Right-multiply by $S_p^{-1}$ and take inner product with $v$:

$$
v^\top \Pi_B^\perp H \;=\; \frac{v^\top \Pi_B^\perp Z \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1} \;=\; \frac{\eta_p^\top Z \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1},
$$

using that $\Pi_B^\perp$ is symmetric, so $v^\top \Pi_B^\perp = (\Pi_B^\perp v)^\top = \eta_p^\top$.

The left side is exactly what we want to control: $v^\top \Pi_B^\perp H = v^\top H - v^\top \Pi_B H$, whose transpose is $H^\top v - H^\top \Pi_B v$. So

$$
H^\top v - H^\top \Pi_B v \;=\; S_p^{-1} \mathcal{X}_p^\top Z^\top \eta_p / \sqrt{n}. \tag{$\dagger$}
$$

Now bound the right side. Its $i$-th entry is

$$
\frac{1}{s_{p,i}\sqrt{n}}\, \chi_{p,i}^\top Z^\top \eta_p \;=\; \frac{1}{s_{p,i}\sqrt{n}} \sum_{l=1}^n (\chi_{p,i})_l \cdot (\eta_p^\top Z_{\cdot l}).
$$

Each summand factors as bounded $\times$ slow-growing: $|(\chi_{p,i})_l| \le 1$ since $|\chi_{p,i}| = 1$, and by Lemma A.1$''$ applied to the bounded vector $\eta_p$, $|\eta_p^\top Z_{\cdot l}| = o(p^{1/2 - \epsilon})$ a.s. for any $\epsilon \in (0, 1/4)$. The sum over $l$ has $n$ (finite) terms, so $|\chi_{p,i}^\top Z^\top \eta_p| = o(p^{1/2 - \epsilon})$ a.s.

By Lemma A.2$'$ Part 4, $s_{p,i} \asymp \sqrt{p}$ a.s., so $1/(s_{p,i}\sqrt{n}) = O(1/\sqrt{p})$ a.s. Combining:

$$
\big| [H^\top v - H^\top \Pi_B v]_i \big| \;=\; O\Big(\frac{p^{1/2 - \epsilon}}{\sqrt{p}}\Big) \;=\; O(p^{-\epsilon}) \;\to\; 0 \quad \text{a.s.}
$$

Holds for each $i = 1, \ldots, k$. The vector $H^\top v - H^\top \Pi_B v \to 0$ a.s. $\square$

**Note.** Setting $v = z = e/\sqrt{p}$ recovers the dispersion-bias-specific statement $H^\top z - H^\top \Pi_B z \to 0$.

## Proof of Part (ii)

The strategy: take limits in the matrix identity $(33')$ projected onto $\mathcal{B}$, using Lemma A.2$'$.

Apply $\Pi_B$ to $(33')$. Now both terms survive: the signal because $\Pi_B B = B$, the noise (projected). After right-multiplying by $S_p^{-1}$:

$$
\Pi_B H \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1} \;+\; \frac{\Pi_B Z \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1}.
$$

The noise term is bounded in operator norm by $\|\Pi_B Z \mathcal{X}_p\|_{\text{op}} / (\sqrt{n}\,s_{p,k})$, and the same Lemma A.1$''$-style argument (apply componentwise to the orthonormal columns of $\tilde B$, each of bounded norm 1) shows this term vanishes a.s. — call this contribution $\varepsilon_p$.

Now left-multiply by $\tilde B^\top$. Since $\tilde B^\top \Pi_B = \tilde B^\top$ (as $\tilde B$ spans $\mathcal{B}$):

$$
\tilde B^\top H \;=\; \tilde B^\top B \cdot \frac{F^\top \mathcal{X}_p}{\sqrt{n}} \cdot S_p^{-1} \;+\; \tilde B^\top \varepsilon_p.
$$

Under Assumption 2.5$'$, $\tilde B^\top B = \operatorname{diag}(|\beta_j|)_j$ (the $j$-th column of $B$ equals $|\beta_j|$ times the $j$-th column of $\tilde B$). Substituting and rescaling:

$$
\tilde B^\top H \;=\; \operatorname{diag}(|\beta_j|/\sqrt{p}) \cdot \frac{F^\top \mathcal{X}_p \sqrt{p}}{\sqrt{n}} \cdot S_p^{-1} \;+\; o(1).
$$

Now take limits factor by factor using Lemma A.2$'$:

- $|\beta_j|/\sqrt{p} \to \alpha_j$ (Assumption 2.2$'$).
- $\mathcal{X}_p \to [X_1/|X_1|, \ldots, X_k/|X_k|]$ a.s. (Lemma A.2$'$ Part 3, requires Assumption 2.6$'$).
- Hence $F^\top \mathcal{X}_p \to F^\top \cdot [X_1/|X_1|, \ldots, X_k/|X_k|]$, with $(i,j)$ entry $X_i^\top X_j / |X_j| = |X_j| \delta_{ij}$ under Assumption 2.6$'$. So $F^\top \mathcal{X}_p \to \operatorname{diag}(|X_j|)$.
- $S_p^{-1} \cdot \sqrt{p} = \operatorname{diag}(\sqrt{p} / s_{p,i}) \to \operatorname{diag}(1/\sqrt{\lambda_i})$ (Lemma A.2$'$ Part 4).

Combining:

$$
(\tilde B^\top H)_\infty = \operatorname{diag}(\alpha_j) \cdot \operatorname{diag}(|X_j|/\sqrt{n}) \cdot \operatorname{diag}(1/\sqrt{\lambda_j}) = \operatorname{diag}\!\Big(\frac{\alpha_j |X_j|}{\sqrt{n \lambda_j}}\Big).
$$

Since $n \lambda_j = \alpha_j^2 |X_j|^2 + \delta^2$ (Lemma A.2$'$ Part 2), the diagonal entries are

$$
\frac{\alpha_j |X_j|}{\sqrt{\alpha_j^2 |X_j|^2 + \delta^2}} \;=\; \psi_{\infty,j} \;\in\; (0, 1) \quad \text{a.s.}
$$

Transposing (which preserves diagonality): $H^\top \tilde B \to \operatorname{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k})$ a.s. $\square$

## Corollary: The Dispersion Bias

Under Assumptions 2.1$'$–2.6$'$, with $c_i := \mu_\infty(\beta_i)/\alpha_i = 1/\sqrt{1+d_\infty^2(\beta_i)} \in (0, 1]$:

$$
|\Pi_B z|^2 \;\to\; \sum_{i=1}^k c_i^2, \qquad |\Pi_H z|^2 \;\to\; \sum_{i=1}^k \psi_{\infty,i}^2 c_i^2,
$$

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\to\; \sum_{i=1}^k (1 - \psi_{\infty,i}^2)\,c_i^2 \;>\; 0 \quad \text{a.s.}
$$

*Proof.* From Assumption 2.2$'$, $\langle b_i, z\rangle_p = (e^\top \beta_i / p) / (|\beta_i|/\sqrt{p}) \to \mu_\infty(\beta_i)/\alpha_i = c_i$. So $|\Pi_B z|^2 = \sum_i \langle b_i, z\rangle_p^2 \to \sum_i c_i^2$.

For $|\Pi_H z|^2 = |H^\top z|^2$: by Part (i) at $v = z$, $H^\top z - H^\top \Pi_B z \to 0$; by Part (ii), $H^\top \tilde B \to \operatorname{diag}(\psi_{\infty,i})$. So $H^\top z \to H^\top \Pi_B z \to \operatorname{diag}(\psi_{\infty,i}) \cdot (c_1, \ldots, c_k)^\top = (\psi_{\infty,1} c_1, \ldots, \psi_{\infty,k} c_k)^\top$, and $|H^\top z|^2 \to \sum_i \psi_{\infty,i}^2 c_i^2$.

Each $1 - \psi_{\infty,i}^2 = \delta^2/(\alpha_i^2 |X_i|^2 + \delta^2) > 0$ a.s. and $c_i > 0$, so the deficit is strictly positive. $\square$

## Reduction to $k = 1$

Setting $k = 1$ collapses every matrix to a scalar. $\Pi_B = bb^\top$, so Part (i) at $v = z$ becomes

$$
\langle h, z\rangle - \langle h, b\rangle \langle b, z\rangle \to 0 \quad \text{a.s.,}
$$

matching the first half of the paper's equation (13). Part (ii) becomes $\langle h, b\rangle_\infty = \alpha|X|/\sqrt{\alpha^2|X|^2 + \delta^2} = \psi_\infty$, matching the paper's formula. The proof structure also collapses: the projection onto $\mathcal{B}^\perp$ in Part (i) becomes the standard $b^\perp$ argument, and Lemma A.1$''$ is just an alternate form of the paper's Lemma A.1 applied to the bounded vector $z - \langle b, z\rangle b$.

## Why this proof is better

A few words on what changed and why.

The original proof I wrote (and Sonnet's expansion of it) followed the paper's $k = 1$ argument literally: dot the SVD identity with each population basis vector $b_m$ and with $z$, get $k+1$ scalar equations, substitute one into the other, observe the residual is noise, bound it. This required Assumption 2.5$'$ to make $b_m^\top B$ collapse to a scaled standard basis row, and required handling each of $k+1$ noise terms separately.

The projection argument bypasses all of that. Project the SVD identity onto $\mathcal{B}^\perp$, and the entire signal term vanishes by a single linear-algebraic identity, $\Pi_B^\perp B = 0$. What remains is a single noise term, projected onto a single bounded vector $\eta_p = \Pi_B^\perp v$. One application of Lemma A.1$''$ controls it.

Three concrete advantages, in order of importance:

First, **Part (i) needs no orthogonality of $B$**. The proof only uses that $B$'s columns span the population subspace $\mathcal{B}$ — which is automatic from the model, no assumption needed. Whether the columns are mutually orthogonal, highly correlated, or anything in between, the projection onto $\mathcal{B}^\perp$ kills the signal term identically. This explains the numerical observation that the matrix residual decays at the same rate even with deliberately correlated raw loadings — something the original proof couldn't explain because it relied on orthogonality where it wasn't needed.

Second, **Part (i) holds for any deterministic bounded $v$**, not just the dispersionless $z = e/\sqrt{p}$. The "dispersion" interpretation is specific to $v = z$, but the underlying matrix factorization is a structural fact about how the sample frame $H$ relates to $\mathcal{B}$. Stating Part (i) generally makes this clear; the dispersion bias is then a corollary obtained by specializing to $v = z$.

Third, **the proof is shorter** — Part (i) takes roughly half a page versus several pages for the dot-product argument. The mathematical content is the same; the reduction is in algebra, not in depth.

I should add: this kind of simplification, where a proof shrinks dramatically when you find the right coordinate-free formulation, is itself diagnostic. It usually means the original argument was navigating around an obstacle that wasn't really there. In this case, the obstacle was the desire to extract per-factor scalar identities (which Part (ii) genuinely needs) using the same tools as for the matrix factorization (which doesn't). Splitting them — projection for Part (i), Davis–Kahan for Part (ii) — clarifies which technical machinery serves which conclusion.

The full proof is shorter, the assumptions are weaker, and the statement is stronger. I'm satisfied this is the right form to commit to.
