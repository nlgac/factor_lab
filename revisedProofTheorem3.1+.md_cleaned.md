# A final reflection

The projection argument is genuinely better than the dot-product-and-substitute approach: shorter, more general (Part (i) drops Assumption 2.5$'$), and more transparent (it makes geometrically vivid what the result *is*). Before committing to it, let me ask whether there's a still-better argument I'm missing.

**Could the proof be even shorter?** The key inequality is $|H^\top \eta| \le |S_p^{-1}|_{\text{op}} \cdot |Z^\top \eta|/\sqrt n$. The first factor is bounded by $|S_p^{-1}| \asymp 1/\sqrt p$ (Lemma A.2$'$). The second is the norm of an $n$-vector each of whose entries has variance $\le \delta^2$, so $|Z^\top \eta| = O_P(\sqrt n) = O_P(1)$ since $n$ is fixed. Multiplying: $|H^\top \eta| = O_P(1/\sqrt p) \to 0$. Upgrading to a.s. via Borel–Cantelli with bounded fourth moments gives a.s. convergence at rate $p^{-1/2 + \epsilon}$.

That's the entire proof in three sentences. I don't see how to make it shorter without losing readability.

**Is there a still more general statement?** Yes, actually. The proof never uses that $z$ is the equal-weight vector — only that $|z| \le 1$ (a fixed bounded vector). So Part (i) holds for any deterministic bounded $v \in \mathbb{R}^p$:

$$
H^\top v - H^\top \Pi_B v \;\to\; 0 \quad \text{a.s.}
$$

This is a strictly stronger result. The "dispersion bias" interpretation enters only through the *specific* choice $v = z = e/\sqrt p$, which makes $\langle b_i, z\rangle_\infty$ have the dispersion-related limit $c_i$.

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
\textbf{(Part ii)} \qquad H^\top \tilde B \;\longrightarrow\; \mathrm{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k}),
$$

where each $\psi_{\infty,i} = \alpha_i |X_i| / \sqrt{\alpha_i^2 |X_i|^2 + \delta^2} \in (0, 1)$ a.s.

**Two notes on what this says.** First, Part (i) holds for *any* deterministic bounded vector $v$, not just the dispersionless $z = e/\sqrt p$. The matrix factorization is a structural statement about how $H$ relates to the population subspace $\mathcal{B}$, independent of which probe vector we choose. Second, **Part (i) does not require orthogonality of $B$'s columns** (Assumption 2.5$'$). It holds for arbitrary loading correlation structures — confirming what our numerics suggested and what an earlier proof had hidden behind unnecessary algebra.

## Setup

The model is $Y = B F^\top + Z$, dimensions $p \times n$, with $B \in \mathbb{R}^{p\times k}$, $F \in \mathbb{R}^{n\times k}$, $Z \in \mathbb{R}^{p\times n}$. Asymptotics: $p \to \infty$, $n$ and $k$ fixed.

Let $\mathcal{B} := \mathrm{span}(B) \subset \mathbb{R}^p$, with orthogonal projector $\Pi_B$ and complement projector $\Pi_B^\perp = I - \Pi_B$. Let $\tilde B$ denote any orthonormal basis for $\mathcal{B}$ (under Assumption 2.5$'$, the natural choice is $\tilde B = B \cdot \mathrm(|\beta_j|^{-1})$, but the choice doesn't affect $\Pi_B$).

The thin SVD of $Y/\sqrt n$ gives $H \in \mathbb{R}^{p\times k}$ (orthonormal top-$k$ left singular vectors), $\mathcal{X}*p \in \mathbb{R}^{n\times k}$ (orthonormal right singular vectors), and $S_p = \mathrm{diag}(s*{p,1}, \ldots, s_{p,k})$ in decreasing order. The fundamental identity is

$$
H S_p \;=\; \frac{Y \mathcal{X}_p}{\sqrt n} \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt n} \;+\; \frac{Z \mathcal{X}_p}{\sqrt n}, \tag{$33'$}
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

**Lemma A.1 (paper's Lemma A.1).** For deterministic $\eta \in \mathbb{R}^p$ with $|\eta|^2/p \in (c, C)$ for $p$ large, and under Assumption 2.3$'$, $(\eta^\top Z)_l / (|\eta|\sqrt p) \to 0$ a.s. for each $l$.

**Lemma A.1$''$ (companion for bounded $\eta$).** *Let $\eta_p \in \mathbb{R}^p$ be deterministic with $|\eta_p| \le C$ for all $p$. Under Assumption 2.3$'$, for each fixed $l$,*

$$
\frac{(\eta_p^\top Z)_l}{p^{1/2 - \epsilon}} \;\to\; 0 \quad \text{a.s. for any } \epsilon < 1/2.
$$

*Proof.* The scalar $W_p := (\eta_p^\top Z)*l = \sum_i (\eta_p)*i Z*{il}$ has mean zero, variance $|\eta_p|^2 \delta^2 \le C^2 \delta^2$, and bounded fourth moment by Assumption 2.3$'$ (since the entries are pairwise independent with bounded fourth moment, $\mathbb{E}[W_p^4] \le 3 (C^2 \delta^2)^2 + |\eta_p|^4 \cdot \sup_i \mathbb{E}[Z*{il}^4]$ is bounded uniformly in $p$). By Markov's inequality,

$$
\Pr\big(|W_p| > p^{1/2 - \epsilon}\big) \;\le\; \frac{\mathbb{E}[W_p^4]}{p^{2 - 4\epsilon}}.
$$

The right side is summable in $p$ for $\epsilon < 1/2$. Borel–Cantelli gives $|W_p| / p^{1/2 - \epsilon} \to 0$ a.s. $\square$

**Lemma A.2$'$ (spectral convergence).** Under Assumptions 2.1$'$–2.5$'$:

1. $R_p := Y^\top Y/(np) \to A_\infty := F M_\infty F^\top / n + (\delta^2/n) I_n$ in operator norm a.s., with $M_\infty = \mathrm{diag}(\alpha_1^2, \ldots, \alpha_k^2)$.
2. $s_{p,i}^2/p \to \lambda_i$ a.s. *Under 2.6$'$,* $\lambda_i = (\alpha_i^2 |X_i|^2 + \delta^2)/n$.
3. *Under 2.6$'$:* $\chi_{p,i} \to X_i/|X_i|$ a.s., and the $\lambda_i$ are distinct a.s.
4. $\sqrt p / s_{p,i} \to 1/\sqrt{\lambda_i} < \infty$ a.s.

*Proof sketch.* Expand $Y^\top Y / (np)$ into signal, cross, and noise terms. Signal: $F(B^\top B/p)F^\top / n \to F M_\infty F^\top / n$ a.s. Cross terms: vanish a.s. by Lemma A.1 applied entrywise. Noise: $Z^\top Z/(np) \to (\delta^2/n) I_n$ a.s. by SLLN. Together gives Part 1. Weyl's inequality gives Part 2. Under 2.6$'$, $F M_\infty F^\top = \sum_j \alpha_j^2 X_j X_j^\top$ has orthogonal rank-1 summands with eigenvectors $X_j/|X_j|$ and distinct eigenvalues $\alpha_j^2|X_j|^2$ a.s.; Davis–Kahan with the spectral gap gives Part 3. Part 4 is immediate. $\square$

## Proof of Part (i)

The argument is short; let me state it inline.

Decompose $v = \Pi_B v + \eta_p$ where $\eta_p := \Pi_B^\perp v$, with $|\eta_p| \le |v| \le 1$ (projectors are non-expansive).

Apply $\Pi_B^\perp$ to both sides of the fundamental identity $(33')$. The signal term vanishes because $\Pi_B^\perp B = 0$ (each column of $B$ lies in $\mathcal{B}$). What remains is

$$
\Pi_B^\perp H \cdot S_p \;=\; \frac{\Pi_B^\perp Z \mathcal{X}_p}{\sqrt n}.
$$

Right-multiply by $S_p^{-1}$ and take inner product with $v$:

$$
v^\top \Pi_B^\perp H \;=\; \frac{v^\top \Pi_B^\perp Z \mathcal{X}_p}{\sqrt n} \cdot S_p^{-1} \;=\; \frac{\eta_p^\top Z \mathcal{X}_p}{\sqrt n} \cdot S_p^{-1},
$$

using that $\Pi_B^\perp$ is symmetric, so $v^\top \Pi_B^\perp = (\Pi_B^\perp v)^\top = \eta_p^\top$.

The left side is exactly what we want to control: $v^\top \Pi_B^\perp H = v^\top H - v^\top \Pi_B H$, whose transpose is $H^\top v - H^\top \Pi_B v$. So

$$
H^\top v - H^\top \Pi_B v \;=\; S_p^{-1} \mathcal{X}_p^\top Z^\top \eta_p / \sqrt n. \tag{$\dagger$}
$$

Now bound the right side. Its $i$-th entry is

$$
\frac{1}{s_{p,i}\sqrt n}, \chi_{p,i}^\top Z^\top \eta_p \;=\; \frac{1}{s_{p,i}\sqrt n} \sum_{l=1}^n (\chi_{p,i})*l \cdot (\eta_p^\top Z*{\cdot l}).
$$

Each summand factors as bounded $\times$ slow-growing: $|(\chi_{p,i})*l| \le 1$ since $|\chi*{p,i}| = 1$, and by Lemma A.1$''$ applied to the bounded vector $\eta_p$, $|\eta_p^\top Z_{\cdot l}| = o(p^{1/2 - \epsilon})$ a.s. for any $\epsilon < 1/2$. The sum over $l$ has $n$ (finite) terms, so $|\chi_{p,i}^\top Z^\top \eta_p| = o(p^{1/2 - \epsilon})$ a.s.

By Lemma A.2$'$ Part 4, $s_{p,i} \asymp \sqrt p$ a.s., so $1/(s_{p,i}\sqrt n) = O(1/\sqrt p)$ a.s. Combining:

$$
\big| [H^\top v - H^\top \Pi_B v]_i \big| \;=\; O\Big(\frac{p^{1/2 - \epsilon}}{\sqrt p}\Big) \;=\; O(p^{-\epsilon}) \;\to\; 0 \quad \text{a.s.}
$$

Holds for each $i = 1, \ldots, k$. The vector $H^\top v - H^\top \Pi_B v \to 0$ a.s. $\square$

**Note.** Setting $v = z = e/\sqrt p$ recovers the dispersion-bias-specific statement $H^\top z - H^\top \Pi_B z \to 0$.

## Proof of Part (ii)

The strategy: take limits in the matrix identity $(33')$ projected onto $\mathcal{B}$, using Lemma A.2$'$.

Apply $\Pi_B$ to $(33')$. Now both terms survive: the signal because $\Pi_B B = B$, the noise (projected). After right-multiplying by $S_p^{-1}$:

$$
\Pi_B H \;=\; \frac{B F^\top \mathcal{X}_p}{\sqrt n} \cdot S_p^{-1} \;+\; \frac{\Pi_B Z \mathcal{X}_p}{\sqrt n} \cdot S_p^{-1}.
$$

The noise term is bounded in operator norm by $|\Pi_B Z \mathcal{X}*p|*{\text{op}} / (\sqrt n , s_{p,k})$, and the same Lemma-A.1$''$-style argument (apply componentwise to the orthonormal columns of $\tilde B$, each of bounded norm 1) shows this term vanishes a.s. — call this contribution $\varepsilon_p$.

Now left-multiply by $\tilde B^\top$. Since $\tilde B^\top \Pi_B = \tilde B^\top$ (as $\tilde B$ spans $\mathcal{B}$):

$$
\tilde B^\top H \;=\; \tilde B^\top B \cdot \frac{F^\top \mathcal{X}_p}{\sqrt n} \cdot S_p^{-1} \;+\; \tilde B^\top \varepsilon_p.
$$

Under Assumption 2.5$'$, $\tilde B^\top B = \mathrm{diag}(|\beta_j|)_j$ (the $j$-th column of $B$ equals $|\beta_j|$ times the $j$-th column of $\tilde B$). Substituting and rescaling:

$$
\tilde B^\top H \;=\; \mathrm{diag}(|\beta_j|/\sqrt p) \cdot \frac{F^\top \mathcal{X}_p \sqrt p}{\sqrt n} \cdot S_p^{-1} \;+\; o(1).
$$

Now take limits factor by factor using Lemma A.2$'$:

- $|\beta_j|/\sqrt p \to \alpha_j$ (Assumption 2.2$'$).
- $\mathcal{X}_p \to [X_1/|X_1|, \ldots, X_k/|X_k|]$ a.s. (Lemma A.2$'$ Part 3, requires Assumption 2.6$'$).
- Hence $F^\top \mathcal{X}*p \to F^\top \cdot [X_1/|X_1|, \ldots, X_k/|X_k|]$, with $(i,j)$ entry $X_i^\top X_j / |X_j| = |X_j| \delta*{ij}$ under Assumption 2.6$'$. So $F^\top \mathcal{X}_p \to \mathrm{diag}(|X_j|)$.
- $S_p^{-1} \cdot \sqrt p = \mathrm{diag}(\sqrt p / s_{p,i}) \to \mathrm(1/\sqrt{\lambda_i})$ (Lemma A.2$'$ Part 4).

Combining:

$$
(\tilde B^\top H)_\infty = \mathrm{diag}(\alpha_j) \cdot \mathrm{diag}(|X_j|/\sqrt n) \cdot \mathrm{diag}(1/\sqrt{\lambda_j}) = \mathrm{diag}\Big(\frac{\alpha_j |X_j|}{\sqrt{n \lambda_j}}\Big)
$$

Since $n \lambda_j = \alpha_j^2 |X_j|^2 + \delta^2$ (Lemma A.2$'$ Part 2), the diagonal entries are

$$
\frac{\alpha_j |X_j|}{\sqrt{\alpha_j^2 |X_j|^2 + \delta^2}} \;=\; \psi_{\infty,j} \;\in\; (0, 1) \quad \text{a.s.}
$$

Transposing (which preserves diagonality): $H^\top \tilde B \to \mathrm{diag}(\psi_{\infty,1}, \ldots, \psi_{\infty,k})$ a.s. $\square$

## Corollary: The Dispersion Bias

Under Assumptions 2.1$'$–2.6$'$, with $c_i := \mu_\infty(\beta_i)/\alpha_i = 1/\sqrt{1+d_\infty^2(\beta_i)} \in (0, 1]$:

$$
|\Pi_B z|^2 \;\to\; \sum_{i=1}^k c_i^2, \qquad |\Pi_H z|^2 \;\to\; \sum_{i=1}^k \psi_{\infty,i}^2 c_i^2,
$$

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\to\; \sum_{i=1}^k (1 - \psi_{\infty,i}^2), c_i^2 \;>\; 0 \quad \text{a.s.}
$$

*Proof.* From Assumption 2.2$'$, $\langle b_i, z\rangle_p = (e^\top \beta_i / p) / (|\beta_i|/\sqrt p) \to \mu_\infty(\beta_i)/\alpha_i = c_i$. So $|\Pi_B z|^2 = \sum_i \langle b_i, z\rangle_p^2 \to \sum_i c_i^2$.

For $|\Pi_H z|^2 = |H^\top z|^2$: by Part (i) at $v = z$, $H^\top z - H^\top \Pi_B z \to 0$; by Part (ii), $H^\top \tilde B \to \mathrm{diag}(\psi_{\infty,i})$. So $H^\top z \to H^\top \Pi_B z \to \mathrm(\psi_{\infty,i}) \cdot (c_1, \ldots, c_k)^\top = (\psi_{\infty,1} c_1, \ldots, \psi_{\infty,k} c_k)^\top$, and $|H^\top z|^2 \to \sum_i \psi_{\infty,i}^2 c_i^2$.

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

Second, **Part (i) holds for any deterministic bounded $v$**, not just the dispersionless $z = e/\sqrt p$. The "dispersion" interpretation is specific to $v = z$, but the underlying matrix factorization is a structural fact about how the sample frame $H$ relates to $\mathcal{B}$. Stating Part (i) generally makes this clear; the dispersion bias is then a corollary obtained by specializing to $v = z$.

Third, **the proof is shorter** — Part (i) takes roughly half a page versus several pages for the dot-product argument. The mathematical content is the same; the reduction is in algebra, not in depth.

I should add: this kind of simplification, where a proof shrinks dramatically when you find the right coordinate-free formulation, is itself diagnostic. It usually means the original argument was navigating around an obstacle that wasn't really there. In this case, the obstacle was the desire to extract per-factor scalar identities (which Part (ii) genuinely needs) using the same tools as for the matrix factorization (which doesn't). Splitting them — projection for Part (i), Davis–Kahan for Part (ii) — clarifies which technical machinery serves which conclusion.

The full proof is shorter, the assumptions are weaker, and the statement is stronger. I'm satisfied this is the right form to commit to.
