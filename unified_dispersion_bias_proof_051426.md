# Multifactor Dispersion Bias with Per-Column Prevalence: A Unified Treatment

*This document consolidates two complementary lines of work: the k-factor dispersion bias result (Theorem 3.1′, referred to as **NG** below, where NG denotes the author of that result) and the multifactor prevalence extension (Theorems 1 and 3 of the companion paper, abbreviated **MPE** and referred to below by its author as **AK**). The result is a single, fully proved theorem that subsumes both, with Part (i) following the NG projection argument and Parts (ii)–(iii) following the AK proof architecture, translated into unified notation. Term definitions appear in §2; the introduction uses terms informally.*

*Author abbreviations: **NG** = [Nick Gunther]; **AK** = [Alec Kircheval]. Full citations appear in the bibliography.*

---

## Table of Contents

1. Introduction
2. Model and Notation
3. Assumptions
4. Main Results
5. Lemmas
6. Proof of Part (i)
7. Proof of Parts (ii) and (iii)
8. Unification of Parts (ii) and (iii)
9. Recovery of NG's Theorem 3.1′
10. Corollaries
11. Discussion
12. Grassmannian Subspace Estimation vs. Frame Estimation
13. Summary

---

## Abstract

Fix a $k$-factor model $Y = BF^\top + Z$ with $p$ assets observed over $n$ time periods. In the asymptotic regime $p \to \infty$ with $n$ and $k$ fixed, the top-$k$ sample principal components $h_1, \ldots, h_k$ are systematically rotated away from the population loading directions $\bar{b}_1, \ldots, \bar{b}_k$. This paper establishes the exact almost-sure limit of $\sin^2\angle(h_j, \bar{b}_j)$ as the sum of two non-negative terms: an *out-of-subspace floor* $\delta^2/(n\rho_j + \delta^2)$, determined solely by the signal-to-noise ratio $\mathrm{SNR}_j = n\rho_j/\delta^2$, and an *in-subspace rotation* arising from the finite-$n$ misalignment of sample and population factor-covariance eigenvectors. The floor is irreducible: adding more assets cannot reduce it. The rotation vanishes when factor returns are orthogonal (recovering the result of NG's Theorem 3.1′) and shrinks to zero as $n \to \infty$. For Grassmannian subspace estimation — measuring the distance between $\mathrm{col}(H)$ and $\mathcal{B}$ as points on $\mathrm{Gr}(k,p)$ — the in-subspace rotation drops out entirely via the identity $d_{\mathrm{Gr}}^2 = \sum_j \|h_j^\perp\|^2$, and the total subspace error equals $\sum_j \delta^2/(n\rho_j + \delta^2)$. Subspace estimation is therefore strictly more efficient than frame estimation whenever factors are correlated.

---

## 1. Introduction

Fix a $k$-factor model in which $p$ assets are observed over $n$ time periods. The central question is: how well do the top-$k$ sample principal components $h_1,\ldots,h_k$ align with the population loading directions $\bar{b}_1,\ldots,\bar{b}_k$?

**The basic mechanism.** Think of the sample PCA as a lens trying to resolve the $k$-dimensional factor subspace $\mathcal{B}$ through noise. Even as $p\to\infty$ (infinitely many assets), the lens does not sharpen to perfect resolution: the ratio of signal to noise is fixed (for fixed $n$), so each sample eigenvector $h_j$ is *systematically rotated away* from its population target $\bar{b}_j$ by a fixed asymptotic angle. This rotation has two independent components.

**Two sources of misalignment** arise in the large-$p$, fixed-$(n,k)$ regime:

1. **Out-of-subspace defect.** Even the entire sample subspace $\mathrm{col}(H)$ does not coincide with the population subspace $\mathcal{B} = \mathrm{col}(B)$. A fraction $\delta^2/(n\rho_j + \delta^2)$ of $h_j$'s squared norm lies outside $\mathcal{B}$ — pure noise leaked into the signal direction. This fraction equals $1/(1+\mathrm{SNR}_j)$ where $\mathrm{SNR}_j = n\rho_j/\delta^2$ is the signal-to-noise ratio for factor $j$. It is irreducible: no matter how many assets $p$ we observe, this floor persists.

2. **In-subspace rotation.** Even the within-$\mathcal{B}$ component of $h_j$ does not align with $\bar{b}_j$. The prevalence-weighted factor-return covariance matrix $\hat{D}$ (defined in §2) is estimated from only $n$ observations; its eigenvectors $\hat{w}_j$ rotate away from the population targets $w_j$ by $\sin^2\angle(\hat{w}_j, w_j)$. This term is a finite-$n$ artifact: it vanishes as $n\to\infty$ but survives even in the $\delta^2\to 0$ limit.

**What the two prior works establish.** NG's Theorem 3.1′ captures source (1) only, because its orthogonality assumptions (on both loadings and factor returns) force source (2) to zero. AK's Theorems 1 and 3 (MPE) capture both sources jointly, in the general setting where loading columns may have unequal norms (characterized by *prevalences* $c_j = \lim\|\beta_j\|^2/p$) and non-orthogonal directions (characterized by the limiting Gram matrix $G_\infty$). The unified result below gives the exact almost-sure limit of $\sin^2\angle(h_j, \bar{b}_j)$ as $p\to\infty$ with $n$, $k$ fixed, showing that NG's Theorem 3.1′ is the special case where source (2) is forced to zero.

All symbols are defined precisely in §2. Worked examples appear in §4.1 immediately following the theorem statement.

---

## 2. Model and Notation

**Data model.** The $p\times n$ data matrix is

$$
Y = BF^\top + Z,
$$

where $B \in \mathbb{R}^{p\times k}$ is the deterministic loading matrix with columns $\beta_1,\ldots,\beta_k$; $F \in \mathbb{R}^{n\times k}$ is the random factor-return matrix with columns equal to the return series of each factor; and $Z\in\mathbb{R}^{p\times n}$ is the idiosyncratic noise. Asymptotics: $p\to\infty$ with $n$ and $k$ fixed.

*Remark (conditioning on $F$).* Throughout this document, $F$ is conditioned on: the result holds for almost every realization of $F$ satisfying the regular-event hypothesis stated at the end of §3. Accordingly, $\rho_j$ — the $j$-th eigenvalue of $\hat{D}$ defined below — is a random variable that depends on the realized factor returns $F$, not a deterministic population constant. The population counterpart $d_j$ (eigenvalue of $D$) is the limit of $\rho_j$ as $n\to\infty$.

**Prevalence and normalized loadings.** Write $A(p) = \mathrm{diag}(\|\beta_j(p)\|)_{j=1}^k$ for the diagonal matrix of loading-column norms, and

$$
b(p) = B\,A(p)^{-1} \in \mathbb{R}^{p\times k}
$$

for the matrix of *unit* loading columns. The *Gram matrix* of normalized loadings is

$$
G(p) = b(p)^\top b(p) \in \mathbb{R}^{k\times k}.
$$

The *prevalence* of factor $j$ is $c_j = \lim_{p\to\infty}\|\beta_j(p)\|^2/p \in (0,\infty)$, and $C = \mathrm{diag}(c_1,\ldots,c_k)$.

*Intuition.* The prevalence $c_j$ measures how strongly factor $j$ is spread across the asset universe. If every asset has loading $a_j$ on factor $j$ (a uniform loading column $\beta_j = a_j\cdot\mathbf{1}_p$), then $\|\beta_j\|^2 = p\,a_j^2$, so $c_j = a_j^2$. If factor $j$ affects only a fixed finite subset of assets, $c_j = 0$ — the factor has zero prevalence and effectively disappears in the large-$p$ limit. The theory requires $c_j > 0$: each factor must be pervasive across assets.

*Example 2.1 (two-factor prevalence).* Suppose $k=2$, $p$ even, and

$$
\beta_1(p) = \bigl(\underbrace{3,\ldots,3}_{p/2},\;\underbrace{1,\ldots,1}_{p/2}\bigr)^\top,\qquad \beta_2(p) = \bigl(\underbrace{-1,\ldots,-1}_{p/2},\;\underbrace{3,\ldots,3}_{p/2}\bigr)^\top.
$$

Then $\|\beta_1\|^2 = p(9+1)/2 = 5p$, so $c_1 = 5$; likewise $c_2 = 5$. Since $\beta_1^\top\beta_2 = p(3)(-1)/2 + p(1)(3)/2 = 0$, the loading columns are orthogonal, so $G(p) = I_2$ and $G_\infty = I_2$. We return to this example numerically in §4.1.

**Signal subspace.** Write $\mathcal{B} = \mathrm{col}(B) \subset \mathbb{R}^p$ for the $k$-dimensional population factor subspace, $\Pi_B = B(B^\top B)^{-1}B^\top$ for the orthogonal projection onto $\mathcal{B}$, and $\Pi_B^\perp = I_p - \Pi_B$ for its complement.

**SVD of normalized loading matrix.** Fix the thin SVD

$$
b(p) = U(p)\,\Sigma(p)\,V(p)^\top, \tag{1}
$$

where $U(p)\in\mathbb{R}^{p\times k}$ has orthonormal columns spanning $\mathcal{B}$, $\Sigma(p)\in\mathbb{R}^{k\times k}$ is diagonal positive, and $V(p)\in O(k)$, the group of $k\times k$ real orthogonal matrices. Write $\Pi_U = U(p)U(p)^\top$ (equal to $\Pi_B$). Note: forming the Gram matrix from the SVD gives $G(p) = b(p)^\top b(p) = V(p)\Sigma(p)^2 V(p)^\top$, so the diagonal entries of $\Sigma(p)^2$ are the eigenvalues of $G(p)$. In particular, $\Sigma(p) \to \Lambda_G^{1/2}$ when $G(p)\to G_\infty = Q\Lambda_G Q^\top$; when $G_\infty = I_k$, $\Sigma(p)\to I_k$ but $V(p)$ need not converge.

**Prevalence-rescaled factor returns.** Define the *prevalence-rescaled factor return matrix*

$$
F^\# = C^{1/2}F^\top \in \mathbb{R}^{k\times n}.
$$

**Prevalence-weighted factor covariance matrices.** Define the $k\times k$ matrices

$$
\hat{D} = C^{1/2}\!\left(\frac{F^\top F}{n}\right)C^{1/2} \quad\text{(sample)}, \qquad D = C^{1/2}\Sigma_F C^{1/2} \quad\text{(population)},
$$

where $\Sigma_F = \lim_{n\to\infty} F^\top F/n$ is the population factor-return covariance. (Since $n$ is fixed and $F$ is conditioned on, $\hat{D}$ is a fixed matrix given $F$; $D$ is the population object it approaches as $n\to\infty$.) Equivalently, $\hat{D} = F^\#(F^\#)^\top/n$.

*Intuition.* $\hat{D}$ blends two sources of information: how strongly each factor is spread cross-sectionally (captured by $C^{1/2}$) and how much it varies over time (captured by $F^\top F/n$). The diagonal entry $\hat{D}_{jj} = c_j\cdot (F_j^\top F_j/n)$ is the $j$-th factor's per-period contribution to cross-sectional variance.

*Remark on loading norms.* The diagonal matrix $A(p) = \mathrm{diag}(\|\beta_j(p)\|)$ satisfies $A(p)/\sqrt{p} \to C^{1/2}$ by Assumption 1. Equivalently, $B^\top B/p = A(p)G(p)A(p)/p \to C^{1/2}G_\infty C^{1/2} = \Gamma_B$. This convergence $\Gamma_p \to \Gamma_B$ is all that the proof of §7 requires; the matrices $A(p)$ and $G(p)$ enter only through their product.

**Rotated-and-reweighted matrices for general $G_\infty$.** Fix a spectral decomposition

$$
G_\infty = Q\Lambda_G Q^\top, \qquad \Lambda_G = \mathrm{diag}(g_1,\ldots,g_k),\quad g_1\ge\cdots\ge g_k>0,\quad Q\in O(k).\\ \tag{2}
$$

Define

$$
\hat{M} = \Lambda_G^{1/2}(Q^\top \hat{D}\,Q)\Lambda_G^{1/2}, \qquad M = \Lambda_G^{1/2}(Q^\top D\,Q)\Lambda_G^{1/2}. \tag{3}
$$

**Interpretation.** The matrix $G_\infty = b(p)^\top b(p)$ (with unit-normalized loading columns) is the *correlation* matrix of the loading columns, not covariance. When loadings are non-orthogonal, the factor covariance $\hat{D}$ is most naturally expressed in coordinates aligned with the eigenbasis of $G_\infty$. The transformation accomplishes this: $Q$ rotates the factor covariance into the eigenbasis of the loading-column correlation $G_\infty$, and $\Lambda_G^{1/2}$ pre- and post-multiplies to reweight each rotated direction by the square root of the corresponding loading-correlation eigenvalue $g_j$. These eigenvalues measure how much the normalized loading variance is concentrated along the $j$-th principal loading direction: larger $g_j$ indicates stronger signal concentration. When $G_\infty = I_k$ (orthogonal loadings), we may take $Q = I_k$, $\Lambda_G = I_k$, giving $\hat{M} = \hat{D}$ and $M = D$.

**Sample PCA objects.** The full SVD of $Y/\sqrt{n}$ is the exact matrix factorization
$$
\frac{Y}{\sqrt{n}} = \tilde{H}\,\tilde{S}\,\tilde{\mathcal{X}}^\top,
$$
where $\tilde{H}\in\mathbb{R}^{p\times r}$ and $\tilde{\mathcal{X}}\in\mathbb{R}^{n\times r}$ have orthonormal columns, $\tilde{S}\in\mathbb{R}^{r\times r}$ is diagonal with entries $s_{p,1}\ge\cdots\ge s_{p,r}>0$, and $r=\mathrm{rank}(Y)\le\min(p,n)$ (with $r=\min(p,n)$ almost surely since $Z$ has full rank). Partition the three factors by keeping the leading $k$ columns:
$$
\tilde{H} = \bigl[H \;\big|\; H_\perp\bigr],
\qquad
\tilde{S} = \begin{bmatrix} S_p & 0 \\ 0 & S_\perp \end{bmatrix},
\qquad
\tilde{\mathcal{X}} = \bigl[\mathcal{X}_p \;\big|\; \mathcal{X}_\perp\bigr],
$$
where $H=[h_1,\ldots,h_k]\in\mathbb{R}^{p\times k}$, $S_p=\mathrm{diag}(s_{p,1},\ldots,s_{p,k})$, and $\mathcal{X}_p=[\chi_{p,1},\ldots,\chi_{p,k}]\in\mathbb{R}^{n\times k}$. The columns of $H$ are the top-$k$ eigenvectors of $YY^\top/n$. Because $\tilde{S}$ is block-diagonal with zero off-diagonal blocks, the product $\tilde{H}\tilde{S} = [HS_p \mid H_\perp S_\perp]$ has no cross terms, and the full SVD reads
$$
\frac{Y}{\sqrt{n}} = H S_p \mathcal{X}_p^\top \;+\; H_\perp S_\perp \mathcal{X}_\perp^\top.
$$
Right-multiplying by $\mathcal{X}_p$ and using $\mathcal{X}_p^\top\mathcal{X}_p=I_k$ and $\mathcal{X}_\perp^\top\mathcal{X}_p=0$ (orthonormality of the columns of $\tilde{\mathcal{X}}$) gives the exact identity

$$
H S_p = \frac{Y\,\mathcal{X}_p}{\sqrt{n}}. \tag{4}
$$

The rank-$k$ term $HS_p\mathcal{X}_p^\top$ is *not* equal to $Y/\sqrt{n}$; the second term $H_\perp S_\perp\mathcal{X}_\perp^\top$ is nonzero and carries the noise contributions. Equation (4) is exact; a truncated reconstruction of $Y/\sqrt{n}$ from the top-$k$ factors alone is not.

The *small Gram matrix*

$$
W^{(p)} = \frac{Y^\top Y}{np} \in \mathbb{R}^{n\times n}
$$

has eigenvalues $s_{p,j}^2/p$ with eigenvectors $\chi_{p,j}$. Working with the $n\times n$ matrix $W^{(p)}$ (rather than the $p\times p$ matrix $YY^\top/(np)$) is analytically convenient: $n$ is fixed, so its limiting spectrum is tractable.

**Population loading direction.** Let $\bar{b}_j$ denote the unit eigenvector of the $p\times p$ population signal covariance

$$
\Sigma_0^{(p)} = \frac{B\,\Sigma_F\,B^\top}{p}
$$

corresponding to its $j$-th largest eigenvalue. Since $\Sigma_0^{(p)}$ is supported on $\mathcal{B}$, we have $\bar{b}_j\in\mathrm{col}(U(p)) = \mathcal{B}$. When the loading columns are mutually orthogonal, $\bar{b}_j$ coincides with the $j$-th normalized loading column $\beta_j/\|\beta_j\|$.

**Symbol summary.**

*Grouping follows the order of introduction in §2. Items first defined outside §2 note their location.*

**Model primitives**

| Symbol                                    | Meaning                                                                                |
|:-----------------------------------------:|:-------------------------------------------------------------------------------------- |
| $p$, $n$, $k$                             | Number of assets, time periods, factors; asymptotics: $p\to\infty$ with $n$, $k$ fixed |
| $Y \in \mathbb{R}^{p\times n}$            | Observed return matrix: $Y = BF^\top + Z$                                              |
| $B \in \mathbb{R}^{p\times k}$, $\beta_j$ | Population loading matrix; $j$-th loading column                                       |
| $F \in \mathbb{R}^{n\times k}$, $F_j$     | Factor-return matrix; $j$-th factor-return series (column of $F$)                      |
| $Z \in \mathbb{R}^{p\times n}$            | Idiosyncratic noise matrix; i.i.d. mean-zero entries with variance $\delta^2$          |
| $\delta^2 > 0$                            | Common idiosyncratic noise variance                                                    |
| $\Sigma_F$                                | Population factor-return covariance: $\Sigma_F = \lim_{n\to\infty} F^\top F/n$         |

**Loading geometry**

| Symbol                  | Meaning                                                                                                                           |
|:-----------------------:|:--------------------------------------------------------------------------------------------------------------------------------- |
| $A(p)$                  | Diagonal matrix of loading-column norms: $A(p) = \mathrm{diag}(\|\beta_j(p)\|)_{j=1}^k$                                           |
| $b(p) = BA(p)^{-1}$     | Unit-normalized loading matrix; columns are $\beta_j/\|\beta_j\|$                                                                 |
| $G(p) = b(p)^\top b(p)$ | Finite-$p$ Gram matrix of unit loading columns; $G(p) \to G_\infty$ by Assumption 2                                               |
| $G_\infty$              | Limiting Gram matrix (positive definite, $k\times k$); see Assumption 2                                                           |
| $Q$, $\Lambda_G$        | Spectral factors of $G_\infty$: $G_\infty = Q\Lambda_G Q^\top$, $\Lambda_G = \mathrm{diag}(g_1,\ldots,g_k)$, $Q\in O(k)$; see (2) |
| $c_j$, $C$              | Prevalence of factor $j$: $c_j = \lim\|\beta_j\|^2/p \in (0,\infty)$; $C = \mathrm{diag}(c_j)$; see Assumption 1                  |
| $\Gamma_B = C^{1/2}G_\infty C^{1/2}$ | Limiting metric on $\mathbb{R}^k$: $B^\top B/p \to \Gamma_B$; used as the inner-product matrix in the §7 proof |

**SVD of unit loading matrix** (equation (1))

| Symbol                                 | Meaning                                                                                          |
|:--------------------------------------:|:------------------------------------------------------------------------------------------------ |
| $U(p) \in \mathbb{R}^{p\times k}$      | Left singular vectors of $b(p)$; columns form an orthonormal basis of $\mathcal{B}$              |
| $\Sigma(p) \in \mathbb{R}^{k\times k}$ | Diagonal singular-value matrix of $b(p)$; $\Sigma(p)^2 = G(p)$, so $\Sigma(p)\to G_\infty^{1/2}$ |
| $V(p) \in O(k)$                        | Right singular vectors of $b(p)$; enters the SVD (1) but does not appear in the §7 proof         |

**Signal subspace**

| Symbol                                | Meaning                                                                                                              |
|:-------------------------------------:|:-------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{B} = \mathrm{col}(B)$       | $k$-dimensional population factor subspace of $\mathbb{R}^p$                                                         |
| $\Pi_B$, $\Pi_B^\perp$                | Orthogonal projection onto $\mathcal{B}$, and its complement $I_p - \Pi_B$                                           |
| $\Sigma_0^{(p)} = B\Sigma_F B^\top/p$ | Population signal covariance ($p\times p$); supported on $\mathcal{B}$                                               |
| $\bar{b}_j$                           | $j$-th population loading direction: unit eigenvector of $\Sigma_0^{(p)}$ for eigenvalue $\lambda_j(\Sigma_0^{(p)})$ |

**Prevalence-weighted factor covariances**

| Symbol                                            | Meaning                                                                                                                     |
|:-------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------- |
| $F^\# = C^{1/2}F^\top \in \mathbb{R}^{k\times n}$ | Prevalence-rescaled factor-return matrix                                                                                    |
| $\hat{D} = C^{1/2}(F^\top F/n)C^{1/2}$            | Sample prevalence-weighted factor covariance ($k\times k$); equals $F^\#(F^\#)^\top/n$                                      |
| $D = C^{1/2}\Sigma_F C^{1/2}$                     | Population prevalence-weighted factor covariance ($k\times k$)                                                              |
| $\hat{M}$, $M$                                    | Rotated-reweighted covariances for general $G_\infty$: see (3); reduce to $\hat{D}$, $D$ when $G_\infty = I_k$              |
| $\rho_j$                                          | $j$-th eigenvalue of $\hat{M}$ (or $\hat{D}$ when $G_\infty=I_k$); $\rho_1>\cdots>\rho_k>0$ by the regular-event hypothesis |
| $\hat{w}_j$, $w_j$                                | $j$-th orthonormal eigenvectors of $\hat{M}$ and $M$ respectively                                                           |
| $d_j$                                             | $j$-th eigenvalue of $M$; population limit $d_j = \lim_{n\to\infty}\rho_j$                                                  |
| $\mathrm{SNR}_j = n\rho_j/\delta^2$               | Per-factor signal-to-noise ratio; determines the out-of-subspace floor $1/(1+\mathrm{SNR}_j)$                               |

**Sample PCA objects** (equation (4))

| Symbol                                                                    | Meaning                                                                                                     |
|:-------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------- |
| $H = [h_1,\ldots,h_k]\in\mathbb{R}^{p\times k}$                           | Top-$k$ sample eigenvectors of $YY^\top/n$; orthonormal columns                                             |
| $S_p = \mathrm{diag}(s_{p,1},\ldots,s_{p,k})$                             | Diagonal matrix of top-$k$ singular values of $Y/\sqrt{n}$                                                  |
| $\mathcal{X}_p = [\chi_{p,1},\ldots,\chi_{p,k}]\in\mathbb{R}^{n\times k}$ | Right singular vectors of $Y/\sqrt{n}$; orthonormal columns                                                 |
| $s_{p,j}$                                                                 | $j$-th largest singular value of $Y/\sqrt{n}$; satisfies $s_{p,j}^2/p \to \rho_j + \delta^2/n$ a.s. by (13) |
| $\chi_{p,j}$                                                              | $j$-th column of $\mathcal{X}_p$; converges to $v_j$ a.s. by (13)                                           |
| $\Pi_H = HH^\top$                                                         | Orthogonal projection onto $\mathrm{col}(H)$                                                                |
| $W^{(p)} = Y^\top Y/(np) \in \mathbb{R}^{n\times n}$                      | Small Gram matrix; eigenvalues $s_{p,j}^2/p$ with eigenvectors $\chi_{p,j}$                                 |

**Limiting small Gram matrix** (Lemmas 7–8)

| Symbol                         | Meaning                                                                               |
|:------------------------------:|:------------------------------------------------------------------------------------- |
| $W_\infty$                     | Almost-sure spectral limit of $W^{(p)}$; $W_\infty = F\Gamma_B F^\top/n + (\delta^2/n)I_n$; see (11) |
| $\tau_j = \rho_j + \delta^2/n$ | $j$-th largest eigenvalue of $W_\infty$ (signal-plus-noise level)                                     |
| $v_j$                          | $j$-th eigenvector of $W_\infty$; see (12); equals $(F^\#)^\top\hat{w}_j/\sqrt{n\rho_j}$ when $G_\infty = I_k$  |

**Shrinkage and alignment** (Corollaries 1 and 5)

| Symbol                                                                | Meaning                                                                                     |
|:---------------------------------------------------------------------:|:------------------------------------------------------------------------------------------- |
| $\psi_{\infty,j}^2 = n\rho_j/(n\rho_j+\delta^2)$                      | Squared alignment ceiling for eigenvector $h_j$; equals $\mathrm{SNR}_j/(1+\mathrm{SNR}_j)$ |
| $\hat{\psi}_{p,j}^2 = 1 - \ell_p^2/s_{p,j}^2$                         | Observable estimator of $\psi_{\infty,j}^2$; see Corollary 1                                |
| $\ell_p^2 = \frac{1}{n-k}\sum_{j=k+1}^n s_{p,j}^2$                    | Mean squared noise singular value; converges to $\delta^2/n$ a.s.                           |
| $\Psi_\infty = \mathrm{diag}(\psi_{\infty,1},\ldots,\psi_{\infty,k})$ | Diagonal shrinkage matrix; see Corollaries 5 and discussion in §9                           |
| $\tilde{B} = b(p)$                                                    | Unit loading column matrix (alternative notation used in §9 and corollaries)                |
| $\Gamma_\infty = \lim_{p\to\infty}\tilde{B}^\top W$                   | Asymptotic alignment matrix between unit loadings and probe frame $W$; see Corollary 5      |

**Angles.** For unit vectors $u,v\in\mathbb{R}^p$, the angle between them satisfies $\cos\angle(u,v) = |\langle u,v\rangle|$ and $\sin^2\angle(u,v) = 1 - \langle u,v\rangle^2$.

---

## 3. Assumptions

The following assumptions are in force throughout. All limits are as $p\to\infty$ with $n$, $k$ fixed.

**Assumption 1 (Per-column prevalence).** For each $j\in\{1,\ldots,k\}$,

$$
\frac{\|\beta_j(p)\|^2}{p} \;\longrightarrow\; c_j \;\in\; (0,\infty).
$$

**Assumption 2 (Gram convergence).** The normalized Gram matrix converges: $G(p)\to G_\infty$ for some positive definite $G_\infty\in\mathbb{R}^{k\times k}$.

**Assumption 3 (Spectral separation).** The matrix $M$ defined in (3) has $k$ distinct positive eigenvalues $m_1 > m_2 > \cdots > m_k > 0$. Under $G_\infty = I_k$ and diagonal $\Sigma_F = \mathrm{diag}(\sigma_1^2,\ldots,\sigma_k^2)$, we have $D = \mathrm{diag}(c_j\sigma_j^2)$ and Assumption 3 reduces to the strict ordering $c_1\sigma_1^2 > c_2\sigma_2^2 > \cdots > c_k\sigma_k^2 > 0$: no two factors contribute equally to cross-sectional variance.

*Example 3.1.* In Example 2.1 with $c_1 = c_2 = 5$ and factor-return standard deviations $\sigma_1 = 0.10$, $\sigma_2 = 0.05$: $c_1\sigma_1^2 = 5\times 0.01 = 0.05 > c_2\sigma_2^2 = 5\times 0.0025 = 0.0125$. Assumption 3 is satisfied.

*Remark (detection threshold).* Assumption 3 requires all eigenvalues $m_j > 0$. In the diagonal case this means $c_j\sigma_j^2 > 0$, which holds as long as each factor has positive variance and positive prevalence. The quantitative formula $\delta^2/(n\rho_j+\delta^2)$ applies when $\rho_j > 0$; this is the analogue of the BBP (Baik–Ben Arous–Péché) detection threshold in this regime. When a factor's signal strength falls below $\delta^2/n$, the corresponding sample eigenvector loses alignment with the population direction entirely, and the floor formula does not apply to that factor.

**Noise assumptions.** The entries of $Z$ are mean-zero, mutually independent within each column, with common variance $\delta^2 > 0$ and uniformly bounded $(2+\eta)$-th moment for some $\eta > 0$:

$$
\sup_p \mathbb{E}|Z_{11}|^{2+\eta} < \infty.
$$

The factor return matrix $F$ and noise $Z$ are independent.

**Regular-event hypothesis.** The matrix $\hat{M}$ has $k$ distinct positive eigenvalues $\rho_1 > \rho_2 > \cdots > \rho_k > 0$ with orthonormal eigenvectors $\hat{w}_1,\ldots,\hat{w}_k$. This event holds almost surely under any law of $F$ whose density is absolutely continuous with respect to Lebesgue measure on $\mathbb{R}^{n\times k}$. The argument: the eigenvalues of $\hat{M}$ are the roots of its characteristic polynomial $\det(\hat{M} - \lambda I_k)$, and the discriminant (the product of squared pairwise differences of roots) is itself a polynomial in the entries of $F^\top F$. This polynomial is non-trivial — it is not identically zero, since diagonal $F^\top F$ with distinct diagonal entries gives distinct eigenvalues of $\hat{M}$. A non-trivial polynomial vanishes on a set of Lebesgue measure zero, so the discriminant is nonzero almost surely.

*Remark (comparison with NG's Theorem 3.1′ assumptions).* Assumption 1 is equivalent to NG's Assumption 2.2′. Assumption 2 with $G_\infty = I_k$ follows from NG's Assumption 2.5′ (orthogonal loading columns). Assumption 3 follows from NG's Assumption 2.6′ together with the ordering $c_1\sigma_1^2 > \cdots > c_k\sigma_k^2$.

*Remark (assumption hierarchy for the three parts of the theorem).* **Part (i)** requires only Assumption 1 and the noise assumptions — no Gram convergence, no spectral separation, and in particular no orthogonality condition on the loading columns or factor returns. Parts (ii)–(iii) require the full set of Assumptions 1–3.

---

## 4. Main Results

The theorem has three parts with distinct assumption requirements. Part (i) is a qualitative subspace alignment result, valid under minimal assumptions. Parts (ii)–(iii) are quantitative per-direction results that require the full assumption set; they give the explicit two-term floor-plus-rotation decomposition under, respectively, diagonal and general Gram matrix $G_\infty$.

**Theorem (Multifactor Dispersion Bias).** *Under Assumptions 1, 2, and 3 and the noise assumptions, conditional on $F$ and almost surely as $p\to\infty$:*

**Part (i): Probe-vector alignment.** *(Requires only Assumption 1 and the noise assumptions.) For any deterministic sequence $v = v(p)\in\mathbb{R}^p$ with $|v|\le 1$,*

$$
H^\top v \;-\; H^\top\Pi_B v \;\longrightarrow\; 0 \quad a.s. \tag{5}
$$

**Part (ii): Per-direction alignment, diagonal-Gram case.** *Under the additional hypothesis $G_\infty = I_k$, for each $j\in\{1,\ldots,k\}$,*

$$
\sin^2\angle(h_j,\,\bar{b}_j) \;\xrightarrow{a.s.}\;
\underbrace{\frac{\delta^2}{n\rho_j + \delta^2}}_{\text{out-of-subspace floor}}
\;+\;
\underbrace{\frac{n\rho_j}{n\rho_j + \delta^2}\,\sin^2\angle(\hat{w}_j,\,e_j)}_{\text{in-subspace rotation}},\\
\tag{6}
$$

*where $\rho_j$ is the $j$-th eigenvalue of $\hat{D} = C^{1/2}(F^\top F/n)C^{1/2}$, $\hat{w}_j$ is its $j$-th eigenvector, and $e_j$ is the $j$-th standard basis vector of $\mathbb{R}^k$.*

**Part (iii): Per-direction alignment, general $G_\infty$.** *For general positive definite $G_\infty$ and each $j\in\{1,\ldots,k\}$,*

$$
\sin^2\angle(h_j,\,\bar{b}_j) \;\xrightarrow{a.s.}\;
\frac{\delta^2}{n\rho_j + \delta^2}
\;+\;
\frac{n\rho_j}{n\rho_j + \delta^2}\,\sin^2\angle(\hat{w}_j,\,w_j),
\tag{7}
$$

*where $\rho_j$ and $\hat{w}_j$ are the $j$-th eigenvalue and eigenvector of $\hat{M}$, and $w_j$ is the $j$-th eigenvector of $M$.*

The right-hand sides of (6) and (7) share the same structure. The first term is the floor $1/(1+\mathrm{SNR}_j)$, determined entirely by the signal-to-noise ratio $\mathrm{SNR}_j = n\rho_j/\delta^2$. The second term is the in-subspace alignment error between sample and population factor-covariance eigenvectors, scaled by $\mathrm{SNR}_j/(1+\mathrm{SNR}_j)$.

### 4.1 Worked Examples

**Example 4.1 ($k=1$, single factor).** Let $k=1$, $\beta = a\cdot\mathbf{1}_p$ for some $a > 0$, so $c = a^2$ and $G(p) = 1 = G_\infty$. The single factor-return series is $F = X\in\mathbb{R}^n$ with $\hat{D} = c\cdot\|X\|^2/n$ (a scalar), so $\rho = c\|X\|^2/n$ and $\hat{w}_1 = e_1 = 1$ trivially. The in-subspace rotation is $\sin^2\angle(1, 1) = 0$. Formula (6) reduces to

$$
\sin^2\angle(h, \bar{b}) \;\to\; \frac{\delta^2}{nc\sigma^2 + \delta^2} \;=\; \frac{1}{1 + \mathrm{SNR}},\quad \mathrm{SNR} = \frac{nc\sigma^2}{\delta^2}.
$$

Take $a = 1$ (so $c = 1$), $n = 60$, $\sigma = 0.10$, $\delta = 1.0$: $\mathrm{SNR} = 60\times 0.01/1 = 0.60$, so

$$
\sin^2\angle(h, \bar{b}) \;\to\; \frac{1}{1.60} \;\approx\; 0.625.
$$

Even with infinitely many assets, the top sample eigenvector $h$ makes an angle of $\arcsin(\sqrt{0.625})\approx 52°$ with the true loading direction. About 62.5% of $h$'s squared norm lies outside the factor subspace — not because estimation is poor, but because the signal-to-noise ratio is inherently limited at $n=60$ periods. This recovers the GPS2022 result (Corollary 2).

**Example 4.2 ($k=2$, diagonal case, orthogonal returns).** Use $k=2$ with the loading setup from Example 2.1: $c_1 = c_2 = 5$, $G_\infty = I_2$, and orthogonal factor returns ($F_1^\top F_2 = 0$). Take $n = 60$, $\sigma_1 = 0.10$, $\sigma_2 = 0.05$, $\delta = 1.0$.

| Factor | $\rho_j = c_j\sigma_j^2$ | $n\rho_j$ | $\text{floor} = \delta^2/(n\rho_j + \delta^2)$ | $\sin^2\angle(\hat{w}_j, e_j)$ |
|:------:|:------------------------:|:---------:|:----------------------------------------------:|:------------------------------:|
| 1      | $5\times0.01 = 0.05$     | $3.00$    | $1/4.00 = 0.250$                               | $0$                            |
| 2      | $5\times0.0025 = 0.0125$ | $0.75$    | $1/1.75 \approx 0.571$                         | $0$                            |

The in-subspace rotation is zero because orthogonal returns make $\hat{w}_j = e_j$. So $\sin^2\angle(h_1, \bar{b}_1) \to 0.250$ and $\sin^2\angle(h_2, \bar{b}_2) \to 0.571$. Factor 2, though sharing the same prevalence, is much more severely misaligned because its SNR of 0.75 barely clears the noise floor.

**Example 4.3 ($k=2$, in-subspace rotation with non-orthogonal returns).** Keep the same loadings but suppose $F_1^\top F_2 \ne 0$, so $\hat{D}$ is non-diagonal. Say $\sigma_{12} = 0.03$: the off-diagonal entry of $\hat{D}$ is $\sqrt{c_1 c_2}\cdot\sigma_{12} = 5\times 0.03 = 0.15$. If $\sin^2\angle(\hat{w}_1, e_1) = 0.05$, then

$$
\sin^2\angle(h_1, \bar{b}_1) \to 0.250 + \frac{3.00}{4.00}\times 0.05 = 0.250 + 0.0375 = 0.2875.
$$

The in-subspace rotation adds a 15% relative increase to the misalignment of factor 1.

---

## 5. Lemmas

The proof rests on two fundamental noise results. **Lemma 1** is the main probabilistic workhorse: a fourth-moment Borel–Cantelli argument showing that a single noise projection $\eta_p^\top Z_{\cdot\ell}/\sqrt{p}$ vanishes a.s. for any bounded deterministic sequence $\eta_p$. **Lemma 4** shows the noise Gram matrix $Z^\top Z/p$ concentrates around $\delta^2 I_n$, proved by the Kolmogorov SLLN. **Corollary 1.1** (following Lemma 1) extends the pointwise bound to the full matrix $U(p)^\top Z$, giving $\|U(p)^\top Z\|_F^2/p \to 0$ a.s. by applying Lemma 1 column-by-column.

### Lemma 1 (Bounded-vector noise concentration)

*Why needed.* Part (i)'s proof requires showing $\eta_p^\top Z_{\cdot\ell}/\sqrt{p}\to 0$ for a deterministic unit vector $\eta_p$. A second-moment Chebyshev bound fails; the fourth moment is required.

*Let $\eta_p\in\mathbb{R}^p$ be a deterministic sequence with $|\eta_p|\le C$ uniformly in $p$. Under the noise assumptions, for each fixed $l\in\{1,\ldots,n\}$,*

$$
\frac{(\eta_p^\top Z)_l}{\sqrt{p}} \;\longrightarrow\; 0 \quad a.s.
$$

*More precisely, $|(\eta_p^\top Z)_l| = o(p^{1/2-\varepsilon})$ a.s. for any fixed $\varepsilon\in(0,1/4)$.*

**Proof.** Fix $l$ and write $W_p = \sum_{i=1}^p a_i Z_{il}$ where $a_i = (\eta_p)_i$ and $\sum_i a_i^2 = |\eta_p|^2 \le C^2$.

*Step 1: Fourth moment.* Expand $\mathbb{E}[W_p^4]$. By independence of $Z_{il}$ within column $l$ and mean-zero, only two index patterns survive: all-equal ($i_1 = i_2 = i_3 = i_4$) and two-pairs. Let $\kappa_4 = \sup_{i,p}\mathbb{E}[Z_{il}^4] < \infty$. Using $\sum_i a_i^4 \le C^4$ and $\sum_{i\ne j}a_i^2 a_j^2 \le C^4$:

$$
\mathbb{E}[W_p^4] \;\le\; C^4\kappa_4 + 3C^4\delta^4 \;=:\; K \;<\;\infty,\quad\text{uniformly in }p.
$$

*Step 2: Markov.* $\Pr(|W_p| > p^{1/2-\varepsilon}) \le K/p^{2-4\varepsilon}.$

*Step 3: Borel–Cantelli.* The series $\sum_p K/p^{2-4\varepsilon}$ converges for $\varepsilon < 1/4$. Borel–Cantelli gives $|W_p| \le p^{1/2-\varepsilon}$ for all large $p$ a.s., hence $W_p/\sqrt{p} \to 0$ a.s. $\square$

*Remark.* A second-moment bound gives $\Pr(|W_p| > p^{1/2-\varepsilon}) \le C^2\delta^2/p^{1-2\varepsilon}$, summable only when $\varepsilon < 0$ — useless. The fourth moment is essential.

---

### Corollary 1.1 (Noise projection onto the signal subspace)

*Let $U(p)\in\mathbb{R}^{p\times k}$ be a deterministic sequence with orthonormal columns spanning $\mathcal{B}$. Then $\|U(p)^\top Z\|_F^2/p \to 0$ a.s. The same bound holds with $b(p)$ in place of $U(p)$: $\|b(p)^\top Z\|_F^2/p \to 0$ a.s.*

**Proof.** Each column $u_l(p)$ of $U(p)$ is a deterministic sequence of unit vectors. Lemma 1 applies to each pair $(l, m)$ with $l\in\{1,\ldots,k\}$, $m\in\{1,\ldots,n\}$: $$|u_l(p)^\top Z_{\cdot m}| = o(p^{1/2-\varepsilon}) \quad\text{a.s.}$$ The $(l,m)$ entry of $U^\top Z$ equals $u_l^\top Z_{\cdot m}$, so summing the $kn$ squared terms: $$\|U^\top Z\|_F^2 = \sum_{l=1}^k\sum_{m=1}^n |u_l^\top Z_{\cdot m}|^2 = o(p^{1-2\varepsilon}) \quad\text{a.s.}$$ Dividing by $p$ gives $\|U^\top Z\|_F^2/p \to 0$ a.s. The same argument applies to $b(p)$: each column $\beta_j/\|\beta_j\|$ is a deterministic unit vector, so Lemma 1 gives $|(\beta_j/\|\beta_j\|)^\top Z_{\cdot m}| = o(p^{1/2-\varepsilon})$ a.s., and summing $kn$ squared terms gives $\|b^\top Z\|_F^2/p \to 0$ a.s. $\square$

*Why Lemma 1 suffices (no Marcinkiewicz–Zygmund needed).* The elementary reason is that a rank-$k$ projector has bounded expected Frobenius norm: $\mathbb{E}\|\Pi_B Z\|_F^2 = \delta^2\mathrm{tr}(\Pi_B) = \delta^2 k$, independent of $p$. Lemma 1's fourth-moment Borel–Cantelli argument promotes this $O(1)$ expectation to almost-sure convergence of the normalized version $\|U^\top Z\|_F^2/p \to 0$.

---

### Lemma 4 (Noise Gram concentration)

*$Z^\top Z/p \to \delta^2 I_n$ a.s. in spectral norm.*

**Proof.** The $(j,\ell)$ entry of $Z^\top Z/p$ equals $(1/p)\sum_{i=1}^p Z_{ij}Z_{i\ell}$. For $j = \ell$ the summands $Z_{ij}^2$ are i.i.d. with mean $\delta^2$; for $j\ne\ell$ they have mean zero. The Kolmogorov SLLN gives entrywise a.s. convergence. Since $n$ is fixed, entrywise convergence implies spectral-norm convergence. $\square$

---

## 6. Proof of Part (i)

*Proof sketch.* Project the PCA equation onto the noise subspace $\mathcal{B}^\perp$. The signal term $BF^\top$ is killed exactly (since $\Pi_B^\perp B = 0$). Only the noise term $Z$ survives, and Lemma 1 shows its contribution vanishes.

**Full proof.** Decompose $v = \Pi_B v + \eta_p$ where $\eta_p = \Pi_B^\perp v$. Since $\Pi_B^\perp$ is non-expansive, $|\eta_p|\le|v|\le 1$.

Apply $\Pi_B^\perp$ to both sides of (4). The signal term vanishes: $\Pi_B^\perp B = 0$ (every column of $B$ lies in $\mathcal{B}$), so

$$
\Pi_B^\perp H\,S_p \;=\; \frac{\Pi_B^\perp Z\,\mathcal{X}_p}{\sqrt{n}}.
$$

Right-multiply by $S_p^{-1}$ and left-multiply by $v^\top$:

$$
H^\top v - H^\top\Pi_B v \;=\; S_p^{-1}\,\frac{\mathcal{X}_p^\top Z^\top \eta_p}{\sqrt{n}}. \tag{8}
$$

For each $i = 1,\ldots,k$, the $i$-th entry of the right side is $\frac{1}{s_{p,i}\sqrt{n}}\sum_{\ell=1}^n(\chi_{p,i})_\ell\,(\eta_p^\top Z_{\cdot\ell})$.

Since $|\chi_{p,i}| = 1$, each coefficient $|(\chi_{p,i})_\ell|\le 1$. By Lemma 1, $|(\eta_p^\top Z_{\cdot\ell})| = o(p^{1/2-\varepsilon})$ a.s. Summing $n$ terms: $|\chi_{p,i}^\top Z^\top\eta_p| = o(p^{1/2-\varepsilon})$ a.s.

*Lower bound on singular values.* By Weyl's inequality applied to $YY^\top/(np) = \Sigma_0^{(p)} + \text{noise}$, the $i$-th eigenvalue of $YY^\top/(np)$ satisfies $s_{p,i}^2/(np) \ge \lambda_i(\Sigma_0^{(p)}) - \|Z^\top Z/(np) - (\delta^2/n)I_p\|_{\mathrm{op}}$. The second term vanishes a.s. by Lemma 4. The $i$-th eigenvalue of $\Sigma_0^{(p)} = B\Sigma_F B^\top/p$ converges to a positive limit under Assumption 1 (it equals $c_{(i)}\sigma_{(i)}^2 + o(1)$ where the subscript denotes the $i$-th ordered value). Hence $s_{p,i}^2/p \ge c > 0$ a.s. for all large $p$, giving $s_{p,i}\asymp\sqrt{p}$ a.s.

Therefore $1/(s_{p,i}\sqrt{n}) = O(1/\sqrt{p})$ a.s., and

$$
\bigl|[H^\top v - H^\top\Pi_B v]_i\bigr| \;=\; O\!\left(\frac{p^{1/2-\varepsilon}}{\sqrt{p}}\right) = O(p^{-\varepsilon}) \;\to\; 0 \quad\text{a.s.}
$$

for each $i$. Hence $H^\top v - H^\top\Pi_B v \to 0$ a.s. $\square$

*Note.* Part (i) requires only Assumption 1 and the noise assumptions. The projection $\Pi_B^\perp$ kills the signal identically — the argument is algebraic, independent of the loading correlation structure. The per-eigenvector results of Parts (ii)–(iii) additionally require Assumptions 2 and 3 to identify which part of $\mathcal{B}$ each sample eigenvector points toward.

---

## 7. Proof of Parts (ii) and (iii)

The proof covers both Parts (ii) ($G_\infty = I_k$) and (iii) (general positive-definite $G_\infty$) in a single argument. The key device is to work in the non-orthonormalized loading frame $\Phi_p(x) = Bx/\sqrt{p}$ rather than the SVD basis $U(p)$. Because $\Phi_p^\top\Phi_p = B^\top B/p = \Gamma_p \to \Gamma_B$ for the full sequence (Assumptions 1–2), the rotational ambiguity created by coalescing singular values of $b(p)$ (when $G_\infty = I_k$) never appears. The proof proceeds in seven steps: (7.1) decompose the misalignment; (7.2–7.3) compute the limiting spectrum of the small Gram matrix $W^{(p)}$; (7.4) apply eigenprojection continuity to pass limits through eigenvectors; (7.5) establish convergence of both the sample and population in-subspace directions in $\Gamma_B$ coordinates; (7.6) compute the floor and in-subspace angle; (7.7) assemble. Steps 7.2–7.4 operate on the fixed-size $n\times n$ matrix $W^{(p)}$, where finite-dimensionality makes spectral analysis tractable.

### 7.1 Parallel/Perpendicular Decomposition

Since $\bar{b}_j\in\mathrm{col}(U(p))$, write $h_j^\perp = (I-\Pi_U)h_j$ and $h_j^\| = \Pi_U h_j$. The angle-decomposition identity gives:

$$
\sin^2\angle(h_j,\,\bar{b}_j) \;=\; \|h_j^\perp\|^2 \;+\; \|h_j^\|\|^2\,\sin^2\!\angle\!\left(\frac{h_j^\|}{\|h_j^\|\|},\,\bar{b}_j\right). \tag{9}
$$

This follows from $\sin^2\angle(h,b) = 1-\langle h,b\rangle^2$, $\langle h_j^\perp, \bar{b}_j\rangle = 0$ (since $\bar{b}_j\in\mathrm{col}(U)$ and $h_j^\perp\perp\mathrm{col}(U)$), and $\|h_j^\perp\|^2 + \|h_j^\|\|^2 = 1$.

### 7.2 Expansion of $W^{(p)}$ and Its Limit

Define $\Gamma_B := C^{1/2}G_\infty C^{1/2}$. Substitute $Y = BF^\top + Z$ into $W^{(p)} = Y^\top Y/(np)$:

$$
W^{(p)} = \underbrace{\frac{F\,(B^\top B/p)\,F^\top}{n}}_{(\mathrm{A})} \;+\; \underbrace{\frac{F\,B^\top Z + Z^\top B\,F^\top}{np}}_{(\mathrm{B})} \;+\; \underbrace{\frac{Z^\top Z}{np}}_{(\mathrm{C})}. \tag{10}
$$

**Term (A).** $B^\top B/p = \Gamma_p \to \Gamma_B$ by Assumptions 1–2, giving

$$
(\mathrm{A}) \;\longrightarrow\; \frac{F\,\Gamma_B\,F^\top}{n}.
$$

When $G_\infty = I_k$: $\Gamma_B = C$, so $F\Gamma_B F^\top/n = FCF^\top/n = (F^\#)^\top F^\#/n$. For general $G_\infty$: $\Gamma_B = C^{1/2}G_\infty C^{1/2}$, so $F\Gamma_B F^\top/n = (F^\#)^\top G_\infty F^\#/n$. Both cases are covered by the single formula $F\Gamma_B F^\top/n$.

**Term (B).** Each column $b_j(p) = \beta_j/\|\beta_j\|$ of $b(p)$ is a deterministic unit vector. Corollary 1.1 gives $\|b(p)^\top Z\|_F^2/p \to 0$ a.s. Since $B^\top Z = A(p)\cdot b(p)^\top Z$ and $A(p)/\sqrt{p} \to C^{1/2}$ (bounded), $\|B^\top Z\|_F/(p\sqrt{n}) \to 0$ a.s., so term (B) vanishes in spectral norm.

**Term (C).** By Lemma 4, $(\mathrm{C}) \to (\delta^2/n)I_n$ a.s.

Combining:

$$
W^{(p)} \;\longrightarrow\; W_\infty \;:=\; \frac{F\,\Gamma_B\,F^\top}{n} \;+\; \frac{\delta^2}{n}\,I_n \quad\text{a.s. in spectral norm.} \tag{11}
$$

### 7.3 Eigenstructure of $W_\infty$

**Lemma 7** (Eigenstructure of $W_\infty$). *The matrix $W_\infty = F\Gamma_B F^\top/n + (\delta^2/n)I_n$ has eigenvalues $\tau_j = \rho_j + \delta^2/n$ for $j\le k$ and $\delta^2/n$ for $j > k$, where $\rho_1 > \cdots > \rho_k > 0$ are the eigenvalues of $\hat{M} = \Lambda_G^{1/2}(Q^\top\hat{D}Q)\Lambda_G^{1/2}$. The top-$k$ eigenvectors are*

$$
v_j \;=\; \frac{F\,C^{1/2}Q\,\Lambda_G^{1/2}\hat{w}_j}{\sqrt{n\rho_j}} \;=\; \frac{(F^\#)^\top Q\,\Lambda_G^{1/2}\hat{w}_j}{\sqrt{n\rho_j}}, \quad j\in\{1,\ldots,k\}, \tag{12}
$$

*where $\hat{w}_j$ is the $j$-th eigenvector of $\hat{M}$ and $(F^\#)^\top = FC^{1/2}\in\mathbb{R}^{n\times k}$.* When $G_\infty = I_k$: $\hat{M} = \hat{D}$, $Q = I_k$, $\Lambda_G = I_k$, and (12) reduces to $v_j = (F^\#)^\top\hat{w}_j/\sqrt{n\rho_j}$.

**Proof.** Write $\Gamma_B = PP^\top$ with $P = C^{1/2}Q\Lambda_G^{1/2} \in \mathbb{R}^{k\times k}$, so $W_\infty - (\delta^2/n)I_n = (FP)(FP)^\top/n$ with $FP \in \mathbb{R}^{n\times k}$.

*Step 1: Nonzero eigenvalues via AB/BA.* The nonzero eigenvalues of $(FP)(FP)^\top/n$ equal those of $(FP)^\top(FP)/n$ (AB/BA: if $ABx = \rho x$ with $\rho\ne 0$, then $BA(Ax) = \rho(Ax)$). Compute: $$\frac{(FP)^\top(FP)}{n} \;=\; P^\top\!\left(\frac{F^\top F}{n}\right)P \;=\; \Lambda_G^{1/2}Q^\top\cdot\hat{D}\cdot Q\Lambda_G^{1/2} \;=\; \hat{M},$$ using $F^\top F/n = C^{-1/2}\hat{D}C^{-1/2}$ (from $\hat{D} = C^{1/2}(F^\top F/n)C^{1/2}$) and $P = C^{1/2}Q\Lambda_G^{1/2}$. Hence the nonzero eigenvalues of $W_\infty - (\delta^2/n)I_n$ are exactly $\rho_1,\ldots,\rho_k$.

*Step 2: Verify $v_j$ is an eigenvector.* Since $\hat{M}\hat{w}_j = \rho_j\hat{w}_j$, we have $(FP)^\top(FP)\hat{w}_j = n\rho_j\hat{w}_j$. Set $v_j = (FP)\hat{w}_j/\sqrt{n\rho_j}$. Then: $$\frac{(FP)(FP)^\top}{n}\,v_j \;=\; \frac{(FP)\,(FP)^\top(FP)\,\hat{w}_j}{n\sqrt{n\rho_j}} \;=\; \frac{(FP)\,n\rho_j\,\hat{w}_j}{n\sqrt{n\rho_j}} \;=\; \rho_j\,v_j.$$ Expanding $FP = FC^{1/2}Q\Lambda_G^{1/2} = (F^\#)^\top Q\Lambda_G^{1/2}$ gives formula (12).

A key identity used in §7.6: from $\hat{M}\hat{w}_j = \rho_j\hat{w}_j$, left-multiplying successively by $\Lambda_G^{-1/2}$ and $Q$ gives: $$\hat{D}\,Q\Lambda_G^{1/2}\hat{w}_j \;=\; Q\,\rho_j\,\Lambda_G^{-1/2}\hat{w}_j. \tag{$*$}$$

*Step 3: Unit norm.* $\|v_j\|^2 = \hat{w}_j^\top(FP)^\top(FP)\hat{w}_j/(n\rho_j) = \hat{w}_j^\top\hat{M}\hat{w}_j/\rho_j = 1$.

*Step 4: Bottom eigenvalues.* Any $u \perp \mathrm{col}(FP)$ satisfies $(FP)(FP)^\top u = 0$, so $W_\infty u = (\delta^2/n)u$. $\square$

*Intuition.* $W_\infty$ is a rank-$k$ perturbation of the noise floor $(\delta^2/n)I_n$. The top-$k$ eigenvalues $\rho_j + \delta^2/n$ are signal-plus-noise; the remaining $n-k$ equal $\delta^2/n$. The gap $\rho_k > 0$ enables eigenvector convergence in step 7.4. The $\hat{M}$ eigenvalues $\rho_j$ encode the signal strengths under both the factor covariance $\hat{D}$ and the loading geometry $G_\infty$; when $G_\infty = I_k$ they reduce to eigenvalues of $\hat{D}$ directly.

### 7.4 Spectral Convergence

By (11), $W^{(p)}\to W_\infty$ a.s. in operator norm on the fixed finite-dimensional space $\mathbb{R}^{n\times n}$. The top-$k$ eigenvalues of $W_\infty$ are simple (Assumption 3 is equivalent to saying $\hat{M}$ has $k$ distinct positive eigenvalues, which by Lemma 7 are exactly the top-$k$ eigenvalues of $W_\infty - (\delta^2/n)I_n$) and separated from the noise level $\delta^2/n$ by the spectral gap $\rho_k > 0$.

**Eigenprojection continuity** (Kato 1995, §II.1). *If $A_p \to A_\infty$ in operator norm and $\tau$ is a simple eigenvalue of $A_\infty$ separated from the rest of the spectrum by a gap $\gamma > 0$, then the eigenprojection of $A_p$ onto the corresponding eigenspace converges in operator norm to that of $A_\infty$. In particular, the eigenvalue converges to $\tau$ and the eigenvector converges (up to sign) to that of $A_\infty$.*

This is the qualitative version only — no explicit angular bound is produced. The eigenvalue convergence alone follows from Weyl's inequality applied to $W^{(p)} \to W_\infty$: since $s_{p,j}^2/p$ is the $j$-th eigenvalue of $W^{(p)}$ and $\tau_j$ is the $j$-th eigenvalue of $W_\infty$, $|s_{p,j}^2/p - \tau_j| \le \|W^{(p)} - W_\infty\|_{\mathrm{op}} \to 0$ a.s.

Applying eigenprojection continuity with gap $\gamma = \rho_k > 0$, for each $j\in\{1,\ldots,k\}$:

$$
\frac{s_{p,j}^2}{p} \;\longrightarrow\; \tau_j = \rho_j + \frac{\delta^2}{n}, \qquad \chi_{p,j} \;\longrightarrow\; v_j \quad\text{a.s., up to sign.} \tag{13}
$$

### 7.5 The $\Gamma_B$-Coordinate Framework

*The loading map.* Define $\Phi_p : \mathbb{R}^k \to \mathcal{B}$ by
$$
\Phi_p(x) \;=\; \frac{Bx}{\sqrt{p}},
$$
so that $\Phi_p^\top\Phi_p = B^\top B/p = \Gamma_p \to \Gamma_B$ (Assumptions 1–2). The map $\Phi_p$ is not orthonormalized: it carries no rotational ambiguity, and $\Gamma_p$ is the natural inner-product matrix on $\mathbb{R}^k$ for expressing lengths and angles inside $\mathcal{B}$. Both the population loading direction and the in-subspace component of $h_j$ can be expressed as $\Phi_p$-images of convergent $k$-vectors, which is the content of the next two paragraphs.

*Population direction.* Since $\bar{b}_j \in \mathcal{B}$ and $B$ has full column rank, write $\bar{b}_j = \Phi_p(a_j^{(p)})$ uniquely. Substituting into the eigenvalue equation $\Sigma_0^{(p)}\bar{b}_j = \lambda_j\bar{b}_j$ (with $\Sigma_0^{(p)} = B\Sigma_F B^\top/p$) and left-multiplying by $B^\top/\sqrt{p}$:
$$
(B^\top B/p)\,\Sigma_F\,(B^\top B/p)\,a_j^{(p)} \;=\; \lambda_j\,(B^\top B/p)\,a_j^{(p)},
$$
which (since $\Gamma_p$ is invertible for large $p$) simplifies to
$$
\Sigma_F\,\Gamma_p\,a_j^{(p)} \;=\; \lambda_j\,a_j^{(p)}.
$$
The normalization $\|\bar{b}_j\|^2 = (a_j^{(p)})^\top\Gamma_p a_j^{(p)} = 1$ holds since $\bar{b}_j$ is a unit vector.

As $\Gamma_p \to \Gamma_B$, the matrix $\Sigma_F\Gamma_p \to \Sigma_F\Gamma_B$ in operator norm. The eigenvalues of $\Sigma_F\Gamma_B$ are simple. To see this: by AB/BA, eigenvalues of $\Sigma_F\Gamma_B$ equal eigenvalues of $\Gamma_B\Sigma_F = C^{1/2}G_\infty D C^{-1/2}$ (using $D = C^{1/2}\Sigma_F C^{1/2}$), which by successive conjugation (first by $C^{-1/2}$, then by $Q^\top$, then by $\Lambda_G^{-1/2}$) equal the eigenvalues of $M = \Lambda_G^{1/2}(Q^\top DQ)\Lambda_G^{1/2}$. Assumption 3 says exactly that $M$ has $k$ distinct positive eigenvalues. Eigenprojection continuity (Kato §II.1) applied to $\Sigma_F\Gamma_p \to \Sigma_F\Gamma_B$ therefore gives convergence of the $j$-th eigenvector for the **full sequence** (no subsequence):
$$
a_j^{(p)} \;\longrightarrow\; a_j^\infty \quad\text{a.s.,}
$$
where $\Sigma_F\Gamma_B a_j^\infty = \lambda_j a_j^\infty$ and $(a_j^\infty)^\top\Gamma_B a_j^\infty = 1$. One can verify that $a_j^\infty = C^{-1/2}Q\Lambda_G^{-1/2}w_j$, where $w_j$ is the $j$-th eigenvector of $M$: substituting into $\Sigma_F\Gamma_B a_j^\infty = \lambda_j a_j^\infty$ and using the eigen-equation for $M$ confirms this. (The $\Gamma_B$-normalization $(a_j^\infty)^\top\Gamma_B a_j^\infty = w_j^\top w_j = 1$ holds since $w_j$ is a unit eigenvector of $M$.)

*Sample direction.* From (4), the in-subspace component of $h_j$ is $\Pi_B h_j = \Pi_B Y\chi_{p,j}/(\sqrt{n}s_{p,j})$. Using $Y = BF^\top + Z$ and $\Pi_B B = B$:
$$
\Pi_B h_j \;=\; \frac{BF^\top\chi_{p,j}}{\sqrt{n}\,s_{p,j}} \;+\; \frac{\Pi_B Z\chi_{p,j}}{\sqrt{n}\,s_{p,j}}.
$$

For the noise term: each column $b_l = \beta_l/\|\beta_l\|$ of $b(p)$ is a deterministic unit vector, so Corollary 1.1 gives $\|b(p)^\top Z\|_F^2/p \to 0$ a.s., hence $\|\Pi_B Z\chi_{p,j}\|/\sqrt{p} \to 0$ a.s. Since $s_{p,j} \asymp \sqrt{p}$ a.s. (from §7.4), this noise term is $o(1)$ a.s.

For the signal term: $BF^\top\chi_{p,j}/(\sqrt{n}s_{p,j}) = \Phi_p(g_j^{(p)})$ where
$$
g_j^{(p)} \;=\; \frac{\sqrt{p}\,F^\top\chi_{p,j}}{\sqrt{n}\,s_{p,j}}.
$$
Therefore $\Pi_B h_j = \Phi_p(g_j^{(p)}) + o(1)$ a.s.

*Convergence of $g_j^{(p)}$.* Since $s_{p,j}^2/p \to \rho_j + \delta^2/n$ a.s. and $\chi_{p,j} \to v_j$ a.s. by (13), we have $\sqrt{p}/s_{p,j} \to 1/\sqrt{\rho_j + \delta^2/n}$ a.s. and $F^\top\chi_{p,j} \to F^\top v_j$. Hence:
$$
g_j^{(p)} \;\longrightarrow\; g_j^\infty \;:=\; \frac{F^\top v_j}{\sqrt{n\rho_j + \delta^2}} \quad\text{a.s.}
$$
This limit holds for the **full sequence**: every factor in (13) converges without subsequence extraction.

### 7.6 Floor and In-Subspace Angle via $\Gamma_B$ Inner Products

With $g_j^{(p)} \to g_j^\infty$, $a_j^{(p)} \to a_j^\infty$, and $\Gamma_p \to \Gamma_B$ all holding for the full sequence, joint continuity of bilinear forms gives:
$$
\|\Pi_B h_j\|^2 \;=\; (g_j^{(p)})^\top\Gamma_p\,g_j^{(p)} + o(1) \;\longrightarrow\; (g_j^\infty)^\top\Gamma_B\,g_j^\infty \quad\text{a.s.}
$$
$$
\langle\Pi_B h_j,\,\bar{b}_j\rangle \;=\; (g_j^{(p)})^\top\Gamma_p\,a_j^{(p)} + o(1) \;\longrightarrow\; (g_j^\infty)^\top\Gamma_B\,a_j^\infty \quad\text{a.s.}
$$

The following three steps evaluate these inner products explicitly, using the formula for $v_j$ from (12).

*Step 1: Compute $g_j^\infty$ explicitly.* From (12), $v_j = FC^{1/2}Q\Lambda_G^{1/2}\hat{w}_j/\sqrt{n\rho_j}$. Apply $F^\top$ (dimension: $k\times n$) to get $F^\top v_j \in \mathbb{R}^k$:
$$
F^\top v_j \;=\; \frac{F^\top F\cdot C^{1/2}Q\Lambda_G^{1/2}\hat{w}_j}{\sqrt{n\rho_j}}.
$$
Use $F^\top F = nC^{-1/2}\hat{D}C^{-1/2}$ (from $\hat{D} = C^{1/2}(F^\top F/n)C^{1/2}$) and the key identity $(*)$ from Lemma 7, $\hat{D}Q\Lambda_G^{1/2}\hat{w}_j = Q\rho_j\Lambda_G^{-1/2}\hat{w}_j$:
$$
F^\top v_j \;=\; \frac{n\,C^{-1/2}\hat{D}\,Q\Lambda_G^{1/2}\hat{w}_j}{\sqrt{n\rho_j}} \;=\; \frac{n\,C^{-1/2}\cdot Q\rho_j\Lambda_G^{-1/2}\hat{w}_j}{\sqrt{n\rho_j}} \;=\; \sqrt{n\rho_j}\cdot C^{-1/2}Q\Lambda_G^{-1/2}\hat{w}_j.
$$
Dividing by $\sqrt{n\rho_j + \delta^2}$:
$$
g_j^\infty \;=\; \sqrt{\frac{n\rho_j}{n\rho_j+\delta^2}}\cdot C^{-1/2}Q\Lambda_G^{-1/2}\hat{w}_j. \tag{14}
$$

*Step 2: In-subspace squared norm and floor.* Compute $(g_j^\infty)^\top\Gamma_B g_j^\infty$. Since $\Gamma_B = C^{1/2}Q\Lambda_G Q^\top C^{1/2}$, the middle factor simplifies as $\Lambda_G^{-1/2}Q^\top C^{-1/2}\cdot C^{1/2}Q\Lambda_G Q^\top C^{1/2}\cdot C^{-1/2}Q\Lambda_G^{-1/2} = I_k$. Therefore:
$$
(g_j^\infty)^\top\Gamma_B g_j^\infty \;=\; \frac{n\rho_j}{n\rho_j+\delta^2}\cdot\hat{w}_j^\top I_k\,\hat{w}_j \;=\; \frac{n\rho_j}{n\rho_j+\delta^2}.
$$
So $\|\Pi_B h_j\|^2 \to n\rho_j/(n\rho_j+\delta^2)$ a.s. The floor then follows from $\|h_j\| = 1$:
$$
\|h_j^\perp\|^2 \;=\; 1 - \|\Pi_B h_j\|^2 \;\longrightarrow\; \frac{\delta^2}{n\rho_j+\delta^2} \quad\text{a.s.} \tag{15}
$$

*Step 3: In-subspace inner product.* Recall $a_j^\infty = C^{-1/2}Q\Lambda_G^{-1/2}w_j$. The same simplification $\Lambda_G^{-1/2}Q^\top C^{-1/2}\cdot\Gamma_B\cdot C^{-1/2}Q\Lambda_G^{-1/2} = I_k$ gives:
$$
(g_j^\infty)^\top\Gamma_B a_j^\infty \;=\; \sqrt{\frac{n\rho_j}{n\rho_j+\delta^2}}\cdot\hat{w}_j^\top I_k\,w_j \;=\; \sqrt{\frac{n\rho_j}{n\rho_j+\delta^2}}\cdot\hat{w}_j^\top w_j.
$$
So $\langle\Pi_B h_j, \bar{b}_j\rangle \to \sqrt{n\rho_j/(n\rho_j+\delta^2)}\cdot\hat{w}_j^\top w_j$ a.s.

*Step 4: In-subspace angle.* Using (9) and the limits from Steps 2–3:
$$
\sin^2\!\angle\!\left(\frac{\Pi_B h_j}{\|\Pi_B h_j\|},\,\bar{b}_j\right) \;=\; 1 - \frac{\langle\Pi_B h_j, \bar{b}_j\rangle^2}{\|\Pi_B h_j\|^2} \;\longrightarrow\; 1 - \frac{(n\rho_j/(n\rho_j+\delta^2))\cdot(\hat{w}_j^\top w_j)^2}{n\rho_j/(n\rho_j+\delta^2)} \;=\; \sin^2\!\angle(\hat{w}_j,\,w_j).
$$

*Diagonal-Gram specialization.* When $G_\infty = I_k$: $Q = I_k$, $\Lambda_G = I_k$, $w_j = e_j$ (since $M = D$ is diagonal under the ordering Assumption 3). Then $a_j^\infty = C^{-1/2}e_j = e_j/\sqrt{c_j}$, $\hat{w}_j^\top w_j = (\hat{w}_j)_j$, and $\sin^2\angle(\hat{w}_j, w_j) = \sin^2\angle(\hat{w}_j, e_j) = 1 - (\hat{w}_j)_j^2$, recovering formula (6).

### 7.7 Assembly

Substituting (15) and Step 4 of §7.6 into the decomposition (9):
$$
\sin^2\angle(h_j,\,\bar{b}_j) \;=\; \|h_j^\perp\|^2 \;+\; \|\Pi_B h_j\|^2\cdot\sin^2\!\angle\!\left(\frac{\Pi_B h_j}{\|\Pi_B h_j\|},\,\bar{b}_j\right)
$$
$$
\;\longrightarrow\; \frac{\delta^2}{n\rho_j+\delta^2} \;+\; \frac{n\rho_j}{n\rho_j+\delta^2}\cdot\sin^2\!\angle(\hat{w}_j,\,w_j) \quad\text{a.s.}
$$

This is formula (7) for general $G_\infty$; formula (6) follows when $G_\infty = I_k$ as noted in §7.6. All limits hold for the full sequence — no subsequence extraction is needed, because the $\Gamma_B$-coordinate framework operates directly on $\Gamma_p \to \Gamma_B$, bypassing the rotational ambiguity of $V(p)$ in the SVD basis. $\square$

---

## 8. Unification of Parts (ii) and (iii)

The proof of §7 covers both Parts (ii) ($G_\infty = I_k$) and (iii) (general positive-definite $G_\infty$) in a single argument.

The two parts differ only in the value of $\Gamma_B = C^{1/2}G_\infty C^{1/2}$:

- **Part (ii)** ($G_\infty = I_k$): $\Gamma_B = C$, $\hat{M} = \hat{D}$, $M = D$, $w_j = e_j$, and $\sin^2\angle(\hat{w}_j, w_j) = \sin^2\angle(\hat{w}_j, e_j) = 1 - (\hat{w}_j)_j^2$. Formula (7) reduces to formula (6).

- **Part (iii)** (general $G_\infty$): $\Gamma_B$ is a general positive-definite matrix, and formula (7) holds as stated.

The proof of §7 treats $\Gamma_B$ as an arbitrary positive-definite matrix throughout — the diagonal vs. non-diagonal distinction plays no role. In the earlier proof architecture (following the SVD basis $U(p)$), Parts (ii) and (iii) required separate treatment because $V(p)$ — the right singular factor of $b(p) = U(p)\Sigma(p)V(p)^\top$ — has a rotational ambiguity when $G_\infty = I_k$ makes the singular values of $b(p)$ coalesce. Working in $\Gamma_B$ coordinates removes this artifact entirely: $\Gamma_p = B^\top B/p$ converges to $\Gamma_B$ for the full sequence, independently of any singular-value ordering, and neither the SVD of $b(p)$ nor the matrix $V(p)$ ever appears in the argument. $\square$

---

## 9. Recovery of NG's Theorem 3.1′

Under NG's Assumptions 2.5′ (orthogonal loading columns) and 2.6′ (orthogonal factor returns), both in-subspace rotation and off-diagonal structure disappear.

**Assumption 2.5′ implies $G_\infty = I_k$.** If $B$'s columns are orthogonal, $G(p)_{ij} = \langle b_i, b_j\rangle = \delta_{ij}$ exactly for all $p$, so $G_\infty = I_k$.

**Assumption 2.6′ kills the in-subspace rotation.** Under 2.6′, factor returns are orthogonal: $F_i^\top F_j = 0$ for $i\ne j$. This makes $F^\top F/n$ diagonal: $(F^\top F/n)_{ij} = (\|F_j\|^2/n)\delta_{ij}$. Hence $\hat{D} = C^{1/2}(F^\top F/n)C^{1/2} = \mathrm{diag}(c_j\|F_j\|^2/n)$ is diagonal, with eigenvectors $\hat{w}_j = e_j$. Similarly $D = \mathrm{diag}(c_j\sigma_j^2)$ is diagonal with $w_j = e_j$. Therefore $\sin^2\angle(\hat{w}_j, e_j) = 0$ and the in-subspace term vanishes.

*Intuition.* When factor returns are orthogonal, the sample factor covariance is diagonal — each factor is its own factor with no mixing. No in-subspace rotation occurs. When returns are correlated, the factor covariance must be diagonalized, producing non-trivial eigenvectors $\hat{w}_j$ and $w_j$ and hence non-zero in-subspace rotation.

With $G_\infty = I_k$ and $\sin^2\angle(\hat{w}_j, e_j) = 0$, formula (6) reduces to:

$$
\sin^2\angle(h_j, \bar{b}_j) \;\to\; \frac{\delta^2}{n\rho_j + \delta^2},
$$

where $\rho_j = c_j\|F_j\|^2/n$. Using NG's notation $\alpha_j^2 := c_j$ and $|X_j|^2 := \|F_j\|^2$, this is $\delta^2/(\alpha_j^2|X_j|^2+\delta^2)$. Since $\bar{b}_j \to b_j := \beta_j/\|\beta_j\|$ under 2.5′:

$$
\cos^2\angle(h_j, b_j) \;\to\; \frac{\alpha_j^2|X_j|^2}{\alpha_j^2|X_j|^2+\delta^2} \;=:\; \psi_{\infty,j}^2.
$$

Assembling into matrix form:

$$
H^\top\tilde{B} \;\to\; \mathrm{diag}(\psi_{\infty,1},\ldots,\psi_{\infty,k}),
$$

which is exactly Part (ii) of NG's Theorem 3.1′. $\square$

*Remark.* NG's Theorem 3.1′ Part (i) follows from Part (i) of the unified theorem with no additional assumptions. In particular, Part (i) of NG's Theorem 3.1′ does not require Assumptions 2.5′ or 2.6′.

---

## 10. Corollaries

### Corollary 1 (Out-of-subspace floor and its observable estimator)

*Under the assumptions of Parts (ii)–(iii), almost surely,*

$$
\liminf_{p\to\infty}\sin^2\angle(h_j,\bar{b}_j) \;\ge\; \frac{\delta^2}{n\rho_j+\delta^2}.
$$

*This lower bound is observable: with $\ell_p^2 := \frac{1}{n-k}\sum_{j=k+1}^n s_{p,j}^2$ and*

$$
\hat{\psi}_{p,j}^2 \;:=\; 1 - \frac{\ell_p^2}{s_{p,j}^2},
$$

*we have $\hat{\psi}_{p,j}^2 \xrightarrow{a.s.} n\rho_j/(n\rho_j+\delta^2) = 1 - \delta^2/(n\rho_j+\delta^2)$.*

**Proof.** The lower bound follows from (6) and (7) since the in-subspace term is non-negative. For the observable estimator: the noise singular values $s_{p,j}$ for $j > k$ satisfy $s_{p,j}^2/p\to\delta^2/n$ a.s. (they are eigenvalues of $W^{(p)}$ converging to $\delta^2/n$). Hence $\ell_p^2/p\to\delta^2/n$ and $s_{p,j}^2/p\to\rho_j+\delta^2/n$, giving $\hat{\psi}_{p,j}^2\to n\rho_j/(n\rho_j+\delta^2)$ a.s. $\square$

*Note.* $\hat{\psi}_{p,j}^2$ requires only the sample singular values of $Y$ — no knowledge of $D$, $G_\infty$, $C$, or $Q$.

### Corollary 2 (GPS2022 recovery, $k=1$)

*Setting $k=1$: $B = \beta\in\mathbb{R}^p$, $F = X\in\mathbb{R}^{n\times 1}$, $c = \lim\|\beta\|^2/p$. Then*

$$
\sin^2\angle(h, \bar{b}) \;\to\; \frac{\delta^2}{c\|X\|^2 + \delta^2}.
$$

*This recovers the main result of Goldberg–Papanicolaou–Shkolnik (2022).*

**Proof.** $\hat{D} = c\|X\|^2/n$ is a scalar, so $\rho_1 = c\|X\|^2/n$ and $\sin^2\angle(\hat{w}_1,e_1) = 0$. Substituting gives $\delta^2/(c\|X\|^2+\delta^2)$. $\square$

### Corollary 3 (Dispersion bias — NG's Theorem 3.1′ corollary)

The *dispersion bias* for portfolio $z$ is $|\Pi_B z|^2 - |\Pi_H z|^2$: the squared loading-subspace exposure minus the squared sample-eigenvector exposure.

*Under the assumptions of §9, with $z = e/\sqrt{p}$ (the equal-weight portfolio) and $c_i^{\mathrm{ew}} := \langle b_i, z\rangle$:*

$$
|\Pi_B z|^2 - |\Pi_H z|^2 \;\to\; \sum_{i=1}^k (1-\psi_{\infty,i}^2)\,(c_i^{\mathrm{ew}})^2 \;>\; 0 \quad a.s.
$$

**Proof.** From Part (i) at $v = z$: $|H^\top z|^2\to|H^\top\Pi_B z|^2$. From §9, $H^\top\tilde{B}\to\mathrm{diag}(\psi_{\infty,i})$ and $\langle b_i, z\rangle \to c_i^{\mathrm{ew}}$. Hence $|H^\top\Pi_B z|^2\to\sum_i\psi_{\infty,i}^2 (c_i^{\mathrm{ew}})^2$ and $|\Pi_B z|^2\to\sum_i(c_i^{\mathrm{ew}})^2$. The deficit $\sum_i(1-\psi_{\infty,i}^2)(c_i^{\mathrm{ew}})^2 > 0$ since each $\psi_{\infty,i}^2 < 1$ and $c_i^{\mathrm{ew}} > 0$. $\square$

*Example.* Using Example 4.2: $(1-0.750)(0.8) + (1-0.429)(0.2) = 0.200 + 0.114 = 0.314$.

### Corollary 4 (Grassmannian subspace distance)

*Under the assumptions of Parts (ii)–(iii), almost surely,*

$$
d_{\mathrm{Gr}}^2\bigl(\mathrm{col}(H),\,\mathcal{B}\bigr) \;:=\; \sum_{j=1}^k \sin^2\theta_j \;\longrightarrow\; \sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2},
$$

*where $\theta_1, \ldots, \theta_k$ are the principal angles between $\mathrm{col}(H)$ and $\mathcal{B}$.*

**Proof.** The squared chordal distance on $\mathrm{Gr}(k,p)$ satisfies the identity

$$
\sum_{j=1}^k \sin^2\theta_j \;=\; \sum_{j=1}^k \|h_j^\perp\|^2,
$$

where $h_j^\perp = (I - \Pi_B)h_j$ is the out-of-subspace component of $h_j$. To see this: the principal angles are defined by the singular values of $H^\top U$ (where $U$ spans $\mathcal{B}$), with $\cos\theta_j = \sigma_j(H^\top U)$. Hence $\sum_j\cos^2\theta_j = \|H^\top U\|_F^2 = \mathrm{tr}(H^\top\Pi_B H) = \|\Pi_B H\|_F^2 = \sum_j\|h_j^\|\|^2$. Since $\|h_j^\perp\|^2 + \|h_j^\|\|^2 = 1$, summing gives $\sum_j\sin^2\theta_j = \sum_j\|h_j^\perp\|^2$.

The limit now follows directly from equation (15): $\|h_j^\perp\|^2\to\delta^2/(n\rho_j+\delta^2)$ a.s. for each $j$, and $k$ is fixed. $\square$

*Remark.* The in-subspace rotation term in the per-vector formula (6)/(7) does not appear in Corollary 4. This is not a coincidence: the Grassmannian distance depends only on $\Pi_H = HH^\top$, which is invariant under orthogonal rotations of the columns of $H$ within $\mathrm{col}(H)$. The in-subspace rotation is precisely such a rotation — it displaces each $h_j$ within $\mathcal{B}$ without changing the subspace $\mathrm{col}(H)$. Corollary 4 therefore provides a strictly weaker error criterion than the per-vector formulas, and achieves a correspondingly smaller limiting error.

---

### Corollary 5 (Frame-Level Dispersion Bias)

**Setup.** Let $W \in \mathbb{R}^{p \times k_W}$ be a deterministic probe frame with $1 \le k_W \le n$ orthonormal columns ($W^\top W = I_{k_W}$). Assume the asymptotic alignment matrix

$$
\Gamma_\infty \;:=\; \lim_{p \to \infty} \tilde{B}^\top W \;\in\; \mathbb{R}^{k \times k_W}
$$

exists, where $\tilde{B} = b(p)$ is the matrix of unit loading columns. (Sufficient condition: $\langle b_j, w_l\rangle \to \gamma_j^{(l)}$ for each $j$, $l$, in which case $(\Gamma_\infty)_{jl} = \gamma_j^{(l)}$.) Part (i) below holds under the weaker condition $\|W\|_{\mathrm{op}} \le 1$ without requiring $W^\top W = I_{k_W}$.

*Note on notation.* $\Gamma_\infty$ is a new $k \times k_W$ matrix; it does not conflict with $C = \mathrm{diag}(c_j)$, the diagonal prevalence matrix from §2. The shorthand $\Psi_\infty := \mathrm{diag}(\psi_{\infty,1},\ldots,\psi_{\infty,k})$ is used throughout; recall $\psi_{\infty,j}^2 = n\rho_j/(n\rho_j+\delta^2)$.

**Part (i): Subspace alignment of probe frame.** *Under Assumption 1 and the noise assumptions only (no orthogonality conditions on $B$ or $F$), and requiring only $\|W\|_{\mathrm{op}} \le 1$:*

$$
H^\top W \;-\; H^\top \Pi_B W \;\longrightarrow\; 0 \quad \text{a.s.}
$$

*in any matrix norm.*

**Part (ii): Frame factorization, NG case.** *Under the additional hypotheses $G_\infty = I_k$ and orthogonal factor returns (the setting of §9, requiring NG's Assumptions 2.5' and 2.6'):*

$$
H^\top W \;\longrightarrow\; \Psi_\infty\, \Gamma_\infty \quad \text{a.s.}
$$

**Four consequences follow.**

*(a) Frame-level Frobenius deficit.*

$$
\|\Pi_B W\|_F^2 \;-\; \|\Pi_H W\|_F^2 \;\longrightarrow\; \sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2}\,(\Gamma_\infty\Gamma_\infty^\top)_{jj} \;\ge\; 0 \quad \text{a.s.,}
$$

with strict positivity whenever $\Gamma_\infty \ne 0$.

*(b) Principal angle shrinkage.* The cosines of the principal angles between $\mathrm{span}(W)$ and $\mathrm{span}(H)$ are asymptotically the singular values of $\Psi_\infty \Gamma_\infty$:

$$
\sigma_l(H^\top W) \;\longrightarrow\; \sigma_l(\Psi_\infty \Gamma_\infty), \qquad l = 1, \ldots, \min(k, k_W).
$$

The corresponding cosines for $\mathrm{span}(W)$ vs.\ $\mathrm{span}(B)$ are $\sigma_l(\Gamma_\infty)$. The probe is therefore systematically less aligned with the sample subspace than with the population subspace.

*(c) Per-factor decomposition.* Writing $(\Gamma_\infty\Gamma_\infty^\top)_{jj} = \sum_l (\gamma_j^{(l)})^2$ for the total exposure of probe frame $W$ to factor $j$:

$$
\|\Pi_B W\|_F^2 - \|\Pi_H W\|_F^2 \;\longrightarrow\; \sum_{j=1}^k \underbrace{\frac{\delta^2}{n\rho_j + \delta^2}}_{\text{floor for factor } j} \cdot \underbrace{\sum_{l=1}^{k_W} (\gamma_j^{(l)})^2}_{\text{frame exposure to factor }j}.
$$

Factor $j$ contributes to the deficit only to the extent that $W$ has nonzero exposure to it.

*(d) Probing the population frame.* When $k_W = k$ and $W = \tilde{B}$ (the population loading frame itself), $\Gamma_\infty = I_k$ and $(\Gamma_\infty\Gamma_\infty^\top)_{jj} = 1$, giving

$$
\|\Pi_B \tilde{B}\|_F^2 - \|\Pi_H \tilde{B}\|_F^2 \;\longrightarrow\; \sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2} \;=\; d_{\mathrm{Gr}}^2(\mathrm{col}(H),\,\mathcal{B}).
$$

The frame-level deficit when the population frame probes itself equals the Grassmannian subspace distance from Corollary 4. This identifies the two error criteria at a canonical choice of probe.

**Proof.**

*Part (i).* Apply Part (i) of the unified theorem to each column $w_l$ of $W$ with bound $|w_l| \le \|W\|_{\mathrm{op}} \le 1$. Each column gives $H^\top w_l - H^\top\Pi_B w_l \to 0$ a.s. The finite intersection of $k_W$ probability-one events has probability one. Assembling column-wise gives the matrix statement; convergence in any matrix norm follows since $H^\top W$ has fixed dimensions $k \times k_W$.

*Part (ii).* From Part (i), $H^\top W = H^\top\Pi_B W + o(1)$ a.s. Under $G_\infty = I_k$ (NG's Assumption 2.5'), the unit loading columns satisfy $G(p) = b(p)^\top b(p) = I_k$ exactly, so $\Pi_B = \tilde{B}\tilde{B}^\top$ and

$$
H^\top\Pi_B W \;=\; (H^\top\tilde{B})(\tilde{B}^\top W).
$$

By the NG recovery in §9 (which additionally requires orthogonal factor returns, Assumption 2.6'), $H^\top\tilde{B} \to \Psi_\infty$ a.s. By the corollary's hypothesis, $\tilde{B}^\top W \to \Gamma_\infty$. Continuity of matrix multiplication gives $H^\top W \to \Psi_\infty\Gamma_\infty$ a.s.

*Consequences (a)–(d).* For (a): $\|\Pi_B W\|_F^2 = \|\tilde{B}^\top W\|_F^2 \to \|\Gamma_\infty\|_F^2 = \mathrm{tr}(\Gamma_\infty\Gamma_\infty^\top)$ and $\|\Pi_H W\|_F^2 = \|H^\top W\|_F^2 \to \|\Psi_\infty\Gamma_\infty\|_F^2 = \mathrm{tr}(\Gamma_\infty^\top\Psi_\infty^2\Gamma_\infty) = \sum_j\psi_{\infty,j}^2(\Gamma_\infty\Gamma_\infty^\top)_{jj}$. Subtracting gives (a) with $1 - \psi_{\infty,j}^2 = \delta^2/(n\rho_j+\delta^2)$. Parts (b)–(c) follow by continuity of singular values and direct expansion. Part (d) substitutes $\Gamma_\infty = I_k$ and invokes Corollary 4. $\square$

**Remarks.**

*1. Three phenomena at $k_W > 1$.*

*Non-isotropic shrinkage.* The shrinkage acts as left-multiplication by $\Psi_\infty$, a diagonal matrix with generally distinct entries. A probe column heavily aligned with the strong factor (large $\psi_{\infty,j}$) suffers least; one aligned with the weak factor suffers most.

*Mixed-probe shrinkage is non-linear.* For a single probe ($k_W = 1$) with equal exposure to both factors — $\Gamma_\infty = (1,\ldots,1)^\top/\sqrt{k}$ — the sample principal cosine is $\|\Psi_\infty\Gamma_\infty\| = (\sum_j\psi_{\infty,j}^2/k)^{1/2}$, the root-mean-square of the individual shrinkages. For $k=2$ with $\psi_{\infty,1} = 0.9$ and $\psi_{\infty,2} = 0.5$: $\|\Psi_\infty\Gamma_\infty\| = \sqrt{(0.81+0.25)/2} \approx 0.728$, while the population cosine is $1$. This is strictly less than the arithmetic mean $(0.9+0.5)/2 = 0.7$ — the mixed probe suffers more than naive averaging would suggest. In contrast, for an orthogonal square probe frame ($k_W = k$, $\Gamma_\infty \in O(k)$), the singular values of $\Psi_\infty\Gamma_\infty$ are always exactly $\{\psi_{\infty,1},\ldots,\psi_{\infty,k}\}$, since $(\Psi_\infty\Gamma_\infty)(\Psi_\infty\Gamma_\infty)^\top = \Psi_\infty^2$. The non-linearity is a feature of mixed, non-orthogonal, or rectangular $\Gamma_\infty$.

*Probe-specific deficit.* A probe orthogonal to $\mathcal{B}$ ($\Gamma_\infty = 0$) experiences zero bias; a probe concentrated on the weakest factor experiences the full floor $\delta^2/(n\rho_k+\delta^2)$ for that component.

*2. Special cases.* For $k_W = 1$ ($W = z$, a single probe): $\Gamma_\infty = (\gamma_1,\ldots,\gamma_k)^\top$ and the deficit reduces to $\sum_j(\delta^2/(n\rho_j+\delta^2))\gamma_j^2$ — the dispersion bias of Corollary 3. For $W$ orthogonal to $\mathcal{B}$: $\Gamma_\infty = 0$, no bias.

*3. Portfolio interpretation.* Each column of $W$ can be a target portfolio. The Frobenius deficit measures the systematic gap between true and estimated factor exposure across the basket. The per-factor decomposition in (c) identifies which factor's noise drives which component of the gap — directly actionable for multi-asset hedging, factor replication, or benchmark tracking.

*4. Extension to general $G_\infty$.* The clean formula $H^\top W \to \Psi_\infty\Gamma_\infty$ holds in the NG case only. In the full diagonal-Gram setting (Part (ii) of the unified theorem, $G_\infty = I_k$ but correlated factor returns), the limit of $H^\top\tilde{B}$ is not $\Psi_\infty$ but $\Psi_\infty\hat{W}$, where $\hat{W} = [\hat{w}_1,\ldots,\hat{w}_k]^\top$ is the matrix of row eigenvectors of $\hat{D}$. The frame-level formula then becomes $H^\top W \to \Psi_\infty\hat{W}\Gamma_\infty$, with the rotation $\hat{W}$ encoding in-subspace mixing from correlated returns. Part (i) of the corollary holds without restriction in all cases, since it requires no diagonal structure.

*5. Frame vs.\ subspace estimation.* The Frobenius deficit in (a) is a frame-level quantity depending on the orientation of $W$. As §12 discusses, the Grassmannian distance $d_{\mathrm{Gr}}^2(\mathrm{col}(H),\mathcal{B})$ is already minimized by $\mathrm{col}(H)$ with no correction needed. Consequence (d) bridges the two: the frame-level deficit, evaluated at the natural probe $W = \tilde{B}$, equals the Grassmannian floor.

*6. Data-dependent probes.* The corollary requires $W$ deterministic. If $W = W(Y)$ depends on the data, $\Pi_B^\perp w_l$ is no longer a deterministic bounded vector and Lemma 1 does not apply column-wise. Extensions to data-dependent probes — e.g., portfolios constructed from the sample — require either independence of $W$ from $Z$ or a uniform-in-$W$ argument over an appropriate class.

---

## 11. Discussion

### Asymptotic regime: HDLSS, not proportional

The theorem operates in the *high-dimension, low-sample-size* (HDLSS) regime: $p \to \infty$ with $n$ and $k$ fixed. This is the setting studied by Jung and Marron (2009, "PCA consistency in high dimension, low sample size context") and the follow-up work of Shen, Shen, and Marron. That literature establishes a qualitative trichotomy: sample PC directions are *consistent*, *strongly inconsistent*, or *subspace-consistent up to a rotation*, depending on signal-to-noise structure. The present theorem makes this trichotomy quantitative: the "floor" $\delta^2/(n\rho_j+\delta^2)$ is HDLSS inconsistency made exact, and the "rotation" $\sin^2\angle(\hat{w}_j, w_j)$ is HDLSS subspace consistency made exact.

The HDLSS regime is *not* the proportional regime $p, n \to \infty$ with $p/n \to \gamma$, which is the setting of the Baik–Ben Arous–Péché (BBP) and Benaych-Georges–Nadakuditi spiked-covariance results. Those results do not apply here: in the fixed-$n$ regime the $n \times n$ Gram matrix $W^{(p)}$ converges to a deterministic limit by the ordinary law of large numbers, with no Marchenko–Pastur bulk and no random-matrix-theoretic limit required. The fixed-$n$ world is the simpler one analytically; importing the proportional-regime machinery would use a harder theorem in the wrong regime.

### Two orthogonal sources of misalignment

The main formula decomposes $\sin^2\angle(h_j,\bar{b}_j)$ into two terms with distinct origins and distinct remedies.

The **out-of-subspace floor** $\delta^2/(n\rho_j+\delta^2) = 1/(1+\mathrm{SNR}_j)$ is irreducible: it is determined entirely by $\mathrm{SNR}_j = n\rho_j/\delta^2$ and cannot be reduced by improving estimation of the factor covariance. As $n\to\infty$, this floor decays to zero. As $\delta^2\to 0$, it also decays. No amount of additional assets $p$ can reduce it.

The **in-subspace rotation** $n\rho_j/(n\rho_j+\delta^2)\cdot\sin^2\angle(\hat{w}_j,w_j)$ is a finite-$n$ artifact. It vanishes in the large-$n$ limit (since $\hat{D}\to D$ by the law of large numbers). In the noiseless limit $\delta^2\to 0$, this term *survives* and equals $\sin^2\angle(\hat{w}_j,w_j)$, dominating the total misalignment.

The two terms suggest different remedies: increase $n$ to reduce the rotation, or increase signal strength $\rho_j$ (e.g., by selecting factors with high cross-sectional prevalence) to reduce the floor.

### The role of loading geometry

The general-$G_\infty$ formula (7) differs from the diagonal-Gram formula (6) in the matrices $\hat{M}$, $M$ replacing $\hat{D}$, $D$. The loading geometry $G_\infty = Q\Lambda_G Q^\top$ rotates and reweights the factor covariance. When all loading directions are equally represented ($G_\infty = I_k$), the reweighting is trivial. In practice, fundamental factor models (Barra, Axioma) commonly produce non-orthogonal loading columns, making (7) the operationally relevant formula.

### Specializations

| Setting                                       | Formula simplifies to                                                     |
|:--------------------------------------------- |:------------------------------------------------------------------------- |
| $k=1$ (GPS2022)                               | $\sin^2\angle(h,\bar{b}) \to \delta^2/(c\|X\|^2+\delta^2)$ (Corollary 2)  |
| $G_\infty=I_k$, orth. returns (NG's Th. 3.1′) | $\sin^2\angle(h_j,\bar{b}_j) \to \delta^2/(n\rho_j+\delta^2)$; floor only |
| $G_\infty=I_k$, general returns               | Formula (6) with rotation $\sin^2\angle(\hat{w}_j,e_j)$                   |
| General $G_\infty$                            | Full formula (7) with $\hat{M}$, $M$                                      |
| $n\to\infty$                                  | $\hat{w}_j\to w_j$; rotation $\to 0$; floor $\to 0$; total $\to 0$        |
| $\delta^2\to 0$                               | Floor $\to 0$; rotation survives with weight $\to 1$                      |
| Grassmannian (Cor. 4)                         | $\sum_j\delta^2/(n\rho_j+\delta^2)$; rotation drops out                   |

### Observable estimation

Corollary 1 provides the estimator $\hat{\psi}_{p,j}^2 = 1 - \ell_p^2/s_{p,j}^2$ for the alignment ceiling $n\rho_j/(n\rho_j+\delta^2)$, requiring only the sample singular values of $Y$. The shrinkage estimator $H\cdot\mathrm{diag}(\hat{\psi}_{p,j})$ is the optimal frame estimator in the sense of minimizing per-column squared alignment error; the derivation is given in the companion document. If the goal is subspace-level estimation — for example, projecting a portfolio onto the factor space or hedging factor exposures — no shrinkage is needed: $\mathrm{col}(H)$ is already the minimum-error subspace estimator, achieving the Grassmannian floor $\sum_j\delta^2/(n\rho_j+\delta^2)$ from Corollary 4 with no additional correction.

---

## 12. Grassmannian Subspace Estimation vs. Frame Estimation

### 12.1 Two Estimation Targets

The theorem supports two estimation goals with different error criteria.

*Frame estimation* aims to recover the matrix $[\bar{b}_1, \ldots, \bar{b}_k]$ as a specific $k$-frame in $\mathbb{R}^p$. The sample estimator is $H = [h_1, \ldots, h_k]$. The total frame error is

$$
\mathcal{E}_{\mathrm{frame}} \;:=\; \sum_{j=1}^k \sin^2\angle(h_j,\, \bar{b}_j).
$$

*Subspace estimation* aims to recover $\mathcal{B} = \mathrm{col}(B)$ as a point on the Grassmannian $\mathrm{Gr}(k,p)$. The sample estimator is $\mathrm{col}(H)$ — the same linear span, treated as an equivalence class under orthogonal rotation. The total subspace error is the squared chordal distance $d_{\mathrm{Gr}}^2(\mathrm{col}(H), \mathcal{B}) = \sum_j \sin^2\theta_j$.

The two targets differ by an in-plane rotation: frame estimation fixes the orientation of the basis within the estimated subspace; subspace estimation does not.

### 12.2 The Floor-Plus-Rotation Decomposition Revisited

The two errors are related by the identity established in Corollary 4 and the main theorem:

$$
\mathcal{E}_{\mathrm{frame}} \;=\; \underbrace{\sum_{j=1}^k \frac{\delta^2}{n\rho_j + \delta^2}}_{\mathcal{E}_{\mathrm{subspace}}} \;+\; \underbrace{\sum_{j=1}^k \frac{n\rho_j}{n\rho_j + \delta^2}\,\sin^2\angle(\hat{w}_j,\, w_j)}_{\mathcal{E}_{\mathrm{rotation}} \;\ge\; 0}.
$$

The subspace error $\mathcal{E}_{\mathrm{subspace}}$ equals the sum of out-of-subspace floors. The rotation excess $\mathcal{E}_{\mathrm{rotation}}$ is the additional penalty paid for demanding a specific orientation within the subspace. Both quantities are almost-sure limits as $p \to \infty$.

### 12.3 When Does the Rotation Term Contribute?

The rotation excess $\mathcal{E}_{\mathrm{rotation}}$ is zero if and only if $\sin^2\angle(\hat{w}_j, w_j) = 0$ for all $j$. Under Assumption 3, this holds if and only if $\hat{M} = M$ (sample and population factor covariance matrices coincide, up to an eigenvalue ordering), which is enforced exactly by NG's Assumption 2.6′ (orthogonal factor returns). In all other cases — correlated factor returns, or non-orthogonal loading columns with general $G_\infty$ — the rotation excess is strictly positive, and frame estimation incurs a penalty not present in subspace estimation.

In applied factor models, factor returns are rarely orthogonal: value and momentum are often mildly correlated, and industry factors frequently share loading directions. The rotation excess is therefore the typical case rather than the exception.

### 12.4 Subspace Estimation Dominates Frame Estimation

The inequality $\mathcal{E}_{\mathrm{subspace}} \le \mathcal{E}_{\mathrm{frame}}$ (a.s.) has a direct statistical interpretation: any use of the individual eigenvectors $h_1, \ldots, h_k$ as proxies for the individual population loading directions $\bar{b}_1, \ldots, \bar{b}_k$ necessarily incurs more total squared error than using only the subspace $\mathrm{col}(H)$. Practical tasks that can be formulated as subspace estimation — portfolio projection, factor neutralization, risk subspace decomposition — should therefore be performed at the subspace level rather than the frame level.

This is not a statement that one estimator is better than another for the subspace task. Both estimators — frame and subspace — use $\mathrm{col}(H)$ for the subspace itself, so they achieve the same $\mathcal{E}_{\mathrm{subspace}}$. The difference is what is demanded: the frame estimator additionally demands per-column alignment, incurring the rotation penalty.

### 12.5 Consistency

Both estimators are inconsistent in the cross-sectional limit $p \to \infty$ with $n$ fixed: the Grassmannian error $\mathcal{E}_{\mathrm{subspace}} \to \sum_j \delta^2/(n\rho_j + \delta^2) > 0$ as $p \to \infty$. Adding more assets does not resolve the subspace more sharply — the signal-to-noise ratio is fixed by $n$.

Both estimators are consistent in the time dimension: as $n \to \infty$, $\rho_j \to d_j > 0$ (the population spike strength) and $\delta^2/(n\rho_j + \delta^2) \to 0$. Additionally, $\hat{w}_j \to w_j$ by the law of large numbers on $\hat{D} \to D$, so $\mathcal{E}_{\mathrm{rotation}} \to 0$ as well. The unique remedy for persistent misalignment is a longer time series, not a larger cross-section.

For fixed $n$, the Grassmannian error $\mathcal{E}_{\mathrm{subspace}} = \sum_j \delta^2/(n\rho_j + \delta^2)$ is the minimum achievable subspace error: the estimator $\mathrm{col}(H)$ attains it, and the identity $\sum_j\sin^2\theta_j = \sum_j\|h_j^\perp\|^2$ shows that this equals the total out-of-subspace squared norm, which is determined entirely by the signal-to-noise structure and cannot be reduced by any choice of estimator for the subspace.

### 12.6 Practical Implications for Factor Modeling

Two concrete adjustments follow from these results.

For *subspace-based tasks* (factor neutralization, risk decomposition, portfolio projection), use $\mathrm{col}(H)$ directly. The projection $\Pi_H z \approx \Pi_B z$ for any portfolio $z$ by Part (i) of the theorem, and the Grassmannian error quantifies how close this approximation is.

For *frame-based tasks* (estimating individual factor loadings, computing factor exposures for attribution), the shrinkage estimator $H \cdot \mathrm{diag}(\hat{\psi}_{p,j})$ (with $\hat{\psi}_{p,j}$ from Corollary 1) reduces each eigenvector toward the origin by the appropriate signal-to-noise scaling, partially compensating for the out-of-subspace floor. This does not eliminate the rotation excess $\mathcal{E}_{\mathrm{rotation}}$, which requires either longer time series or a factor model that enforces orthogonal returns.

---

## 13. Summary

The unified theorem establishes the almost-sure limit of $\sin^2\angle(h_j, \bar{b}_j)$ as $p \to \infty$ with $n$ and $k$ fixed, for a general $k$-factor model with per-column loading prevalences. The limit decomposes into two non-negative terms: the out-of-subspace floor $\delta^2/(n\rho_j + \delta^2)$, which is fixed by the signal-to-noise ratio and irreducible in $p$, and the in-subspace rotation $(n\rho_j/(n\rho_j + \delta^2))\sin^2\angle(\hat{w}_j, w_j)$, which is a finite-$n$ artifact that vanishes as $n \to \infty$.

The theorem extends the GPS2022 single-factor result through a three-level assumption hierarchy. GPS2022 establishes the $k=1$ formula $\sin^2\angle(h, \bar{b}) \to \delta^2/(c\|X\|^2 + \delta^2)$ (Corollary 2); the only structurally meaningful quantities are the scalar prevalence $c$ and the factor's squared return norm. NG's Theorem 3.1′ extends to $k \ge 1$ factors under the hypothesis that loading columns are orthogonal ($G_\infty = I_k$) and factor returns are orthogonal (Assumption 2.6′), which forces the in-subspace rotation to zero; each $\sin^2\angle(h_j, \bar{b}_j)$ then reduces to its floor $\delta^2/(n\rho_j + \delta^2)$ and the factors decouple exactly (§9). The unified theorem removes both orthogonality restrictions: loading columns may be non-orthogonal with an arbitrary limiting Gram matrix $G_\infty$, and factor returns may be correlated. In this fully general setting the in-subspace rotation $\sin^2\angle(\hat{w}_j, w_j)$ is positive and is weighted by $n\rho_j/(n\rho_j + \delta^2)$, making the per-column angle strictly larger than the floor whenever $\hat{M} \ne M$. The asymptotic regime — $p \to \infty$ with $n$ and $k$ fixed — is distinct from the Marchenko–Pastur setting in which $p/n \to \gamma \in (0,\infty)$: here $n$ does not grow with $p$, the noise bulk does not collapse, and the signal-to-noise ratio is set entirely by $n$ rather than by the aspect ratio.

Four corollaries extract the principal consequences. Corollary 1 supplies the observable shrinkage estimator $\hat{\psi}_{p,j} = \sqrt{\max(0,\, 1 - \ell_p^2/s_{p,j}^2)}$, which estimates the alignment ceiling $\sqrt{n\rho_j/(n\rho_j + \delta^2)}$ from the sample singular values of $Y$ alone, with no knowledge of $B$, $F$, or $G_\infty$. Corollary 3 translates the per-column angle formula into a dispersion bias statement: the squared projection $|\Pi_H z|^2$ falls short of $|\Pi_B z|^2$ by $\sum_j (1 - \psi_{\infty,j}^2)\, c_j^2$ almost surely. Corollary 4 shows that for Grassmannian subspace estimation the in-subspace rotation drops out entirely via the identity $d_{\mathrm{Gr}}^2(\mathrm{col}(H), \mathcal{B}) = \sum_j \|h_j^\perp\|^2$: the total subspace error is $\sum_j \delta^2/(n\rho_j + \delta^2)$, a sum of per-factor floors with no rotation contribution. Corollary 5 extends the frame-level analysis to a $k$-column probe $W \in \mathbb{R}^{p \times k_W}$: the Frobenius deficit $\|\Pi_B W\|_F^2 - \|\Pi_H W\|_F^2$ converges to $\sum_j (\delta^2/(n\rho_j + \delta^2))(\Gamma_\infty \Gamma_\infty^\top)_{jj}$, a weighted sum of floors in which the weights $(\Gamma_\infty\Gamma_\infty^\top)_{jj}$ measure how much the probe aligns with each factor direction; at the canonical choice $W = \tilde{B}$, the deficit equals $d_{\mathrm{Gr}}^2$, linking the frame-level and subspace-level criteria.

Two practical conclusions follow.

*First*, the out-of-subspace floor $\delta^2/(n\rho_j + \delta^2)$ is irreducible in the cross-section: adding more assets (larger $p$) does not reduce it, because the signal-to-noise ratio is determined by $n$ alone. The unique lever is a longer time series.

*Second*, the floor and the rotation excess respond to different estimation strategies. Tasks that can be formulated at the subspace level — factor neutralization, risk subspace projection, portfolio hedging — achieve this floor without any rotation penalty and require no correction; the estimator $\mathrm{col}(H)$ is already optimal for the Grassmannian distance criterion. Tasks that demand individual loading directions — factor attribution, per-column exposure calculation — incur the additional rotation excess $\mathcal{E}_{\mathrm{rotation}} = \sum_j (n\rho_j/(n\rho_j + \delta^2))\sin^2\angle(\hat{w}_j, w_j)$. This excess can be partially reduced by replacing $H$ with the shrinkage estimator $H \cdot \mathrm{diag}(\hat{\psi}_{p,j})$ (which corrects for the floor component of the dispersion bias), but the rotation excess itself cannot be eliminated without either orthogonal factor returns or a longer time series that drives $\hat{M} \to M$.