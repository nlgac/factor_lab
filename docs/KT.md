# Knowledge Transfer — Factor Lab

*Supersedes `KT_proof_theorem3.1prime.md`, `KT_extension_to_nonorthogonal_factors.md`, and `KT_update_2026-05-04.md`. Self-contained briefing for a fresh session.*

> Last updated: 2026-07-23 | Trigger: \ukt | Staleness: Stale — §1, §2, §3, §5 required substantial rewrite; see HIL NOTICE below.

⚠️ HIL NOTICE — 2026-07-23
This document was last substantively updated 2026-06-18, when `main-9.tex` was the active manuscript. In the intervening ~5 weeks the manuscript line moved through `main-11.tex` → the `main-14`/`main-14a`/`main-14-revised`/`main-14_restored` family → `main-16.tex` (now current, 2360 lines, titled "Estimation Error in Latent High-Dimensional Factor Models," authors Bernstein/Goldberg/Gunther/Kercheval/Lan/Lin/Yao). §1–§3 and §5 below have been rewritten from (a) this session's direct work on `main-16.tex` and (b) a file-timestamp scan of `paper/`, but **no session-by-session record exists for most of the `main-11`→`main-14` development** (roughly late June–mid July 2026) — only the end state is reconstructed. If a fresh session needs the *history* of decisions made during that gap (not just current state), check `git log`/file diffs directly or ask Ken rather than trusting §5.

---

## 1. Project Context

This project proves and develops the **estimation-error theory for high-dimensional latent factor models**: in a $k$-factor model $y=Bf+z$ (population) sampled as $Y\in\R^{p\times n}$, the sample top-$k$ principal directions $h_j$ are systematically misaligned with the population principal directions $b_j$ (eigenvectors of the signal covariance $\Sigma_0=B\Sigma_fB^\top$), with an exact finite-$p$ split into an out-of-subspace "floor" and an in-subspace "rotation" term, both with explicit $p\to\infty$ limits ($n,k$ fixed). See §2.0 for the current theorem.

The project has three interlocking strands:

1. **The proof** — originally Theorem 3.1′ (NG, single author) unified with AK's general-$G^\infty_B$ result in `unified_dispersion_bias_proof_051926_cleaned.md` (§2.1–2.3, old notation); now superseded as the *active* proof by Theorem 1 (`thm:main`) in `main-16.tex` (§2.0, current notation) — a reorganized, publication-track version with its own Assumption/Lemma structure. The two lines are mathematically related (same floor+rotation phenomenon) but use different notation and are not yet reconciled document-to-document; see §3.4/§6.
2. **The correction** — the James-Stein correction $\hat\Pi_B^{\mathrm{JS}} z = HD_\psi^{-1}H^\top z$ (§2.4, old notation), developed through the $k$-frame probe extension, in `dispersion_bias_correction_cleaned.md`. **Not yet present in `main-16.tex`** — this strand has not been carried forward into the active manuscript (see §6, Open Work).
3. **Manuscript preparation** — **Now active**: `paper/main-16.tex` (2360 lines), titled "Estimation Error in Latent High-Dimensional Factor Models," authors Alex Bernstein, Lisa R. Goldberg, Nicholas Gunther (NLG/Ken), Alec Kercheval, Tian Lan, Ethan Lin, Darwin Yao. Supersedes `main-9.tex` and the `main-11.tex` → `main-14`/`main-14a`/`main-14-revised`/`main-14_restored` line (§3.1). Current work is **collaborator review and refinement**, not new proof development: a placement dispute with Alec Kercheval over Corollary 3 (resolved — retained, justified via shared Bjorck–Golub machinery with Corollary 2, and via a "leakage" interpretation of its off-diagonal terms; §7), a defense of retaining the Grassmann/Stiefel background material (manuscript's Section 9, which originated from a Lisa Goldberg question, not added unprompted), migration of `\nlgcmt{}` review comments between manuscript versions, an evaluation of an external AI-generated critique ("Kimi Executive Summary — The Three Biggest Wins"), and two new modular `\input`-appendices (`assumptions_lemmas_appendix.tex`, `full_symbol_glossary.tex`) added as easily-removable single-`\input`-line sections. See §3.1 and §5.

---

## 2. The Mathematics

### 2.0 Current manuscript theorem (`main-16.tex`, active as of 2026-07-23)

**Model.** $y=Bf+z\in\R^p$, $B\in\R^{p\times k}$ rank $k$, $f,z$ mean-zero finite-second-moment, uncorrelated; $\Sigma_f=\E[ff^\top]$, $\Delta_z=\E[zz^\top]$, $\Sigma_y=\Sigma_0+\Delta_z$ with $\Sigma_0:=B\Sigma_fB^\top=b\Delta_0b^\top$ (spectral decomposition, $k$ distinct eigenvalues). The columns $b_j$ of $b$ are the **principal directions** — the estimation targets — with $\col(b)=\col(B)=:\mathcal B$. Since $b=BC$ for a unique invertible $C$, setting $\phi=C^{-1}f$ rewrites the model in **principal-direction coordinates** $y=b\phi+z$. Sample side: $Y\in\R^{p\times n}$, $h_j$ = top-$k$ left singular vectors of $Y/\sqrt n$ (equivalently top-$k$ eigenvectors of the sample covariance $S^{(p,n)}$). Error metric: $\sin^2\angle(h_j,b_j)$.

**Six Assumptions** (`asm:fm`–`asm:reg`, all required as a block for Theorem 1 and all Corollaries): `asm:fm` [Signal] — $F$ has common finite covariance $\Sigma_f$, otherwise free (serial dependence, heteroskedasticity allowed); purely definitional, makes $\Sigma_0$/$b_j$ well-defined. `asm:noise` [Noise] — conditional on $F$, noise entries independent, mean zero, uniformly bounded 4th moments; the workhorse noise assumption. `asm:delta` [Average specific variance] — cross-sectional average noise variance converges to a common $\delta^2$ across dates; makes the floor term a clean scalar. `asm:gram` [Loading Gram convergence] — $B^\top B/p\to G_B\succ0$; underlies every eigenvalue in the paper. `asm:sep` [Population separation] — $k$ distinct population eigenvalues; makes $b_j$ well-defined individually. `asm:reg` [Regular event] — $k$ distinct *realized* (finite-$n$) eigenvalues; the sample analogue of `asm:sep`, a mild genericity condition (holds a.s. if $F$ has a density). Full usage detail, including exactly which downstream Lemma/Corollary each assumption feeds: `paper/assumptions_and_lemmas.md` and `paper/assumptions_lemmas_appendix.tex`.

**Seven Lemmas** (Appendix "Supporting Lemmas"): `lem:gramdual` [Gram duality] — the single most load-bearing lemma; the mechanical device that reduces the ambient $p$-dimensional eigenproblem to a fixed $n\times n$ (or $k\times k$) dual via $(\lambda,v)\leftrightarrow(\lambda,A^\top v/\sqrt\lambda)$. `lem:basis` [Change of basis] — explicit $C=(B^\top B)^{-1}B^\top b$. `lem:phi` [Principal direction factor coordinates] — bridges raw (non-convergent) loadings $B$ to the convergent principal basis $b$. `lem:noise` [Noise concentration] — $a^\top Z_{\cdot l}/\sqrt p\to0$ a.s., the LLN building block for noise negligibility. `lem:econv` [Eigenpair convergence and sign pinning] — converts matrix convergence into eigenvector convergence, with the sign-pinning device needed since eigenvectors are only defined up to sign. `lem:pcconv` [Principal-coordinate convergence] — the master signal-side convergence result: $\bar\Phi:=\Phi/\sqrt p\to\bar\Phi^\infty$ (closed form), $N^{(p,n)}\to N^{(n)}$, and the Gram-duality link between $N^{(n)}$'s eigenvectors $\nu_j^{(n)}$ and $W^{(n),0}$'s eigenvectors $w_j^{(n)}$. **Note the naming trap**: "realized signal Gram" is this paper's established name for $N^{(p,n)}/N^{(n)}$ specifically (principal-coordinate object) — it is *not* a name for the raw $FF^\top/n$, which is merely the input that $N^{(n)}$ is built from via a sandwich with $C$-type rotations (caught 2026-07-23 reviewing an external AI's remark revision that conflated the two). `lem:uncorr` [All-pairs uncorrelatedness] — explicitly *not used in the proofs*; illustrative only, motivating the path-conditioning in `asm:noise`.

**Theorem 1** (`thm:main`): for fixed $j$, exact split at every $p$: $\sin^2\angle(h_j,b_j)=\sin^2\angle(h_j,\mathcal B)+\cos^2\angle(h_j,\mathcal B)\sin^2\angle(\Pi_Bh_j,b_j)$. Conditional on $F$, a.s. as $p\to\infty$: $\sin^2\angle(h_j,\mathcal B)\to\delta^2/(n\lambda_j^{(n)}+\delta^2)$ (floor) and $\sin^2\angle(\Pi_Bh_j,b_j)\to\sin^2\angle(\nu_j^{(n)},e_j)$ (rotation), where $(\lambda_j^{(n)},\nu_j^{(n)})$ is the $j$th eigenpair of the realized signal Gram $N^{(n)}$. Combined: $\sin^2\angle(h_j,b_j)\to\delta^2/(n\lambda_j^{(n)}+\delta^2)+[n\lambda_j^{(n)}/(n\lambda_j^{(n)}+\delta^2)]\sin^2\angle(\nu_j^{(n)},e_j)$. A signal-to-noise-ratio reformulation ($\mathrm{SNR}_j^{(n)}=n\lambda_j^{(n)}/\delta^2$) is given as an immediate corollary remark. As $n\to\infty$ (Corollary `cor:n-limit`), both terms vanish.

**Four Corollaries**: `cor:obsfloor` [Observable floor], `cor:subspace` [Subspace error], `cor:noiseless`, `cor:purenoise` [the noise does not rotate $h_j$ within $\mathcal B$]. These replace the old-notation Corollaries 1–5 of §2.3 below — the correspondence between the two Corollary sets has not been worked out and may not be one-to-one (see §6).

**A random-variable framing** (remark following Theorem 1, revised 2026-07-23): the asymptotic error is itself a random variable whose randomness (by `lem:pcconv`) enters only through the fixed $k\times k$ object $FF^\top/n$, not the full path $F$ — so a prior on $FF^\top/n$ (e.g. from an i.i.d. assumption on factor draws) pushes forward through Theorem 1 to confidence bounds on $\sin^2\angle(h_j,b_j)$ without needing more data.

### 2.1 Model (original dispersion-bias/correction framework — pre-`main-9` notation; strand 1–2, not currently being revised)

$$
Y = BF^\top + Z, \qquad Y \in \mathbb{R}^{p \times n},\quad p \to \infty,\quad n,k \;\text{fixed}.
$$

$B \in \mathbb{R}^{p \times k}$ (loadings, columns $\beta_1,\ldots,\beta_k$), $F \in \mathbb{R}^{n \times k}$ (factor returns, columns $X_1,\ldots,X_k$), $Z$ (noise, i.i.d. mean-zero, variance $\delta^2$, bounded 4th moments). $H \in \mathbb{R}^{p \times k}$: top-$k$ left singular vectors of $Y/\sqrt{n}$.

Key matrices: $G^\infty_B = \lim_{p\to\infty} B^\top B/p$ (loading Gram limit, positive definite). $\hat{M} = (G^\infty_B)^{1/2}(F^\top F/n)(G^\infty_B)^{1/2}$ (sample signal matrix); $M = (G^\infty_B)^{1/2}\Sigma_F(G^\infty_B)^{1/2}$ (population limit). Eigenvalues of $\hat{M}$: $\hat\lambda_1 > \cdots > \hat\lambda_k > 0$, eigenvectors $\hat{w}_j$. Eigenvalues of $M$: $\lambda_1 > \cdots > \lambda_k > 0$, eigenvectors $w_j$. In the diagonal case ($G^\infty_b = I_k$), $G^\infty_B = C = \mathrm{diag}(c_j)$ and $\lambda_j = c_j\sigma_j^2$ where $c_j = \lim\|\beta_j\|^2/p$.

### 2.2 Unified Theorem (Three Parts)

**Part (i)** *(minimal assumptions)*: For any deterministic $v$ with $|v|\le 1$,
$$
H^\top v - H^\top\Pi_B v \to 0 \quad\text{a.s.}
$$

**Part (ii)** *(general $G^\infty_B$; replaces earlier "Part (ii-diag)" + "Part (iii)" split)*:
$$
\sin^2\angle(h_j, b_j) \xrightarrow{\text{a.s.}}
\underbrace{\frac{\delta^2}{n\hat\lambda_j+\delta^2}}_{\text{out-of-subspace floor}}
+
\underbrace{\frac{n\hat\lambda_j}{n\hat\lambda_j+\delta^2}\sin^2\angle(\hat{w}_j, w_j)}_{\text{in-subspace rotation}},
$$
where $\hat{w}_j$ are eigenvectors of $\hat{M}$ and $w_j$ are eigenvectors of $M$. In the diagonal case ($G^\infty_b = I_k$), $w_j = e_j$ and the rotation reduces to $\sin^2\angle(\hat{w}_j, e_j)$.

The **floor** $\delta^2/(n\hat\lambda_j+\delta^2) = 1/(1+\widehat{\mathrm{SNR}}_j)$ is irreducible — more assets do not reduce it. The **rotation** term vanishes when $\hat{M} = M$ (equivalently, as $n\to\infty$); in the noiseless limit $\delta^2\to 0$ it survives with weight 1.

### 2.3 Key Corollaries

**Corollary 3 (Dispersion bias — NG case)**: With $z = e/\sqrt{p}$ (equal-weight portfolio) and $c_j = \langle b_j, z\rangle_\infty$,
$$
|\Pi_B z|^2 - |\Pi_H z|^2 \to \sum_j (1-\psi_{\infty,j}^2)c_j^2 > 0 \quad\text{a.s.}
$$
where $\psi_{\infty,j} = \sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)} \in (0,1)$.

**Corollary 4 (Grassmannian subspace distance)**:
$$
d_{\mathrm{Gr}}^2(\mathrm{col}(H),\mathcal{B}) \to \sum_j \frac{\delta^2}{n\hat\lambda_j+\delta^2} \quad\text{a.s.}
$$
This equals the sum of out-of-subspace floors only — the in-subspace rotation cancels in the Grassmannian metric. Subspace estimation is therefore strictly more efficient than frame estimation when factors are correlated.

**Corollary 5 (Frame-level dispersion bias)**: For a probe frame $\tilde{W} \in \mathbb{R}^{p\times k_W}$ with $\tilde{W}^\top \tilde{W} = I_{k_W}$ and frame alignment matrix $\Gamma_\infty = \lim B^\top \tilde{W}/\sqrt{p} \cdot (\text{appropriate normalization})$:

- *Part (i)*: $H^\top W - H^\top\Pi_B W \to 0$ a.s. (minimal assumptions).
- *Part (ii) (NG case)*: $H^\top W \to \Psi_\infty\Gamma_\infty$ a.s., where $\Psi_\infty = \mathrm{diag}(\psi_{\infty,j})$.
- *Frobenius deficit*: $\|\Pi_B \tilde{W}\|_F^2 - \|\Pi_H \tilde{W}\|_F^2 \to \sum_j [\delta^2/(n\hat\lambda_j+\delta^2)](\Gamma_\infty\Gamma_\infty^\top)_{jj}$.
- *Principal angle shrinkage*: $\sigma_l(H^\top \tilde{W}) \to \sigma_l(\Psi_\infty\Gamma_\infty) \le \psi_{\infty,1}\sigma_l(\Gamma_\infty)$.
- *Grassmannian bridge*: Setting $\tilde{W} = B_\infty$ (population loading frame) gives deficit $= d_{\mathrm{Gr}}^2$.
- *General $G^\infty_B$*: $H^\top \tilde{W} \to \Psi_\infty\hat{W}^\top\Gamma_U$ where $\Gamma_U = \lim U^\top \tilde{W}$ and $\hat{W} \in O(k)$ is the limiting rotation of sample eigenvectors.

### 2.4 The James-Stein Correction

**Single probe** ($k_W = 1$, $W = z$): Define $\hat\psi_j = \sqrt{\max(0, 1 - \hat\delta^2 p/s_{p,j}^2)}$ where $s_{p,j}$ are singular values of $Y$. The JSE correction is:
$$
\hat\Pi_B^{\mathrm{JS}} z := HD_\psi^{-1}H^\top z = \sum_j \frac{h_j^\top z}{\hat\psi_j}h_j.
$$
This inflates each factor-$j$ coordinate by $1/\hat\psi_j$ (stronger correction for weaker factors). The squared norm $|\hat\Pi_B^{\mathrm{JS}} z|^2 \to |\Pi_B z|^2$ a.s. — the scalar bias is eliminated.

**$k$-frame probe** ($W \in \mathbb{R}^{p\times k_W}$): The same operator applied column-by-column:
$$
\hat\Pi_B^{\mathrm{JS}} W = HD_\psi^{-1}H^\top W = \sum_j h_j(h_j^\top W)/\hat\psi_j.
$$
The Frobenius norm $\|\hat\Pi_B^{\mathrm{JS}} W\|_F^2 \to \|\Pi_B W\|_F^2$ a.s. The corrected principal cosines $\sigma_l(D_\psi^{-1}H^\top W) \to \sigma_l(\Gamma_\infty)$ — principal angle shrinkage is fully restored.

**What cannot be corrected**: The directional residual $\Pi_B \tilde{W} - \hat\Pi_B^{\mathrm{JS}} \tilde{W}$ lies in $\mathcal{B}\cap\mathcal{H}^\perp$ and its squared Frobenius norm equals the Frobenius deficit — irreducible, bounded below by $\sum_j[\delta^2/(n\hat\lambda_j+\delta^2)]\|\hat{w}_j^\top\Gamma_U\|^2$.

---

## 3. Document Inventory

All files in `C:\Users\nlgun\personal\nlgcode\factor_lab\` unless noted.

### 3.1 Primary Mathematical Documents (current)

| File | Purpose | Status |
|---|---|---|
| `unified_dispersion_bias_proof_051926_cleaned.md` | Full proof of unified theorem (Parts i–iii) + Corollaries 1–5 + §12 Grassmannian vs. frame estimation. ~900 lines. | **Primary proof reference** (old notation; migration pending) |
| `dispersion_bias_correction_cleaned.md` | James-Stein correction document. | **Primary correction reference** |
| `dispersion_bias_correction_v2_051226.md` | Complete §7 ($k$-frame probe extension) — the most recent version with full Frobenius deficit theorem, JSE correction, principal angle shrinkage. | **Most complete correction document** |
| `Proof_Theorem_3.1_prime_v3.md` | Full proof of NG's Theorem 3.1′ only (~935 lines, manuscript-quality). | Still valid; superseded mathematically by the unified proof but cleaner for the NG-only result |
| `multifactor_dispersion_prevalence_v7.pdf` | AK's paper. $k=3$, general $G^\infty_B$. Has observable bounds (Cor 2, Thm 2, Prop 1) not yet in the unified proof. | External reference; see §4.3 below. Note: `multifactor_dispersion_v8.pdf` also present on disk — may be a newer version; verify. |
| `proof_walkthrough_k3_cleaned.md` | Step-by-step illustrated walkthrough of Theorem Part (ii) with $k=3$, $p=500$, $n=60$ concrete example. Follows Appendix B.3. **Notation migration applied 2026-05-26.** | **Primary expository reference for the proof** |
| `theorem_part_ii_3_expanded.md` | Expanded proof of Theorem 1 Part (ii): statement, full 7-step proof with Lemmas 1/4/7 (fully proved with Borel–Cantelli + Kolmogorov SLLN), inline k=3 callouts, Corollary 4, worked example. | **Primary accessible proof document** |
| `latex/theorem_part_ii_3_expanded.tex` | LaTeX conversion of the above via `md_to_latex.py`. Blockquote-wrapped tables and Unicode ✓ handled correctly. | LaTeX artefact — regenerate as needed |
| `main-9.tex` | 1357 lines. Titled "Quantifying Principal Component Concentration Bias." Presents model, 5 assumptions, Theorem (floor + rotation), Lemma 1 (noise, a.s. via 4th moments + Borel-Cantelli), Lemma on dual Gram convergence, 4-step proof architecture. Uses notation $W_n^{(p)}$, $\theta_{n,j}^{(p)}$, $\kappa_{n,j}^2 = \lambda_{n,j}/(\lambda_{n,j}+\delta^2/n)$. | Superseded by `main-11.tex`→`main-16.tex` line below |
| `main-11.tex` | 960 lines. Intermediate manuscript revision between `main-9` and the `main-14` family. | Superseded by `main-14`/`main-16` |
| `main-14.tex`, `main-14._ed.tex` | 1797 lines each. Manuscript stage prior to `main-16`; both share an identical `\thanks`-placement bug (blank first page) later fixed. | Superseded by `main-16.tex` |
| `main-14a.tex` | 1797 lines. Same structure/labels as `main-14.tex`; carries 16 `\nlgcmt{}` review comments, source for the comment-migration into `main-16.tex` (9 moved directly, 5 already present, 2 flagged as ambiguous/orphaned — see §5). | Historical — comment source, not edited further |
| `main-14-revised.tex` | 606 lines (vs. 1797 original). An aggressive AI (Kimi) revision per "Kimi Executive Summary" critique. **Regression found**: all 9 non-Theorem-1 proofs (Corollaries `obsfloor`/`subspace`/`noiseless`, Lemmas `gramdual`/`phi`/`noise`/`econv`/`pcconv`, Prop `dual`) were dropped to bare statements — not requested by the critique itself. Also had a real `enumitem` `[(i)]`/`[(a)]` compile bug (fix: explicit `[label=(\roman*)]` syntax). | Superseded by `main-14_restored.txt`; kept as a record of the regression |
| `main-14_restored.txt` | 826 lines. Restoration of `main-14-revised.tex` with all 9 proofs reinstated, notation reverted from Kimi's $U/L$ rename back to $b/B/h/H$, `enumitem` fix applied. Compiles cleanly (17pp). Two residual gaps found 2026-07-2x and reported to Ken: a dangling `\eqref{eq:obsbound}` (should be `eq:obsfloor`) and two missing `\begin{figure}` blocks for `fig:decomp_p_sweep`/`fig:decomp_n_sweep` (gap predates the restoration). | Historical — fixes reported, not yet reapplied to `main-16.tex`; verify before assuming resolved |
| `main-16.tex` | **2360 lines. Active formal manuscript**, titled "Estimation Error in Latent High-Dimensional Factor Models." Authors: Bernstein, Goldberg, Gunther, Kercheval, Lan, Lin, Yao. Structure: Introduction; Context/related lit; Factor model & estimation target; Data/scaling/assumptions (6 Assumptions); Two small Gram matrices; Main theorem (Theorem 1, `thm:main`); Proof of Theorem 1; Data-driven estimators; The noiseless case; Simulation; Practitioner applications; Appendix (7 Supporting Lemmas, Symbol Tables 3–4, "AxiomProver" by Ken Ono). See §2.0 for the current theorem/assumption/lemma content. | **Active formal manuscript** |
| `assumptions_and_lemmas.md` | Markdown summary of all 6 Assumptions/7 Lemmas in `main-16.tex`, written from the perspective of downstream use (what breaks without each one), plus a dependency "at a glance" table. | Created 2026-07-23 |
| `assumptions_lemmas_appendix.tex` | LaTeX version of the above, structured as a new Appendix section (`\label{sec:reader-guide}`), with two separate "at a glance" tables (Assumptions, Lemmas) each giving name + downstream dependents. Designed to be `\input{}`-ed into `main-16.tex` and removed by deleting one line. **Not yet `\input`-ed into the delivered `main-16.tex`** — Ken manages that insertion himself. | Created 2026-07-23; verified via full two-pass `pdflatex` compile against a copy of `main-16.tex` |
| `full_symbol_glossary.tex` | Comprehensive ~70-symbol glossary (`\label{sec:full-glossary}`), 12 thematic subsections each with its own table (Symbol / Meaning / First appears). Same modular `\input{}` design as above. | Created 2026-07-23; verified via full compile (39pp, no fatal errors) |
| `side_explanations_memo.md` | Running log of conceptual side-explanations from working sessions, each tagged to the paper location it clarifies (e.g. symmetric eigenvalue problem, over-specified vs. ill-conditioned, $\col(b)=\col(B)$ proof). Ken asked this be maintained ongoing. | Created 2026-07-14; **update whenever a substantive side-explanation is given during paper work — check staleness each session** |
| `concentration-11.pdf`, `concentration-16.pdf` | Companion/earlier paper versions (same author list as `main-16.tex`). `concentration-11.pdf` used to verify the $\col(b)=\col(B)$ proof (Eq. 4). | Reference |
| `geometry_of_algorithms.pdf` | Edelman/Arias/Smith, "The Geometry of Algorithms with Orthogonality Constraints" (cited as `edelmanariassmith1998`). Used to ground the argument that the manuscript's Grassmann/Stiefel background (Section 9) is not far afield — quotes re: the symmetric eigenvalue problem and eigenvector ill-conditioning under repeated/close eigenvalues. | Reference |
| `Kimi Executive Summary_The Three Biggest Wins.md` | External AI critique: "Notation Crisis / Structure Inversion / Missing Intuition Bridges." Assessed 2026-07-2x: structural reordering + intuition-bridge additions judged low-risk/high-value; the $b\to U$, $B\to L$ notation rename judged costly churn against established collaborator notation, not adopted. | Reference — critique itself still valid; execution (`main-14-revised.tex`) had the proof-dropping regression, not the critique's fault |
| `diff.tex`, `main-14a_vs_main-16_latexdiff.tex/.pdf`, `main_14_to_16.html`, `DiffA-*`, `DiffB-*` | `latexdiff`-generated redlines between manuscript versions. | Generated artifacts — regenerate as needed, don't hand-edit |
| `proof_summary_5ideas.tex` | ~480-line pedagogical companion, produces 12-page PDF. "5 ideas" summary: (1) SVD duality, (2) algebraic inversion, (3) spiked spectrum, (4) isometry, (5) nested projections + Pythagorean assembly + Chebyshev appendix. **Three-color observability scheme** throughout: `\obs{}` blue = directly from Y; `\est{}` green = estimable from spectrum of $W_n^{(p)}$; black = unobserved. Applied to Table 1 (signal matrices) and Appendix longtable. `\obs{n}` applied to standalone $n$ as multiplier in theorem equations. All `\includegraphics` use `../figures/` paths (TeXmaker symlink fix). `proof_summary_5ideas - Copy.tex` is backup of pre-edit state. **Note**: Chebyshev appendix proves in-probability convergence only; main-9's Lemma 1 uses 4th moments + Borel-Cantelli for a.s. — gap is open (see §6 item 7). | **Active pedagogical companion — last revised 2026-06-18** |
| `memo_wn_and_snr.tex` | Technical memo on $W_n^{(p)}$ (the $n\times n$ dual Gram) and the SNR quantity $\kappa_{n,j}^2$. Supporting reference for main-9. | Created ≤2026-06-17 |
| `condensed_proof_skeleton.tex` | Short condensed proof skeleton of the dispersion-bias theorem. | Supporting reference |
| `duality_graph.tex` | TikZ diagram illustrating the $p\times p \leftrightarrow n\times n$ SVD duality (Gram reduction). | Figure |
| `gemini_theorem_proof_20260611_cleaned.tex` | Gemini-generated proof attempt, cleaned. | External/comparison draft |
| `gemini_theorem_proof_20260611_expanded_cleaned.tex` | Expanded cleaned version of the above. | External/comparison draft |
| `fiber_bundle_geometry.tex` | Exposition of the fiber bundle interpretation of the factor model proof geometry. | Exploratory — created ≤2026-06-17 |
| `statmech_dictionary.tex` | "Variance as Energy" dictionary: stat-mech analogy mapping factor model objects to statistical mechanics objects. | Exploratory — created ≤2026-06-17 |
| `DEFINED_TERMS.tex` | Complete table of all defined terms, symbols, and named objects (originally from `main.tex`). | Created 2026-06-05 |
| `step5excerpt.tex` | Standalone expanded exposition of Step 5 (loading-frame coordinates): symbol glossary, Group A substeps 5A.1–5A.4, Group B substeps 5B.1–5B.3, explicit input/output interface. | Created 2026-06-05 |
| `floor-rotation.tex` | Original AK paper draft (reviewed by NLG; issues catalogued). | Predecessor to main-9.tex |
| `floor_rotation_nlged.tex` | NLG-revised version with numbered green `\nlgcmt{}` comments. | Reviewed draft |
| `floor_rotation_nlged_v2.tex` | Copy with typos as `\typo{}` markers and substantive comments renumbered 1–10. | Alternate revision |
| `Notation Migration Guide.md` | Complete two-table guide: symbol renames + affected locations for all documents. | Reference |
| `proof_expansion_plan.md` | 9-section plan for expanding `theorem_part_ii_3.pdf`. Implemented in `theorem_part_ii_3_expanded.md`. | Implemented |

### 3.2 Supporting / Historical

| File | Notes |
|---|---|
| `dispersion_bias_correction_v1_050726.md` | Prior version of the correction doc; §7 was placeholder. Superseded. |
| `reflection_fundamental_nonorthogonal_factors.md` | Deep analysis of doubly-extended $\Xi$ for fundamental factor models. Still relevant for open Task 1. |
| `KT_proof_theorem3.1prime.md` | Session KT through 2026-04-26. Superseded by this file. |
| `KT_extension_to_nonorthogonal_factors.md` | KT for non-orthogonal extension (dropping 2.6′). Superseded by this file; §6 sketch now formalized in unified proof. |
| `KT_update_2026-05-04.md` | KT through 2026-05-04 session. Superseded by this file. |
| `revisedProofTheorem3.1+.md_cleaned.md` | Cleaned source of v3. Superseded. Five known issues listed in KT_update_2026-05-04.md §3. |

### 3.3 Simulation Code

| File | Purpose |
|---|---|
| `bias_correction_demo.py` | Main simulation: k=2 illustration, MSE tables, stability floor $\tau$. Reproduces v2 correction tables. |
| `sim_theorem_partii.py` | Canonical simulation for the theorem equation (renamed from `sim_theorem_eq.py`). Accepts JSON spec file. |
| `factor_sims.py` | General simulation engine. |
| `reformat_math.py` | Math formatter: canonical `$$` blocks, semicolons, inline joins. Idempotent. |
| `reformat_math_extra.py` | Extended math formatter; accepts `-o console` to pipe output to stdout (UTF-8 safe). Fixes: blockquote table pipe protection (`> \| ...\|` lines now correctly handled); `\emph{}` inside math replaced with `\text{}`. |
| `reflow_md.py` | Reflows hard-wrapped Markdown prose to long lines. Accepts `-` for stdin and `-o` for output path; pipe-composable with `reformat_math_extra.py`. |
| `md_to_latex.py` | Converts math-heavy Markdown to compilable LaTeX. Fixes (2026-05-26): blockquote-wrapped table rows correctly converted to `tabular` environments (not emitted as raw `\begin{quote}` text); Unicode characters (✓, —, …, ×, etc.) substituted with LaTeX equivalents. Run: `python md_to_latex.py input.md [output.tex]`. |
| `proof_walkthrough_k3.py` | Generates all numerical outputs for the k=3 walkthrough. |
| `proof_walkthrough_figures.py` | Generates all figures (`walkthrough_figs/fig_w0*.png`) for `proof_walkthrough_k3_cleaned.md`. |
| `proof/make_fig_isometry.py` | Generates `figures/fig_isometry_pythagorean_split.pdf`. Panel 1 includes the $G_B^{(p)}$-unit ellipse $\{x:x^\top G_B^{(p)}x=1\}$ (pink, #e7298a), parameterized as `G_inv_sqrt @ unit_circle`. Output: 47454 bytes. Run directly as `python3 scripts/proof/make_fig_isometry.py` from project root (exec()-via-bash fails due to `__file__` scope). |
| `rotation_check.py` | Verifies that finite-sample eigenvector misalignment of $M_n$ is substantial at small $n$ even with diagonal $G_B$ and $\Sigma_f$. Uses `factor_lab` (FactorModelBuilder + FlexibleReturnsSimulator). Produces two tables: mean $\sin^2\angle(\hat{w}_j, e_j)$ by $n$, and single-draw floor+rotation decomposition. |
| `rotation_check_o.py` | Ad-hoc standalone version of the above (pure numpy, no factor_lab). Includes `%right-angle` column showing misalignment as % of 90°. |
| `unwrap_prose.py` | Utility: removes mid-paragraph line breaks from `.tex` files while preserving math environments, blank lines, and command lines. Backs up original. Run: `python3 unwrap_prose.py file.tex`. |

### 3.4 Notation

A notation migration plan for `proof_walkthrough_k3_cleaned.md` was developed in session 2026-05-23. Key decisions:

- **Gram family unified**: $G^{(p)}_B$, $G^\infty_B$ (unnormalized loading Gram); $G^{(p)}_b$, $G^\infty_b$ (normalized); $G^{(n)}_F$, $\Sigma_F$ (factor Gram / population covariance). Replaces $\Gamma_p$, $\Gamma_B$, $G(p)$, $G_\infty$, and inline $F^\top F/n$.
- **Eigenvalue family unified**: $\lambda_j$ (population spike $c_j\sigma_j^2$, replaces $d_j$); $\hat\lambda_j$ (eigenvalue of $\hat M$, replaces $\rho_j$); $\hat\lambda_j + \delta^2/n$ written inline (replaces $\tau_j$).
- **$W_\infty \to W$**: limit object carries no decoration; $W^{(p)} \to W$.
- **$\chi_{p,j} \to v^{(p)}_j$**, $s_{p,j} \to s^{(p)}_j$: superscript $(p)$ for finite-$p$ sequences.
- **$a_j^\infty \to a_j$**, **$g_j^\infty \to g_j$**: limit objects lose the $\infty$ superscript.
- **$D$, $\hat D$ dropped**: appear only as aliases for $M$, $\hat M$ in the diagonal case; not load-bearing.
- **$P$ dropped**: replaced inline by $C^{1/2}$ (appears only twice).
- **SNR**: $\mathrm{SNR}_j = n\lambda_j/\delta^2$ (population); $\widehat{\mathrm{SNR}}_j = n\hat\lambda_j/\delta^2$ (realized).
- **Decoration convention**: superscript $(p)$ = varies with $p$; hat = finite-$n$ sample; no decoration = limit.
- **$\hat{\cdot}$ convention overridden in `main.tex`** (2026-06-05): in the manuscript, $\hat{M}$ is renamed $M_n$, $\hat\lambda_j$ becomes $\lambda_{n,j}$, $\hat{w}_j$ becomes $w_{n,j}$, and $\widehat{\mathrm{SNR}}_j$ becomes $\mathrm{SNR}_{n,j}$. Rationale: the subscript $n$ more clearly signals that $p\to\infty$ has already been taken and $n$ is the only remaining approximation. The hat convention is retained in all documents *other than* `main.tex` until a global migration is decided.

The full migration table (old symbol → new symbol, with every affected location) is in `Notation Migration Guide.md` (created session 2026-05-23).

**Migration status:**
- `proof_walkthrough_k3_cleaned.md` — **applied 2026-05-26** ✓
- `theorem_part_ii_1.md` — **created with migrated notation 2026-05-26** ✓
- `unified_dispersion_bias_proof_051926_cleaned.md` — pending
- `dispersion_bias_correction_cleaned.md` — pending

---

## 4. What Each Document Contains (Detail)

### 4.1 `unified_dispersion_bias_proof_051926_cleaned.md`

Structure (13 sections):

| § | Content |
|---|---|
| 1 | Introduction: two sources of misalignment, comparison of NG and AK results |
| 2 | Model, notation, $\hat{D}$, $G_\infty$, $\rho_j$, $\hat{w}_j$ |
| 3 | Assumptions 1–3 (noise, loading regularity, spectral gap) |
| 4 | Theorem (Parts i, ii-diag, iii) + Examples 4.1 (k=1), 4.2 (k=2 diagonal), 4.3 (k=2 with rotation) |
| 5 | Lemmas 1–4 (noise concentration, Borel–Cantelli, matrix noise, Gram convergence) |
| 6 | Proof of Part (i) |
| 7 | Proof of Part (ii-diag): parallel/perp decomposition, $W^{(p)}$ expansion, eigenstructure |
| 8 | Proof of Part (iii): general $G_\infty$, $U$-basis, rotation $\hat{W}$ |
| 9 | Recovery of NG's Theorem 3.1′ |
| 10 | Corollaries 1–5 |
| 11 | Discussion: two sources, loading geometry, specializations, observable estimation |
| 12 | Grassmannian subspace estimation vs. frame estimation |
| 13 | Summary |

### 4.2 `dispersion_bias_correction_v2_051226.md` (most complete; `dispersion_bias_correction_cleaned.md` is an earlier version)

Structure (7 sections):

| § | Content |
|---|---|
| 1 | What is the bias: scalar formula, irreducible component |
| 2 | James-Stein correction: structure, why irreducible part cannot be recovered, $\hat\psi_j$ estimation, Ledoit-Wolf connection |
| 3 | Illustration: $k=2$ model (two-block loadings, $n=60$, $\delta=1$) |
| 4 | MSE tables |
| 5 | Simulation code |
| 6 | General $G_\infty$: $U$-basis, $\Gamma_U$, $\hat{W}$, bias formula, invariance theorem |
| 7 | $k$-frame probe: formulation (§7.1), Frobenius deficit theorem (§7.2), JSE correction (§7.3), principal angle shrinkage (§7.4), Grassmannian bridge (§7.5), $k=2$ illustration (§7.6) |

### 4.3 `multifactor_dispersion_prevalence_v7.pdf` (AK's paper)

Strengths relative to the unified proof: has **observable bounds** not yet incorporated into `unified_dispersion_bias_proof_050726.md`:
- **Corollary 2**: $\ell_p^2/s_{p,j}^2$ as a computable lower bound on $\langle h_j, b_j^{(\beta)}\rangle^2$.
- **Theorem 2**: Two-sided bracket for the squared cosine with bias correction $B_j = \sum_{i\ne j}d_id_j/(d_i-d_j)^2$.
- **Proposition 1**: CLT — $n[\langle h_j,b_j^{(\beta)}\rangle^2_\infty - \varphi_j(1-B_j/n)] \to N(0, 2B_j^2)$.

These are the key missing piece for making the results statistically operational.

---

## 5. Completed Work (Chronological)

**[GAP — not individually reconstructed]** Between 2026-06-18 and roughly 2026-07-11, the manuscript line was rewritten from `main-9.tex` through `main-11.tex` into the `main-14` family (per file timestamps: `main-11.tex` Jun 29, `main-14.tex`/`main-14._ed.tex` Jul 12-20, `main-14a.tex` Jul 21). No session notes exist for this period's individual decisions. *(inferred from file scan only)*.

**Session 2026-07-2x (manuscript review and collaborator response, culminating 2026-07-23)** — spans several sessions on `main-16.tex`; exact session boundaries not tracked:
- Resolved a placement dispute with collaborator Alec Kercheval over Corollary 3: initially claimed it "sums the same angles as Corollary 2" — corrected after Alec caught the error (Corollary 2 sums $\sin^2\angle(h_j,\mathcal B)$, subspace-level; Corollary 3 sums $\sin^2\angle(b_j,h_j)$, matched-pair; the real connection is that Corollary 3's proof *reuses* Corollary 2's Bjorck–Golub machinery). Final framing added a "leakage" interpretation via Corollary 3's dependence on $M$'s off-diagonal terms.
- Drafted and refined a persuasive case for retaining the manuscript's Grassmann/Stiefel background (Section 9): grounded via `geometry_of_algorithms.pdf` (symmetric eigenvalue problem framing), led with Section 9's genesis as a direct question from Lisa Goldberg (not added unprompted), and generalized the audience justification to "a reader with interests similar to at least two of our collaborators."
- Defined "the symmetric eigenvalue problem" precisely and related it to the paper's sample-eigenvector setting; clarified the sample covariance $S^{(p,n)}$ is *rank-deficient* (exact, from $p\gg n$) rather than "ill-conditioned" (which would refer to near-degenerate eigenvalue gaps — a distinct issue).
- Reviewed `concentration-11.pdf`, proved $\col(b)=\col(B)$ (Eq. 4) rigorously, then validated Ken's own simpler rank-counting argument as correct and more elegant.
- Fixed two instances of a blank-page-1 LaTeX bug (both traced to `\thanks{}` placement — must nest inside `\title{...}`'s closing brace, not follow it or stand alone between `\author`/`\affil`); verified via actual `pdflatex` compiles in `/tmp`.
- Migrated `\nlgcmt{}` review comments from `main-14a.tex` (16 total) into `main-16.tex`: 9 moved to matching locations, 5 already present, 2 flagged as ambiguous (one structurally resolved by main-16's Assumption split, one orphaned — its host Remark no longer exists) rather than guessed.
- Compared `main-14.tex` vs. `main-14-revised.tex` against the "Kimi Executive Summary" critique; found and reported a serious regression (all 9 non-Theorem-1 proofs dropped, not requested by the critique) plus a real `enumitem` compile bug. Gave an independent assessment of the critique document itself, separate from the regression.
- Reviewed uploaded `main-14_restored.txt`; confirmed all 9 proofs reinstated and notation correctly reverted; found and reported two residual gaps (dangling `\eqref{eq:obsbound}`, two missing figure blocks).
- Wrote `assumptions_and_lemmas.md` (all 6 Assumptions/7 Lemmas of `main-16.tex`, from a downstream-use perspective) and converted it to a modular `\input`-appendix `assumptions_lemmas_appendix.tex` (two "at a glance" tables, Assumptions/Lemmas, at Ken's request to split and add names) — both verified via full `pdflatex` compile against a copy of `main-16.tex`.
- Built `full_symbol_glossary.tex` (~70 symbols, 12 thematic subsections), same modular `\input` design; verified via full compile.
- Reviewed Ken's own notational fix (`$\bar\Phi^{\infty}=\limp\Phi/\sqrt p$` → `$\bar\Phi^{\infty}\equiv\limp\bar\Phi,\ \bar\Phi\equiv\Phi/\sqrt p$`) — agreed it was correct for consistency, but noted the paper's standing convention is `:=` not `\equiv` for definitions; wrote a persuasive rationale for collaborators.
- Revised the Theorem 1 remark on the angular error being a random-variable (dependent on $F$ only through $FF^\top/n$ via `lem:pcconv`); compared against an independent Kimi revision of the same remark (Kimi's version correctly promoted the same $FF^\top/n$-only-dependence fact, and added a nice "confidence bounds without waiting for more data" framing, but mislabeled $FF^\top/n$ as "the realized signal Gram matrix" — a name already reserved in this paper for $N^{(p,n)}$, a related but distinct principal-coordinate object). Final merged version applied to `main-16.tex` — but at Ken's request, delivered as an `\nlgcmt{}` proposal alongside the original text, not a silent replacement.
- Wrote/maintained `side_explanations_memo.md` per Ken's standing request to log side-explanations against their paper location.

**Session 2026-06-18 (color scheme, figure fix, Davis-Kahan analysis)**:
- Applied three-color observability scheme to `proof_summary_5ideas.tex`: `\obs{}` (blue = directly from $Y$), `\est{}` (green = estimable from spectrum of $W_n^{(p)}$), black (unobserved population objects). Applied to Table 1 (signal matrices, eigenvalues, eigenvectors) and the full Appendix longtable. Caption of Table 1 updated to include color key.
- Fixed TeXmaker silent compilation failure: all three `\includegraphics` paths changed from bare filenames (Linux symlinks pointing to `paper/`) to `../figures/` paths where real PDFs live. Error code 0x80070780 = NTFS cannot follow Linux symlinks.
- Added $G_B^{(p)}$-unit ellipse to Panel 1 of `fig_isometry_pythagorean_split.pdf` — caption had referenced it but figure no longer showed it. Edited `scripts/proof/make_fig_isometry.py`: computed `G_inv_sqrt = Q @ diag(1/sqrt(eigvals)) @ Q.T`, parameterized ellipse as `G_inv_sqrt @ unit_circle`, plotted in pink (#e7298a). Regenerated figure: 47454 bytes. Previous figure had been overwritten by a blank 6K version (tight_layout + set_aspect("equal") interaction; fixed by running inline rather than via exec()).
- Applied `\obs{n}` to standalone $n$ as a multiplier/divisor in all main theorem equations (noise floor, kappa formula, sibling display, equation (d)). $n$ is directly observable (column dimension of $Y$) and was incorrectly black.
- Mathematical discussion: Davis-Kahan bound $\sin^2\angle(w_{n,j},w_j) \leq \|M_n-M\|^2/\mathrm{gap}_j^2$. Denominator $\mathrm{gap}_j = \min_{l\neq j}|\lambda_j - \lambda_{n,l}|$ is approximately green (proxied by observed eigenvalue separation). Numerator $\|M_n-M\|$ is black even as a product norm: recovering it requires $\lambda_j$ (population eigenvalues of $M$), which are only accessible as $n\to\infty$, not $p\to\infty$ with $n$ fixed. Observable scaling: $\|M_n-M\| \lesssim \lambda_{n,1}\cdot\sqrt{k/n}$ (up to a constant), but the $O_P$ notation is imprecise in this model — $F$ is fixed, so $\|M_n-M\|$ is a deterministic constant, not a random variable in the $p\to\infty$ sense.
- `proof_summary_5ideas.tex` now compiles to 12 pages, 427018 bytes.

**Session 2026-06-17 (Chebyshev appendix, main-9 review, Kolm-Ritter comparison)**:
- Wrote full Chebyshev proof that $\frac{1}{p}Z^\top\Pi_B Z \xrightarrow{p} 0$: splits into off-diagonal (Chebyshev, bound $C^2/p^2$ using $\|U^\top\Delta_z U\|_F^2 \leq [\mathrm{tr}(\Pi_B\Delta_z)]^2$ and $\mathrm{tr}(\Pi_B\Delta_z)\to C = \mathrm{tr}(G_B^{-1}G_{B,\delta}) < \infty$) and diagonal (Markov, mean $= C/p \to 0$). No CLT, no 4th moments required. Inserted as appendix into `proof_summary_5ideas.tex` (bash-mount-desync workaround: head/tail splice).
- Reviewed `main-9.tex` in full and compared with `proof_summary_5ideas.tex`. Key finding: main-9 has 4-step proof architecture (angular decomp → duality exact at every p → limiting inputs a.s. → assembly), versus 5ideas' narrative "5 ideas" format. main-9 contains Lemma 1 (a.s. convergence via 4th moments + Borel-Cantelli), Cor cor:noiseless (exact equality in noiseless case), Lemma lem:dual (Gram reduction), Lemma lem:econverge (eigenpair convergence). 5ideas' Chebyshev appendix proves only in-probability convergence — gap vs. main-9's a.s. result (see §6 item 7).
- Read and summarized `refrences/Kolm-Ritter-2026.txt` ("Hidden Factors in Portfolio Risk Models", Kolm & Ritter 2026, 8143 lines). Key contrast: KR uses proportional regime ($N,T\to\infty$, $\gamma=N/T$ fixed, BBP/Marchenko-Pastur); main-9 uses fixed-$n$ regime ($n$ fixed, $p\to\infty$, elementary LLN). KR's main contribution is Monte Carlo calibration of finite-sample BBP thresholds. KR gives scalar Paul (2007) alignment $a^2 = (1-\gamma/\theta^2)/(1+\gamma/\theta)$; main-9 gives the Pythagorean two-term decomposition (floor + rotation) which KR lacks. Note: folder is spelled `refrences/` (typo) not `references/`.

**Session 2026-06-05 (continued — Step 5 expanded exposition)**:
- Created `step5excerpt.tex`: standalone expanded exposition of Step 5 (loading-frame coordinates). Contains: complete symbol glossary for all Step 5 symbols; Group A (algebraic reduction, substeps 5A.1 Coordinate substitution / 5A.2 Gram premultiplication / 5A.3 Gram inversion / 5A.4 Eigenvalue rescaling); Group B (eigenvalue scaling note, substeps 5B.1 AB/BA dimension reduction / 5B.2 Growth rate identification / 5B.3 Limit passage); explicit "Output for Step 6" closing block with the two inner-product limits assembled.
- Key fixes applied during review: replaced circular invertibility parenthetical in 5A.3 with the correct $\lambda_{\min}(B^\top B/p)\to\lambda_{\min}(G_B)>0$ argument; removed redundant similarity-relation proof of $\Sigma_fG_B \sim M$ (already established in 5B); added sign-convention note ($a_j\to\pm a_j^\infty$, sign fixed by $a_j^\top G_B^{1/2}w_j\ge 0$) to Kato invocation; replaced `$B_{\mathrm{here}}$` clash with $P$/$Q$ naming in AB/BA applications; fixed convergence argument in 5B.3 to invoke eigenvalue continuity explicitly.
- Added `\subsubsection*` headings for Group A and Group B.
- Expanded introductory Purpose paragraph to list the three concrete deliverables from Steps 2–4 (SVD identity, eigenvalue limit, eigenvector limit) and preview the two limits Step 5 produces.

**Session 2026-06-03 to 2026-06-05 (manuscript development)**:
- Reviewed `floor-rotation.tex` and produced `floor_rotation_nlged.tex` (NLG revision with numbered green comments) and `floor_rotation_nlged_v2.tex` (typo markers separated from substantive comments, counter renumbered 1–10).
- Created and debugged `rotation_check.py` (uses factor_lab) and `rotation_check_o.py` (standalone numpy). Both verify: at $n=5$, factor 2 has mean $\sin^2\approx 0.55$; at $n=50$ still $\approx 0.13$ — substantial rotation with fully diagonal $G_B$ and $\Sigma_f$. Non-diagonal Gram is NOT required.
- Developed `main.tex` from scratch as the primary manuscript: model (§Problem Formulation), five assumptions, Theorem (floor + rotation), Lemma 1 (noise concentration), 7-step proof. Multiple revision passes.
- Notation decisions for `main.tex`: $\hat{M}\to M_n$, $\hat\lambda_j\to\lambda_{n,j}$, $\hat{w}_j\to w_{n,j}$, $\widehat{\mathrm{SNR}}_j\to\mathrm{SNR}_{n,j}$ (subscript $n$ = $p\to\infty$ already taken, $n$ is only remaining approximation).
- Created `DEFINED_TERMS.tex`: complete table of all symbols and named terms in `main.tex` with locations and open-issue flags.
- Key mathematical clarifications (for manuscript exposition):
  - Almost sure vs. convergence in probability: a.s. controls the entire sample path; needed for pointwise assembly in Step 7.
  - $\lambda_{n,j}$ is NOT the $j$th factor variance in general; it is the $j$th eigenvalue of $M_n = G_B^{1/2}(FF^\top/n)G_B^{1/2}$, blending sample factor covariance with loading structure. It equals the factor variance only when $G_B = I_k$.
  - $\lambda_{n,j}\to\lambda_j$ a.s. as $n\to\infty$ by SLLN + Weyl's inequality.
  - Kato eigenprojection continuity reference: Kato (1966), Davis-Kahan (1970 SIAM J. Numer. Anal. 7(1):1–46), Yu-Wang-Samworth (2015, Biometrika 102(2):315–323).
- Approved revision plan for `main.tex` (9 items): notation rename (1a–1e), definition placement (2a–2d). Items 3–6 (remove duplicate Step 5, Kato reference, etc.) pending.
- Created `unwrap_prose.py`: LaTeX line-break removal utility with `--dry-run` and backup.
- Updated skills: `output-discipline` (Rule 4: offer Python script for mechanical bulk edits), `hil-practice` (Rule 5: debugging spiral — write task_brief.md and stop after 3 failed attempts).

**Session 2026-05-26 (session 2 — tooling fixes)**:
- Fixed `reformat_math_extra.py`: blockquote table rows (`> | ... |`) were not having pipe characters protected inside inline math because the parser checked `line.startswith('|')` after stripping `> `; fixed by stripping `'> \t'` before the check. Also added `\emph{} → \text{}` replacement inside math spans.
- Fixed `md_to_latex.py` (multiple issues):
  - Blockquote-wrapped table rows (`> | ... |`) were being emitted as `\begin{quote}` prose instead of `tabular` environments. Fix: detect `inner.startswith('|')` in the blockquote handler; accumulate rows in `_table_rows` with `_table_in_blockquote = True`; `_flush_table()` wraps with `\end{quote}...\begin{quote}` so the surrounding blockquote context is preserved.
  - Unicode characters not supported by pdflatex (✓, —, –, …, ×, ±, ≈, ≤, ≥, ∞, smart quotes) were passed through raw. Fixed by adding `_UNICODE_SUBS` list applied in `_escape_text()`.
  - File had been silently truncated by earlier edit-tool operations; repaired by rewriting the full file via `Write` tool and stripping trailing null bytes.
- Generated `latex/theorem_part_ii_3_expanded.tex` from the expanded proof document; all tables and math render correctly.

**Session 2026-05-26**:
- Applied full notation migration (per `Notation Migration Guide.md`) to `proof_walkthrough_k3_cleaned.md`. All old symbols replaced: $\rho_j \to \hat\lambda_j$, $\hat{D} \to \hat{M}$, $G_\infty \to G^\infty_B$, $\Gamma_p \to G^{(p)}_B$, $d_j \to \lambda_j$, $W_\infty \to W$, $\chi_{p,j} \to v^{(p)}_j$, $s_{p,j} \to s^{(p)}_j$, and related changes.
- Resolved eigenvector naming question: confirmed $w_j$/$\hat{w}_j$ should be retained (not renamed to $v_j$/$\hat{v}_j$, which would collide with $v^{(p)}_j$ = eigenvectors of the dual matrix $W$).
- Created `theorem_part_ii_1.md`: full markdown transcription of `theorem_part_ii_1.pdf` with migrated notation ($G^\infty_B$, $\hat\lambda_j$, $\lambda_j$, $\widehat{\mathrm{SNR}}_j$, $w_j$).
- Created `proof_expansion_plan.md`: 9-section detailed plan for expanding `theorem_part_ii_3.pdf` into an accessible proof document with 7-step structure, k=3 inline callouts, and worked example.
- Created `theorem_part_ii_3_expanded.md`: implemented the expansion plan in full — statement (from PDF), auxiliary lemmas (Lemmas 1, 4, 7 with Lemma 7 proved in detail), 7 proof steps each with k=3 callouts, Corollary 4, and complete worked example section.

**Sessions through 2026-04-25**: Proved $k$-factor generalization (Theorem 3.1′). Established projection argument ($\Pi_B^\perp$ applied to SVD identity) as the correct approach — removes need for Assumption 2.5′ in Part (i). Fixed $\epsilon < 1/4$ error in Borel–Cantelli. Created `Proof_Theorem_3.1_prime_v3.md` and `reformat_math.py`.

**Session 2026-04-26**: Added loading-scale interpretation to v3. KT updated.

**Session 2026-05-04**: 
- Analyzed doubly-extended $\Xi$ for fundamental factor models ($B$-correlated, $F$-correlated). See `reflection_fundamental_nonorthogonal_factors.md`.
- Identified notational inconsistency in KT extension: display has $Q_\Xi\,\mathrm{diag}(\tilde\psi)$ but the indexed calculation assembles to $\mathrm{diag}(\tilde\psi)\,Q_\Xi^\top$. The bias formula is correct either way, but the intermediate display is wrong.
- Clarified z-scoring degeneracy ($c_j = 0$ for z-scored fundamental loadings → zero bias for equal-weight portfolio).
- Fixed two-faces error in `dispersion_bias_correction_cleaned.md` ($|\hat\Pi_B^{\mathrm{JS}}z - \Pi_Bz|^2 \ne \sum(1-\psi_i^2)c_i^2$; the latter is the *projection* bias, not the JSE-vector residual).
- Created `dispersion_bias_correction_v2.md` (first version of correction document).

**Session 2026-05-07** (approximate):
- Created `unified_dispersion_bias_proof_050726.md`: merged NG's Theorem 3.1′ with AK's general $G_\infty$ result into a single three-part theorem. Added Corollaries 1–5 including Corollary 4 (Grassmannian distance) and Corollary 5 (frame-level dispersion bias).
- Documented the Grassmannian bridge: $W = \tilde{B}$ in Corollary 5 gives deficit $= d_{\mathrm{Gr}}^2 = \sum_j\delta^2/(n\rho_j+\delta^2)$, connecting frame estimation to Corollary 4.
- Compared unified proof to AK's PDF: unified has general $k$ and unified theorem structure; AK's PDF has observable bounds (Cor 2, Thm 2, Prop 1) not yet incorporated.

**Session 2026-05-23**:
- Created `proof_walkthrough_k3_cleaned.md`: step-by-step illustrated walkthrough of Theorem Part (ii), Eq. (5), for $k=3$, $p=500$, $n=60$. Follows Appendix B.3 of the unified proof, giving general argument then concrete numbers at each step. Covers Steps B.3.1–B.3.7, Corollary 4, and a verification section.
- Created `proof_walkthrough_k3.py` and `proof_walkthrough_figures.py`: generate all numerical outputs and figures. Figures saved to `walkthrough_figs/`. Fixed a bug in `fig_eigvec_alignment`: $F$ must be held fixed across $p$ values to avoid trivially perfect convergence.
- Refactored `reformat_math_extra.py`: added `-o`/`--out` flag accepting a filename or `"console"` (stdout, UTF-8 safe via `sys.stdout.buffer`).
- Refactored `reflow_md.py`: replaced `sys.argv` parsing with `argparse`; added stdin support (`-` as source) and `-o` flag. Correct pipe command: `python reformat_math_extra.py paper.md -o console | python reflow_md.py - -o final.md`.
- Developed complete notation migration plan for `proof_walkthrough_k3_cleaned.md` (see §3.4). Plan not yet applied to the document.
- Created `ERRATA.md`: tracks errata against `latex/main.tex`.

**Session 2026-05-12/13**:
- Created `dispersion_bias_correction_v2_051226.md` with complete §7 ($k$-frame probe extension). Key §7 content: Frobenius deficit theorem (§7.2), JSE correction with factor-specific $1/\hat\psi_j$ inflation (§7.3), principal angle shrinkage and its restoration by JSE (§7.4), Grassmannian bridge (§7.5), $k=2$ illustration with two frames (§7.6).
- Reflowed prose to eliminate hard mid-sentence line breaks throughout (941 → 673 → 702 lines after additions).
- Fixed definition gap: added explicit definition of "Frobenius deficit" at top of §7.2.
- Confirmed: factor-specific correction — the JSE correction *does* vary by factor, through $\hat{D}_\psi^{-1} = \mathrm{diag}(1/\hat\psi_1,\ldots,1/\hat\psi_k)$; weaker factors receive larger inflation.

---

## 6. Open Work (Priority Order)

**1. Incorporate AK's observable bounds into the unified proof.** Corollary 2 (lower bound $\ell_p^2/s_{p,j}^2$), Theorem 2 (two-sided bracket with $B_j$ bias correction), and Proposition 1 (CLT) from `multifactor_dispersion_prevalence_v7.pdf` are the missing statistical inference layer. Currently the unified proof establishes the limits but gives no CLT or finite-sample bounds.

**2. Carry out the doubly-extended Part (ii) for fundamental factor models.** The sketch in `reflection_fundamental_nonorthogonal_factors.md` handles non-orthogonal loadings *and* non-orthogonal factor returns via $\Xi = \Gamma_\infty^{1/2}(F^\top F)\Gamma_\infty^{1/2}$. Needs to be written as a proper theorem extending Part (iii) of the unified result.

**3. Fix five proof issues in the cleaned proof** (all small; listed in `KT_update_2026-05-04.md` §3):
   - Assumption 2.3′: "pairwise independent" should be "mutually independent within each column."
   - Lemma A.2′ Part 2: define $\lambda_i$ without 2.6′ first; closed form only under 2.6′.
   - Weyl invocation: spell out the Courant–Fischer corollary and the symmetric-matrix version.
   - Sign convention: ensure `revisedProofTheorem3.1+.md_cleaned.md` explicitly states $e^\top h_i/p \ge 0$.
   - Atomlessness in 2.1′: align wording with KT extension's requirement for joint atomlessness.

**4. Apply notation migration to primary proof documents.** `proof_walkthrough_k3_cleaned.md` is done (2026-05-26). Still pending: `unified_dispersion_bias_proof_051926_cleaned.md` and `dispersion_bias_correction_v2_051226.md` (the most complete correction document — both confirmed present on disk). Use `Notation Migration Guide.md` as the source.

**5. Manuscript preparation**: LaTeX conversion of `unified_dispersion_bias_proof_051926_cleaned.md`, length check against SIAM J. Financial Math style (target 6–8 pages per the earlier brief), citation hygiene (Davis–Kahan via Bhatia or Stewart–Sun; GPS2022 explicit citation for Lemma A.1).

**6a. Complete `main.tex` cleanup (approved but not yet applied)**:
- Remove duplicate Step 5 block (red `{\color{red}...}` block, lines ~586–658) and its malformed `\begin{lemma*}` without `\end{lemma*}`.
- Remove red draft-lemmas in Steps 2, 3, 4 (or formalize as proper numbered lemmas before Theorem 1).
- Resolve remaining blue AK comments (notation, Kato reference, $\lambda_{n,j}$ description).
- Correct SNR gloss: replace "the $j$th factor variance $\lambda_{n,j}$" with accurate description (see §7).
- Add Kato et al. references with pinpoint cites (Kato 1966 Ch. I §5, Davis-Kahan 1970, Yu-Wang-Samworth 2015).
- Apply `unwrap_prose.py` to remove mid-paragraph line breaks from `main.tex`.

**6b. Notation migration decision**: $M_n$ notation is now used in `main.tex`. Decide whether to backport to `unified_dispersion_bias_proof_051926_cleaned.md` and other documents, or maintain separate conventions. If backporting, update `Notation Migration Guide.md` first.

**7. Fix a.s./in-probability gap in `proof_summary_5ideas.tex` Chebyshev appendix.** The appendix currently proves only in-probability convergence for each entry of $\frac{1}{p}Z^\top\Pi_B Z$. main-9's Lemma 1 achieves a.s. convergence using 4th moments + Borel-Cantelli ($\sum_p P(|X_{ij}| \geq \varepsilon) \leq C^2/(\varepsilon^2) \sum_p 1/p^2 < \infty$). Fix requires: (a) add assumption $\sup_i \mathbb{E}[Z_{ai}^4] \leq \kappa_4 < \infty$; (b) upgrade off-diagonal bound to show $O(1/p^2)$ summability; (c) invoke Borel-Cantelli. Medium effort.

**6. Observable estimation for the $k$-frame correction.** Section §7 of the correction document estimates the Frobenius deficit but does not give a CLT or confidence interval for $\|\Pi_B W\|_F^2$. The AK observable bounds (Task 1 above) would fill this gap.

**8. Reconcile the two Corollary sets.** `main-16.tex`'s Corollaries (`cor:obsfloor`, `cor:subspace`, `cor:noiseless`, `cor:purenoise`) have not been mapped against the old-notation Corollaries 1–5 (§2.1–2.3) or against the James-Stein correction machinery (§2.4/strand 2). It is not yet established whether the correction strand still applies cleanly to the `main-16.tex` formulation, or needs rederiving.

**9. Reapply `main-14_restored.txt`'s two residual fixes to `main-16.tex`.** The dangling `\eqref{eq:obsbound}` (→ `eq:obsfloor`) and the two missing figure blocks (`fig:decomp_p_sweep`, `fig:decomp_n_sweep`) were found and reported against `main-14_restored.txt`; unverified whether they were carried forward into `main-16.tex` or still need fixing there.

**10. Resolve the two orphaned `\nlgcmt{}` comments from the `main-14a.tex` migration.** (a) The "Assumption 1 combines three things" comment — likely moot, since `main-16.tex` already splits old Assumption 1 into `asm:fm`/`asm:noise`, but not explicitly confirmed with Ken. (b) The "$f^{(l)}$-free" comment — its host Remark no longer exists in `main-16.tex` (replaced by a different remark referencing `lem:uncorr`); needs Ken's judgment on whether the underlying concern still applies elsewhere.

**11. Decide on the pending `\nlgcmt{}` remark-revision proposal.** The Theorem 1 remark on the angular error's random-variable status (line ~761 of `main-16.tex`) now carries a proposed rewrite as an `\nlgcmt{}` comment (merged from this session's revision + a Kimi revision), with the original text left in place per Ken's request. Awaiting Ken's decision on whether to adopt it as the final remark text.

---

## 7. Key Technical Details to Remember

**The $\epsilon < 1/4$ threshold.** Borel–Cantelli requires $\Pr(|W_p| > p^{1/2-\epsilon}) \le K/p^{2-4\epsilon}$, summable only when $2-4\epsilon > 1$, i.e. $\epsilon < 1/4$. Earlier drafts wrote $\epsilon < 1/2$. This is wrong.

**The exact SVD identity.** $HS_p = Y\mathcal{X}_p/\sqrt{n}$ is an exact equality, not an approximation. Right-multiply the full SVD by the top-$k$ right singular vectors; the tail columns vanish by orthogonality.

**Why Part (i) doesn't need 2.5′.** Old approach: dot with $b_m$, need $b_m^\top B$ to collapse (requires 2.5′). New approach: apply $\Pi_B^\perp$ to the full matrix identity; $\Pi_B^\perp B = 0$ for any $B$, orthogonal columns or not.

**KT extension display error.** In `KT_extension_to_nonorthogonal_factors.md` §6, the displayed limit $H^\top\tilde{B} \to Q_\Xi\,\mathrm{diag}(\tilde\psi)$ is wrong. The indexed calculation assembles to $\mathrm{diag}(\tilde\psi)\,Q_\Xi^\top$. The bias formula is correct because it uses $M^\top M$ which is the same either way.

**Z-scoring degeneracy.** For z-scored fundamental loadings, $\mu_p(\beta_j) = 0$ so $c_j = 0$ for all $j$, and the equal-weight portfolio dispersion bias is zero by construction. The interesting quantity for fundamental factor practitioners is the bias for *factor-tilted* portfolios with $c_j \ne 0$.

**Grassmannian vs. frame estimation.** The Grassmannian metric $d_{\mathrm{Gr}}^2 = \sum_j\delta^2/(n\hat\lambda_j+\delta^2)$ accumulates only the irreducible floors; the in-subspace rotation cancels. Frame estimation adds the rotation term, which is non-zero when factors are correlated. So: measuring subspace alignment by $d_{\mathrm{Gr}}^2$ is *strictly easier* than measuring it by per-direction cosines when $G^\infty_B \ne I_k$.

**JSE correction is factor-specific.** The operator $D_\psi^{-1} = \mathrm{diag}(1/\hat\psi_1,\ldots,1/\hat\psi_k)$ inflates each factor-$j$ coordinate differently. Weaker factors ($\hat\psi_j$ closer to 0) receive larger inflation. This is a feature: the correction is proportionally stronger where the bias is worst.

**$\lambda_{n,j}$ is NOT the $j$th factor variance.** $\lambda_{n,j}$ is the $j$th eigenvalue of $M_n = G_B^{1/2}(FF^\top/n)G_B^{1/2}$. By AB/BA, this equals the $j$th eigenvalue of $(FF^\top/n)G_B$ — a blend of sample factor covariance and loading Gram. Only when $G_B = I_k$ does $\lambda_{n,j}$ reduce to the $j$th eigenvalue of $FF^\top/n$ (the sample factor variance). The SNR ratio $n\lambda_{n,j}/\delta^2$ should be described as the $j$th signal eigenvalue scaled by sample size, relative to average noise — not as a factor variance ratio. Population analog: $\lambda_j$ is the $j$th eigenvalue of $M = G_B^{1/2}\Sigma_f G_B^{1/2}$, similarly a blend. Convergence: $\lambda_{n,j}\to\lambda_j$ a.s. as $n\to\infty$ (SLLN gives $FF^\top/n\to\Sigma_f$ a.s., then Weyl).

**$F$ is fixed; $\|M_n-M\|$ is deterministic, not random.** In this model $p\to\infty$ with $n,k$ fixed and $F\in\mathbb{R}^{k\times n}$ fixed (one realization). So $\|G_B^{1/2}(FF^\top/n-\Sigma_f)G_B^{1/2}\|$ is a deterministic (but unobserved) constant — it is NOT a random variable in the $p\to\infty$ probability space. Writing $O_P(\cdot)$ for it is imprecise; the $O_P$ applies only if $F$ is treated as random (i.e., averaging over hypothetical draws of $F$). The Davis-Kahan bound denominator ($\mathrm{gap}_j$) is approximately green; the numerator ($\|M_n-M\|$) is black for fixed $n$. The term $\sin^2\angle(w_{n,j},w_j)$ goes to 0 as $n\to\infty$ (SLLN on $FF^\top/n\to\Sigma_f$, then Weyl + eigenvector continuity), at rate $O(1/\sqrt{n})$ under standard moment conditions on $F$.

**Principal angle shrinkage is a new phenomenon at $k_W > 1$.** The scalar bias ($k_W = 1$) says only that $|H^\top z|^2 < |\Pi_B z|^2$. For a frame ($k_W > 1$), the singular values of $H^\top \tilde{W}$ are all strictly shrunk: $\sigma_l(H^\top \tilde{W}) \to \sigma_l(\Psi_\infty\Gamma_\infty) \le \psi_{\infty,1}\sigma_l(\Gamma_\infty)$. The JSE correction restores them: $\sigma_l(D_\psi^{-1}H^\top \tilde{W}) \to \sigma_l(\Gamma_\infty)$.

---

**Rank-deficient vs. ill-conditioned (main-16.tex's setting).** The sample covariance $S^{(p,n)}$ in the current manuscript's regime ($p\to\infty$, $n$ fixed) is *rank-deficient* by construction ($\mathrm{rank}(S^{(p,n)})\le n\ll p$) — an exact, structural fact, not a numerical-conditioning issue. "Ill-conditioned" would instead describe near-degenerate top-$k$ eigenvalue gaps, which is a separate (and, per `asm:reg`, generically avoided) concern. Don't conflate the two when describing why sample eigenvectors are hard to estimate here.

**Corollary 3's "leakage" interpretation (main-16.tex).** Corollary 3 sums $\sin^2\angle(b_j,h_j)$ (matched-pair angles) via the diagonal/off-diagonal decomposition of $M$ (the relevant Gram matrix); its dependence on $M$'s *off-diagonal* terms has a natural reading as misalignment "leakage" between factor directions, distinct from Corollary 2's subspace-level $\sin^2\angle(h_j,\mathcal B)$ result. Corollary 3's proof reuses Corollary 2's proof machinery (Bjorck–Golub) — that's the real connection between the two, not that they sum the same angles (an error caught by Alec Kercheval, see §5).

**$\bar\Phi$/$\bar\Phi^\infty$ notation (main-16.tex).** Current convention: $\bar\Phi:=\Phi/\sqrt p$, $\bar\Phi^\infty\equiv\lim_p\bar\Phi$ — the averaged object $\bar\Phi$ is defined once and used consistently, rather than writing $\bar\Phi^\infty=\lim_p\Phi/\sqrt p$ inline each time. Note the paper's standing definitional symbol is `:=`, not `\equiv` — flagged as a minor inconsistency in Ken's edit, not yet resolved.

## 8. Reading Order for a Fresh Session

To come up to speed, read in this order:

1. **This file** (`docs/KT.md`) — project overview, esp. §2.0 for the current theorem.
1b. `paper/main-16.tex` — **the active formal manuscript**. Read the section/label map in §2.0 first; the manuscript itself is 2360 lines.
1c. `paper/assumptions_and_lemmas.md` — the 6 Assumptions/7 Lemmas of `main-16.tex` explained from a downstream-use perspective; faster orientation than reading the Appendix cold.
1d. `paper/side_explanations_memo.md` — running log of conceptual clarifications tied to specific `main-16.tex` locations; check for anything relevant before re-deriving from scratch.
2. If working on the *older* strand (dispersion-bias proof/correction, pre-`main-9` notation — not yet reconciled with `main-16.tex`, see §6 item 8): `unified_dispersion_bias_proof_051926_cleaned.md` §1–4, then `dispersion_bias_correction_cleaned.md` §1–2/§7 *(verify filename — see §3.1)*, then `unified_dispersion_bias_proof_051926_cleaned.md` §10–12.
2b. `theorem_part_ii_3_expanded.md` — accessible proof of the older Part (ii) with full derivation and k=3 worked example (old-strand notation).
3. `reflection_fundamental_nonorthogonal_factors.md` — if working on fundamental factor extension.
4. `multifactor_dispersion_prevalence_v7.pdf` §3.6–3.7 — for observable bounds (old-strand Task 1).

`Proof_Theorem_3.1_prime_v3.md` is still the cleanest standalone proof of the NG case (old strand) and is worth reading if the proof mechanics are unclear. `paper/main-9.tex` is superseded — read only for historical comparison, not as a starting point.

---

*End of KT.*
