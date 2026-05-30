# Knowledge Transfer — Factor Lab

*Supersedes `KT_proof_theorem3.1prime.md`, `KT_extension_to_nonorthogonal_factors.md`, and `KT_update_2026-05-04.md`. Self-contained briefing for a fresh session.*

> Last updated: 2026-05-26 (session 2) | Trigger: manual | Staleness: Drifting — §3.1 (correction file verified, .tex output noted), §3.3 (tool descriptions updated), §5 (new session work), §6 (resolved correction-file uncertainty).

---

## 1. Project Context

This project proves and develops the **multifactor dispersion bias**: in a $k$-factor return model $Y = BF^\top + Z$, the top-$k$ sample principal components $H = [h_1,\ldots,h_k]$ are systematically rotated away from the population loading directions $\bar{b}_1,\ldots,\bar{b}_k$, with exact asymptotic limits as $p\to\infty$ with $n,k$ fixed. The consequence is that portfolio-level factor-exposure estimates are downward-biased, and a James-Stein-type correction exists.

The project has three interlocking strands:

1. **The proof** — Theorem 3.1′ (NG, single author, $k$-factor, diagonal $G^\infty_B$) and its unification with AK's result (general $G^\infty_B$), now in `unified_dispersion_bias_proof_051926_cleaned.md`.
2. **The correction** — The James-Stein correction $\hat\Pi_B^{\mathrm{JS}} z = HD_\psi^{-1}H^\top z$, developed through the $k$-frame probe extension, in `dispersion_bias_correction_cleaned.md` *(note: `dispersion_bias_correction_v2_051226.md` referenced in earlier KT is not found on disk — verify which file is current)*.

3. **Manuscript preparation** — LaTeX conversion, citation hygiene, SIAM J. Financial Math format. Not yet started.

---

## 2. The Mathematics

### 2.1 Model

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
| `multifactor_dispersion_prevalence_v7.pdf` | AK's paper. $k=3$, general $G^\infty_B$. Has observable bounds (Cor 2, Thm 2, Prop 1) not yet in the unified proof. | External reference; see §4.3 below |
| `proof_walkthrough_k3_cleaned.md` | Step-by-step illustrated walkthrough of Theorem Part (ii) with $k=3$, $p=500$, $n=60$ concrete example. Follows Appendix B.3. **Notation migration applied 2026-05-26.** | **Primary expository reference for the proof** |
| `theorem_part_ii_3_expanded.md` | Expanded proof of Theorem 1 Part (ii): statement, full 7-step proof with Lemmas 1/4/7 (fully proved with Borel–Cantelli + Kolmogorov SLLN), inline k=3 callouts, Corollary 4, worked example. | **Primary accessible proof document** |
| `latex/theorem_part_ii_3_expanded.tex` | LaTeX conversion of the above via `md_to_latex.py`. Blockquote-wrapped tables and Unicode ✓ handled correctly. | LaTeX artefact — regenerate as needed |
| `theorem_part_ii_1.md` | Theorem statement and notation from `theorem_part_ii_1.pdf`, converted to markdown with migrated notation. | Created 2026-05-26 |
| `Notation Migration Guide.md` | Complete two-table guide: symbol renames + affected locations for all documents. | Reference |
| `proof_expansion_plan.md` | 9-section plan for expanding `theorem_part_ii_3.pdf` into an accessible proof document. | Implemented in `theorem_part_ii_3_expanded.md` |

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
- **$\hat{\cdot}$ retained** (not replaced by $^{(n)}$) for finite-$n$ quantities, given its standard statistical meaning and use in the theorem statement.

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

**6. Observable estimation for the $k$-frame correction.** Section §7 of the correction document estimates the Frobenius deficit but does not give a CLT or confidence interval for $\|\Pi_B W\|_F^2$. The AK observable bounds (Task 1 above) would fill this gap.

---

## 7. Key Technical Details to Remember

**The $\epsilon < 1/4$ threshold.** Borel–Cantelli requires $\Pr(|W_p| > p^{1/2-\epsilon}) \le K/p^{2-4\epsilon}$, summable only when $2-4\epsilon > 1$, i.e. $\epsilon < 1/4$. Earlier drafts wrote $\epsilon < 1/2$. This is wrong.

**The exact SVD identity.** $HS_p = Y\mathcal{X}_p/\sqrt{n}$ is an exact equality, not an approximation. Right-multiply the full SVD by the top-$k$ right singular vectors; the tail columns vanish by orthogonality.

**Why Part (i) doesn't need 2.5′.** Old approach: dot with $b_m$, need $b_m^\top B$ to collapse (requires 2.5′). New approach: apply $\Pi_B^\perp$ to the full matrix identity; $\Pi_B^\perp B = 0$ for any $B$, orthogonal columns or not.

**KT extension display error.** In `KT_extension_to_nonorthogonal_factors.md` §6, the displayed limit $H^\top\tilde{B} \to Q_\Xi\,\mathrm{diag}(\tilde\psi)$ is wrong. The indexed calculation assembles to $\mathrm{diag}(\tilde\psi)\,Q_\Xi^\top$. The bias formula is correct because it uses $M^\top M$ which is the same either way.

**Z-scoring degeneracy.** For z-scored fundamental loadings, $\mu_p(\beta_j) = 0$ so $c_j = 0$ for all $j$, and the equal-weight portfolio dispersion bias is zero by construction. The interesting quantity for fundamental factor practitioners is the bias for *factor-tilted* portfolios with $c_j \ne 0$.

**Grassmannian vs. frame estimation.** The Grassmannian metric $d_{\mathrm{Gr}}^2 = \sum_j\delta^2/(n\hat\lambda_j+\delta^2)$ accumulates only the irreducible floors; the in-subspace rotation cancels. Frame estimation adds the rotation term, which is non-zero when factors are correlated. So: measuring subspace alignment by $d_{\mathrm{Gr}}^2$ is *strictly easier* than measuring it by per-direction cosines when $G^\infty_B \ne I_k$.

**JSE correction is factor-specific.** The operator $D_\psi^{-1} = \mathrm{diag}(1/\hat\psi_1,\ldots,1/\hat\psi_k)$ inflates each factor-$j$ coordinate differently. Weaker factors ($\hat\psi_j$ closer to 0) receive larger inflation. This is a feature: the correction is proportionally stronger where the bias is worst.

**Principal angle shrinkage is a new phenomenon at $k_W > 1$.** The scalar bias ($k_W = 1$) says only that $|H^\top z|^2 < |\Pi_B z|^2$. For a frame ($k_W > 1$), the singular values of $H^\top \tilde{W}$ are all strictly shrunk: $\sigma_l(H^\top \tilde{W}) \to \sigma_l(\Psi_\infty\Gamma_\infty) \le \psi_{\infty,1}\sigma_l(\Gamma_\infty)$. The JSE correction restores them: $\sigma_l(D_\psi^{-1}H^\top \tilde{W}) \to \sigma_l(\Gamma_\infty)$.

---

## 8. Reading Order for a Fresh Session

To come up to speed, read in this order:

1. **This file** (`docs/KT.md`) — project overview.
2. `unified_dispersion_bias_proof_051926_cleaned.md` §1–4 — theorem statement and examples.
3. `dispersion_bias_correction_cleaned.md` §1–2, §7 — correction and k-frame extension *(verify filename — see §3.1)*.
4. `unified_dispersion_bias_proof_051926_cleaned.md` §10–12 — corollaries and Grassmannian discussion.
4b. `theorem_part_ii_3_expanded.md` — accessible proof of Part (ii) with full derivation and k=3 worked example.
5. `reflection_fundamental_nonorthogonal_factors.md` — if working on fundamental factor extension.
6. `multifactor_dispersion_prevalence_v7.pdf` §3.6–3.7 — for observable bounds (Task 1).

`Proof_Theorem_3.1_prime_v3.md` is still the cleanest standalone proof of the NG case and is worth reading if the proof mechanics are unclear.

---

*End of KT.*
