# Knowledge Transfer — Factor Lab

> Last updated: 2026-10-07T16:45Z | Trigger: manual (full rewrite) | Staleness: Drifting (paths and content re-verified 2026-10-07; only the 2026-06-19→09-27 history is unrecorded, *(inferred)* from file dates)

⚠️ HIL NOTICE — please confirm (Ken):
1. June-era open items 3, 4, 7 (old Markdown proofs; Chebyshev a.s. gap in `proof_summary_5ideas.tex`) are classified OBSOLETE/low priority by assumption (you did not answer).
2. The 2026-06-19→09-27 lineage in §5 and §4 is *(inferred)* from file dates and the `\date` lines, not from notes.
3. Purpose of `paper/diff.html` / `paper/diff.tex` is unconfirmed (§4).

## 1. Project Overview

**What it is.** In a statistical k-factor model Y = BF + Z (p assets × n observations), the leading principal components of the sample covariance are used to estimate the factor ("principal") directions. With p → ∞ and n, k fixed, we prove that their error has an exact almost-sure limit made of two parts: an irreducible **floor** (noise swamps part of each signal direction; more assets do not help) and a **rotation** (the sample factor covariance differs from the population one, so the realized directions are rotated within the factor subspace). The floor is estimable from Y; the rotation is not. This is the **multifactor dispersion bias**.

**The paper.** `paper/main-25.tex`, "Principal component error in high-dimensional factor models", version date 2026-10-06 (Ken confirmed it is the current draft on 2026-10-07). Authors: Alex Bernstein, Lisa R. Goldberg, Ken ("NLG"; the `\nlgcmt` comments are his), Alec Kercheval ("AK"), Tian Lan, Yian Lin, Dayi Yao. Reviewer macros in the source: `\nlgcmt` (Ken, green), `\akcmt`, `\lrbcmt`, `\abcmt`, `\nlgcur`. Structure: Introduction; factor model; data assumptions; Gram reduction; Errors in principal directions (Theorem 1 `thm:main`; subsection "Proof of the error decomposition", label `sec:main-proof`, called **§6.2** in all sessions; starts at line 812 of `main-25.tex`, Theorem 1 at line 751, `lem:gramdual` at line 1546); estimable out-of-subspace error (Thms `thm:obsfloor`, `thm:rotrange`); non-estimable in-subspace error; Simulation; Conclusion; Supporting results (incl. correlated specific returns); Indeterminacy of the principal frame; Symbol table.

**People.** Alec Kercheval (AK) and Lisa Goldberg (LG) are the main co-authors/reviewers. LG's critique: practitioners hold (b, Φ), not (B, F); answered by an insert in §6.2 (§5 below). Ken sent Alec the Gram-duality note on 2026-10-07.

**Status.** Active: polishing the proof of Theorem 1 and collaborator notes. The James-Stein correction is **not** part of the paper (Ken, 2026-10-07; project history in `docs/KT_history.md`).

## 2. Goals & Constraints

**Goals.** (1) Make Theorem 1 and its proof (§6.2 of `main-25.tex`) final. (2) Decide whether the proof takes the shorter "simplified" structure (§5, §6). (3) Keep co-author notes (Alec, Lisa) consistent with `main-25.tex`.

**Constraints.**
- Regime: p → ∞, n and k fixed; **F is the fixed realized k×n path** and is conditioned on; "almost sure" refers to Z only.
- Z: independent entries, mean 0, variance δ², bounded fourth moments (Assumption `noise`; Assumption `noise′` allows cross-sectional correlation with ‖Δ^{(p)}‖ ≤ C_Δ).
- Notation is that of `main-25.tex` (cheat-sheet in §4). Math is written in LaTeX. `paper/*.tex` was compiled in a scratch build directory, so PDFs in `paper/` can be stale.
- Working mode (Ken's standing rules): Claude reviews and supplies anchored small-edit LaTeX; Ken applies edits himself; Claude edits his files only when asked; do exactly what is asked; ask before unrequested changes.

**Non-goals.** James-Stein/bias correction inside the paper; proportional-regime (p/n → c: BBP, Marchenko–Pastur) results; Grassmann-vs-Stiefel estimation material (kept as notes).

## 3. Prototypes & Examples

| Item | Status | One line |
|---|---|---|
| `paper/theorem1_simplified_proof.tex` | Candidate structure; edited 2026-10-07 | Exact coordinate ξ_j := b̄ᵀh_j; one limit gives floor and rotation; no e_j term. Compiles (8 pp.); its PDF is not regenerated. |
| `paper/theorem1_frame_free_proof.tex` | Reference, unchanged | Same idea with no orthonormal frame (coordinate G⁻¹Bᵀh/√p, G-angle); conceptual, not in the paper's notation. |
| `proof_working/proof_walkthrough_k3_cleaned.md` | Reference | Illustrated k = 3 worked example (p = 500, n = 60); old notation, migrated 2026-05-26. |
| `scripts/sim/rotation_check.py` (+ `rotation_check_o.py`, project root) | Working | Monte-Carlo check of floor + rotation; first uses the `factor_lab` package, second is plain numpy. |
| `condensed_proof_skeleton.tex` | Superseded in spirit by the simplified proof | Gram duality → one LLN limit → trace assembly (3 pp., 2026-06-11). |
| `paper/main-25.tex` §8 simulation | In the paper | Three-factor US-equity simulation validating the asymptotics; code `fl_experiment_runner.py`, `fl_experiment_setup.py` (project root). |

## 4. Architecture & Key Files

**Theorem 1 (`thm:main`, main-25 notation).** Y = BF + Z, G_B^{(p)} = BᵀB/p → G_B ≻ 0, M̂ = G_B^{1/2}(FFᵀ/n)G_B^{1/2} has eigenpairs (λ_j, ω_j), M = G_B^{1/2}Σ_f G_B^{1/2} has (μ_j, β_j); h_j = j-th sample PC, b_j = j-th principal direction. Then, almost surely, as p → ∞:
sin²∠_p(h_j, b_j) → δ²/(nλ_j + δ²) + c_j² · sin²∠(ω_j, β_j), c_j² = nλ_j/(nλ_j + δ²).
First term = floor (irreducible, estimable); second = rotation (vanishes if M̂ = M, i.e. as n → ∞; not estimable from Y).

**Notation cheat-sheet (old KT / older documents → `main-25.tex`).**

| Old | main-25 |
|---|---|
| Y = BFᵀ + Z, F n×k | Y = BF + Z, F k×n (fixed realized path) |
| G_B^∞ | G_B (finite p: G_B^{(p)} = BᵀB/p) |
| M̂ with (λ̂_j, ŵ_j); M with (λ_j, w_j) | M̂ with (λ_j, ω_j); M with (μ_j, β_j); Σ_F → Σ_f |
| κ_j (or ψ_∞,j) | c_j := √(nλ_j/(nλ_j+δ²)) |
| W_n^{(p)}, θ_{n,j} | W^{(p)} = YᵀY/(np), n×n dual, eigenpairs (θ_j^{(p)}, w_j^{(p)}) |
| — | loading frame b̄ = B(BᵀB)^{−1/2}; Π_B = b̄b̄ᵀ = bbᵀ (`\Pib`, `\Pibp`); β_j^{(p)} = b̄ᵀb_j; B = √p b̄ (G_B^{(p)})^{1/2}; duality link G_B^{1/2}F w_j = √(nλ_j) ω_j |
| ‖·‖_F | ‖·‖_Fr (`\|\cdot\|_{\mathrm{Fr}}`) |

**Where things are now** (verified 2026-10-07; paths relative to the project root). Full older inventory: `docs/KT_history.md`.

| Group | File | Note |
|---|---|---|
| Current manuscript | `paper/main-25.tex` | 144 KB, saved 2026-10-07 16:09. Has weakened `lem:gramdual`, Lisa insert (~line 831), `c_j`. Bibliography: `\bibliography{references_2}` (line 2476); `references_2.bib` is *not found* on disk (`paper/references.bib` and `paper/references_old.bib` exist). |
| Companions | `paper/KT_main24_crosscorr_review.md` | KT of the 2026-09-29 review of `main-24` (correlated specific returns, App. A.5). |
| | `paper/nlg_cmts_100226.md` | Ken's six review comments on §6.2 (applied in `main-24a`/`main-25`). |
| | `paper/Notation Migration Guide.md` | Old-symbol → new-symbol tables (May 2026 conventions). |
| Reference proofs | `paper/theorem1_simplified_proof.tex`, `paper/theorem1_frame_free_proof.tex` | See §3. |
| | `proof_working/unified_dispersion_bias_proof_051926_cleaned.md` | Old full proof (NG + AK unification), old notation. |
| | `proof_working/Proof_Theorem_3.1_prime_v3.md` | Single-author k-factor proof (diagonal case). |
| | `multifactor_dispersion_prevalence_v7.pdf` | AK's earlier paper (k = 3, observable bounds); `multifactor_dispersion_v8.pdf` also present, relation unclear. |
| Historical manuscripts | `paper/main-24a.tex` (Ken's copy of Alec's `paper/main-24.tex`, version 2026-09-28), `paper/main-20.tex` (09-14), `paper/main-19_extended.tex`, `paper/main-19_original.tex` (09-01), `paper/main-17.tex` (08-21), `paper/main-16.tex` (08-19; many variants `main-16*.tex`), `paper/main-15.tex` (07-20), `paper/main-14.tex` (07-12), `paper/main-11.tex`, `paper/main-9.tex` (06-17) | Lineage is *(inferred)* from `\date` lines and mtimes. `paper/main-distributed.tex` (09-14) is a distributed copy of the main-20 era. |
| | `main-8.tex`, `archive/main-8.tex`, `main.tex`, `floor-rotation.tex`, `proof_summary_5ideas.tex` | Pre-paper drafts and the "5 ideas" summary (project root, not `paper/`). |
| Diffs | `paper/diff.html` | Redline (87 insertions, 55 deletions) of a main-24-lineage text; purpose unconfirmed. `paper/diff.tex` is empty (0 bytes). Latexdiffs: `paper/main1920diff.tex`, `paper/main1920diff_v2.tex`. |
| Code and tools | `factor_lab/` (package), `scripts/sim/`, `scripts/proof/`, `scripts/util/`, `sim_theorem_partii.py`, `tests/`, `specs/`, `pyproject.toml` | Simulation engine, JSON-spec driven sims, proof figures, Markdown/LaTeX helpers. |
| Backups | `paper/theorem1_simplified_proof.tex.bak_pre_nlgpolish`, `.bak_pre_gramfix`, `.bak_pre_items145` (in `paper/`); `_pull_backup_20260902_233940/`; `docs/KT_pre_rewrite_2026-10-07.md` | Safe to ignore. |
| Archive | `docs/KT_history.md` | Full previous KT (session log 2026-04-25→10-07, corollaries 3–5, James-Stein, old inventory, open-item list). |

**Gotchas.**
- *Gram duality needs only a simple j-th eigenvalue.* The old hypothesis (all nonzero eigenvalues of AAᵀ simple) fails because θ_{k+1..n} → δ²/n; `main-25` states the weakened `lem:gramdual`; Alec was told.
- *The a.s. threshold p₀ is random* (depends on the realization of Z).
- *F is fixed, so ‖M̂ − M‖ is a deterministic constant.* O_P is wrong for it unless F is treated as random.
- *Rotation error is not estimable from Y* (only λ_j, w_j, θ_j, δ², hence floor and c_j, are). With F random and independent of Z, Theorem 1 holds conditionally and the limit is a random variable R_j(F).
- *ξ_j clash.* In `main-25` ξ_j is only the systematic coordinate (Π_B h_j = b̄ξ_j + e_j, with e_j = o(1)); in the simplified proof ξ_j := b̄ᵀh_j is exact and a_j is the systematic part.
- *λ_j is not "the j-th factor variance"*: it is the j-th eigenvalue of G_B^{1/2}(FFᵀ/n)G_B^{1/2}, a blend with G_B.
- *Parametrization.* Invariance under (B,F) ↦ (BR, R⁻¹F) holds at finite p (b̄ ↦ b̄Q, β_j^{(p)} ↦ Qᵀβ_j^{(p)}); the practitioner embedding B = √p b, F = Φ/√p gives G_B = I_k and b̄ = b.
- *Correlated specific returns:* the simplified proof uses independence only through its Fact 3; growing ‖Δ^{(p)}‖ is not covered.

## 5. Recent Decisions & Rationale

1. **2026-10-07 — Gram-duality hypothesis weakened.** Only the j-th eigenvalue must be simple (θ_j for j ≤ k); the proof only needs top-k. Why: the bottom eigenvalues of W^{(p)} cluster at δ²/n. Applied in `main-25` `lem:gramdual`, in the simplified proof (Fact 1 parts (i),(ii)), and sent to Alec.
2. **2026-10-07 — `main-25.tex` is the working draft.** It supersedes `main-24a`/`main-24`; Ken will redline it against `main-24` himself.
3. **2026-10-07 — James-Stein correction not part of the paper.** Why: scope; retained only in `docs/KT_history.md`.
4. **2026-10-07 — Simplified proof polished at Ken's request** (nlgcmt polish; Gram fix; Fact 2 proof, Parametrization paragraph, correlated-returns remark). Backed up; main-25 untouched by Claude.
5. **2026-10-01…07 — Proof structure.** All three §6.2 variants (paper, simplified, frame-free) are correct. Recommendation (Ken has not decided): adopt the simplified exact-coordinate structure in the b̄ notation, rename the systematic coordinate a_j^{(p)}, keep the noise calculation for the floor as a remark.
6. **2026-10-03…07 — Random F / only Y observed.** F random and independent of Z: conditional version of Theorem 1; by Fubini sin²∠ − R_j(F) → 0 a.s., expectation by bounded convergence; k = 1 Gaussian: 1/(1 + rχ²_n). Only Y observed: floor and c_j estimable, rotation not; for Gaussian F the rotation has a matrix-Bingham posterior (checked by Monte Carlo, k = 2, n = 6).
7. **2026-10-02…03 — Lisa Goldberg's critique** answered by the practitioner embedding and an insert about finite-p invariance (present in `main-25` near line 831). An earlier draft overstated invariance of the limit objects; corrected to finite-p invariance only.

## 6. Open Questions & Blockers

Owner = Ken unless stated; date = when raised.

1. **Restructure Steps 1–3 per the simplified proof?** (2026-10-01; Ken to decide; Claude writes anchored small edits on request). If yes, rename ξ_j to avoid the clash.
2. **Finish §6.2 cleanup in `main-25`** (2026-10-07; Ken applies, Claude supplies text): `$x:=\dots,\ x$ is therefore bounded` (~line 1021) → `$x:=\dots$, $x$ is therefore bounded`; leftover brace wrappers from removed `\nlgcmt` (`{Goal: …}` in step headings, `{, denoted …}`, `{(here $n$ is fixed …)}`, `{ The coordinates $B,F$ …`).
3. **Random-F / only-Y results — needs Ken's answers** (2026-10-03): (a) what law for F (Gaussian? other)? (b) what is the goal of the Bingham/posterior analysis (inference on the rotation, uncertainty bands, or remark only)? Then decide remark vs theorem in §7.
4. **Observable bounds from AK's paper** (June item 1, 2026-06): bounds are in `main-25` §7 (`thm:obsfloor`, `thm:rotrange`); the CLT (Proposition 1 of `multifactor_dispersion_prevalence_v7.pdf`) is not (grep for CLT/central limit finds none). James–Stein appears in `main-25` only as a related-literature remark (~lines 225–226), not as a result. Unclear whether a CLT is wanted — needs human input.
5. **Fundamental-factor (non-orthogonal loadings and factors) extension** (June item 2): STILL OPEN as research; not in `main-25` (no occurrence). Notes: `reflection_fundamental_nonorthogonal_factors.md`. Low priority.
6. **Simplified-proof housekeeping** (2026-10-07): check its Assumptions (sep)/(reg) match `asm:sep`/`asm:reg` of `main-25`; remaining typos ("Gram eigenvector transformation $(A:=\dots)$", "eg."); §5 "What the simplification removes" is working-note text; regenerate its PDF.
7. **Bibliography** (2026-10-07): `main-25` line 2476 uses `\bibliography{references_2}`; no `references_2.bib` on disk, only `paper/references.bib` and `paper/references_old.bib`. Probably lives on Overleaf *(inferred)* — confirm, or copy it into `paper/` so the paper builds locally.
8. **Git hygiene** (2026-10-07): HEAD `ebf02a3` (2026-09-03); only `docs/KT.md` is tracked-and-modified, everything under `paper/` is untracked. Commit? Ken's call.
9. **June-era items classified by assumption (HIL notice):** OBSOLETE — items 3, 4, 5, 6 (JS observable CI), 6a, 6b, 7 and the Bolzano–Weierstrass Kato swap (superseded by `main-25`, not mentioned there); superseded by item 1 above — condensed-proof adoption (6c); unapplied exploratory — fiber-bundle §4 fix in `fiber_bundle_geometry.tex` (file dated 2026-06-08, before the 06-11 review). Details in `docs/KT_history.md`.

## 7. Next Steps

1. Ken: decide open question 1 (restructure per simplified proof) — it determines whether `main-25` §6.2 Steps 1–3 are rewritten.
2. Apply the §6.2 cleanup edits (open question 2) in `main-25.tex`; Claude can supply anchored text.
3. Answer open question 3(a),(b) so Claude can write the random-F / only-Y remark.
4. Locate `references_2.bib` (or point `\bibliography` at `references.bib`) and confirm `main-25.tex` builds.
5. Decide on the CLT (open question 4) and, optionally, commit the work (open question 8).
6. On the next `\ukt`: record any new decisions; do not re-derive the history above.

## 8. Last Updated

2026-10-07T16:45Z | Trigger: manual (full rewrite, then six review fixes) | Staleness: Drifting. Rewrote the whole file: paths re-verified against disk, old KT moved verbatim to `docs/KT_history.md`, June-era items classified, gotchas added, HIL notice reduced to three items.
