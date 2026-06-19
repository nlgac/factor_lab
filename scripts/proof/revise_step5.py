#!/usr/bin/env python3
"""
revise_step5.py
===============
Apply the agreed Step 5 revisions to main.tex:
  1. Add roadmap sentence and expand metric description.
  2. Fix eigenvalue notation (lambda_j^(p) = eigenvalue of Sigma_0^(p)/p).
  3. Show left-multiplication step explicitly.
  4. Expand similarity argument.
  5. Add k=1 remark before Step 6.
  6. Remove duplicate red block.

Usage:
    python3 revise_step5.py main.tex              # edit in place
    python3 revise_step5.py main.tex --mark       # wrap changes in \\chg{}
    python3 revise_step5.py main.tex --dry-run    # show line-count diff only
"""

import sys
import re
import shutil
from pathlib import Path

def wrap(text, mark):
    """Wrap text in \\chg{} if mark is True, else return as-is."""
    if not mark:
        return text
    return r'\chg{' + text + r'}'

def revise(content, mark=False):

    # ------------------------------------------------------------------
    # 1. Step 5 opening: roadmap + concrete metric description
    # ------------------------------------------------------------------
    old_opening = (
        r'\medskip\noindent\textbf{Step 5 (Loading-frame coordinates).}' + '\n'
        r'Define $\Phi_p:\mathbb{R}^k\to\mathcal{B}$ by $\Phi_p(x):=Bx/\sqrt{p}$.' + '\n'
        r'Then $\Phi_p^{\!\top}\Phi_p = B^{\!\top}B/p\to G_B$, so $G_B$ is the' + '\n'
        r'limiting metric tensor for the image of $\Phi_p$.'
    )

    new_opening_core = (
        'Steps~2--4 established convergence results in $\\mathbb{R}^n$. '
        'This step establishes convergence in the factor loading frame --- '
        'the $k$-dimensional coordinate system spanned by the columns of $B$ --- '
        'which is what we need to compute the inner products '
        '$\\langle\\Pi_B h_j,b_j\\rangle$ and $\\|\\Pi_B h_j\\|^2$ in Step~6.\n\n'
        'Define the scaled coordinate map $\\Phi_p:\\mathbb{R}^k\\to\\mathcal{B}$ '
        'by $\\Phi_p(x):=Bx/\\sqrt{p}$. '
        'Then $\\Phi_p^{\\!\\top}\\Phi_p = B^{\\!\\top}B/p\\to G_B$, so '
        'for any two vectors $u=\\Phi_p(x)$ and $v=\\Phi_p(y)$ in $\\mathcal{B}$, '
        'their Euclidean inner product satisfies '
        '$\\langle u,v\\rangle = x^\\top(B^{\\!\\top}B/p)y\\to x^\\top G_B y$. '
        'Inner products between vectors in $\\mathcal{B}$ therefore correspond, '
        'in the limit, to $G_B$-weighted inner products of their $\\mathbb{R}^k$ '
        'coordinates. Since $\\Phi_p$ is injective for large $p$ '
        '(as $B$ has full column rank), '
        'every vector in $\\mathcal{B}$ has a unique coordinate representation.'
    )

    new_opening = (
        r'\medskip\noindent\textbf{Step 5 (Loading-frame coordinates).}' + '\n'
        + wrap(new_opening_core, mark)
    )

    content = content.replace(old_opening, new_opening)

    # ------------------------------------------------------------------
    # 2. Population direction: fix eigenvalue definition + expand derivation
    # ------------------------------------------------------------------
    old_pop = (
        r'\textit{Population direction.}' + '\n'
        r'Since $\mathrm{col}(\Sigma_0^{(p)})\subseteq\mathcal{B}$, every eigenvector' + '\n'
        r' of $\Sigma_0^{(p)}$ for a nonzero eigenvalue lies in $\mathcal{B}$, so' + '\n'
        r'$b_j\in\mathcal{B}$; write $b_j=\Phi_p(a_j)$ uniquely (as $B$ has full' + '\n'
        r'column rank for large $p$).  Let $\lambda_j^{(p)}$ denote the $j$-th' + '\n'
        r'eigenvalue of $\Sigma_0^{(p)}$.  The eigenequation' + '\n'
        r'$\Sigma_0^{(p)}b_j=\lambda_j^{(p)}b_j$, left-multiplied by' + '\n'
        r'$\sqrt{p}\,B^{\!\top}$, reduces to' + '\n'
        r'\[' + '\n'
        r'  \Sigma_f\!\left(\frac{B^{\!\top}B}{p}\right)\!a_j = \lambda_j^{(p)}\,a_j,' + '\n'
        r'  \qquad' + '\n'
        r'  a_j^{\!\top}\!\left(\frac{B^{\!\top}B}{p}\right)\!a_j = 1.' + '\n'
        r'\]' + '\n'
        r'As $p\to\infty$ the matrix $\Sigma_f(B^{\!\top}B/p)\to\Sigma_f G_B$,' + '\n'
        r'whose eigenvalues equal those of $M$ (via' + '\n'
        r'$G_B^{1/2}\cdot\Sigma_f G_B\cdot G_B^{-1/2}=M$), hence are simple by' + '\n'
        r'Assumption~\ref{asm:sep}.  Eigenprojection continuity gives' + '\n'
        r'$a_j\to a_j^{\infty}$ a.s., where'
    )

    new_pop_core = (
        r'\textit{Population direction.}' + '\n'
        'Since $\\mathrm{col}(\\Sigma_0^{(p)})\\subseteq\\mathcal{B}$, '
        'every eigenvector of $\\Sigma_0^{(p)}$ for a nonzero eigenvalue '
        'lies in $\\mathcal{B}$, so $b_j\\in\\mathcal{B}$; '
        'write $b_j=\\Phi_p(a_j)=Ba_j/\\sqrt{p}$ uniquely. '
        'Denote by $\\lambda_j^{(p)}$ the $j$-th eigenvalue of '
        '$\\Sigma_0^{(p)}/p$, so that the eigenequation reads '
        '$\\Sigma_0^{(p)}b_j = p\\lambda_j^{(p)}b_j$. '
        'Substituting $b_j = Ba_j/\\sqrt{p}$ into this equation gives '
        '$B\\Sigma_f B^\\top(Ba_j/\\sqrt{p}) = p\\lambda_j^{(p)}(Ba_j/\\sqrt{p})$; '
        'simplifying and left-multiplying by $B^\\top$ then cancelling using '
        'full column rank of $B$ reduces the $p$-dimensional eigenequation '
        'to the $k$-dimensional system\n'
        '\\[\n'
        '  \\Sigma_f\\!\\left(\\frac{B^{\\!\\top}B}{p}\\right)\\!a_j = \\lambda_j^{(p)}\\,a_j,\n'
        '  \\qquad\n'
        '  a_j^{\\!\\top}\\!\\left(\\frac{B^{\\!\\top}B}{p}\\right)\\!a_j = 1.\n'
        '\\]\n'
        'As $p\\to\\infty$, $\\Sigma_f(B^{\\!\\top}B/p)\\to\\Sigma_f G_B$. '
        'The matrix $\\Sigma_f G_B$ is similar to $M = G_B^{1/2}\\Sigma_f G_B^{1/2}$ '
        'via $\\Sigma_f G_B = G_B^{-1/2}\\,M\\,G_B^{1/2}$, '
        'so it has the same eigenvalues as $M$, which are simple by '
        'Assumption~\\ref{asm:sep}. '
        'Eigenprojection continuity gives $\\lambda_j^{(p)}\\to\\lambda_j$ '
        'and $a_j\\to a_j^{\\infty}$ a.s., where'
    )

    content = content.replace(old_pop, new_pop_core if not mark else new_pop_core)
    # For the population direction, we replace directly (it already has \textit which
    # we keep outside the \chg wrapper to avoid nesting issues with environments)
    # Re-do: mark just the changed prose, not the math environments
    if mark:
        # We already wrote new_pop_core without wrapping; now we need to
        # handle it differently. Let's do a direct replacement with marking
        # on the prose portions only — simplest: wrap the whole block
        # but outside the displayed math. We'll do it as one \chg around the text portions.
        # Since this is complex, we wrap just the changed explanatory sentences.
        content = content.replace(
            'Denote by $\\lambda_j^{(p)}$ the $j$-th eigenvalue of '
            '$\\Sigma_0^{(p)}/p$, so that the eigenequation reads '
            '$\\Sigma_0^{(p)}b_j = p\\lambda_j^{(p)}b_j$. '
            'Substituting $b_j = Ba_j/\\sqrt{p}$ into this equation gives '
            '$B\\Sigma_f B^\\top(Ba_j/\\sqrt{p}) = p\\lambda_j^{(p)}(Ba_j/\\sqrt{p})$; '
            'simplifying and left-multiplying by $B^\\top$ then cancelling using '
            'full column rank of $B$ reduces the $p$-dimensional eigenequation '
            'to the $k$-dimensional system',
            '\\chg{Denote by $\\lambda_j^{(p)}$ the $j$-th eigenvalue of '
            '$\\Sigma_0^{(p)}/p$, so that the eigenequation reads '
            '$\\Sigma_0^{(p)}b_j = p\\lambda_j^{(p)}b_j$. '
            'Substituting $b_j = Ba_j/\\sqrt{p}$ into this equation gives '
            '$B\\Sigma_f B^\\top(Ba_j/\\sqrt{p}) = p\\lambda_j^{(p)}(Ba_j/\\sqrt{p})$; '
            'simplifying and left-multiplying by $B^\\top$ then cancelling using '
            'full column rank of $B$ reduces the $p$-dimensional eigenequation '
            'to the $k$-dimensional system}'
        )
        content = content.replace(
            'The matrix $\\Sigma_f G_B$ is similar to $M = G_B^{1/2}\\Sigma_f G_B^{1/2}$ '
            'via $\\Sigma_f G_B = G_B^{-1/2}\\,M\\,G_B^{1/2}$, '
            'so it has the same eigenvalues as $M$, which are simple by '
            'Assumption~\\ref{asm:sep}. '
            'Eigenprojection continuity gives $\\lambda_j^{(p)}\\to\\lambda_j$ '
            'and $a_j\\to a_j^{\\infty}$ a.s., where',
            '\\chg{The matrix $\\Sigma_f G_B$ is similar to $M = G_B^{1/2}\\Sigma_f G_B^{1/2}$ '
            'via $\\Sigma_f G_B = G_B^{-1/2}\\,M\\,G_B^{1/2}$, '
            'so it has the same eigenvalues as $M$, which are simple by '
            'Assumption~\\ref{asm:sep}. '
            'Eigenprojection continuity gives $\\lambda_j^{(p)}\\to\\lambda_j$ '
            'and $a_j\\to a_j^{\\infty}$ a.s., where}'
        )

    # ------------------------------------------------------------------
    # 3. Remove the red duplicate block (everything from the second
    #    {\color{red} after eq:ginfty through the closing brace before Step 6)
    # ------------------------------------------------------------------
    dup_start = (
        '\n{\color{red}\n'
        r'\begin{lemma*} For $j = 1, 2, \ldots, k$, there are unique vectors $a_j^{\infty}  =G_B^{-1/2}w_j \in R^k$ for which:'
    )
    # Find the duplicate block and remove it
    idx_start = content.find(dup_start)
    if idx_start != -1:
        # The block ends with ...G_B^{-1/2}\mchg{w_{n,j}}.}
        # followed by \n\n\medskip\noindent\textbf{Step 6
        dup_end_marker = r'G_B^{-1/2}\mchg{w_{n,j}}.}' + '\n\n'
        idx_end = content.find(dup_end_marker, idx_start)
        if idx_end != -1:
            idx_end += len(dup_end_marker)
            content = content[:idx_start] + '\n\n' + content[idx_end:]

    # ------------------------------------------------------------------
    # 4. Add k=1 remark just before Step 6
    # ------------------------------------------------------------------
    step6_marker = r'\medskip\noindent\textbf{Step 6 (Inner products).}'

    remark_text = (
        r'\begin{remark}[The single-factor case]' + '\n'
        r'When $k=1$ the coordinate limits $a_j^{\infty}$ and $g_j^{\infty}$ '
        r'simplify to scalars. The loading matrix $B$ is a $p\times 1$ vector '
        r'and $G_B = \lim\|B\|^2/p$ is the mean-square loading (prevalence). '
        r'The coordinate map $\Phi_p$ simply scales the beta vector, and '
        r'$w_1 = w_{n,1} = 1$ (scalars). '
        r'The population limit is $a_1^{\infty} = G_B^{-1/2}$: the true factor '
        r'direction in loading-frame coordinates is the inverse square root of '
        r'the prevalence. '
        r'The sample limit is '
        r'$g_1^{\infty} = \sqrt{\mchg{\mathrm{SNR}_{n,1}}/(1+\mchg{\mathrm{SNR}_{n,1}})}\;G_B^{-1/2}$, '
        r'which is the population coordinate $G_B^{-1/2}$ multiplied by the '
        r'weight $\sqrt{\widehat{\mathrm{SNR}}/(1+\widehat{\mathrm{SNR}})}<1$. '
        r'This factor, less than one, reflects the geometric shrinkage of the '
        r'estimated eigenvector within $\mathcal{B}$: the sample PC falls short '
        r'of the true loading direction by exactly the in-subspace weight of '
        r'Theorem~1.' + '\n'
        r'\end{remark}' + '\n\n'
    )

    remark_wrapped = wrap(remark_text, mark)
    content = content.replace(step6_marker,
                               remark_wrapped + step6_marker)

    return content


def main():
    dry_run = '--dry-run' in sys.argv
    mark    = '--mark'    in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not args:
        print('Usage: python3 revise_step5.py <file.tex> [--mark] [--dry-run]')
        sys.exit(1)

    path = Path(args[0])
    if not path.exists():
        print(f'File not found: {path}'); sys.exit(1)

    original = path.read_text(encoding='utf-8')
    result   = revise(original, mark=mark)

    if dry_run:
        print(f'Lines: {original.count(chr(10))} -> {result.count(chr(10))}')
        changed = original != result
        print(f'Content changed: {changed}')
        return

    if not dry_run:
        shutil.copy(path, path.with_suffix('.tex.step5bak'))
        path.write_text(result, encoding='utf-8')
        print(f'Done. Lines: {original.count(chr(10))} -> {result.count(chr(10))}. '
              f'Backup: {path.with_suffix(".tex.step5bak")}')

if __name__ == '__main__':
    main()
