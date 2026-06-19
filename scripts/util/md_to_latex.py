#!/usr/bin/env python3
"""
md_to_latex.py  —  Convert math-heavy Markdown to a compilable LaTeX document.

Tuned for the research/proof documents in this project, which embed LaTeX math
($...$ and $$...$$) in Markdown prose.  Math content is passed through verbatim;
only structural and typographic Markdown markup is translated.

Usage:
    python md_to_latex.py input.md [output.tex]

If output.tex is omitted, writes to <stem>.tex beside the input file.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# -- LaTeX preamble ────────────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{longtable}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}
\usepackage{parskip}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
\newtheorem{assumption}[theorem]{Assumption}
"""

# -- Math-span locator (adapted from reformat_math_extra.py) ──────────────────

def _find_math_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return sorted (start, end, kind) for every top-level math span.

    Kind is 'display' ($$...$$) or 'inline' ($...$).  Uses the same
    stack-based scanner as reformat_math_extra.py.
    """
    spans: List[Tuple[int, int, str]] = []
    stack: List[Tuple[str, int]] = []
    i, n = 0, len(text)

    while i < n:
        if text[i] == '\\':
            i += 2
            continue
        if text[i] != '$':
            i += 1
            continue

        is_display = (i + 1 < n and text[i + 1] == '$')
        tok   = 'display' if is_display else 'inline'
        tlen  = 2 if is_display else 1

        if not stack:
            stack.append((tok, i))
        else:
            top_tok, top_pos = stack[-1]
            if top_tok == tok:
                stack.pop()
                if not stack:
                    spans.append((top_pos, i + tlen, tok))
            elif top_tok == 'display' and tok == 'inline':
                stack.append((tok, i))           # nested inline inside display
            elif top_tok == 'inline' and tok == 'display':
                stack.pop()
                if not stack:
                    spans.append((top_pos, i + tlen, 'inline'))  # mismatch
        i += tlen

    return sorted(spans, key=lambda s: s[0])


# -- Inline text transforms ────────────────────────────────────────────────────

_BOLD_RE   = re.compile(r'\*\*(.+?)\*\*', re.DOTALL)
_ITALIC_RE = re.compile(r'(?<!\*)\*([^*\n]+?)\*(?!\*)')
_CODE_RE   = re.compile(r'`([^`]+)`')
_LINK_RE   = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

# Characters that must be escaped in LaTeX text mode.
# Backslash and braces are excluded: they may appear as valid LaTeX the
# author wrote directly, and escaping them would corrupt those commands.
_ESCAPE_MAP = str.maketrans({'%': r'\%', '&': r'\&', '_': r'\_'})

# Unicode characters not natively supported by pdflatex; mapped to LaTeX equivalents.
# Requires \usepackage{amssymb} (already in preamble via amsmath).
_UNICODE_SUBS = [
    ('✓', r'$\checkmark$'),   # ✓
    ('✗', r'$\times$'),       # ✗
    ('—', r'---'),            # — em dash
    ('–', r'--'),             # – en dash
    ('’', r"'"),              # ' right single quote
    ('‘', r'`'),              # ' left single quote
    ('“', r'``'),             # " left double quote
    ('”', r"''"),             # " right double quote
    ('…', r'\ldots{}'),       # …
    ('×', r'$\times$'),       # ×
    ('±', r'$\pm$'),          # ±
    ('≈', r'$\approx$'),      # ≈
    ('≤', r'$\leq$'),         # ≤
    ('≥', r'$\geq$'),         # ≥
    ('∞', r'$\infty$'),       # ∞
]


def _escape_text(s: str) -> str:
    """Escape LaTeX-special characters in a plain-text (non-math) string."""
    s = s.translate(_ESCAPE_MAP)
    for char, replacement in _UNICODE_SUBS:
        s = s.replace(char, replacement)
    return s


def _apply_markup(s: str) -> str:
    """Apply Markdown inline markup to *s* (assumed already escaped).

    Called after _escape_text, so content inside code/bold/italic spans
    already has underscores etc. escaped — do not double-escape.
    """
    s = _LINK_RE.sub(lambda m: rf'\href{{{m.group(2)}}}{{{m.group(1)}}}', s)
    s = _CODE_RE.sub(lambda m: rf'\texttt{{{m.group(1)}}}', s)
    s = _BOLD_RE.sub(lambda m: rf'\textbf{{{m.group(1)}}}', s)
    s = _ITALIC_RE.sub(lambda m: rf'\textit{{{m.group(1)}}}', s)
    return s


def _transform_line(line: str) -> str:
    """Apply text transforms to *line*, leaving all $...$ math spans intact.

    Strategy: replace math spans with NUL-delimited placeholders, apply
    escape + markup to the whole string (so bold/italic can span across
    an inline math token), then restore the original math.
    """
    spans = _find_math_spans(line)
    if not spans:
        return _apply_markup(_escape_text(line))

    placeholders: dict[str, str] = {}
    parts, prev = [], 0
    for i, (start, end, _kind) in enumerate(spans):
        ph = f'\x00M{i}\x00'
        placeholders[ph] = line[start:end]
        parts.append(line[prev:start])
        parts.append(ph)
        prev = end
    parts.append(line[prev:])

    assembled   = ''.join(parts)
    transformed = _apply_markup(_escape_text(assembled))
    for ph, math in placeholders.items():
        transformed = transformed.replace(ph, math)
    return transformed


# -- Display-math emitter ──────────────────────────────────────────────────────

_ALIGN_ENV_RE = re.compile(r'\\begin\{(align|gather|multline|split)')


def _emit_display_math(body: str) -> List[str]:
    """Wrap display-math *body* in an appropriate LaTeX environment.

    Heuristic:
      - Already contains \\begin{align...} etc. → emit body as-is.
      - Contains \\\\ (line breaks for alignment) → align* environment.
      - Otherwise → \\[...\\].
    """
    stripped = body.strip()
    if _ALIGN_ENV_RE.search(stripped):
        return [stripped]
    if r'\\' in stripped:
        return [r'\begin{align*}', stripped, r'\end{align*}']
    return [r'\[', stripped, r'\]']


# -- Table emitter ─────────────────────────────────────────────────────────────

_TABLE_SEP_RE = re.compile(r'^\|[\s|:_-]+\|$')


def _table_align(cell: str) -> str:
    c = cell.strip()
    if c.startswith(':') and c.endswith(':'):
        return 'c'
    if c.endswith(':'):
        return 'r'
    return 'l'


def _split_table_row(row: str) -> List[str]:
    """Split a markdown table row on | separators, respecting math mode."""
    spans = _find_math_spans(row)

    # Build a set of positions that are inside math spans
    math_positions = set()
    for start, end, _kind in spans:
        math_positions.update(range(start, end))

    # Split on | only if it's not in math mode and not escaped
    cells = []
    current = []
    i = 0
    while i < len(row):
        if row[i] == '\\' and i + 1 < len(row):
            # Handle escape sequences: don't split on \|
            current.append(row[i:i+2])
            i += 2
        elif row[i] == '|' and i not in math_positions:
            # Found an unescaped, non-math pipe
            cells.append(''.join(current).strip())
            current = []
            i += 1
        else:
            current.append(row[i])
            i += 1

    # Add final cell
    if current or cells:
        cells.append(''.join(current).strip())

    # Remove outer empty cells from leading and trailing pipes
    if cells and cells[0] == '':
        cells.pop(0)
    if cells and cells[-1] == '':
        cells.pop()

    return cells


def _emit_table(rows: List[str]) -> List[str]:
    """Convert a list of markdown table rows to a LaTeX tabular block."""
    data_rows = [r for r in rows if not _TABLE_SEP_RE.match(r.strip())]
    sep_row   = next((r for r in rows if _TABLE_SEP_RE.match(r.strip())), None)

    if not data_rows:
        return []

    header_cells = _split_table_row(data_rows[0])
    n_cols = len(header_cells)
    aligns = ([_table_align(c) for c in _split_table_row(sep_row)]
              if sep_row else ['l'] * n_cols)
    # Pad aligns in case column count mismatches
    aligns = (aligns + ['l'] * n_cols)[:n_cols]

    col_spec = '|' + '|'.join(aligns) + '|'
    out = [r'\begin{center}', f'\\begin{{tabular}}{{{col_spec}}}', r'\hline']
    for row in data_rows:
        cells = [_transform_line(c) for c in _split_table_row(row)]
        out.append(' & '.join(cells) + r' \\')
        out.append(r'\hline')
    out += [r'\end{tabular}', r'\end{center}', '']
    return out


# -- List helpers ──────────────────────────────────────────────────────────────

_UL_RE = re.compile(r'^(\s*)[-*]\s+(.*)')
_OL_RE = re.compile(r'^(\s*)\d+\.\s+(.*)')


def _list_env(ltype: str) -> str:
    return 'itemize' if ltype == 'ul' else 'enumerate'


def _open_list(out: List[str], stack: list, ltype: str, indent: int) -> None:
    out.append(f'\\begin{{{_list_env(ltype)}}}')
    stack.append((ltype, indent))


def _close_lists(out: List[str], stack: list, to_indent: int = -1) -> None:
    """Close list environments until stack top has indent <= to_indent."""
    while stack and stack[-1][1] > to_indent:
        out.append(f'\\end{{{_list_env(stack[-1][0])}}}')
        stack.pop()


# -- Heading map ───────────────────────────────────────────────────────────────

_HEADING_CMDS = {1: None, 2: r'\section', 3: r'\subsection', 4: r'\subsubsection'}
_HEADING_RE   = re.compile(r'^(#{1,4})\s+(.*)')
_HR_RE        = re.compile(r'^---+\s*$')
_BQ_RE        = re.compile(r'^>\s?(.*)')
_FENCE_RE     = re.compile(r'^```(\w*)')
_DISPLAY_RE   = re.compile(r'^\s*\$\$\s*$')


# -- Main converter ────────────────────────────────────────────────────────────

class _Converter:
    """Stateful line-by-line Markdown-to-LaTeX converter."""

    def __init__(self) -> None:
        self._out:          List[str] = []
        self._list_stack:   list      = []      # [(type, indent), ...]
        self._table_rows:   List[str] = []
        self._display_body: List[str] = []
        self._state:        str       = 'normal'  # normal | display | code
        self._title:        Optional[str] = None
        self._in_blockquote: bool     = False
        self._table_in_blockquote: bool = False  # table rows accumulated inside a blockquote
        self._skip_toc:     bool      = False   # suppress the manual TOC list

    # -- state flush helpers ───────────────────────────────────────────────────

    def _flush_table(self) -> None:
        if self._table_rows:
            if self._table_in_blockquote and self._in_blockquote:
                # Table is embedded inside a blockquote: close the quote, emit the
                # table (which must live outside \begin{quote}), then reopen the quote.
                self._out.append(r'\end{quote}')
                self._out.extend(_emit_table(self._table_rows))
                self._out.append(r'\begin{quote}')
            else:
                self._out.extend(_emit_table(self._table_rows))
            self._table_rows = []
            self._table_in_blockquote = False

    def _flush_lists(self) -> None:
        _close_lists(self._out, self._list_stack)

    def _flush_blockquote(self) -> None:
        if self._in_blockquote:
            self._out.append(r'\end{quote}')
            self._in_blockquote = False

    def _flush_all(self) -> None:
        self._flush_table()
        self._flush_lists()
        self._flush_blockquote()

    # -- line dispatch ─────────────────────────────────────────────────────────

    def feed(self, line: str) -> None:
        # -- inside a display-math block ──────────────────────────────────────
        if self._state == 'display':
            if _DISPLAY_RE.match(line):          # closing $$
                self._out.extend(_emit_display_math('\n'.join(self._display_body)))
                self._display_body = []
                self._state = 'normal'
            else:
                self._display_body.append(line)
            return

        # -- inside a code fence ───────────────────────────────────────────────
        if self._state == 'code':
            if re.match(r'^```\s*$', line):
                self._out.append(r'\end{verbatim}')
                self._state = 'normal'
            else:
                self._out.append(line)           # verbatim: no transforms
            return

        # -- normal state ──────────────────────────────────────────────────────

        # Opening display-math block (lone $$ on its own line)
        if _DISPLAY_RE.match(line):
            self._flush_all()
            self._state = 'display'
            return

        # Opening code fence
        m = _FENCE_RE.match(line)
        if m:
            self._flush_all()
            self._out.append(r'\begin{verbatim}')
            self._state = 'code'
            return

        # Horizontal rule
        if _HR_RE.match(line):
            self._flush_all()
            self._out.append(r'\medskip\noindent\rule{\linewidth}{0.4pt}\medskip')
            return

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            self._flush_all()
            level     = len(m.group(1))
            raw_text  = m.group(2)
            text      = _transform_line(raw_text)
            if level == 1:
                self._title = raw_text           # raw title for \title{}
            elif re.sub(r'^\d+\.\s*', '', raw_text).strip().lower() == 'table of contents':
                self._out.append(r'\tableofcontents')
                self._skip_toc = True            # swallow the following list
            else:
                self._skip_toc = False
                cmd = _HEADING_CMDS.get(level, r'\subsubsection')
                self._out.append(f'{cmd}{{{text}}}')
            return

        # Blockquote (or blockquote-wrapped table row)
        m = _BQ_RE.match(line)
        if m:
            inner = m.group(1)
            if inner.startswith('|'):
                # Table row inside a blockquote — accumulate the stripped row.
                # _flush_table() will wrap it in \end{quote}...\begin{quote} to
                # keep the surrounding blockquote context intact.
                if not self._in_blockquote:
                    # Shouldn't normally happen, but handle gracefully.
                    self._flush_lists()
                self._table_in_blockquote = True
                self._table_rows.append(inner)
                return
            # Plain blockquote line — flush any pending table first (which may
            # reopen the quote environment), then continue accumulating quote text.
            self._flush_table()
            self._flush_lists()
            if not self._in_blockquote:
                self._out.append(r'\begin{quote}')
                self._in_blockquote = True
            if inner.strip():  # skip blank "> " lines (they don't need output)
                self._out.append(_transform_line(inner))
            return

        # Close blockquote when we exit > lines
        if self._in_blockquote and not _BQ_RE.match(line):
            self._flush_blockquote()

        # Table row
        if line.startswith('|'):
            self._flush_lists()
            self._table_rows.append(line)
            return

        # Flush pending table when a non-table line arrives
        if self._table_rows:
            self._flush_table()

        # Unordered list item
        m = _UL_RE.match(line)
        if m:
            if not self._skip_toc:
                indent, content = len(m.group(1)), _transform_line(m.group(2))
                self._manage_list('ul', indent, content)
            return

        # Ordered list item
        m = _OL_RE.match(line)
        if m:
            if not self._skip_toc:
                indent, content = len(m.group(1)), _transform_line(m.group(2))
                self._manage_list('ol', indent, content)
            return

        # Blank line
        if not line.strip():
            # Lists survive blank lines between items; flush other contexts.
            self._flush_table()
            self._flush_blockquote()
            self._out.append('')
            return

        # Plain text — close any open lists first
        self._flush_lists()
        self._out.append(_transform_line(line))

    def _manage_list(self, ltype: str, indent: int, content: str) -> None:
        """Open or adjust list environments, then emit \\item."""
        if not self._list_stack:
            _open_list(self._out, self._list_stack, ltype, indent)
        elif indent > self._list_stack[-1][1]:
            _open_list(self._out, self._list_stack, ltype, indent)
        elif indent < self._list_stack[-1][1]:
            _close_lists(self._out, self._list_stack, to_indent=indent)
        # If same indent but different type, close and reopen
        if self._list_stack and self._list_stack[-1][0] != ltype:
            self._out.append(f'\\end{{{_list_env(self._list_stack[-1][0])}}}')
            self._list_stack.pop()
            _open_list(self._out, self._list_stack, ltype, indent)
        self._out.append(f'\\item {content}')

    def result(self) -> Tuple[Optional[str], List[str]]:
        """Flush all pending state and return (title, body_lines)."""
        self._flush_all()
        # Close any still-open display or code block (malformed input)
        if self._state == 'display' and self._display_body:
            self._out.extend(_emit_display_math('\n'.join(self._display_body)))
        if self._state == 'code':
            self._out.append(r'\end{verbatim}')
        return self._title, self._out


# -- Document assembly ─────────────────────────────────────────────────────────

def _assemble(title: Optional[str], body_lines: List[str]) -> str:
    parts = [_PREAMBLE]
    if title:
        parts.append(f'\\title{{{_transform_line(title)}}}')
        parts.append(r'\author{}')
        parts.append(r'\date{}')
        parts.append('')
    parts.append(r'\begin{document}')
    if title:
        parts.append(r'\maketitle')
    parts.append('')
    # Collapse runs of more than two blank lines
    prev_blank = False
    for line in body_lines:
        if line == '':
            if not prev_blank:
                parts.append('')
            prev_blank = True
        else:
            parts.append(line)
            prev_blank = False
    parts += ['', r'\end{document}', '']
    return '\n'.join(parts)


# -- Public entry point ────────────────────────────────────────────────────────

def convert(text: str) -> str:
    """Convert a Markdown string to a complete LaTeX document string."""
    conv = _Converter()
    for line in text.replace('\r\n', '\n').split('\n'):
        conv.feed(line)
    title, body = conv.result()
    return _assemble(title, body)


# -- CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert a math-heavy Markdown file to LaTeX.'
    )
    parser.add_argument('input',  type=Path, help='Input .md file')
    parser.add_argument('output', type=Path, nargs='?',
                        help='Output .tex file (default: <stem>.tex)')
    args = parser.parse_args()

    if not args.input.is_file():
        sys.exit(f'Error: not found: {args.input}')

    out_path = args.output or args.input.with_suffix('.tex')
    text = args.input.read_text(encoding='utf-8')
    latex = convert(text)
    out_path.write_text(latex, encoding='utf-8')
    print(f'Written: {out_path}')


if __name__ == '__main__':
    main()
