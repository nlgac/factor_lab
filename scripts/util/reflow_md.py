"""
reflow_md.py
============
Reflow hard-wrapped Markdown prose back to long lines.

Diagnosis
---------
The target file has prose paragraphs hard-wrapped at ~72 characters per line.
This pattern is produced by editors that honour a fixed ``textwidth`` setting
(vim, Emacs, etc.) or by tools that emit fixed-column output.  In Markdown a
bare newline inside a paragraph renders as a space, so the document displays
correctly — but in a raw text view every paragraph is a ladder of short lines.

Fix
---
Join the wrapped lines within each prose paragraph into a single long line.
All structural Markdown elements are left exactly as they are:

  * blank lines                        – paragraph / block separators
  * ATX headings  (``# …``)
  * horizontal rules  (``---``, ``***``, ``___``)
  * display-math blocks  (``$$`` … ``$$``)
  * fenced code blocks  (`` ``` `` … `` ``` ``)
  * table rows  (``| … |``)
  * bullet-list items  (``- ``, ``* ``, ``+ ``)  and their indented continuations
  * numbered-list items  (``1. ``, ``2. ``, …)   and their indented continuations

Usage
-----
Command-line::

    python reflow_md.py input.md                   # reflows in place
    python reflow_md.py input.md -o output.md      # writes to output.md
    python reflow_md.py - -o output.md             # reads from stdin
    python reformat_math_extra.py p.md -o console | python reflow_md.py - -o final.md

If ``-o`` is omitted the input file is modified in place.  Pass ``-`` as the
source to read from stdin; diagnostics then go to stderr so they don't
corrupt the pipe.

Programmatic::

    from reflow_md import reflow, diagnose
    print(diagnose(text))
    fixed = reflow(text)
"""

import argparse
import re
import sys
from pathlib import Path


# ── Line classifiers ─────────────────────────────────────────────────────────

_BLANK      = re.compile(r'^\s*$')
_HEADING    = re.compile(r'^#{1,6} ')
_HRULE      = re.compile(r'^[-*_]{3,}\s*$')   # ---, ***, ___ (no content after)
_BULLET     = re.compile(r'^[-*+] ')           # bullet items  (- text, * text, + text)
_NUMBERED   = re.compile(r'^\d+\. ')           # numbered items (1. text, 2. text, …)
_TABLE_ROW  = re.compile(r'^\|')
_FENCE      = re.compile(r'^```')
_MATH_DELIM = re.compile(r'^\$\$\s*$')        # $$ alone on a line
_INDENT     = re.compile(r'^  ')              # 2+ leading spaces → list continuation


def _classify(line: str) -> str:
    """Return a category tag for one raw line (outside fenced/math blocks)."""
    if _BLANK.match(line):      return 'blank'
    if _FENCE.match(line):      return 'fence'
    if _MATH_DELIM.match(line): return 'math_delim'
    if _HEADING.match(line):    return 'heading'
    if _HRULE.match(line):      return 'hrule'
    if _TABLE_ROW.match(line):  return 'table'
    if _BULLET.match(line):     return 'bullet'
    if _NUMBERED.match(line):   return 'numbered'
    if _INDENT.match(line):     return 'indent'
    return 'prose'


# ── Core reflower ────────────────────────────────────────────────────────────

def reflow(text: str) -> str:
    """Join hard-wrapped prose lines within Markdown paragraphs.

    Parameters
    ----------
    text:
        Raw Markdown text with hard-wrapped prose paragraphs.

    Returns
    -------
    str
        Markdown text with prose paragraphs reflowed to single long lines.
        All structural Markdown elements are preserved exactly.
    """
    lines    = text.split('\n')
    out: list[str]  = []
    para: list[str] = []   # buffered prose fragments waiting to be joined

    in_fence = False   # inside ``` … ``` block
    in_math  = False   # inside $$ … $$ block
    in_list  = False   # immediately after a bullet or numbered-list item

    def flush() -> None:
        """Emit the buffered prose paragraph as one joined line."""
        if para:
            out.append(' '.join(para))
            para.clear()

    for line in lines:

        # ── fenced code block ────────────────────────────────────────────────
        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
            in_list  = False
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        # ── display math block ───────────────────────────────────────────────
        if _MATH_DELIM.match(line):
            flush()
            in_math = not in_math
            in_list = False
            out.append(line)
            continue

        if in_math:
            out.append(line)
            continue

        # ── remaining line types ─────────────────────────────────────────────
        kind = _classify(line)

        if kind == 'blank':
            flush()
            in_list = False
            out.append(line)
            continue

        if kind in ('heading', 'hrule', 'table'):
            flush()
            in_list = False
            out.append(line)
            continue

        if kind in ('bullet', 'numbered'):
            flush()
            in_list = True
            out.append(line)
            continue

        if kind == 'indent':
            if in_list and out:
                # Continuation of the current list item: append to its line.
                out[-1] = out[-1] + ' ' + line.strip()
            else:
                # Indented line outside a list context: treat as prose.
                para.append(line.strip())
            continue

        # kind == 'prose'
        if in_list:
            # An un-indented prose line after a list item ends list context.
            in_list = False
        para.append(line.strip())

    flush()
    return '\n'.join(out)


# ── Diagnosis ────────────────────────────────────────────────────────────────

def diagnose(text: str) -> str:
    """Return a short diagnostic report for a hard-wrapped Markdown file."""
    all_lines    = text.splitlines()
    nonempty     = [l for l in all_lines if l.strip()]
    lengths      = [len(l) for l in nonempty]

    if not lengths:
        return "File appears to be empty."

    lengths.sort()
    median = lengths[len(lengths) // 2]
    short  = [n for n in lengths if 0 < n < 85]
    long_  = [n for n in lengths if n >= 85]
    pct_s  = 100 * len(short) // len(lengths)
    pct_l  = 100 * len(long_)  // len(lengths)

    return (
        f"Non-blank lines   : {len(lengths)}\n"
        f"Median length     : {median} chars\n"
        f"Lines  < 85 chars : {len(short):4d}  ({pct_s}%)\n"
        f"Lines >= 85 chars : {len(long_):4d}  ({pct_l}%)\n"
        "\n"
        f"Cause: prose paragraphs are hard-wrapped at ~{median} chars per line,\n"
        "consistent with an editor 'textwidth' setting (e.g. vim/Emacs at 72).\n"
        "Bare newlines inside a Markdown paragraph render as spaces, so the\n"
        "document displays correctly but has many unnecessarily short raw lines."
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflow hard-wrapped Markdown prose back to long lines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python reflow_md.py input.md                   "
            "# reflows in place\n"
            "  python reflow_md.py input.md -o output.md      "
            "# writes to output.md\n"
            "  python reflow_md.py - -o output.md             "
            "# reads from stdin\n"
            "  python reformat_math_extra.py p.md -o console | python reflow_md.py - -o final.md"
        ),
    )
    parser.add_argument(
        "input",
        help='Input Markdown file, or "-" to read from stdin.',
    )
    parser.add_argument(
        "-o", "--out",
        default=None,
        metavar="FILE",
        help="Output file. Defaults to modifying the input file in place. "
             "Ignored (stdout used) when input is \"-\" and this flag is omitted.",
    )
    args = parser.parse_args()

    from_stdin = args.input == "-"

    if from_stdin:
        # reformat_math_extra writes UTF-8 bytes to stdout.buffer; read the same way.
        original = sys.stdin.buffer.read().decode('utf-8')
        src_label = "<stdin>"
    else:
        src = Path(args.input)
        if not src.is_file():
            sys.exit(f"Error: Input file not found: {src}")
        original = src.read_text(encoding='utf-8')
        src_label = src.name

    to_stdout = from_stdin and args.out is None
    diag = sys.stderr if to_stdout else sys.stdout

    print("── Diagnosis ──────────────────────────────────────────────────────", file=diag)
    print(diagnose(original), file=diag)
    print(file=diag)

    reflowed = reflow(original)

    orig_lines     = original.splitlines()
    reflowed_lines = reflowed.splitlines()
    n_changed = sum(1 for a, b in zip(orig_lines, reflowed_lines) if a != b)
    n_removed = len(orig_lines) - len(reflowed_lines)

    if to_stdout:
        sys.stdout.buffer.write(reflowed.encode('utf-8'))
        dst_label = "stdout"
    else:
        dst = Path(args.out) if args.out else Path(args.input)
        dst.write_text(reflowed, encoding='utf-8')
        dst_label = str(dst)

    print("── Result ─────────────────────────────────────────────────────────", file=diag)
    print(f"Lines changed : {n_changed}", file=diag)
    print(f"Lines removed : {n_removed}  ({len(orig_lines)} → {len(reflowed_lines)})", file=diag)
    print(f"Written to    : {dst_label}", file=diag)


if __name__ == '__main__':
    main()
