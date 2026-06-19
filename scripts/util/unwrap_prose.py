#!/usr/bin/env python3
"""
unwrap_prose.py
===============
Remove mid-paragraph line breaks from a LaTeX file, leaving blank lines,
math environments, and lines starting with LaTeX commands untouched.

Usage:
    python3 unwrap_prose.py floor_rotation_nlged.tex
    python3 unwrap_prose.py floor_rotation_nlged.tex --dry-run
"""

import re
import sys
import shutil
from pathlib import Path

# Environments whose content we never touch
MATH_ENVS = {
    'align', 'align*', 'equation', 'equation*',
    'gather', 'gather*', 'multline', 'multline*',
    'displaymath', 'array', 'cases',
}
VERBATIM_ENVS = {'verbatim', 'lstlisting', 'minted'}


def is_protected_line(line: str) -> bool:
    """True if this line must never be joined to its predecessor."""
    s = line.strip()
    if s == '':
        return True
    if s.startswith('%'):
        return True
    # Any line that begins with a backslash command
    if s.startswith('\\'):
        return True
    # Lines that are continuation of display math or alignment
    if s.startswith('&') or s.startswith('+') or s.startswith('='):
        return True
    return False


def unwrap(text: str) -> str:
    lines = text.split('\n')
    out = []
    in_math = False
    in_verbatim = False
    env_stack: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # Track \[ ... \] inline math
        if s == '\\[':
            in_math = True
        if s == '\\]':
            in_math = False
            out.append(line)
            i += 1
            continue

        # Track \begin / \end
        bm = re.match(r'\\begin\{([^}]+)\}', s)
        em = re.match(r'\\end\{([^}]+)\}', s)
        if bm:
            env = bm.group(1)
            env_stack.append(env)
            if env in MATH_ENVS:
                in_math = True
            if env in VERBATIM_ENVS:
                in_verbatim = True
        if em:
            env = em.group(1)
            if env_stack and env_stack[-1] == env:
                env_stack.pop()
            if env in MATH_ENVS:
                in_math = False
            if env in VERBATIM_ENVS:
                in_verbatim = False

        # Inside math or verbatim: emit as-is
        if in_math or in_verbatim:
            out.append(line)
            i += 1
            continue

        # Protected line: emit as-is
        if is_protected_line(line):
            out.append(line)
            i += 1
            continue

        # Plain prose line: try to join with next line
        while i + 1 < len(lines):
            next_line = lines[i + 1]
            ns = next_line.strip()
            if ns == '' or is_protected_line(next_line):
                break
            # Don't join if we're at a sentence end that naturally breaks
            # (this keeps things readable; LaTeX doesn't care either way)
            line = line.rstrip() + ' ' + ns
            i += 1  # consumed next_line

        out.append(line)
        i += 1

    return '\n'.join(out)


def main():
    dry_run = '--dry-run' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print("Usage: python3 unwrap_prose.py <file.tex> [--dry-run]")
        sys.exit(1)

    path = Path(args[0])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    original = path.read_text(encoding='utf-8')
    result = unwrap(original)

    if dry_run:
        orig_lines = original.count('\n')
        new_lines = result.count('\n')
        print(f"Lines: {orig_lines} -> {new_lines} ({orig_lines - new_lines} removed)")
        print("(dry run — file not modified)")
    else:
        backup = path.with_suffix('.tex.bak')
        shutil.copy(path, backup)
        path.write_text(result, encoding='utf-8')
        orig_lines = original.count('\n')
        new_lines = result.count('\n')
        print(f"Done. Lines: {orig_lines} -> {new_lines}. Backup: {backup}")


if __name__ == '__main__':
    main()
