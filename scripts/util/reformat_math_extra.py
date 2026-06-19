#!/usr/bin/env python3
"""
reformat_math.py -- Canonically format LaTeX math blocks in Markdown files.

This script parses Markdown, isolates math blocks (skipping code fences), and applies
structural and stability formatting to ensure robust rendering across markdown parsers.

Usage:
    python reformat_math.py input.md [output.md]
    
If output.md is omitted, the script safely defaults to: input_cleaned.md

Stack Parser Model:
-------------------
The parser scans left-to-right with a one-slot-deep stack per nesting level.
Tokens are $$ (display) and $ (inline); \\ escapes skip the next character.

  Stack empty    + $$  ->  push display
  Stack empty    + $   ->  push inline
  IN_DISPLAY     + $$  ->  pop, record display span           (well-formed)
  IN_DISPLAY     + $   ->  push NESTED inline                 (e.g. \\tag{$8'$})
  IN_INLINE      + $   ->  pop, record inline span            (well-formed)
  IN_INLINE      + $$  ->  pop, record mismatch_d_dd span     ($...$$ typo)

Nested inline closes when its matching $ is found; outer display continues.
Only TOP-LEVEL spans (stack depth returning to 0) are recorded.
Unclosed openers at end-of-text become 'orphan' entries and trigger warnings.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# --- Constants & Compiled Regex ---
FENCE_RE = re.compile(r'(```.*?```)', re.DOTALL)
EXCESS_BLANK_RE = re.compile(r'\n{3,}')
SEMICOLON_RE = re.compile(r'(?<!\\);')


# --- Domain Models ---
@dataclass
class MathSpan:
    """Represents a located span of math text within the document."""
    start: int
    end: Optional[int]  # None indicates an unclosed (orphan) math delimiter
    kind: str           # 'display', 'inline', 'mismatch_d_dd', 'orphan'


# --- Parsing (The Functional Core) ---
def find_math_spans(text: str) -> List[MathSpan]:
    """
    Scans text left-to-right using a stack to accurately pair nested math delimiters.
    Returns a sorted list of top-level MathSpans.
    """
    spans: List[MathSpan] = []
    stack: List[Tuple[str, int]] = []  # Stores tuple of (kind, start_index)
    i, n = 0, len(text)

    while i < n:
        if text[i] == '\\':
            i += 2  # Skip escaped characters entirely
            continue
            
        if text[i] != '$':
            i += 1
            continue

        is_display = (i + 1 < n and text[i + 1] == '$')
        tok = 'display' if is_display else 'inline'
        tlen = 2 if is_display else 1

        if not stack:
            stack.append((tok, i))
        else:
            top_tok, top_pos = stack[-1]
            
            if top_tok == tok:
                stack.pop()
                if not stack:  # Top-level span closed
                    spans.append(MathSpan(top_pos, i + tlen, tok))
            elif top_tok == 'display' and tok == 'inline':
                stack.append((tok, i)) # Nested inline (e.g., inside \text{})
            elif top_tok == 'inline' and tok == 'display':
                stack.pop()
                if not stack:  # Mismatched closure ($...$$)
                    spans.append(MathSpan(top_pos, i + tlen, 'mismatch_d_dd'))
                    
        i += tlen

    # Flush unclosed openers as orphans
    for _, op in stack:
        spans.append(MathSpan(op, None, 'orphan'))

    return sorted(spans, key=lambda s: s.start)


# --- Formatting Rules ---
def _apply_common_math_fixes(content: str) -> str:
    """Applies universal stability fixes to raw math content (DRY)."""
    # 1. Safely scope the parallel superscript to avoid double-superscript errors
    content = content.replace(r'^\|', r'^{\parallel}')
    
    # 1.5. AUTO-REPAIR HEURISTIC: Catch missing closing norms
    # Fixed regex: Escaped caret \^ to match the literal character, not start-of-string.
    content = re.sub(r'(\^\{\\parallel\})(\^|_)', r'\1\\Vert \2', content)
    
    # 2. Replace norm bars with \Vert and a trailing space to prevent macro collisions
    content = content.replace(r'\|', r'\Vert ')
    
    # 3. Remove the space if the norm bar is immediately followed by a superscript or subscript.
    content = content.replace(r'\Vert ^', r'\Vert^')
    content = content.replace(r'\Vert _', r'\Vert_')
    
    # 4. Fix semicolons
    content = SEMICOLON_RE.sub(r'\\;', content)

    # 5. Replace \emph{} with \text{} inside math (LaTeX \emph is a text-mode command)
    content = re.sub(r'\\emph\{', r'\\text{', content)

    return content


def format_math_span(raw_text: str, kind: str, warnings: List[str], offset: int) -> str:
    """Routes a raw math string through formatting rules and performs stack-based validation."""
    
    def check_delimiter_stack(text: str):
        """
        Validates structure using a Pushdown Automaton. 
        Tracks \Vert alongside { and } to catch crossing scopes and missing pulls.
        """
        stack = []
        i, n = 0, len(text)
        
        while i < n:
            if text[i] == '\\':
                # Check for \Vert macro
                if text[i:i+5] == r'\Vert':
                    if stack and stack[-1] == r'\Vert':
                        stack.pop()  # Pull (Close norm)
                    else:
                        stack.append(r'\Vert')  # Push (Open norm)
                    i += 5
                    continue
                i += 2  # Skip other escaped characters
                continue
                
            ch = text[i]
            if ch == '{':
                stack.append('{')
            elif ch == '}':
                if not stack:
                    pass # Ignore orphan closing braces
                elif stack[-1] == '{':
                    stack.pop()
                elif stack[-1] == r'\Vert':
                    # Scope Violation: Closing a brace while a norm is still open
                    snippet = text[max(0, i-15):min(n, i+15)].replace('\n', ' ')
                    warnings.append(f"Scope violation near index {offset + i}: Closed '}}' while '\\Vert' was still pushed. (Context: '{snippet}')")
                    stack.pop() # Pop the \Vert to attempt recovery
                    if stack and stack[-1] == '{':
                        stack.pop()
            i += 1

        # Anything left on the stack is unclosed
        if r'\Vert' in stack:
            snippet = text[:40].replace('\n', ' ') + "..." if len(text) > 40 else text.replace('\n', ' ')
            warnings.append(f"Unclosed '\\Vert' (missing pull) detected in block starting at index {offset}. (Snippet: '{snippet}')")

    if kind == 'display':
        content = _apply_common_math_fixes(raw_text[2:-2].strip())
        check_delimiter_stack(content)
        return f'\n\n$$\n{content}\n$$\n\n'

    if kind == 'inline':
        content = raw_text.replace('\n', ' ') if '\n' in raw_text else raw_text
        content = _apply_common_math_fixes(content)
        check_delimiter_stack(content)
        return content

    if kind == 'mismatch_d_dd':
        content = _apply_common_math_fixes(raw_text[1:-2].strip())
        check_delimiter_stack(content)
        if '\n' in content:
            warnings.append(f'Mismatched $...$$ (multi-line) near index {offset} -> converted to display block.')
            return f'\n\n$$\n{content}\n$$\n\n'
        warnings.append(f'Mismatched $...$$ near index {offset} -> converted to inline.')
        return f'${content}$'

    return raw_text


def protect_table_pipes(text: str) -> str:
    """Prevents markdown table parsers from breaking on vertical bars inside inline math."""
    out = []
    for line in text.split('\n'):
        stripped = line.lstrip('> \t')
        if stripped.startswith('|') and '$' in line:
            in_math, chars = False, []
            i = 0
            while i < len(line):
                ch = line[i]
                if ch == '\\' and i + 1 < len(line):
                    chars.extend([ch, line[i+1]])
                    i += 2
                    continue
                if ch == '$':
                    in_math = not in_math
                elif ch == '|' and in_math:
                    ch = r'\vert'
                chars.append(ch)
                i += 1
            line = ''.join(chars)
        out.append(line)
    return '\n'.join(out)


# --- Orchestration ---
def process_markdown(text: str) -> Tuple[str, List[str]]:
    """Main pipeline: splits code blocks, processes math, and cleans whitespace."""
    text = text.replace('\r\n', '\n')
    warnings: List[str] = []
    segments = FENCE_RE.split(text)

    for idx, seg in enumerate(segments):
        if idx % 2 != 0:
            continue  # Skip Markdown code fences

        spans = find_math_spans(seg)
        out_seg, prev = [], 0

        for span in spans:
            out_seg.append(seg[prev:span.start])
            
            if span.end is None:
                warnings.append(f"Unmatched math delimiter near character index {span.start}: {repr(seg[span.start:span.start+40])}")
                out_seg.append(seg[span.start:])
                prev = len(seg)
                break

            formatted = format_math_span(seg[span.start:span.end], span.kind, warnings, span.start)
            out_seg.append(formatted)
            prev = span.end

        out_seg.append(seg[prev:])
        segments[idx] = ''.join(out_seg)

    result = ''.join(segments)
    result = protect_table_pipes(result)
    result = EXCESS_BLANK_RE.sub('\n\n', result).strip('\n') + '\n'
    
    return result, warnings


# --- Imperative Shell (CLI) ---
def main():
    parser = argparse.ArgumentParser(
        description="Canonically format display-math blocks in Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python reformat_math_extra.py paper.md                  "
            "# writes paper_cleaned.md\n"
            "  python reformat_math_extra.py paper.md -o out.md        "
            "# writes out.md\n"
            "  python reformat_math_extra.py paper.md -o console       "
            "# stdout; pipe-friendly\n"
            "  python reformat_math_extra.py paper.md -o console | python reflow_md.py - final.md"
        ),
    )
    parser.add_argument("input", type=Path, help="Path to the input Markdown file.")
    parser.add_argument(
        "-o", "--out",
        default=None,
        metavar="FILE|console",
        help=(
            'Output destination. Pass a file path to write there, or "console" to '
            'write reformatted text to stdout (warnings and status go to stderr, '
            'making the output pipe-safe). Defaults to {stem}_cleaned{suffix}.'
        ),
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.is_file():
        sys.exit(f"Error: Input file not found: {input_path}")

    to_console = args.out == "console"
    # When piping, warnings and status must not contaminate stdout.
    diag = sys.stderr if to_console else sys.stdout

    original = input_path.read_text(encoding='utf-8')
    result, warnings = process_markdown(original)

    if warnings:
        print(f"WARNINGS in {input_path.name}:", file=diag)
        for w in warnings:
            print(f"  - {w}", file=diag)

    normalised = original.replace('\r\n', '\n').strip('\n') + '\n'
    if result == normalised:
        print(f"No changes needed: {input_path.name}", file=diag)
        if to_console:
            sys.stdout.buffer.write(result.encode('utf-8'))  # still emit so pipe isn't empty
        return

    if to_console:
        # Force UTF-8 on Windows where the console codec may be cp1252.
        sys.stdout.buffer.write(result.encode('utf-8'))
        n_dd = result.count('$$') // 2
        print(f"Success: Reformatted {n_dd} display-math block(s) -> stdout", file=diag)
    else:
        output_path = Path(args.out) if args.out else input_path.with_name(
            f"{input_path.stem}_cleaned{input_path.suffix}"
        )
        output_path.write_text(result, encoding='utf-8')
        n_dd = result.count('$$') // 2
        print(f"Success: Reformatted {n_dd} display-math block(s) -> {output_path}")



if __name__ == '__main__':
    main()
