#!/usr/bin/env python3
"""
reformat_math.py -- Professional LaTeX Math Formatter for Markdown

This script implements a robust pipeline for cleaning and stabilizing LaTeX 
math blocks in Markdown files. It uses a dual-stack parsing model:
1. A top-level stack to isolate math blocks from Markdown text and code fences.
2. A secondary Pushdown Automaton to validate and repair delimiters (\Vert, { }) 
   inside the isolated math blocks.

Usage:
    python reformat_math.py input.md [output.md]
    (If output.md is omitted, defaults to input_cleaned.md)
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class MathSpan:
    """Represents a located span of math text within a document segment."""
    start: int
    end: Optional[int]  # None indicates an unclosed (orphan) delimiter
    kind: str           # 'display', 'inline', 'mismatch_d_dd', 'orphan'


class MathRefactorer:
    """
    Encapsulates the logic for parsing, validating, and refactoring 
    math content within a single Markdown document.
    """
    FENCE_PATTERN = re.compile(r'(```.*?```)', re.DOTALL)
    EXCESS_BLANK_PATTERN = re.compile(r'\n{3,}')
    SEMICOLON_PATTERN = re.compile(r'(?<!\\);')

    def __init__(self, text: str):
        self.text = text.replace('\r\n', '\n')
        self.warnings: List[str] = []

    def run(self) -> str:
        """Executes the full refactoring pipeline."""
        # Split text into segments: [markdown, code_fence, markdown, ...]
        segments = self.FENCE_PATTERN.split(self.text)
        
        # Only process segments that are NOT inside code fences (even indices)
        processed = [
            self._process_markdown_segment(seg) if i % 2 == 0 else seg
            for i, seg in enumerate(segments)
        ]
        
        result = "".join(processed)
        result = self._protect_table_pipes(result)
        result = self.EXCESS_BLANK_PATTERN.sub('\n\n', result)
        return result.strip('\n') + '\n'

    def _process_markdown_segment(self, segment: str) -> str:
        """Finds and reformats math blocks within a non-code segment."""
        spans = self._find_top_level_spans(segment)
        output, prev_end = [], 0

        for span in spans:
            output.append(segment[prev_end:span.start])
            
            if span.end is None:
                self.warnings.append(f"Orphan delimiter at index {span.start}")
                output.append(segment[span.start:])
                prev_end = len(segment)
                break

            content = segment[span.start:span.end]
            formatted = self._reformat_span(content, span)
            output.append(formatted)
            prev_end = span.end

        output.append(segment[prev_end:])
        return "".join(output)

    def _find_top_level_spans(self, text: str) -> List[MathSpan]:
        """Implements the top-level stack parser for math delimiters."""
        spans, stack, i, n = [], [], 0, len(text)
        while i < n:
            if text[i] == '\\' and i + 1 < n:
                i += 2
                continue
            if text[i] == '$':
                is_display = (i + 1 < n and text[i+1] == '$')
                kind = 'display' if is_display else 'inline'
                tlen = 2 if is_display else 1

                if not stack:
                    stack.append((kind, i))
                else:
                    top_kind, top_pos = stack[-1]
                    if top_kind == kind:
                        stack.pop()
                        if not stack:
                            spans.append(MathSpan(top_pos, i + tlen, kind))
                    elif top_kind == 'display' and kind == 'inline':
                        stack.append((kind, i))  # Allow nested inline (e.g. \tag{$8$})
                    elif top_kind == 'inline' and kind == 'display':
                        stack.pop()
                        if not stack:
                            spans.append(MathSpan(top_pos, i + tlen, 'mismatch_d_dd'))
                i += tlen
            else:
                i += 1
        
        for _, pos in stack:
            spans.append(MathSpan(pos, None, 'orphan'))
        return sorted(spans, key=lambda x: x.start)

    def _reformat_span(self, raw: str, span: MathSpan) -> str:
        """Applies stability fixes and validates internal delimiter structure."""
        # 1. Strip delimiters
        inner = raw[2:-2] if span.kind == 'display' else raw[1:-1]
        if span.kind == 'mismatch_d_dd': inner = raw[1:-2]
        
        # 2. Apply "Gold-Standard" fixes and Heuristics
        content = inner.strip()
        if span.kind == 'inline': content = content.replace('\n', ' ')
        
        content = content.replace(r'^\|', r'^{\parallel}')
        # Auto-repair: Inject missing \Vert after parallel superscripts
        content = re.sub(r'(\\parallel})(\^|_)', r'\1\\Vert\2', content)
        content = content.replace(r'\|', r'\Vert ')
        content = content.replace(r'\Vert ^', r'\Vert^')
        content = content.replace(r'\Vert _', r'\Vert_')
        content = self.SEMICOLON_PATTERN.sub(r'\\;', content)

        # 3. Validation via Pushdown Automaton
        self._validate_delimiters(content, span.start)

        # 4. Wrap in canonical delimiters
        if span.kind == 'display' or (span.kind == 'mismatch_d_dd' and '\n' in content):
            return f"\n\n$$\n{content}\n$$\n\n"
        return f"${content}$"

    def _validate_delimiters(self, text: str, offset: int):
        """Pushdown Automaton to catch scope violations and missing pulls."""
        stack = []
        for i, match in enumerate(re.finditer(r'\\Vert|\{|\}', text)):
            token = match.group()
            if token == r'\Vert':
                if stack and stack[-1] == r'\Vert': stack.pop()
                else: stack.append(r'\Vert')
            elif token == '{':
                stack.append('{')
            elif token == '}':
                if not stack: continue
                if stack[-1] == '{': stack.pop()
                elif stack[-1] == r'\Vert':
                    self.warnings.append(f"Scope violation at {offset+match.start()}: '}}' closed while '\\Vert' open.")
                    stack.pop() # Recovery
                    if stack and stack[-1] == '{': stack.pop()
        
        if r'\Vert' in stack:
            self.warnings.append(f"Unclosed '\\Vert' in block starting at {offset}")

    def _protect_table_pipes(self, text: str) -> str:
        """Ensures math pipes don't break Markdown table column parsing."""
        lines = []
        for line in text.split('\n'):
            if line.lstrip().startswith('|') and '$' in line:
                in_math, chars, i = False, [], 0
                while i < len(line):
                    ch = line[i]
                    if ch == '\\' and i+1 < len(line):
                        chars.extend([ch, line[i+1]]); i += 2; continue
                    if ch == '$': in_math = not in_math
                    elif ch == '|' and in_math: ch = r'\vert'
                    chars.append(ch); i += 1
                line = "".join(chars)
            lines.append(line)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Clean and stabilize Markdown LaTeX math.")
    parser.add_argument("input", type=Path, help="Input Markdown file")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (optional)")
    args = parser.parse_args()

    if not args.input.is_file():
        sys.exit(f"Error: {args.input} not found.")

    output_path = args.output or args.input.with_name(f"{args.input.stem}_cleaned{args.input.suffix}")
    
    refactorer = MathRefactorer(args.input.read_text(encoding='utf-8'))
    result = refactorer.run()

    for warning in refactorer.warnings:
        print(f"WARNING [{args.input.name}]: {warning}")

    output_path.write_text(result, encoding='utf-8')
    print(f"Success: Processed {args.input.name} -> {output_path.name}")


if __name__ == "__main__":
    main()