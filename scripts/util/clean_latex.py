#!/usr/bin/env python
"""Delete LaTeX build artifacts, keeping .tex sources and .pdf output.

Usage:
    python clean_latex.py [DIR] [--yes]

DIR defaults to the current directory. Without --yes, lists files and asks
for confirmation before deleting.
"""

import sys
from pathlib import Path

# Common auxiliary extensions produced by pdflatex/biblatex/synctex.
AUX_EXTENSIONS = {
    ".aux", ".log", ".out", ".toc", ".lof", ".lot", ".bbl", ".bcf",
    ".blg", ".fls", ".fdb_latexmk", ".synctex.gz", ".run.xml",
    ".nav", ".snm", ".vrb", ".figlist", ".makefile", ".auxlock",
}


def find_aux_files(directory: Path) -> list[Path]:
    return [
        f for f in directory.iterdir()
        if f.is_file() and "".join(f.suffixes[-2:]) in AUX_EXTENSIONS
        or f.is_file() and f.suffix in AUX_EXTENSIONS
    ]


def main() -> None:
    args = sys.argv[1:]
    auto_confirm = "--yes" in args
    args = [a for a in args if a != "--yes"]
    directory = Path(args[0]) if args else Path(".")

    targets = find_aux_files(directory)
    if not targets:
        print("No LaTeX auxiliary files found.")
        return

    print(f"Found {len(targets)} file(s) to delete:")
    for f in targets:
        print(f"  {f}")

    if not auto_confirm:
        if input("Delete these files? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return

    for f in targets:
        f.unlink()
    print(f"Deleted {len(targets)} file(s).")


if __name__ == "__main__":
    main()
