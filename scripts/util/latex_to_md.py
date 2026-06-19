import re
import subprocess
import sys
from pathlib import Path

def preprocess_latex(tex_content: str) -> str:
    """
    Cleans up custom macros and specific formatting before handing off to Pandoc.
    """
    # 1. Accept Revisions: Strip \chg{} and \mchg{} wrappers but keep the inner text
    # The regex handles up to one level of nested braces.
    tex_content = re.sub(r'\\m?chg\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', tex_content)
    
    # 2. Preserve Comments: Convert \nlgcmt{} and \typo{} into Markdown blockquotes
    tex_content = re.sub(r'\\nlgcmt\{((?:[^{}]|\{[^{}]*\})*)\}', r'\n> **NLG Comment:** \1\n', tex_content)
    tex_content = re.sub(r'\\typo\{((?:[^{}]|\{[^{}]*\})*)\}', r'\n> **Typo:** \1\n', tex_content)
    
    # 3. Clean up custom color wrappers if any remain
    tex_content = re.sub(r'\\colortext\[.*?\]\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', tex_content)
    
    return tex_content

def convert_tex_to_md(input_path: str, output_path: str):
    tex_file = Path(input_path)
    if not tex_file.exists():
        print(f"Error: Could not find file {input_path}")
        sys.exit(1)

    print(f"Reading {tex_file.name}...")
    raw_tex = tex_file.read_text(encoding='utf-8')
    
    print("Pre-processing custom macros...")
    clean_tex = preprocess_latex(raw_tex)

    # Save an intermediate file for Pandoc to process
    temp_tex = tex_file.with_suffix('.temp.tex')
    temp_tex.write_text(clean_tex, encoding='utf-8')

    print("Running Pandoc AST conversion...")
    # 'markdown+tex_math_dollars' ensures LaTeX math is passed through cleanly
    cmd = [
        'pandoc',
        str(temp_tex),
        '-f', 'latex',
        '-t', 'markdown+tex_math_dollars',
        '-s', # Standalone document
        '-o', str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Success! Markdown saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Pandoc encountered an error:\n{e.stderr}")
    finally:
        # Cleanup the intermediate file
        if temp_tex.exists():
            temp_tex.unlink()

if __name__ == "__main__":
    # Example usage: python tex2md.py main.tex output.md
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input.tex> <output.md>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_tex_to_md(input_file, output_file)