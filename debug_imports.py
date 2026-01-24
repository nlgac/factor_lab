"""
Debug script to find what's causing the import error.
Save this as debug_imports.py and run it.
"""

import sys
from pathlib import Path
import os

print("=" * 70)
print("DEBUGGING FACTOR_LAB IMPORTS")
print("=" * 70)

# 1. Show current directory
print(f"\n1. Current directory: {os.getcwd()}")

# 2. Show Python path
print(f"\n2. Python sys.path:")
for i, p in enumerate(sys.path):
    print(f"   {i}: {p}")

# 3. Check for conftest.py files
project_root = Path.cwd()
print(f"\n3. Looking for conftest.py files from: {project_root}")

conftest_files = list(project_root.rglob("conftest.py"))
print(f"   Found {len(conftest_files)} conftest.py files:")
for cf in conftest_files:
    print(f"   - {cf}")
    # Show first 10 lines
    with open(cf, 'r') as f:
        lines = f.readlines()[:10]
    for i, line in enumerate(lines, 1):
        print(f"     {i}: {line.rstrip()}")
    print()

# 4. Check for __init__.py files
print(f"\n4. Looking for __init__.py files:")
init_files = list(project_root.rglob("__init__.py"))
for init_file in init_files[:5]:  # Show first 5
    print(f"   - {init_file}")
    # Check if it has imports from .types
    with open(init_file, 'r') as f:
        content = f.read()
    if 'from .types import' in content:
        print(f"     ⚠ This file imports from .types")
        # Show the import lines
        for i, line in enumerate(content.split('\n')[:20], 1):
            if 'from' in line or 'import' in line:
                print(f"     {i}: {line}")

# 5. Try importing factor_lab
print(f"\n5. Attempting to import factor_lab...")
try:
    import factor_lab
    print(f"   ✓ SUCCESS: factor_lab imported from {factor_lab.__file__}")
    print(f"   ✓ Version: {factor_lab.__version__}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# 6. Check if pytest is loading the wrong __init__.py
print(f"\n6. Checking for duplicate __init__.py files in tests/:")
tests_init = project_root / "tests" / "__init__.py"
if tests_init.exists():
    print(f"   ⚠ WARNING: tests/__init__.py exists!")
    print(f"   This file should probably NOT exist.")
    print(f"   Content:")
    with open(tests_init, 'r') as f:
        content = f.read()
    print(content[:500])
else:
    print(f"   ✓ No tests/__init__.py (this is correct)")

# 7. Final diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("""
If you see any of these issues:
1. tests/__init__.py exists → DELETE IT
2. Multiple conftest.py with imports → Keep only path setup
3. factor_lab import failed → Python can't find the package
4. Wrong __init__.py loaded → Check sys.path order
""")
