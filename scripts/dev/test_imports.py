#!/usr/bin/env python
"""Quick import test for refactored modules."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import sim_theorem_partii as sim
    print("✓ sim_theorem_partii imported successfully")
except ImportError as e:
    print(f"✗ Failed to import sim_theorem_partii: {e}")
    sys.exit(1)

try:
    import fl_graphics as gfx
    print("✓ fl_graphics imported successfully")
except ImportError as e:
    print(f"✗ Failed to import fl_graphics: {e}")
    sys.exit(1)

# Check key module contents
required_sim = ["K", "SIGMA2", "TAU2", "DELTA2", "build_model", "SineAlignmentAnalysis",
                "Eq20RHSAnalysis", "simulate", "print_summary"]
for attr in required_sim:
    if not hasattr(sim, attr):
        print(f"✗ sim_theorem_partii missing: {attr}")
        sys.exit(1)
print(f"✓ sim_theorem_partii has all {len(required_sim)} required attributes")

required_gfx = ["load_results", "plot_convergence", "plot_scatter", "plot_components", "plot_all"]
for attr in required_gfx:
    if not hasattr(gfx, attr):
        print(f"✗ fl_graphics missing: {attr}")
        sys.exit(1)
print(f"✓ fl_graphics has all {len(required_gfx)} required attributes")

print("\nAll imports successful!")
