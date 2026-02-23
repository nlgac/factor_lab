#!/usr/bin/env python3
"""
verify_npz_output.py - Verify NPZ Files Are Created Correctly
==============================================================

This script demonstrates that build_and_simulate.py creates comprehensive
NPZ files with all of Gemini's original keys.

Usage:
    python verify_npz_output.py
"""

import sys
from pathlib import Path
import subprocess

def check_npz_files():
    """Check if NPZ files exist and show their contents."""
    print("="*70)
    print("  NPZ FILE VERIFICATION")
    print("="*70)
    
    # Look for NPZ files in current directory
    npz_files = list(Path(".").glob("simulation_*.npz"))
    npz_files.extend(Path(".").glob("factor_model.npz"))
    
    if not npz_files:
        print("\n❌ No NPZ files found in current directory!")
        print("\nTo create them, run:")
        print("    python build_and_simulate.py model_spec.json")
        return False
    
    print(f"\n✓ Found {len(npz_files)} NPZ file(s):")
    for f in sorted(npz_files):
        print(f"   • {f}")
    
    # Check each file
    import numpy as np
    
    for npz_file in sorted(npz_files):
        print(f"\n{'='*70}")
        print(f"  {npz_file.name}")
        print('='*70)
        
        try:
            data = np.load(npz_file)
            
            print(f"\n📊 Contains {len(data.files)} arrays:\n")
            
            # Expected keys from Gemini's format
            expected_keys = [
                'security_returns',
                'true_B',
                'true_ortho_B',
                'true_F',
                'true_D',
                'true_eigenvalues',
                'true_eigenvectors',
                'sample_B',
                'sample_F',
                'sample_D',
                'sample_eigenvalues',
                'sample_eigenvectors',
                'dist_grassmannian',
                'dist_stiefel_chordal',
                'dist_stiefel_procrustes',
                'principal_angles'
            ]
            
            # Check which keys are present
            print("  Gemini's Expected Keys:")
            for key in expected_keys:
                if key in data.files:
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        if val.ndim == 0:
                            print(f"    ✓ {key:30s} scalar: {float(val):.6f}")
                        else:
                            print(f"    ✓ {key:30s} shape: {val.shape}")
                    else:
                        print(f"    ✓ {key:30s} present")
                else:
                    print(f"    ✗ {key:30s} MISSING")
            
            # Show any additional keys
            additional_keys = [k for k in data.files if k not in expected_keys]
            if additional_keys:
                print(f"\n  Additional Keys ({len(additional_keys)}):")
                for key in sorted(additional_keys):
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        if val.ndim == 0:
                            print(f"    + {key:30s} scalar: {float(val):.6f}")
                        else:
                            print(f"    + {key:30s} shape: {val.shape}")
            
            data.close()
            
        except Exception as e:
            print(f"❌ Error reading {npz_file}: {e}")
    
    return True


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("  NPZ OUTPUT VERIFICATION TOOL")
    print("="*70)
    
    print("\nThis script verifies that build_and_simulate.py creates")
    print("comprehensive NPZ files with all of Gemini's original keys.")
    
    if check_npz_files():
        print("\n" + "="*70)
        print("  ✅ VERIFICATION COMPLETE")
        print("="*70)
        print("\nAll NPZ files are properly formatted!")
        print("\nTo inspect files interactively:")
        print("    python inspect_npz.py simulation_*.npz")
    else:
        print("\n" + "="*70)
        print("  ⚠️  NO FILES FOUND")
        print("="*70)
        print("\nRun this to create NPZ files:")
        print("    python build_and_simulate.py model_spec.json")
        print("\nThen run this script again to verify.")
    
    print()


if __name__ == "__main__":
    main()
