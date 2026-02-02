#!/usr/bin/env python3
"""
inspect_npz.py - Utility to Inspect NPZ Files
==============================================

Usage:
    python inspect_npz.py simulation_gaussian.npz
    python inspect_npz.py simulation_*.npz
"""

import sys
from pathlib import Path
import numpy as np


def inspect_npz(filepath: Path):
    """Inspect contents of NPZ file."""
    print(f"\n{'='*70}")
    print(f"  File: {filepath.name}")
    print('='*70)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    # Load
    data = np.load(filepath)
    
    # Print all keys
    print(f"\n📊 Contains {len(data.files)} arrays:")
    print()
    
    # Organize by category
    categories = {
        'Raw Data': [],
        'True Model': [],
        'Sample Model': [],
        'Manifold Distances': [],
        'Eigenvalue Metrics': [],
        'Eigenvector Metrics': [],
        'Other': []
    }
    
    for key in sorted(data.files):
        if key.startswith('security_'):
            categories['Raw Data'].append(key)
        elif key.startswith('true_'):
            categories['True Model'].append(key)
        elif key.startswith('sample_'):
            categories['Sample Model'].append(key)
        elif key.startswith('dist_'):
            categories['Manifold Distances'].append(key)
        elif 'eigenvalue' in key:
            categories['Eigenvalue Metrics'].append(key)
        elif 'eigenvector' in key or 'vector_' in key or key == 'principal_angles':
            categories['Eigenvector Metrics'].append(key)
        else:
            categories['Other'].append(key)
    
    # Print organized
    for category, keys in categories.items():
        if keys:
            print(f"  {category}:")
            for key in keys:
                val = data[key]
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        print(f"    • {key:30s} scalar: {float(val):.6f}")
                    else:
                        print(f"    • {key:30s} shape: {str(val.shape):15s} dtype: {val.dtype}")
                else:
                    print(f"    • {key:30s} {type(val).__name__}")
            print()
    
    # Summary statistics
    print(f"{'='*70}")
    print("  Summary")
    print('='*70)
    
    # Check for key metrics
    if 'dist_grassmannian' in data:
        print(f"\n  Grassmannian Distance:  {float(data['dist_grassmannian']):.6f}")
    if 'dist_stiefel_procrustes' in data:
        print(f"  Procrustes Distance:    {float(data['dist_stiefel_procrustes']):.6f}")
    if 'dist_stiefel_chordal' in data:
        print(f"  Chordal Distance:       {float(data['dist_stiefel_chordal']):.6f}")
    
    if 'eigenvalue_rmse' in data:
        print(f"\n  Eigenvalue RMSE:        {float(data['eigenvalue_rmse']):.6f}")
    
    if 'mean_correlation' in data:
        print(f"\n  Mean Eigenvector Corr:  {float(data['mean_correlation']):.4f}")
    
    if 'true_B' in data and 'sample_B' in data:
        print(f"\n  Model Dimensions:")
        true_B = data['true_B']
        print(f"    k (factors): {true_B.shape[0]}")
        print(f"    p (assets):  {true_B.shape[1]}")
    
    if 'security_returns' in data:
        returns = data['security_returns']
        print(f"    T (periods): {returns.shape[0]}")
    
    print()
    data.close()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <file.npz>")
        print("   or: python inspect_npz.py simulation_*.npz")
        sys.exit(1)
    
    # Handle wildcards
    import glob
    files = []
    for pattern in sys.argv[1:]:
        files.extend(glob.glob(pattern))
    
    if not files:
        print(f"❌ No files found matching: {sys.argv[1:]}")
        sys.exit(1)
    
    # Inspect each file
    for filepath in sorted(files):
        inspect_npz(Path(filepath))
    
    if len(files) > 1:
        print(f"\n{'='*70}")
        print(f"  Inspected {len(files)} files")
        print('='*70 + "\n")


if __name__ == "__main__":
    main()
