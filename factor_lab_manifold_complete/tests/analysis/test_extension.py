#!/usr/bin/env python3
"""
Test script for factor_variances extension feature.

Tests that factor_variances is automatically extended by repeating
the last variance when fewer variances than k_factors are provided.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import perturbation_study
sys.path.insert(0, str(Path(__file__).parent))

from perturbation_study import PerturbationSpec


def test_extension():
    """Test that factor_variances is extended correctly."""
    print("=" * 70)
    print("TEST: Factor Variances Extension")
    print("=" * 70)
    
    # Test 1: Extension needed (3 → 5)
    print("\n📝 Test 1: Extension from 3 to 5 values")
    print("-" * 70)
    config1 = {
        "p_assets": 500,
        "k_factors": 5,
        "factor_variances": [0.0324, 0.01, 0.0025]
    }
    spec1 = PerturbationSpec(**config1)
    expected1 = [0.0324, 0.01, 0.0025, 0.0025, 0.0025]
    print(f"Expected: {expected1}")
    print(f"Got:      {spec1.factor_variances}")
    assert spec1.factor_variances == expected1, "Test 1 failed!"
    print("✅ PASS")
    
    # Test 2: No extension needed (exact match)
    print("\n📝 Test 2: No extension needed (exact match)")
    print("-" * 70)
    config2 = {
        "p_assets": 500,
        "k_factors": 3,
        "factor_variances": [0.0324, 0.01, 0.0025]
    }
    spec2 = PerturbationSpec(**config2)
    expected2 = [0.0324, 0.01, 0.0025]
    print(f"Expected: {expected2}")
    print(f"Got:      {spec2.factor_variances}")
    assert spec2.factor_variances == expected2, "Test 2 failed!"
    print("✅ PASS (no console output expected)")
    
    # Test 3: Single variance extended to many
    print("\n📝 Test 3: Single variance → 10 factors")
    print("-" * 70)
    config3 = {
        "p_assets": 500,
        "k_factors": 10,
        "factor_variances": [0.01]
    }
    spec3 = PerturbationSpec(**config3)
    expected3 = [0.01] * 10
    print(f"Expected: {expected3}")
    print(f"Got:      {spec3.factor_variances}")
    assert spec3.factor_variances == expected3, "Test 3 failed!"
    print("✅ PASS")
    
    # Test 4: Two variances extended to many
    print("\n📝 Test 4: Two variances → 20 factors")
    print("-" * 70)
    config4 = {
        "p_assets": 1000,
        "k_factors": 20,
        "factor_variances": [0.1, 0.001]
    }
    spec4 = PerturbationSpec(**config4)
    expected4 = [0.1] + [0.001] * 19
    print(f"Expected: {expected4[:5]} + [0.001] * 15")
    print(f"Got:      {spec4.factor_variances[:5]} + [0.001] * 15")
    assert spec4.factor_variances == expected4, "Test 4 failed!"
    print("✅ PASS")
    
    # Test 5: More variances than factors (no truncation in spec)
    print("\n📝 Test 5: More variances than factors")
    print("-" * 70)
    config5 = {
        "p_assets": 500,
        "k_factors": 2,
        "factor_variances": [0.04, 0.02, 0.01, 0.005]
    }
    spec5 = PerturbationSpec(**config5)
    # Should keep all variances (no truncation in spec)
    expected5 = [0.04, 0.02, 0.01, 0.005]
    print(f"Expected: {expected5}")
    print(f"Got:      {spec5.factor_variances}")
    assert spec5.factor_variances == expected5, "Test 5 failed!"
    print("✅ PASS (no extension, extra variances kept)")
    
    # Test 6: Default (None) - should not extend
    print("\n📝 Test 6: Default (None) - no extension")
    print("-" * 70)
    config6 = {
        "p_assets": 500,
        "k_factors": 3,
        "factor_variances": None
    }
    spec6 = PerturbationSpec(**config6)
    print(f"Got: {spec6.factor_variances}")
    assert spec6.factor_variances is None, "Test 6 failed!"
    print("✅ PASS (remains None)")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


def example_usage():
    """Show example usage with console output."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE: Console Output When Extension Happens")
    print("=" * 70)
    
    print("\n📋 Creating spec with 5 factors but only 3 variances...")
    config = {
        "p_assets": 500,
        "k_factors": 5,
        "factor_variances": [0.0324, 0.01, 0.0025]
    }
    
    print("\nInitializing PerturbationSpec...")
    print("-" * 70)
    spec = PerturbationSpec(**config)
    print("-" * 70)
    
    print(f"\n✓ Final factor_variances: {spec.factor_variances}")
    print(f"✓ Length: {len(spec.factor_variances)}")
    print(f"✓ Matches k_factors: {len(spec.factor_variances) == spec.k_factors}")


if __name__ == "__main__":
    test_extension()
    example_usage()
