#!/usr/bin/env python3
"""
Quick Test Runner for Factor Lab Refactoring

Usage:
    python test_runner.py              # Run all tests
    python test_runner.py --quick      # Quick check only
    python test_runner.py --coverage   # With coverage report
    python test_runner.py --baseline   # Create baseline
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def run_command(cmd, description="Running command"):
    """Run a shell command and return success status."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{description}{Colors.RESET}")
    print(f"{Colors.BLUE}Command: {' '.join(cmd)}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n{Colors.GREEN}✓ {description} - SUCCESS{Colors.RESET}\n")
        return True
    else:
        print(f"\n{Colors.RED}✗ {description} - FAILED{Colors.RESET}\n")
        return False


def count_tests():
    """Count total number of tests."""
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return None
    
    # Parse output to count tests
    lines = result.stdout.split('\n')
    for line in lines:
        if 'tests collected' in line or 'test collected' in line:
            # Extract number from line like "153 tests collected"
            parts = line.split()
            if parts:
                try:
                    return int(parts[0])
                except ValueError:
                    pass
    return None


def quick_check():
    """Run a quick validation check."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Running Quick Check...{Colors.RESET}\n")
    
    checks = []
    
    # 1. Import check
    print(f"{Colors.BLUE}1. Testing imports...{Colors.RESET}")
    result = subprocess.run(
        ["python", "-c", "from factor_lab import *; print('OK')"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and 'OK' in result.stdout:
        print(f"{Colors.GREEN}   ✓ All imports work{Colors.RESET}")
        checks.append(True)
    else:
        print(f"{Colors.RED}   ✗ Import failed{Colors.RESET}")
        print(f"{Colors.RED}   {result.stderr}{Colors.RESET}")
        checks.append(False)
    
    # 2. Test collection
    print(f"\n{Colors.BLUE}2. Collecting tests...{Colors.RESET}")
    test_count = count_tests()
    if test_count is not None:
        print(f"{Colors.GREEN}   ✓ Found {test_count} tests{Colors.RESET}")
        checks.append(True)
    else:
        print(f"{Colors.RED}   ✗ Failed to collect tests{Colors.RESET}")
        checks.append(False)
    
    # 3. Quick test run (fail fast)
    print(f"\n{Colors.BLUE}3. Running quick test (first failure stops)...{Colors.RESET}")
    success = run_command(
        ["python", "-m", "pytest", "tests/", "-x", "-q"],
        "Quick test run"
    )
    checks.append(success)
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    if all(checks):
        print(f"{Colors.GREEN}{Colors.BOLD}✓ QUICK CHECK PASSED{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ QUICK CHECK FAILED{Colors.RESET}")
        return False


def run_full_tests(verbose=True):
    """Run full test suite."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Running Full Test Suite...{Colors.RESET}\n")
    
    cmd = ["python", "-m", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd, "Full test suite")


def run_with_coverage():
    """Run tests with coverage report."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Running Tests with Coverage...{Colors.RESET}\n")
    
    success = run_command(
        ["python", "-m", "pytest", "tests/", 
         "--cov=factor_lab", 
         "--cov-report=term",
         "--cov-report=html"],
        "Test suite with coverage"
    )
    
    if success:
        print(f"\n{Colors.GREEN}Coverage report generated in htmlcov/index.html{Colors.RESET}")
    
    return success


def create_baseline():
    """Create baseline test report."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Creating Baseline Report...{Colors.RESET}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = Path("test_baselines")
    baseline_dir.mkdir(exist_ok=True)
    
    # Count tests
    test_count = count_tests()
    
    # Run tests and save output
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    baseline_file = baseline_dir / f"baseline_{timestamp}.txt"
    with open(baseline_file, 'w') as f:
        f.write(f"Baseline created: {datetime.now()}\n")
        f.write(f"Total tests: {test_count}\n")
        f.write(f"Status: {'PASSED' if result.returncode == 0 else 'FAILED'}\n")
        f.write("="*70 + "\n\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n" + "="*70 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)
    
    print(f"{Colors.GREEN}Baseline saved to: {baseline_file}{Colors.RESET}")
    
    # Also save coverage baseline
    subprocess.run(
        ["python", "-m", "pytest", "tests/", 
         "--cov=factor_lab", 
         "--cov-report=term"],
        capture_output=False
    )
    
    coverage_file = baseline_dir / f"coverage_{timestamp}.txt"
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", 
         "--cov=factor_lab", 
         "--cov-report=term"],
        capture_output=True,
        text=True
    )
    
    with open(coverage_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"{Colors.GREEN}Coverage baseline saved to: {coverage_file}{Colors.RESET}")
    
    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Factor Lab Test Runner")
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick check only'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Create baseline report'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (less verbose)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Factor Lab Test Runner{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    # Determine what to run
    if args.baseline:
        success = create_baseline()
    elif args.quick:
        success = quick_check()
    elif args.coverage:
        success = run_with_coverage()
    else:
        success = run_full_tests(verbose=not args.quiet)
    
    # Final summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    if success:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED{Colors.RESET}")
        print(f"{Colors.GREEN}You can proceed with the next refactoring step.{Colors.RESET}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ CHECKS FAILED{Colors.RESET}")
        print(f"{Colors.RED}Fix issues before proceeding with refactoring.{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
