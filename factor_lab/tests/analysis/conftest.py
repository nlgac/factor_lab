"""Pytest configuration for standalone testing."""
import sys
from pathlib import Path

# Add parent directory to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
