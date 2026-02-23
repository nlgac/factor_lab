#!/usr/bin/env python3
"""
setup.py - Backward compatibility shim
======================================

This file exists for backward compatibility with tools that don't yet
support PEP 621 (pyproject.toml).

Modern installation should use pyproject.toml:
    pip install .
    pip install -e .
    pip install -e ".[all]"

This setup.py simply delegates to setuptools' pyproject.toml support.
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists only for backward compatibility
setup()
