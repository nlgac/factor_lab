# Packaging Files - Summary

## Overview

I've created a complete, modern Python packaging setup for Factor Lab Manifold Complete following best practices and PEP standards.

---

## Files Created

### 1. `pyproject.toml` (Primary Configuration)

**Purpose**: Modern Python package configuration (PEP 621)

**Contents**:
- Package metadata (name, version, description, authors)
- Dependencies (core, optional, dev)
- Build system configuration
- Tool configurations (pytest, black, ruff, mypy, coverage)
- Entry points and URLs

**Key Features**:
- ✅ PEP 621 compliant
- ✅ Optional dependencies (plotly, dev, docs, all)
- ✅ Complete tool configurations
- ✅ Type hints support
- ✅ Extensible for future needs

**Installation with optional dependencies**:
```bash
pip install .                  # Core only
pip install ".[plotly]"       # With interactive visualizations
pip install ".[dev]"          # With dev tools
pip install -e ".[all]"       # Everything (editable)
```

---

### 2. `requirements.txt` (Core Dependencies)

**Purpose**: Minimum required dependencies for package functionality

**Contents**:
```
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0
matplotlib>=3.3.0,<4.0.0
seaborn>=0.11.0,<1.0.0
```

**Use when**: 
- Basic installation
- Production deployment
- Minimal footprint needed

**Installation**:
```bash
pip install -r requirements.txt
```

---

### 3. `requirements-optional.txt` (Optional Features)

**Purpose**: Dependencies for enhanced features

**Contents**:
```
plotly>=5.0.0,<6.0.0
pandas>=1.3.0,<3.0.0
```

**Features enabled**:
- Interactive HTML dashboards
- Enhanced data handling in examples

**Use when**: You want interactive visualizations

**Installation**:
```bash
pip install -r requirements.txt -r requirements-optional.txt
```

---

### 4. `requirements-dev.txt` (Development Tools)

**Purpose**: Tools for development, testing, and documentation

**Contents**:
- Testing: pytest, pytest-cov, pytest-xdist
- Formatting: black
- Linting: ruff, flake8
- Type checking: mypy
- Documentation: sphinx, sphinx-rtd-theme, myst-parser
- Utilities: ipython, ipdb

**Use when**: 
- Developing the package
- Contributing code
- Running tests
- Building documentation

**Installation**:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

---

### 5. `requirements-all.txt` (Everything)

**Purpose**: All dependencies in one file

**Contents**: References to all other requirements files

**Use when**: You want complete setup

**Installation**:
```bash
pip install -r requirements-all.txt
```

**Equivalent to**:
```bash
pip install -e ".[all]"
```

---

### 6. `MANIFEST.in` (Distribution Files)

**Purpose**: Specify non-Python files to include in distribution

**Includes**:
- Documentation (README, LICENSE, docs/)
- Examples (demo.py, examples/)
- Tests (tests/)
- Configuration files (*.json)
- Requirements files

**Use when**: Building distributions with `python -m build`

---

### 7. `setup.py` (Backward Compatibility)

**Purpose**: Support tools that don't yet understand pyproject.toml

**Contents**: Minimal shim that delegates to pyproject.toml

**Note**: This is for compatibility only. Modern installations should use pyproject.toml.

---

### 8. `INSTALL.md` (Installation Guide)

**Purpose**: Comprehensive installation documentation

**Contents**:
- 6 installation methods
- Dependency tier explanations
- System requirements
- Verification procedures
- Troubleshooting guide
- Development setup

**Use when**: Someone needs to install the package

---

## Installation Scenarios

### Scenario 1: End User (Basic)

**Goal**: Use the package for analysis

**Steps**:
```bash
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete
pip install -r requirements.txt
python demo.py
```

**What they get**:
- Core functionality
- Static visualizations
- All analyses

**What they don't get**:
- Interactive HTML dashboards
- Development tools

---

### Scenario 2: End User (Full Features)

**Goal**: Use all features including interactive visualizations

**Steps**:
```bash
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete
pip install -r requirements.txt -r requirements-optional.txt
python demo.py
```

**Or using pyproject.toml**:
```bash
pip install ".[plotly]"
```

**What they get**:
- Everything from Scenario 1
- Interactive HTML dashboards
- Enhanced data handling

---

### Scenario 3: Developer

**Goal**: Develop, test, and contribute

**Steps**:
```bash
tar -xzf factor_lab_manifold_complete.tar.gz
cd factor_lab_manifold_complete

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Verify
pytest tests/ -v
```

**What they get**:
- Everything from Scenarios 1 & 2
- All development tools
- Code quality tools
- Documentation tools
- Editable installation (changes take effect immediately)

---

### Scenario 4: Production Deployment

**Goal**: Deploy to production environment

**Steps**:
```bash
# Install only what's needed
pip install -r requirements.txt

# Or build and install wheel
python -m build
pip install dist/factor_lab_manifold_complete-2.2.0-py3-none-any.whl
```

**Considerations**:
- Pin exact versions for reproducibility
- Use virtual environment or container
- Test thoroughly before deployment

---

## Version Constraints Explained

### Constraint Types

**`>=X.Y.Z`** - Minimum version
```
numpy>=1.21.0  # Must be 1.21.0 or newer
```

**`<X.Y.Z`** - Maximum version (exclusive)
```
numpy<2.0.0  # Can be 1.x.x but not 2.0.0+
```

**`>=X.Y.Z,<W.V.U`** - Range
```
numpy>=1.21.0,<2.0.0  # Between 1.21.0 and 2.0.0
```

### Why These Constraints?

**Core Dependencies**:
- `numpy>=1.21.0,<2.0.0` - NumPy 2.0 has breaking changes
- `scipy>=1.7.0,<2.0.0` - Future-proof against major changes
- `matplotlib>=3.3.0,<4.0.0` - Stable API in 3.x series
- `seaborn>=0.11.0,<1.0.0` - Pre-1.0, conservative range

**Development Tools**:
- More lenient constraints
- Tools, not runtime dependencies
- Less critical for compatibility

---

## pyproject.toml vs requirements.txt

### Use pyproject.toml when:
- ✅ Publishing to PyPI
- ✅ Installing as package: `pip install .`
- ✅ Managing multiple optional dependency sets
- ✅ Following modern Python standards
- ✅ Want single source of truth

### Use requirements.txt when:
- ✅ Pinning exact versions for reproducibility
- ✅ Legacy systems or tools
- ✅ Simple deployment scenarios
- ✅ Direct installation: `pip install -r requirements.txt`
- ✅ Clear separation of dependency tiers

### Best Practice: Use Both!
- **pyproject.toml**: Package metadata and flexible dependencies
- **requirements.txt**: Concrete, reproducible environments
- They complement each other!

---

## Building and Publishing (Optional)

### Build Distribution

```bash
# Install build tool
pip install build

# Build wheel and source distribution
python -m build

# Result:
# dist/
#   factor_lab_manifold_complete-2.2.0-py3-none-any.whl
#   factor_lab_manifold_complete-2.2.0.tar.gz
```

### Test Installation from Wheel

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/factor_lab_manifold_complete-2.2.0-py3-none-any.whl

# Test
python -c "import factor_lab; print(factor_lab.__version__)"
pytest tests/ -v
```

### Publish to PyPI (if desired)

```bash
# Install twine
pip install twine

# Check distribution
twine check dist/*

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Install from Test PyPI and test
pip install --index-url https://test.pypi.org/simple/ factor-lab-manifold-complete

# If all good, upload to real PyPI
twine upload dist/*
```

---

## Quick Reference

### Installation Commands

| Use Case | Command |
|----------|---------|
| **Basic** | `pip install -r requirements.txt` |
| **With Plotly** | `pip install -r requirements.txt -r requirements-optional.txt` |
| **Development** | `pip install -e ".[all]"` |
| **Production** | `pip install .` |

### Build Commands

| Action | Command |
|--------|---------|
| **Build distributions** | `python -m build` |
| **Check package** | `twine check dist/*` |
| **Install locally** | `pip install dist/*.whl` |

### Development Commands

| Action | Command |
|--------|---------|
| **Editable install** | `pip install -e ".[dev]"` |
| **Run tests** | `pytest tests/ -v` |
| **Format code** | `black factor_lab/ tests/` |
| **Lint** | `ruff factor_lab/ tests/` |
| **Type check** | `mypy factor_lab/` |

---

## File Deployment

To deploy these files to the package:

```bash
# From outputs directory
cp pyproject.toml /path/to/factor_lab_manifold_complete/
cp setup.py /path/to/factor_lab_manifold_complete/
cp MANIFEST.in /path/to/factor_lab_manifold_complete/
cp requirements*.txt /path/to/factor_lab_manifold_complete/
cp INSTALL.md /path/to/factor_lab_manifold_complete/
```

---

## Maintenance

### Updating Dependencies

**When to update**:
- Security vulnerabilities
- Bug fixes in dependencies
- New features needed
- Major version releases

**How to update**:
```bash
# Check for outdated packages
pip list --outdated

# Update requirements.txt
vim requirements.txt  # Update version constraints

# Update pyproject.toml
vim pyproject.toml  # Update dependencies section

# Test thoroughly
pip install -e ".[all]"
pytest tests/ -v
```

### Version Bumping

**Update version in**:
1. `pyproject.toml` - `[project] version = "X.Y.Z"`
2. `factor_lab/__init__.py` - `__version__ = "X.Y.Z"`

**Version scheme** (Semantic Versioning):
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes

---

## Best Practices

### ✅ Do:
- Use virtual environments
- Pin versions for production
- Test after any dependency update
- Document breaking changes
- Keep requirements files in sync with pyproject.toml
- Use editable install for development

### ❌ Don't:
- Install system-wide without venv
- Use `pip install --upgrade` blindly
- Mix pip and conda carelessly
- Forget to test after updates
- Use overly broad version constraints

---

## Summary

This packaging setup provides:

✅ **Modern standards** - PEP 621, pyproject.toml  
✅ **Flexibility** - Multiple installation methods  
✅ **Clarity** - Separate dependency tiers  
✅ **Compatibility** - Backward compatible setup.py  
✅ **Developer friendly** - Complete dev tools  
✅ **Production ready** - Clean, tested, documented  

**Status**: Ready to deploy and use immediately!

---

*Created: February 2, 2026*  
*Package Version: 2.2.0*  
*Python: 3.8+*
