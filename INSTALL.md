# ExaTune Installation Guide

## Quick Installation

### Option 1: Install from Source (Recommended)

```bash
# Navigate to the ExaTune directory
cd /path/to/exatune

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install ExaTune with all dependencies
pip install -e .

# Verify installation
python -c "import exatune; print(f'ExaTune {exatune.__version__} installed successfully!')"
```

### Option 2: Install with Development Dependencies

```bash
# Same as above, but install with dev dependencies
pip install -e ".[dev]"

# This includes testing, linting, and documentation tools
```

### Option 3: Install Only Core Dependencies

```bash
# Install just the requirements
pip install -r requirements.txt

# Then install ExaTune in development mode
pip install -e . --no-deps
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'rich'`

**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Or install ExaTune which will pull dependencies
pip install -e .
```

### Issue: `ModuleNotFoundError: No module named 'exatune'`

**Solution:**
```bash
# Install ExaTune in editable mode
pip install -e .
```

### Issue: Permission errors during installation

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -e .

# OR use --user flag (not recommended)
pip install --user -e .
```

### Issue: `pip` command not found

**Solution:**
```bash
# Use python -m pip instead
python3 -m pip install -e .
```

## Step-by-Step Installation (Detailed)

### 1. Check Python Version

```bash
python3 --version
# Should be Python 3.8 or higher
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv exatune_env

# Activate it
source exatune_env/bin/activate  # macOS/Linux
# OR
exatune_env\Scripts\activate  # Windows

# Verify activation (should show venv path)
which python
```

### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install ExaTune

```bash
# Navigate to ExaTune directory
cd /path/to/exatune

# Install with all dependencies
pip install -e .

# This will install:
# - Core dependencies from requirements.txt
# - ExaTune package in editable mode
# - Command-line tool 'exatune'
```

### 5. Verify Installation

```bash
# Check ExaTune is installed
pip list | grep exatune

# Check command-line tool
exatune --version

# Run test suite
python test_exatune_toy.py
```

## Dependencies

### Core Dependencies (Automatically Installed)

- **scikit-learn** (≥1.0.0) - Machine learning models
- **xgboost** (≥1.5.0) - Gradient boosting models
- **numpy** (≥1.20.0) - Numerical computing
- **pandas** (≥1.3.0) - Data manipulation
- **pyarrow** (≥6.0.0) - Parquet file support
- **pydantic** (≥2.0.0) - Configuration validation
- **pyyaml** (≥6.0) - YAML parsing
- **jinja2** (≥3.0.0) - Template rendering
- **joblib** (≥1.1.0) - Model serialization
- **click** (≥8.0.0) - CLI framework
- **rich** (≥10.0.0) - Rich text output
- **tqdm** (≥4.60.0) - Progress bars
- **matplotlib** (≥3.4.0) - Plotting

### Optional Dependencies

Install with: `pip install -e ".[viz]"`
- **plotly** (≥5.0.0) - Interactive visualizations
- **bokeh** (≥2.4.0) - Interactive dashboards
- **umap-learn** (≥0.5.0) - Dimensionality reduction
- **seaborn** (≥0.11.0) - Statistical visualizations

### Development Dependencies

Install with: `pip install -e ".[dev]"`
- **pytest** (≥7.0.0) - Testing framework
- **pytest-cov** (≥3.0.0) - Coverage reporting
- **black** (≥22.0.0) - Code formatting
- **isort** (≥5.10.0) - Import sorting
- **flake8** (≥4.0.0) - Linting
- **mypy** (≥0.950) - Type checking
- **pre-commit** (≥2.17.0) - Git hooks

## Testing Your Installation

### Quick Test

```bash
python -c "
import exatune
from exatune.core.config import ExaTuneConfig
from exatune.core.grid_generator import GridGenerator
print('✅ All imports successful!')
print(f'✅ ExaTune version: {exatune.__version__}')
"
```

### Comprehensive Test

```bash
# Run the test suite
python test_exatune_toy.py

# Expected output: 7/7 tests passed (100%)
```

### Test CLI

```bash
# Test CLI is available
exatune --version

# Validate example configuration
exatune validate examples/configs/random_forest_config.yaml
```

## Common Installation Patterns

### For Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### For Production Use

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### For Experimentation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[viz]"
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # or exatune_env, or whatever you named it

# If installed with --user flag
pip uninstall exatune
```

## Next Steps After Installation

1. **Read the Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
2. **Run Examples**: Try `python examples/basic_sklearn_example.py`
3. **Test Functionality**: Run `python test_exatune_toy.py`
4. **Read Documentation**: See [README.md](README.md) for full documentation

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify Python version: `python3 --version` (need ≥3.8)
3. Check if in virtual environment: `which python`
4. Try clean install: Delete venv, recreate, reinstall
5. Check GitHub issues: [Report a bug](https://github.com/BORN-Ontario/exatune/issues)

## Installation on Compute Canada / Digital Research Alliance of Canada

Compute Canada clusters (Cedar, Graham, Narval, Beluga, Niagara) use a custom
wheel system that intercepts packages like `numpy`, `scipy`, `pyarrow`,
`matplotlib`, and `scikit-learn` with dummy wheels. These packages must be loaded
as **environment modules** before creating your virtual environment.

### Step 1: Load Required Modules

```bash
# Load the compiler, Python, and scientific stack
module load gcc/11.3.0 python/3.10 scipy-stack/2023b arrow/14.0.1

# Verify modules are loaded
module list
```

> **Important:** The exact module versions available depend on your cluster.
> Use `module spider python` and `module spider arrow` to find available versions.

The `scipy-stack` module provides: `numpy`, `scipy`, `pandas`, `scikit-learn`,
`matplotlib`, `joblib`, and more. The `arrow` module provides `pyarrow`.

### Step 2: Create Virtual Environment with System Site Packages

```bash
# --system-site-packages lets the venv see module-provided packages
python -m venv --system-site-packages ~/exatune_env
```

### Step 3: Activate and Install

```bash
source ~/exatune_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install ExaTune, skipping packages already provided by modules
pip install -e . --no-build-isolation
```

If pip still tries to install a module-provided package and fails, install
without the problematic dependency and add it manually:

```bash
pip install -e . --no-deps
pip install pydantic pyyaml jinja2 click rich tqdm xgboost
```

### Step 4: Verify Installation

```bash
python -c "
import pyarrow; print(f'PyArrow {pyarrow.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
import numpy; print(f'NumPy {numpy.__version__}')
import exatune; print(f'ExaTune {exatune.__version__}')
print('All imports successful!')
"
```

### Step 5: Add Modules to Your Job Scripts

When submitting SLURM jobs, the same modules must be loaded. Add them to your
ExaTune config:

```yaml
slurm:
  modules:
    - "gcc/11.3.0"
    - "python/3.10"
    - "scipy-stack/2023b"
    - "arrow/14.0.1"
  python_environment: "~/exatune_env"
```

Or, if using the CLI directly, ensure modules are loaded in your SLURM script
before activating the virtual environment.

### Compute Canada Troubleshooting

**Error: `dummy wheel` / `metadata-generation-failed` for pyarrow, numpy, scipy, etc.**

This means the package is provided by a Compute Canada module. Load the
appropriate module and recreate your venv with `--system-site-packages`:

| Package | Module to Load |
|---------|---------------|
| `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib` | `scipy-stack` |
| `pyarrow` | `arrow` |
| `xgboost` | Usually installs via pip (no module needed) |

```bash
deactivate  # if venv is active
module load gcc python scipy-stack arrow
python -m venv --system-site-packages ~/exatune_env
source ~/exatune_env/bin/activate
pip install -e .
```

**Error: Module version conflicts**

Use `module spider <package>` to see available versions and their dependencies:

```bash
module spider arrow
module spider scipy-stack
```

Some module versions require specific `gcc` or `python` versions. The `spider`
output will tell you which combinations are valid.

**Error: `pip install` hangs or is very slow**

On login nodes with limited resources, builds can be slow. Use `--no-build-isolation`
or request an interactive job:

```bash
salloc --time=0:30:00 --mem=4G --cpus-per-task=2
module load gcc python scipy-stack arrow
source ~/exatune_env/bin/activate
pip install -e .
```

**Ensuring consistency between login and compute nodes**

All Compute Canada clusters use a shared filesystem (`/home`, `/project`,
`/scratch`), so packages installed in your venv on the login node are available
on compute nodes — as long as the same modules are loaded.

### Recommended `.bashrc` / Module Setup

Add this to your `~/.bashrc` or create a setup script:

```bash
# File: ~/setup_exatune.sh
module load gcc/11.3.0 python/3.10 scipy-stack/2023b arrow/14.0.1
source ~/exatune_env/bin/activate
```

Then use: `source ~/setup_exatune.sh`

---

## System Requirements

- **Python**: 3.8 or higher
- **OS**: macOS, Linux, or Windows
- **Memory**: 2GB minimum, 8GB recommended
- **Disk Space**: ~500MB for full installation with dependencies

For HPC/SLURM usage:
- Access to a SLURM cluster
- Shared filesystem across compute nodes
- Python environment accessible from compute nodes
