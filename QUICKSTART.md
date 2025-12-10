# ExaTune Quick Start Guide

This guide will help you get started with ExaTune in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Access to a SLURM HPC cluster

## Installation

### Step 1: Clone or Navigate to Repository

```bash
cd /path/to/exatune
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install ExaTune

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"
```

This will install:
- ExaTune package
- All core dependencies (scikit-learn, xgboost, pandas, etc.)
- Development tools (pytest, black, mypy, etc.)
- Visualization libraries (matplotlib, plotly, etc.)

### Step 4: Verify Installation

```bash
exatune version
```

You should see: `ExaTune version 0.1.0`

## Quick Example: Random Forest on Iris

### Option 1: Using Pre-configured YAML

```bash
# Validate the configuration
exatune validate examples/configs/random_forest_config.yaml

# This will show you the experiment summary including:
# - Model configuration
# - Hyperparameter grid size
# - Expected number of SLURM jobs
```

### Option 2: Using Python API

Run the example script:

```bash
python examples/basic_sklearn_example.py
```

This will:
1. Create an experiment configuration
2. Generate job scripts
3. Save configuration for later use

## Understanding the Configuration

Let's look at a minimal configuration:

```yaml
experiment:
  name: "my_experiment"
  output_dir: "./results/my_experiment"

model:
  type: "sklearn"  # or "xgboost"
  class: "RandomForestClassifier"
  task: "classification"

hyperparameters:
  n_estimators: [10, 50, 100]
  max_depth: [3, 5, 10]

dataset:
  name: "iris"  # Built-in dataset
  test_size: 0.2

evaluation:
  cv_folds: 5
  scoring: "accuracy"

slurm:
  partition: "compute"
  time: "00:30:00"
  memory: "4G"
```

This creates 3 × 3 = 9 hyperparameter combinations, each trained with 5-fold cross-validation.

## Running an Experiment

### Step 1: Create Configuration

Create a file `my_config.yaml` with your experiment settings (see above).

### Step 2: Validate Configuration

```bash
exatune validate my_config.yaml
```

This checks for:
- Valid YAML syntax
- Required fields present
- Correct parameter types
- Reasonable values

### Step 3: Generate Jobs

```bash
exatune run my_config.yaml --dry-run
```

The `--dry-run` flag generates job scripts without submitting them. This is useful for:
- Checking generated scripts
- Testing without SLURM
- Estimating resource requirements

### Step 4: Submit to SLURM (When Ready)

```bash
exatune run my_config.yaml
```

This will:
1. Generate job scripts
2. Submit to SLURM
3. Save job IDs for monitoring

### Step 5: Monitor Progress

```bash
exatune status my_experiment
```

This shows:
- Number of pending/running/completed jobs
- Estimated time to completion
- Any failed jobs

### Step 6: Collect Results

```bash
exatune collect my_experiment
```

This will:
1. Aggregate all job results
2. Create a DataFrame
3. Save as Parquet and CSV
4. Display best configuration

## Directory Structure After Running

```
results/
└── my_experiment/
    ├── config.yaml           # Your configuration
    ├── metadata.json         # Experiment metadata
    ├── jobs/                 # SLURM job scripts
    │   ├── job_000000.sh
    │   ├── job_000001.sh
    │   └── ...
    ├── logs/                 # SLURM output logs
    │   ├── job_000000_12345.out
    │   └── ...
    ├── results/              # Collected results
    │   ├── all_results.parquet
    │   ├── all_results.csv
    │   └── result_*.json     # Individual results
    └── checkpoints/          # Experiment checkpoints
        └── latest.json
```

## Working Without SLURM

If you don't have access to SLURM, you can still use ExaTune:

1. **Generate jobs in dry-run mode:**
   ```bash
   exatune run my_config.yaml --dry-run
   ```

2. **Inspect generated scripts:**
   ```bash
   cat results/my_experiment/jobs/job_000000.sh
   ```

3. **Run locally (for small grids):**
   - Use the Python API
   - Run jobs sequentially on your machine
   - Modify scripts to remove SLURM directives

## Creating Your Own Experiment

### 1. Choose Your Model

**scikit-learn:**
```yaml
model:
  type: "sklearn"
  class: "RandomForestClassifier"  # or SVC, LogisticRegression, etc.
  task: "classification"
```

**XGBoost:**
```yaml
model:
  type: "xgboost"
  class: "XGBClassifier"  # or XGBRegressor
  task: "classification"
```

### 2. Define Hyperparameter Space

**Simple (list of values):**
```yaml
hyperparameters:
  n_estimators: [50, 100, 200]
  max_depth: [3, 5, 10, null]
```

**Advanced (with ranges):**
```yaml
hyperparameters:
  learning_rate:
    min: 0.01
    max: 0.3
    step: 0.05
    type: "float"
    scale: "log"
  max_depth:
    min: 3
    max: 10
    step: 1
    type: "int"
    scale: "linear"
```

### 3. Configure Your Dataset

**Built-in dataset:**
```yaml
dataset:
  name: "iris"  # or breast_cancer, diabetes, etc.
  test_size: 0.2
```

**Custom dataset:**
```yaml
dataset:
  path: "/path/to/your/data.csv"
  target_column: "target"
  test_size: 0.2
```

### 4. Set SLURM Resources

```yaml
slurm:
  partition: "compute"       # Your cluster's partition
  time: "01:00:00"          # HH:MM:SS format
  memory: "4G"              # Memory per job
  cpus_per_task: 2          # CPUs per job
  account: "your_account"   # If required
  email: "you@example.com"  # For notifications
  email_type: "FAIL"        # When to email
```

## Common Commands

```bash
# Validate configuration
exatune validate my_config.yaml

# Run experiment (dry run)
exatune run my_config.yaml --dry-run

# Submit to SLURM
exatune run my_config.yaml

# Check status
exatune status my_experiment

# Collect results
exatune collect my_experiment

# Generate visualizations (coming soon)
exatune visualize my_experiment --params n_estimators max_depth

# Analyze landscape (coming soon)
exatune analyze my_experiment
```

## Tips & Best Practices

1. **Start Small:** Begin with a small grid (< 100 combinations) to test
2. **Use Dry Run:** Always use `--dry-run` first to check job scripts
3. **Validate First:** Run `exatune validate` before submission
4. **Monitor Progress:** Check `exatune status` periodically
5. **Set Realistic Time Limits:** Estimate job duration and add buffer
6. **Use Array Jobs:** For large grids, SLURM array jobs are more efficient

## Getting Help

```bash
# Show all commands
exatune --help

# Help for specific command
exatune run --help
```

## Troubleshooting

### "Configuration file not found"
- Check file path is correct
- Use absolute path if relative path doesn't work

### "SLURM is not available"
- Ensure you're on a SLURM cluster
- Check `sbatch --version` works
- Use `--dry-run` to test without SLURM

### "Module not found: exatune"
- Ensure you installed with `pip install -e .`
- Activate your virtual environment
- Check `pip list | grep exatune`

### Jobs not submitting
- Check SLURM partition name is correct
- Verify account/QOS if required
- Check resource limits (time, memory)

## Next Steps

1. **Read the README:** Full documentation in [README.md](README.md)
2. **Explore Examples:** Check [examples/](examples/) directory
3. **Read Project Description:** See the original PDF for research context
4. **Check Project Status:** Review [PROJECT_STATUS.md](PROJECT_STATUS.md)
5. **Start Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)

## Example Workflow

```bash
# 1. Create your configuration
nano my_experiment.yaml

# 2. Validate it
exatune validate my_experiment.yaml

# 3. Dry run to check
exatune run my_experiment.yaml --dry-run

# 4. Check the generated scripts
ls -l results/my_experiment/jobs/

# 5. Submit to SLURM
exatune run my_experiment.yaml

# 6. Monitor (wait for completion)
exatune status my_experiment

# 7. Collect results
exatune collect my_experiment

# 8. Analyze (inspect CSV/Parquet files)
head results/my_experiment/results/all_results.csv
```

Happy hyperparameter tuning! 🚀
