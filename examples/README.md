# ExaTune Examples

This directory contains comprehensive examples and benchmarks for using ExaTune to explore hyperparameter landscapes on HPC clusters.

## Directory Structure

```
examples/
├── README.md                              # This file
├── basic_sklearn_example.py               # Basic sklearn example (existing)
├── configs/                               # Configuration files
│   ├── random_forest_config.yaml          # Random Forest config (existing)
│   ├── xgboost_config.yaml                # XGBoost config (existing)
│   └── benchmarks/                        # Benchmark configurations
│       ├── iris_classification.yaml       # Iris dataset - Random Forest
│       ├── digits_classification.yaml     # Digits dataset - SVM
│       ├── wine_classification.yaml       # Wine dataset - Gradient Boosting
│       ├── breast_cancer_classification.yaml  # Breast Cancer - XGBoost
│       ├── diabetes_regression.yaml       # Diabetes regression - Random Forest
│       ├── california_housing_regression.yaml # Housing regression - XGBoost
│       └── custom_dataset_template.yaml   # Template for custom datasets
├── scripts/                               # Python scripts
│   ├── create_sample_custom_dataset.py   # Generate synthetic datasets
│   └── benchmarks/                        # Benchmark scripts
│       ├── run_classification_benchmarks.py
│       ├── run_regression_benchmarks.py
│       └── compare_models.py              # Multi-model comparison
└── datasets/                              # Custom dataset directory (empty)
```

---

## Quick Start

### 1. Basic Usage (Built-in Datasets)

Run a simple experiment with a built-in sklearn dataset:

```bash
# Using the basic example
cd examples
python basic_sklearn_example.py

# Using the CLI
exatune run configs/random_forest_config.yaml
```

### 2. Run Benchmark Experiments

#### Classification Benchmarks

```bash
# Run iris benchmark (generate jobs only)
python scripts/benchmarks/run_classification_benchmarks.py iris

# Run and submit breast cancer benchmark
python scripts/benchmarks/run_classification_benchmarks.py breast_cancer --submit

# Run all classification benchmarks
python scripts/benchmarks/run_classification_benchmarks.py all --submit
```

#### Regression Benchmarks

```bash
# Run diabetes regression benchmark
python scripts/benchmarks/run_regression_benchmarks.py diabetes

# Run California housing benchmark
python scripts/benchmarks/run_regression_benchmarks.py california_housing --submit

# Run all regression benchmarks
python scripts/benchmarks/run_regression_benchmarks.py all --submit
```

### 3. Compare Multiple Models

Compare different models on the same dataset:

```bash
# Compare all models on iris dataset
python scripts/benchmarks/compare_models.py iris --submit

# Compare specific models on digits dataset
python scripts/benchmarks/compare_models.py digits --models random_forest xgboost --submit

# After experiments complete, compare results
python scripts/benchmarks/compare_models.py iris --compare-results
```

### 4. Use Custom Datasets

#### Create a Sample Dataset

```bash
# Create synthetic classification dataset
python scripts/create_sample_custom_dataset.py classification \
    --samples 1000 \
    --features 20 \
    --classes 3 \
    --format csv \
    --create-config

# Create synthetic regression dataset
python scripts/create_sample_custom_dataset.py regression \
    --samples 1000 \
    --features 20 \
    --format parquet \
    --create-config
```

#### Use Your Own Dataset

1. Prepare your dataset in CSV or Parquet format
2. Copy and modify the template config:
   ```bash
   cp configs/benchmarks/custom_dataset_template.yaml configs/my_dataset_config.yaml
   ```
3. Edit the config to point to your dataset
4. Run the experiment:
   ```bash
   exatune run configs/my_dataset_config.yaml
   ```

---

## Available Benchmarks

### Classification Benchmarks

| Dataset | Samples | Features | Classes | Model | Est. Grid Size |
|---------|---------|----------|---------|-------|----------------|
| **Iris** | 150 | 4 | 3 | Random Forest | ~1,200 |
| **Digits** | 1,797 | 64 | 10 | SVM | ~480 |
| **Wine** | 178 | 13 | 3 | Gradient Boosting | ~720 |
| **Breast Cancer** | 569 | 30 | 2 | XGBoost | ~15,120 |

### Regression Benchmarks

| Dataset | Samples | Features | Target | Model | Est. Grid Size |
|---------|---------|----------|--------|-------|----------------|
| **Diabetes** | 442 | 10 | Continuous | Random Forest | ~2,520 |
| **California Housing** | 20,640 | 8 | Continuous | XGBoost | ~7,200 |

---

## Configuration Files

### Basic Configuration Structure

All ExaTune configurations follow this structure:

```yaml
experiment:
  name: "experiment_name"
  output_dir: "./results/experiment_name"
  random_seed: 42

model:
  type: "sklearn"  # or "xgboost"
  class_name: "sklearn.ensemble.RandomForestClassifier"
  task: "classification"  # or "regression"
  fixed_params:
    random_state: 42

hyperparameters:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15, null]

dataset:
  name: "iris"  # or path: "/path/to/dataset.csv"
  test_size: 0.2
  random_state: 42

evaluation:
  cv_folds: 5
  cv_strategy: "stratified"  # or "kfold"
  scoring: "accuracy"
  additional_metrics:
    - "f1_macro"
    - "precision_macro"

slurm:
  partition: "compute"
  time: "01:00:00"
  memory: "8G"
  cpus_per_task: 2
```

### Built-in Datasets

ExaTune supports the following sklearn built-in datasets:

**Classification:**
- `iris`: Iris flower species (150 samples, 4 features, 3 classes)
- `digits`: Handwritten digits (1797 samples, 64 features, 10 classes)
- `wine`: Wine recognition (178 samples, 13 features, 3 classes)
- `breast_cancer`: Breast cancer diagnosis (569 samples, 30 features, 2 classes)

**Regression:**
- `diabetes`: Diabetes progression (442 samples, 10 features)
- `california_housing`: California housing prices (20640 samples, 8 features)

### Custom Datasets

To use your own dataset:

1. **Format Requirements:**
   - CSV or Parquet format
   - All numeric features
   - Single target column
   - No missing values (or handle beforehand)

2. **Configuration:**
   ```yaml
   dataset:
     path: "/path/to/your/dataset.csv"
     target_column: "target"
     feature_columns:  # Optional - defaults to all except target
       - "feature1"
       - "feature2"
     test_size: 0.2
     random_state: 42
   ```

3. **Example:**
   See [configs/benchmarks/custom_dataset_template.yaml](configs/benchmarks/custom_dataset_template.yaml)

---

## Model Types and Hyperparameters

### sklearn Models

**Random Forest Classifier/Regressor:**
```yaml
model:
  type: "sklearn"
  class_name: "sklearn.ensemble.RandomForestClassifier"
  task: "classification"

hyperparameters:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15, null]
  min_samples_split: [2, 5, 10]
  min_samples_leaf: [1, 2, 4]
  max_features: ["sqrt", "log2", null]
```

**Gradient Boosting:**
```yaml
model:
  type: "sklearn"
  class_name: "sklearn.ensemble.GradientBoostingClassifier"
  task: "classification"

hyperparameters:
  n_estimators: [50, 100, 200]
  learning_rate: [0.01, 0.1, 0.2]
  max_depth: [3, 5, 7]
  subsample: [0.7, 0.8, 1.0]
```

**Support Vector Machine:**
```yaml
model:
  type: "sklearn"
  class_name: "sklearn.svm.SVC"
  task: "classification"

hyperparameters:
  C: [0.1, 1.0, 10.0]
  kernel: ["rbf", "linear", "poly"]
  gamma: ["scale", "auto"]
```

### XGBoost Models

**XGBoost Classifier:**
```yaml
model:
  type: "xgboost"
  task: "classification"
  fixed_params:
    objective: "binary:logistic"  # or "multi:softprob"
    random_state: 42

hyperparameters:
  n_estimators: [100, 200, 300]
  learning_rate: [0.01, 0.1, 0.2]
  max_depth: [3, 5, 7]
  subsample: [0.7, 0.8, 1.0]
  colsample_bytree: [0.7, 0.8, 1.0]
  gamma: [0, 0.1, 0.5]
  reg_alpha: [0, 0.1, 1.0]
  reg_lambda: [0.1, 1.0, 5.0]
```

**XGBoost Regressor:**
```yaml
model:
  type: "xgboost"
  task: "regression"
  fixed_params:
    objective: "reg:squarederror"
    eval_metric: "rmse"
    tree_method: "hist"  # Faster for large datasets
```

---

## Evaluation Metrics

### Classification Metrics

**Primary Metrics (for `scoring`):**
- `accuracy`: Overall accuracy
- `f1_macro`: F1 score (macro-averaged)
- `f1_weighted`: F1 score (weighted by class support)
- `roc_auc`: Area under ROC curve (binary classification)
- `roc_auc_ovr`: ROC AUC for multi-class (one-vs-rest)

**Additional Metrics:**
- `precision_macro`, `precision_weighted`
- `recall_macro`, `recall_weighted`
- `f1_micro`: F1 score (micro-averaged)

### Regression Metrics

**Primary Metrics:**
- `r2`: R² score (coefficient of determination)
- `neg_mean_squared_error`: Negative MSE
- `neg_root_mean_squared_error`: Negative RMSE
- `neg_mean_absolute_error`: Negative MAE

**Additional Metrics:**
- `explained_variance`: Explained variance score
- `max_error`: Maximum residual error

---

## Workflow Examples

### Complete Workflow: Single Experiment

```bash
# 1. Validate configuration
exatune validate configs/benchmarks/iris_classification.yaml

# 2. Generate and submit jobs
exatune run configs/benchmarks/iris_classification.yaml

# 3. Monitor progress
exatune status --experiment-dir ./results/benchmarks/iris_rf

# 4. Collect results (after jobs complete)
exatune collect --experiment-dir ./results/benchmarks/iris_rf

# 5. Results are saved in:
#    - ./results/benchmarks/iris_rf/results/all_results.parquet
#    - ./results/benchmarks/iris_rf/results/all_results.csv
```

### Complete Workflow: Model Comparison

```bash
# 1. Run comparison (generates and submits jobs for all models)
python scripts/benchmarks/compare_models.py iris \
    --models random_forest xgboost gradient_boosting \
    --submit

# 2. Monitor each model's progress
exatune status --experiment-dir ./results/comparison/iris/random_forest
exatune status --experiment-dir ./results/comparison/iris/xgboost
exatune status --experiment-dir ./results/comparison/iris/gradient_boosting

# 3. After completion, compare results
python scripts/benchmarks/compare_models.py iris \
    --compare-results \
    --models random_forest xgboost gradient_boosting

# 4. Results saved in:
#    - ./results/comparison/iris/comparison_summary.csv
```

### Dry Run (Test Before Submission)

```bash
# Test configuration and job generation without submitting
python scripts/benchmarks/run_classification_benchmarks.py iris --dry-run

# This will:
# - Validate the configuration
# - Generate job scripts
# - Show summary
# - NOT submit jobs to SLURM
```

---

## SLURM Configuration

### Resource Requirements

Default SLURM configurations for each benchmark:

| Benchmark | Time | Memory | CPUs | Notes |
|-----------|------|--------|------|-------|
| Iris | 00:30:00 | 4G | 2 | Small dataset, fast |
| Digits | 01:00:00 | 8G | 2 | SVM can be slow |
| Wine | 01:30:00 | 8G | 2 | Gradient Boosting |
| Breast Cancer | 02:00:00 | 8G | 2 | Large XGBoost grid |
| Diabetes | 01:00:00 | 8G | 2 | Small regression dataset |
| California Housing | 04:00:00 | 16G | 4 | Large dataset (20K samples) |

### Customizing SLURM Resources

Edit the `slurm` section in your config:

```yaml
slurm:
  partition: "compute"      # SLURM partition
  time: "02:00:00"         # Wall time (HH:MM:SS)
  memory: "16G"            # Memory per job
  cpus_per_task: 4         # CPUs per job
  nodes: 1                 # Usually 1 for single-node jobs
  account: "def-username" # Your compute account (optional)
  email: "user@email.com"  # Email for notifications (optional)
  email_type: "FAIL"       # When to email: FAIL, ALL, BEGIN, END
  modules:                 # Environment modules to load
    - "python/3.9"
    - "scipy-stack"
```

---

## Output Structure

After running an experiment, you'll find:

```
results/
└── experiment_name/
    ├── config.yaml                    # Copy of configuration
    ├── metadata.json                  # Experiment metadata
    ├── jobs/                          # Job scripts
    │   ├── job_000000.sh
    │   ├── job_000001.sh
    │   └── ...
    ├── logs/                          # SLURM output logs
    │   ├── job_000000.out
    │   ├── job_000000.err
    │   └── ...
    ├── results/                       # Results
    │   ├── result_000000.json        # Individual job results
    │   ├── result_000001.json
    │   ├── results.parquet           # Aggregated results
    │   ├── all_results.parquet       # Final results with rankings
    │   └── all_results.csv           # CSV format
    └── checkpoints/                   # Checkpoints
        └── after_submission.json
```

---

## Tips and Best Practices

### 1. Start Small

Test with a small hyperparameter grid first:

```yaml
hyperparameters:
  n_estimators: [50, 100]  # Just 2 values
  max_depth: [5, 10]       # Just 2 values
# Grid size: 2 × 2 = 4 configurations
```

### 2. Use Dry Run Mode

Always test with `--dry-run` first:

```bash
python scripts/benchmarks/run_classification_benchmarks.py iris --dry-run
```

### 3. Estimate Grid Size

Calculate grid size before running:
- Grid size = product of all hyperparameter value counts
- Example: 3 parameters with [5, 4, 3] values = 5 × 4 × 3 = 60 configs

### 4. Monitor Resource Usage

After a few jobs complete:
```bash
# Check job efficiency
sacct -j <job_id> --format=JobID,Elapsed,TotalCPU,MaxRSS,State
```

### 5. Handle Large Grids

For grids >1000 configurations:
- Use job arrays: `slurm.array_size: 50` (runs 50 at a time)
- Increase walltime: longer experiments need more time
- Use efficient models: tree-based methods are faster than SVM

### 6. Reproducibility

Always set random seeds:
```yaml
experiment:
  random_seed: 42

dataset:
  random_state: 42

model:
  fixed_params:
    random_state: 42
```

---

## Troubleshooting

### Common Issues

**1. "SLURM not available"**
```bash
# Check if SLURM is installed
which sbatch

# If not on HPC cluster, use local mode (Phase 6 feature - coming soon)
```

**2. "Module not found" errors**
```yaml
# Add Python modules to SLURM config
slurm:
  modules:
    - "python/3.9"
    - "scipy-stack"
```

**3. Jobs fail with "Memory exceeded"**
```yaml
# Increase memory allocation
slurm:
  memory: "16G"  # or higher
```

**4. "Target column not found"**
```yaml
# For custom datasets, specify target column explicitly
dataset:
  path: "/path/to/data.csv"
  target_column: "your_target_column_name"
```

**5. Jobs timeout**
```yaml
# Increase walltime
slurm:
  time: "04:00:00"  # 4 hours
```

---

## Getting Help

- **Documentation:** See [../README.md](../README.md)
- **Issues:** https://github.com/BORN-Ontario/exatune/issues
- **Examples:** This directory contains working examples for all common scenarios

---

## Next Steps

After running benchmarks:

1. **Phase 3: Visualization** (coming soon)
   - Visualize hyperparameter landscapes
   - Create interactive dashboards
   - Generate publication-quality plots

2. **Phase 4: Analysis** (coming soon)
   - Compute landscape metrics
   - Analyze parameter sensitivity
   - Identify optimal regions

3. **Phase 5: Advanced Features** (coming soon)
   - Bayesian optimization
   - Multi-objective optimization
   - Warm-start from previous runs

---

*Last Updated: January 12, 2026*
*ExaTune Version: 0.1.0*
