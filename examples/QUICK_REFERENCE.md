# ExaTune Examples - Quick Reference

Quick command reference for common tasks.

## Run Benchmarks

### Classification
```bash
# Single benchmark
python scripts/benchmarks/run_classification_benchmarks.py iris --submit

# All classification benchmarks
python scripts/benchmarks/run_classification_benchmarks.py all --submit
```

### Regression
```bash
# Single benchmark
python scripts/benchmarks/run_regression_benchmarks.py diabetes --submit

# All regression benchmarks
python scripts/benchmarks/run_regression_benchmarks.py all --submit
```

## Model Comparison
```bash
# Compare models (4 models on iris)
python scripts/benchmarks/compare_models.py iris --submit

# Compare specific models
python scripts/benchmarks/compare_models.py digits --models random_forest xgboost --submit

# Compare results after completion
python scripts/benchmarks/compare_models.py iris --compare-results
```

## Custom Datasets

### Create Sample Dataset
```bash
# Classification
python scripts/create_sample_custom_dataset.py classification \
    --samples 1000 --features 20 --classes 3 --create-config

# Regression
python scripts/create_sample_custom_dataset.py regression \
    --samples 1000 --features 20 --create-config
```

### Use Your Dataset
```bash
# 1. Edit template config
cp configs/benchmarks/custom_dataset_template.yaml configs/my_config.yaml

# 2. Update dataset path and parameters in my_config.yaml

# 3. Run experiment
exatune run configs/my_config.yaml
```

## Monitor and Collect

```bash
# Check status
exatune status --experiment-dir ./results/experiment_name

# Collect results
exatune collect --experiment-dir ./results/experiment_name
```

## Dry Run (Test First)
```bash
# Test without submitting
python scripts/benchmarks/run_classification_benchmarks.py iris --dry-run
```

## Available Benchmarks

### Classification
- `iris` - 150 samples, 3 classes, Random Forest
- `digits` - 1,797 samples, 10 classes, SVM
- `wine` - 178 samples, 3 classes, Gradient Boosting
- `breast_cancer` - 569 samples, 2 classes, XGBoost

### Regression
- `diabetes` - 442 samples, Random Forest
- `california_housing` - 20,640 samples, XGBoost

### Models (for comparison)
- `random_forest`
- `gradient_boosting`
- `svm`
- `xgboost`

## File Locations

- **Configs**: `configs/benchmarks/*.yaml`
- **Scripts**: `scripts/benchmarks/*.py`
- **Results**: `results/benchmarks/<dataset>/<model>/`
- **Comparison**: `results/comparison/<dataset>/<model>/`
