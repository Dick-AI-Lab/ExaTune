# ExaTune Examples - Index

Quick navigation for all example files and documentation.

## 📚 Documentation

- **[README.md](README.md)** - Complete examples documentation (start here!)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page command reference
- **[EXAMPLES_SUMMARY.md](EXAMPLES_SUMMARY.md)** - Implementation details and statistics

## 🚀 Quick Start

New to ExaTune? Start here:

1. Read [README.md](README.md) - Section: "Quick Start"
2. Try: `python scripts/benchmarks/run_classification_benchmarks.py iris --dry-run`
3. Review the generated jobs in `./results/benchmarks/iris_rf/jobs/`
4. Submit: `python scripts/benchmarks/run_classification_benchmarks.py iris --submit`

## 📝 Configuration Files

### Benchmark Configs ([configs/benchmarks/](configs/benchmarks/))

#### Classification
- **[iris_classification.yaml](configs/benchmarks/iris_classification.yaml)** - Iris dataset, Random Forest (150 samples)
- **[digits_classification.yaml](configs/benchmarks/digits_classification.yaml)** - Digits dataset, SVM (1,797 samples)
- **[wine_classification.yaml](configs/benchmarks/wine_classification.yaml)** - Wine dataset, Gradient Boosting (178 samples)
- **[breast_cancer_classification.yaml](configs/benchmarks/breast_cancer_classification.yaml)** - Breast Cancer, XGBoost (569 samples)

#### Regression
- **[diabetes_regression.yaml](configs/benchmarks/diabetes_regression.yaml)** - Diabetes dataset, Random Forest (442 samples)
- **[california_housing_regression.yaml](configs/benchmarks/california_housing_regression.yaml)** - Housing dataset, XGBoost (20,640 samples)

#### Templates
- **[custom_dataset_template.yaml](configs/benchmarks/custom_dataset_template.yaml)** - Template for custom datasets with detailed comments

### Basic Configs ([configs/](configs/))
- **[random_forest_config.yaml](configs/random_forest_config.yaml)** - Basic Random Forest example
- **[xgboost_config.yaml](configs/xgboost_config.yaml)** - Basic XGBoost example

## 🐍 Python Scripts

### Benchmark Scripts ([scripts/benchmarks/](scripts/benchmarks/))

- **[run_classification_benchmarks.py](scripts/benchmarks/run_classification_benchmarks.py)** - Run classification benchmarks
  ```bash
  python scripts/benchmarks/run_classification_benchmarks.py iris --submit
  ```

- **[run_regression_benchmarks.py](scripts/benchmarks/run_regression_benchmarks.py)** - Run regression benchmarks
  ```bash
  python scripts/benchmarks/run_regression_benchmarks.py diabetes --submit
  ```

- **[compare_models.py](scripts/benchmarks/compare_models.py)** - Compare multiple models
  ```bash
  python scripts/benchmarks/compare_models.py iris --submit
  ```

### Utility Scripts ([scripts/](scripts/))

- **[create_sample_custom_dataset.py](scripts/create_sample_custom_dataset.py)** - Generate synthetic datasets
  ```bash
  python scripts/create_sample_custom_dataset.py classification --create-config
  ```

### Basic Examples
- **[basic_sklearn_example.py](basic_sklearn_example.py)** - Simple sklearn example

## 📊 Benchmark Reference

### Classification Benchmarks

| Name | Command | Dataset | Samples | Model | Grid Size |
|------|---------|---------|---------|-------|-----------|
| iris | `... iris` | Iris | 150 | Random Forest | ~1,200 |
| digits | `... digits` | Digits | 1,797 | SVM | ~480 |
| wine | `... wine` | Wine | 178 | Gradient Boosting | ~720 |
| breast_cancer | `... breast_cancer` | Breast Cancer | 569 | XGBoost | ~15,120 |

### Regression Benchmarks

| Name | Command | Dataset | Samples | Model | Grid Size |
|------|---------|---------|---------|-------|-----------|
| diabetes | `... diabetes` | Diabetes | 442 | Random Forest | ~2,520 |
| california_housing | `... california_housing` | CA Housing | 20,640 | XGBoost | ~7,200 |

## 🎯 Common Tasks

### Run a Single Benchmark
```bash
cd examples
python scripts/benchmarks/run_classification_benchmarks.py iris --submit
```

### Run All Benchmarks
```bash
# Classification
python scripts/benchmarks/run_classification_benchmarks.py all --submit

# Regression
python scripts/benchmarks/run_regression_benchmarks.py all --submit
```

### Compare Models
```bash
# Run comparison
python scripts/benchmarks/compare_models.py iris --submit

# View results after completion
python scripts/benchmarks/compare_models.py iris --compare-results
```

### Create Custom Dataset
```bash
python scripts/create_sample_custom_dataset.py classification \
    --samples 1000 --features 20 --classes 3 --create-config
```

### Dry Run (Test First)
```bash
python scripts/benchmarks/run_classification_benchmarks.py iris --dry-run
```

## 📁 Output Locations

After running experiments, find results at:

```
results/
├── benchmarks/
│   ├── iris_rf/               # Iris benchmark results
│   ├── digits_svm/            # Digits benchmark results
│   └── ...
└── comparison/
    ├── iris/                  # Model comparison results
    │   ├── random_forest/
    │   ├── xgboost/
    │   └── comparison_summary.csv
    └── ...
```

## 🔍 Find What You Need

**Want to...**

- **Get started quickly?** → [README.md](README.md) "Quick Start" section
- **See all commands?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Understand implementation?** → [EXAMPLES_SUMMARY.md](EXAMPLES_SUMMARY.md)
- **Run classification?** → [run_classification_benchmarks.py](scripts/benchmarks/run_classification_benchmarks.py)
- **Run regression?** → [run_regression_benchmarks.py](scripts/benchmarks/run_regression_benchmarks.py)
- **Compare models?** → [compare_models.py](scripts/benchmarks/compare_models.py)
- **Use custom data?** → [custom_dataset_template.yaml](configs/benchmarks/custom_dataset_template.yaml)
- **Create test data?** → [create_sample_custom_dataset.py](scripts/create_sample_custom_dataset.py)

## 🆘 Getting Help

1. Check [README.md](README.md) "Troubleshooting" section
2. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for correct commands
3. Validate config: `exatune validate your_config.yaml`
4. Try dry run: `python script.py ... --dry-run`
5. Check main documentation: [../README.md](../README.md)

## 📚 Related Documentation

- **[../README.md](../README.md)** - Main ExaTune documentation
- **[../INSTALL.md](../INSTALL.md)** - Installation guide
- **[../QUICKSTART.md](../QUICKSTART.md)** - ExaTune quickstart
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contributing guide
- **[../IMPLEMENTATION_PROGRESS.md](../IMPLEMENTATION_PROGRESS.md)** - Project status

---

*Quick Index - ExaTune Examples*
*Last Updated: January 12, 2026*
