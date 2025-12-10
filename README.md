# ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**ExaTune** is a Python package for exhaustive hyperparameter landscape exploration using high-performance computing (HPC) resources. Rather than settling for approximate optimization methods, ExaTune leverages SLURM-orchestrated HPC clusters to map complete hyperparameter performance landscapes, revealing insights into model stability, parameter sensitivity, and optimization dynamics.

## Motivation

Traditional hyperparameter optimization techniques (grid search, random search, Bayesian optimization, genetic algorithms) balance computational efficiency against exhaustive search, often trading completeness for speed. While these methods can find good solutions, they may overlook:

- **Regions of instability** where small parameter changes cause large performance swings
- **Multiple performance optima** that reveal model behavior patterns
- **Parameter interactions** that affect generalizability
- **Chaotic sensitivity regions** that impact reproducibility

With modern HPC infrastructure, we can now revisit exhaustive exploration at scale. By distributing each hyperparameter configuration as an independent job, ExaTune maps entire performance landscapes, enabling researchers to study them as rich, multi-dimensional surfaces with smooth regions, sharp cliffs, and chaotic pockets.

## Key Features

- **Exhaustive Search**: Generate and evaluate all hyperparameter combinations from configuration files
- **HPC Integration**: Seamless SLURM job submission, monitoring, and result collection
- **ML Framework Support**: Compatible with scikit-learn and XGBoost models
- **Flexible Storage**: Results saved as JSON, Parquet, or SQLite for efficient analysis
- **Rich Visualizations**:
  - 2D heatmaps for parameter pairs
  - 3D surface plots for triple-parameter exploration
  - Interactive dashboards with Plotly/Bokeh
  - Dimensionality reduction (PCA, t-SNE, UMAP) for high-dimensional landscapes
- **Landscape Analysis**: Quantify smoothness, multimodality, parameter sensitivity, and interaction effects
- **Dual Interface**: Both Python API and command-line interface (CLI)
- **Production Ready**: Type-checked, tested, and documented for research and production use

## Installation

### From PyPI (when released)

```bash
pip install exatune
```

### From Source

```bash
git clone https://github.com/BORN-Ontario/exatune.git
cd exatune
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/BORN-Ontario/exatune.git
cd exatune
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### 1. Define Your Experiment Configuration

Create a YAML configuration file (`config.yaml`):

```yaml
experiment:
  name: "random_forest_iris"
  output_dir: "./results"

model:
  type: "sklearn"
  class: "RandomForestClassifier"

hyperparameters:
  n_estimators: [10, 50, 100, 200]
  max_depth: [3, 5, 10, 15, null]
  min_samples_split: [2, 5, 10]
  min_samples_leaf: [1, 2, 4]

dataset:
  name: "iris"
  test_size: 0.2
  random_state: 42

evaluation:
  cv_folds: 5
  scoring: "accuracy"

slurm:
  partition: "compute"
  time: "00:30:00"
  memory: "4G"
  cpus_per_task: 2
```

### 2. Run via Python API

```python
from exatune import Experiment

# Create and run experiment
experiment = Experiment.from_config("config.yaml")
experiment.submit_jobs()
experiment.monitor_progress()
results = experiment.collect_results()

# Visualize landscape
from exatune.visualization import plot_heatmap, plot_surface

plot_heatmap(results, x="n_estimators", y="max_depth", metric="accuracy")
plot_surface(results, x="n_estimators", y="max_depth", z="min_samples_split")
```

### 3. Run via CLI

```bash
# Submit experiment
exatune run config.yaml

# Monitor progress
exatune status random_forest_iris

# Collect results
exatune collect random_forest_iris

# Generate visualizations
exatune visualize random_forest_iris --params n_estimators max_depth

# Analyze landscape
exatune analyze random_forest_iris --metrics smoothness multimodality sensitivity
```

## Use Cases

### Healthcare Research

In healthcare ML applications, small improvements in predictive performance can translate into significant clinical impact:

- **Early Disease Detection**: Identify optimal models for rare condition screening
- **Risk Stratification**: Precisely tune models for patient risk prediction
- **Resource Allocation**: Optimize predictive models for efficient resource distribution

ExaTune's comprehensive landscape view strengthens reproducibility and interpretability, ensuring models are performant, trustworthy, and resilient across diverse patient populations.

### General ML Research

- **Benchmark Studies**: Systematically compare algorithms across parameter spaces
- **Model Behavior Analysis**: Understand how hyperparameters affect stability and generalization
- **Optimization Method Evaluation**: Use complete landscapes as ground truth for testing new optimization algorithms
- **Publication-Quality Results**: Generate comprehensive visualizations and quantitative landscape metrics

## Architecture

```
exatune/
├── core/           # Experiment orchestration and configuration
├── models/         # ML model wrappers (scikit-learn, XGBoost)
├── hpc/            # SLURM integration and job management
├── storage/        # Result storage backends (JSON, Parquet, SQLite)
├── visualization/  # Plotting and interactive dashboards
├── analysis/       # Landscape metrics and sensitivity analysis
└── cli/            # Command-line interface
```

## Requirements

- Python 3.8+
- Access to SLURM HPC cluster (for distributed execution)
- scikit-learn and/or XGBoost (depending on models used)

## Documentation

Full documentation is available at [Read the Docs](https://exatune.readthedocs.io) (coming soon).

- [Installation Guide](docs/installation.rst)
- [Quick Start Tutorial](docs/quickstart.rst)
- [HPC Setup Guide](docs/hpc_setup.rst)
- [Configuration Reference](docs/configuration.rst)
- [API Reference](docs/api_reference.rst)
- [Examples](docs/examples.rst)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/exatune.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`
6. Make your changes and add tests
7. Run tests: `pytest`
8. Submit a pull request

## Citation

If you use ExaTune in your research, please cite:

```bibtex
@software{exatune2025,
  title = {ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC},
  author = {Lonsway, Katie and Dick, Kevin and Howley, Heather},
  year = {2025},
  organization = {BORN Ontario},
  url = {https://github.com/BORN-Ontario/exatune}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed at BORN Ontario as part of the Fall 2025 Internship Program
- Intern: Katie Lonsway
- Supervisors: Dr. Kevin Dick, Heather Howley
- HPC resources provided by Digital Research Alliance of Canada (DRAC)

## Contact

- Issues: [GitHub Issues](https://github.com/BORN-Ontario/exatune/issues)
- Email: kevin.dick@bornontario.ca

---

**Note**: This project is under active development. The API may change before the 1.0 release.
