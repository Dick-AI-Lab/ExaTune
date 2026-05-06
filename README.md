# ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**ExaTune** is a Python package for exhaustive hyperparameter landscape exploration using high-performance computing (HPC) resources. Rather than settling for approximate optimization methods, ExaTune leverages SLURM-orchestrated HPC clusters to map complete hyperparameter performance landscapes, revealing insights into model stability, parameter sensitivity, and optimization dynamics.

## Motivation

Traditional hyperparameter optimization techniques (grid search, random search, Bayesian optimization) balance computational efficiency against exhaustive coverage. While these methods find good solutions, they may overlook:

- **Regions of instability** where small parameter changes cause large performance swings
- **Multiple performance optima** that reveal model behavior patterns
- **Parameter interactions** that affect generalizability
- **Chaotic sensitivity regions** that impact reproducibility

With modern HPC infrastructure, we can revisit exhaustive exploration at scale. ExaTune distributes each hyperparameter configuration as an independent SLURM job, mapping entire performance landscapes as rich, multi-dimensional surfaces.

## Key Features

- **Exhaustive Search**: Generate and evaluate all hyperparameter combinations from YAML configuration files
- **HPC Integration**: Seamless SLURM job submission, monitoring, and result collection with Compute Canada module support
- **ML Framework Support**: scikit-learn and XGBoost model wrappers
- **Flexible Storage**: Results saved as JSON (per-job) and Parquet (aggregated)
- **Rich Visualizations**: Heatmaps, 3D surfaces, contour plots, parallel coordinates, slice plots, importance charts, and summary dashboards
- **Landscape Analysis**: Quantify smoothness (Moran's I, ruggedness), multimodality (local optima, fitness-distance correlation), parameter importance (variance/correlation/range), and interaction effects
- **Dual Interface**: Python API and Click-based CLI
- **Comprehensive Testing**: 350+ tests across 23 test files

## Installation

### From Source

```bash
git clone https://github.com/BORN-Ontario/exatune.git
cd exatune
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Visualization Extras

```bash
pip install -e ".[viz]"     # adds seaborn, plotly, bokeh, umap-learn
pip install -e ".[all]"     # everything
```

### On Compute Canada / DRAC Clusters

Compute Canada intercepts packages like numpy, scipy, and pyarrow with dummy wheels. Load modules first:

```bash
module load gcc/11.3.0 python/3.10 scipy-stack/2023b arrow/14.0.1
python -m venv --system-site-packages ~/exatune_env
source ~/exatune_env/bin/activate
pip install -e .
```

Add modules to your ExaTune config so SLURM jobs load them too:

```yaml
slurm:
  modules: ["gcc/11.3.0", "python/3.10", "scipy-stack/2023b", "arrow/14.0.1"]
  python_environment: "~/exatune_env"
```

## Quick Start

### 1. Define Your Experiment

Create `config.yaml`:

```yaml
experiment:
  name: "random_forest_iris"
  output_dir: "./results"

model:
  type: "sklearn"
  class: "RandomForestClassifier"
  task: "classification"

hyperparameters:
  n_estimators: [10, 50, 100, 200]
  max_depth: [3, 5, 10, 15, null]
  min_samples_split: [2, 5, 10]

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
experiment.print_summary()
experiment.generate_jobs()
job_ids = experiment.submit_jobs()
experiment.monitor_progress()
results = experiment.collect_results()
best = experiment.get_best_config()
```

### 3. Visualize Results

```python
from exatune.visualization import (
    plot_dashboard, plot_heatmap, plot_importance,
    plot_parallel_coordinates, plot_all_slices,
)

# Comprehensive dashboard
plot_dashboard(results, metric="mean_score", output_path="dashboard.png")

# 2D heatmap
plot_heatmap(results, x_param="n_estimators", y_param="max_depth",
             metric="mean_score", output_path="heatmap.png")

# Parameter importance
plot_importance(results, method="variance", output_path="importance.png")

# Parallel coordinates
plot_parallel_coordinates(results, highlight_best=10, output_path="parallel.png")
```

### 4. Analyze Landscape

```python
from exatune.analysis import (
    compute_experiment_summary, format_summary_text,
    find_local_optima, compute_autocorrelation,
)

# Full analysis
summary = compute_experiment_summary(results, direction="maximize")
print(format_summary_text(summary))

# Specific analyses
optima = find_local_optima(results, direction="maximize")
smoothness = compute_autocorrelation(results)
```

### 5. Run via CLI

```bash
exatune validate config.yaml           # Validate configuration
exatune run config.yaml --dry-run      # Generate jobs without submitting
exatune run config.yaml                # Submit to SLURM
exatune status random_forest_iris      # Check progress
exatune collect random_forest_iris     # Aggregate results
exatune visualize random_forest_iris --type dashboard
exatune analyze random_forest_iris --report ./analysis_output
```

## Architecture

```
exatune/
├── core/           # Configuration (Pydantic), experiment orchestration, grid generation
├── models/         # ML model wrappers (scikit-learn, XGBoost)
├── hpc/            # SLURM client, worker script, Jinja2 job templates
├── storage/        # JSON and Parquet storage backends
├── visualization/  # Heatmaps, surfaces, contours, slices, parallel coords, dashboard
├── analysis/       # Smoothness, multimodality, importance, interactions, reports
└── cli/            # Click-based CLI (run, status, collect, visualize, analyze, validate)
```

## Configuration Reference

### Hyperparameter Specification

Discrete values:
```yaml
n_estimators: [10, 50, 100, 200]
max_features: ["sqrt", "log2", null]
```

Continuous ranges:
```yaml
learning_rate:
  min: 0.001
  max: 1.0
  step: 0.1
  type: "float"
  scale: "log"      # or "linear"
```

### Built-in Datasets

`iris`, `digits`, `wine`, `breast_cancer`, `diabetes`, `california_housing`

### Custom Datasets

```yaml
dataset:
  path: "/path/to/data.csv"
  target_column: "target"
  test_size: 0.2
```

## Analysis Metrics

| Category | Metric | What It Answers |
|----------|--------|----------------|
| Smoothness | Moran's I | Is the landscape smooth enough for local search? |
| Smoothness | Ruggedness index | How variable are neighboring configurations? |
| Multimodality | Local optima count | How many peaks exist? |
| Multimodality | Fitness-distance correlation | Is the landscape deceptive? |
| Importance | Variance-based (eta-squared) | Which parameters explain the most variance? |
| Importance | Spearman correlation | Which parameters have monotonic effects? |
| Interactions | Pairwise ANOVA | Which parameter pairs interact? |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/BORN-Ontario/exatune.git
cd exatune
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@software{exatune2026,
  title = {ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC},
  author = {Dick, Kevin and Lonsway, Katie},
  year = {2026},
  organization = {BORN Ontario},
  url = {https://github.com/Dick-AI-Lab/ExaTune}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
