"""
ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC

A Python package for exhaustive hyperparameter landscape exploration using
high-performance computing resources. ExaTune leverages SLURM-orchestrated
HPC clusters to map complete hyperparameter performance landscapes.
"""

from exatune.__version__ import __author__, __email__, __license__, __version__
from exatune.core.config import ExaTuneConfig
from exatune.core.experiment import Experiment
from exatune.core.grid_generator import GridGenerator

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "ExaTuneConfig",
    "Experiment",
    "GridGenerator",
]

VERSION = __version__
