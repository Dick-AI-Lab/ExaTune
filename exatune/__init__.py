"""
ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC

A Python package for exhaustive hyperparameter landscape exploration using
high-performance computing resources. ExaTune leverages SLURM-orchestrated
HPC clusters to map complete hyperparameter performance landscapes.
"""

from exatune.__version__ import __author__, __email__, __license__, __version__

# Core API exports (will be populated as modules are implemented)
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Version info
VERSION = __version__
