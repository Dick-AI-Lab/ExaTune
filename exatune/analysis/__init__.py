"""Landscape analysis and metrics calculation.

Provides tools for quantifying hyperparameter landscape characteristics
including smoothness, multimodality, parameter importance, and interaction effects.
"""

from exatune.analysis.importance import (
    compute_correlation_importance,
    compute_importance_summary,
    compute_range_importance,
    compute_variance_importance,
)
from exatune.analysis.interactions import (
    compute_pairwise_interactions,
    detect_significant_interactions,
)
from exatune.analysis.multimodality import (
    compute_fitness_distance_correlation,
    compute_multimodality_metrics,
    find_local_optima,
)
from exatune.analysis.report import generate_report
from exatune.analysis.smoothness import (
    compute_autocorrelation,
    compute_gradient_statistics,
    compute_ruggedness,
)
from exatune.analysis.summary import compute_experiment_summary, format_summary_text

__all__ = [
    "compute_autocorrelation",
    "compute_gradient_statistics",
    "compute_ruggedness",
    "find_local_optima",
    "compute_multimodality_metrics",
    "compute_fitness_distance_correlation",
    "compute_variance_importance",
    "compute_correlation_importance",
    "compute_range_importance",
    "compute_importance_summary",
    "compute_pairwise_interactions",
    "detect_significant_interactions",
    "compute_experiment_summary",
    "format_summary_text",
    "generate_report",
]
