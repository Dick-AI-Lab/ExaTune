"""Visualization tools for hyperparameter landscapes.

All plotting functions accept a pandas DataFrame (from collect_results()),
return matplotlib Figure objects, and optionally save to file.
"""

from exatune.visualization.dashboard import plot_dashboard
from exatune.visualization.distributions import (
    plot_score_histogram,
    plot_score_violin,
    plot_train_vs_test,
)
from exatune.visualization.importance import plot_importance, plot_interaction_heatmap
from exatune.visualization.landscape import plot_contour, plot_heatmap, plot_surface
from exatune.visualization.parallel import plot_parallel_coordinates
from exatune.visualization.slices import plot_all_slices, plot_conditional_slice, plot_slice

__all__ = [
    "plot_heatmap",
    "plot_surface",
    "plot_contour",
    "plot_slice",
    "plot_all_slices",
    "plot_conditional_slice",
    "plot_importance",
    "plot_interaction_heatmap",
    "plot_score_histogram",
    "plot_score_violin",
    "plot_train_vs_test",
    "plot_parallel_coordinates",
    "plot_dashboard",
]
