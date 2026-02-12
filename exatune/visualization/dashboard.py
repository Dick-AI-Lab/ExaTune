"""Multi-panel summary dashboard combining multiple visualization types.

Generates a single comprehensive figure that summarizes the key aspects
of a hyperparameter landscape: score distribution, parameter importance,
slice plots, and heatmap.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from exatune.visualization._utils import (
    get_metric_column,
    infer_hyperparameter_columns,
    save_figure,
    validate_results_dataframe,
)
from exatune.visualization.distributions import plot_score_histogram, plot_train_vs_test
from exatune.visualization.importance import _compute_variance_importance, plot_importance
from exatune.visualization.landscape import plot_heatmap
from exatune.visualization.slices import plot_slice


def plot_dashboard(
    df: pd.DataFrame,
    metric: str = "mean_score",
    params: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (20, 16),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Generate a comprehensive dashboard summarizing the landscape.

    Layout:
    - Top row: Score histogram | Parameter importance
    - Middle row: Slice plots for top 3 most important parameters
    - Bottom row: Heatmap of top 2 params | Train vs test scatter

    Args:
        df: Results DataFrame.
        metric: Primary metric to visualize.
        params: Parameters to include (None = auto-select most important).
        figsize: Figure size.
        output_path: Save path.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, metric=metric)
    metric_col = get_metric_column(filtered, metric)

    if params is None:
        params = infer_hyperparameter_columns(filtered)

    if not params:
        raise ValueError("No hyperparameter columns found.")

    # Rank parameters by importance for layout decisions
    importance = _compute_variance_importance(filtered, metric_col, params)
    sorted_params = list(importance.keys())

    # Determine number of slices (up to 3)
    n_slices = min(3, len(sorted_params))
    has_train = "mean_train_score" in filtered.columns
    has_heatmap = len(sorted_params) >= 2

    # Build grid layout
    # Row 0: histogram + importance (2 cols)
    # Row 1: slices (n_slices cols, spanning)
    # Row 2: heatmap + train-vs-test (2 cols)
    n_rows = 3 if (has_heatmap or has_train) else 2

    fig = plt.figure(figsize=figsize)

    # --- Top row ---
    ax_hist = fig.add_subplot(n_rows, 2, 1)
    plot_score_histogram(filtered, metric=metric, ax=ax_hist, show_kde=True)

    ax_imp = fig.add_subplot(n_rows, 2, 2)
    plot_importance(filtered, metric=metric, method="variance", params=sorted_params, ax=ax_imp)

    # --- Middle row: slice plots ---
    if n_slices > 0:
        for i in range(n_slices):
            ax_slice = fig.add_subplot(n_rows, max(n_slices, 2), max(n_slices, 2) + i + 1)
            plot_slice(filtered, param=sorted_params[i], metric=metric, ax=ax_slice)

    # --- Bottom row ---
    if n_rows == 3:
        col_offset = max(n_slices, 2) * 2
        if has_heatmap:
            ax_heat = fig.add_subplot(n_rows, 2, n_rows * 2 - 1)
            plot_heatmap(
                filtered,
                x_param=sorted_params[0],
                y_param=sorted_params[1],
                metric=metric,
                ax=ax_heat,
                annotate=False,
            )

        if has_train:
            ax_tt = fig.add_subplot(n_rows, 2, n_rows * 2)
            plot_train_vs_test(filtered, metric=metric)
            # We need to plot on existing axes; re-do inline
            train_scores = filtered["mean_train_score"].values
            test_scores = filtered["mean_score"].values
            ax_tt.scatter(test_scores, train_scores, alpha=0.5, color="steelblue", s=20)
            lims = [
                min(train_scores.min(), test_scores.min()) - 0.01,
                max(train_scores.max(), test_scores.max()) + 0.01,
            ]
            ax_tt.plot(lims, lims, "k--", alpha=0.5)
            ax_tt.set_xlabel(f"Test {metric}")
            ax_tt.set_ylabel(f"Train {metric}")
            ax_tt.set_title("Train vs Test")
            ax_tt.set_aspect("equal")
            plt.close(plot_train_vs_test(filtered, metric=metric))

    fig.suptitle("ExaTune Landscape Dashboard", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    return fig
