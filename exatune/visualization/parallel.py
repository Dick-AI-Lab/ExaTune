"""Parallel coordinates visualization for high-dimensional landscapes.

Shows all hyperparameter configurations as lines across parallel axes,
colored by performance metric. Essential for understanding relationships
in spaces with more than 2-3 hyperparameters.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_metric_column,
    infer_hyperparameter_columns,
    save_figure,
    validate_results_dataframe,
)


def plot_parallel_coordinates(
    df: pd.DataFrame,
    params: Optional[List[str]] = None,
    metric: str = "mean_score",
    top_n: Optional[int] = None,
    highlight_best: int = 5,
    alpha: float = 0.3,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot parallel coordinates showing all hyperparameter configurations.

    Each line represents one configuration. Lines are colored by metric value.
    Best configurations are highlighted with thicker lines.

    Args:
        df: Results DataFrame.
        params: Which parameters to include (None = all hyperparameters).
        metric: Metric for coloring lines.
        top_n: If set, only show the top N configurations.
        highlight_best: Number of best configs to highlight.
        alpha: Transparency for non-highlighted lines.
        cmap: Colormap for metric coloring.
        title: Plot title.
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

    # Sort by metric (descending) and optionally limit
    plot_df = filtered.sort_values(metric_col, ascending=False).reset_index(drop=True)
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    # Encode parameters to [0, 1] range
    encoded = pd.DataFrame()
    param_ticks = {}  # Store tick positions and labels per param

    for param in params:
        col = plot_df[param]
        unique_vals = sorted(col.unique(), key=lambda x: (isinstance(x, str), x))

        if len(unique_vals) == 1:
            encoded[param] = 0.5
            param_ticks[param] = {0.5: str(unique_vals[0])}
        else:
            # Map values to evenly spaced positions in [0, 1]
            val_to_pos = {v: i / (len(unique_vals) - 1) for i, v in enumerate(unique_vals)}
            encoded[param] = col.map(val_to_pos)
            param_ticks[param] = {pos: str(val) for val, pos in val_to_pos.items()}

    # Normalize metric for coloring
    metric_vals = plot_df[metric_col].values
    if metric_vals.max() > metric_vals.min():
        norm_metric = (metric_vals - metric_vals.min()) / (metric_vals.max() - metric_vals.min())
    else:
        norm_metric = np.ones_like(metric_vals) * 0.5

    if cmap is None:
        cmap = "viridis"
    colormap = plt.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each configuration as a line
    x_positions = range(len(params))
    best_indices = list(range(min(highlight_best, len(plot_df))))

    for idx in range(len(plot_df) - 1, -1, -1):  # Draw worst first so best is on top
        y_vals = [encoded[param].iloc[idx] for param in params]
        color = colormap(norm_metric[idx])
        lw = 2.5 if idx in best_indices else 0.8
        a = 0.9 if idx in best_indices else alpha
        ax.plot(x_positions, y_vals, color=color, linewidth=lw, alpha=a)

    # Set axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=10)

    # Add parameter value labels on each axis
    for i, param in enumerate(params):
        ticks = param_ticks[param]
        for pos, label in ticks.items():
            ax.annotate(
                label,
                xy=(i, pos),
                xytext=(-5, 0),
                textcoords="offset points",
                fontsize=7,
                ha="right",
                va="center",
                alpha=0.7,
            )

    # Draw vertical axis lines
    for i in x_positions:
        ax.axvline(i, color="gray", linewidth=0.5, alpha=0.3)

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized parameter value")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(metric_vals.min(), metric_vals.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=metric_col, shrink=0.8)

    if title is None:
        title = f"Parallel Coordinates ({len(plot_df)} configurations)"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
