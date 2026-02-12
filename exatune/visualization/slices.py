"""Slice plots: 1D performance curves varying one parameter at a time.

These plots show how performance changes as a single hyperparameter varies,
marginalizing (averaging) over all other hyperparameters. This reveals
the main effect of each parameter.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_figure_and_axes,
    get_metric_column,
    infer_hyperparameter_columns,
    save_figure,
    validate_results_dataframe,
)


def plot_slice(
    df: pd.DataFrame,
    param: str,
    metric: str = "mean_score",
    show_individual: bool = False,
    show_std: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot performance vs one hyperparameter (marginalizing over others).

    For each unique value of the parameter, aggregates the metric over all
    other hyperparameter values, showing mean and optionally std band.

    Args:
        df: Results DataFrame.
        param: Hyperparameter to plot on x-axis.
        metric: Metric for y-axis.
        show_individual: Whether to show individual data points.
        show_std: Whether to show +/- std shading.
        title: Plot title.
        figsize: Figure size.
        output_path: Save path.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, param_columns=[param], metric=metric)
    metric_col = get_metric_column(filtered, metric)

    grouped = filtered.groupby(param)[metric_col]
    means = grouped.mean()
    stds = grouped.std()

    fig, ax = get_figure_and_axes(ax, figsize)

    x_positions = range(len(means))
    x_labels = [str(v) for v in means.index]

    ax.plot(x_positions, means.values, "o-", color="steelblue", linewidth=2, markersize=8)

    if show_std and stds is not None:
        ax.fill_between(
            x_positions,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.2,
            color="steelblue",
        )

    if show_individual:
        for i, val in enumerate(means.index):
            points = filtered[filtered[param] == val][metric_col]
            jitter = np.random.normal(0, 0.05, size=len(points))
            ax.scatter(
                [i + j for j in jitter],
                points.values,
                alpha=0.3,
                color="gray",
                s=15,
                zorder=1,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(param)
    ax.set_ylabel(metric_col)

    if title is None:
        title = f"{metric_col} vs {param}"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_all_slices(
    df: pd.DataFrame,
    params: Optional[List[str]] = None,
    metric: str = "mean_score",
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot slice plots for all (or specified) hyperparameters in a grid.

    Args:
        df: Results DataFrame.
        params: Parameters to plot (None = all inferred hyperparameters).
        metric: Metric for y-axis.
        ncols: Number of columns in subplot grid.
        figsize_per_plot: Size per individual subplot.
        output_path: Save path.

    Returns:
        matplotlib Figure with subplot grid.
    """
    filtered = validate_results_dataframe(df, metric=metric)
    metric_col = get_metric_column(filtered, metric)

    if params is None:
        params = infer_hyperparameter_columns(filtered)

    if not params:
        raise ValueError("No hyperparameter columns found to plot.")

    nrows = (len(params) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for idx, param in enumerate(params):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        grouped = filtered.groupby(param)[metric_col]
        means = grouped.mean()

        x_positions = range(len(means))
        ax.plot(x_positions, means.values, "o-", color="steelblue", linewidth=1.5, markersize=6)

        stds = grouped.std()
        if stds is not None:
            ax.fill_between(
                x_positions,
                means.values - stds.values,
                means.values + stds.values,
                alpha=0.2,
                color="steelblue",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(v) for v in means.index], rotation=45, ha="right", fontsize=7)
        ax.set_title(param, fontsize=10)
        ax.set_ylabel(metric_col, fontsize=8)

    # Hide unused subplots
    for idx in range(len(params), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Hyperparameter Slice Plots ({metric_col})", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_conditional_slice(
    df: pd.DataFrame,
    param: str,
    condition_param: str,
    metric: str = "mean_score",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot performance vs one parameter, colored/grouped by another.

    This reveals interaction effects: if the lines are parallel,
    the parameters do not interact; if they cross, they interact strongly.

    Args:
        df: Results DataFrame.
        param: Primary parameter (x-axis).
        condition_param: Conditioning parameter (color/group).
        metric: Metric for y-axis.
        title: Plot title.
        figsize: Figure size.
        output_path: Save path.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(
        df, param_columns=[param, condition_param], metric=metric
    )
    metric_col = get_metric_column(filtered, metric)

    fig, ax = plt.subplots(figsize=figsize)

    condition_values = sorted(filtered[condition_param].unique(), key=lambda x: (isinstance(x, str), x))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(condition_values), 10)))

    for i, cval in enumerate(condition_values):
        subset = filtered[filtered[condition_param] == cval]
        grouped = subset.groupby(param)[metric_col].mean()
        x_positions = range(len(grouped))
        ax.plot(
            x_positions,
            grouped.values,
            "o-",
            color=colors[i % len(colors)],
            label=f"{condition_param}={cval}",
            linewidth=1.5,
            markersize=6,
        )
        # Set consistent x-ticks from first series
        if i == 0:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(v) for v in grouped.index], rotation=45, ha="right")

    ax.set_xlabel(param)
    ax.set_ylabel(metric_col)
    ax.legend(fontsize=8, loc="best")

    if title is None:
        title = f"{metric_col} vs {param} (by {condition_param})"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
