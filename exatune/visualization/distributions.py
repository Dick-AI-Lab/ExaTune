"""Score distribution visualizations.

Histograms, violin plots, and train-vs-test scatter plots for understanding
the distribution of performance across the hyperparameter space.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_figure_and_axes,
    get_metric_column,
    save_figure,
    validate_results_dataframe,
)


def plot_score_histogram(
    df: pd.DataFrame,
    metric: str = "mean_score",
    bins: int = 30,
    show_kde: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot histogram of score distribution across all configurations.

    Args:
        df: Results DataFrame.
        metric: Metric to plot.
        bins: Number of histogram bins.
        show_kde: Whether to overlay a KDE curve (uses scipy if available).
        title: Plot title.
        figsize: Figure size.
        output_path: Save path.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, metric=metric)
    metric_col = get_metric_column(filtered, metric)
    values = filtered[metric_col].dropna().values

    fig, ax = get_figure_and_axes(ax, figsize)

    ax.hist(values, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", density=show_kde)

    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_range, kde(x_range), color="darkred", linewidth=2, label="KDE")
            ax.legend()
        except ImportError:
            pass  # scipy not available, skip KDE

    # Add summary statistics
    mean_val = values.mean()
    ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.4f}")
    ax.axvline(values.max(), color="green", linestyle="--", alpha=0.5, label=f"Max: {values.max():.4f}")

    ax.set_xlabel(metric_col)
    ax.set_ylabel("Density" if show_kde else "Count")
    ax.legend(fontsize=9)

    if title is None:
        title = f"Distribution of {metric_col} ({len(values)} configurations)"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_score_violin(
    df: pd.DataFrame,
    group_by: str,
    metric: str = "mean_score",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot violin plots of score distribution grouped by a hyperparameter.

    Args:
        df: Results DataFrame.
        group_by: Hyperparameter to group by.
        metric: Metric for y-axis.
        title: Plot title.
        figsize: Figure size (auto-calculated if None).
        output_path: Save path.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, param_columns=[group_by], metric=metric)
    metric_col = get_metric_column(filtered, metric)

    groups = sorted(filtered[group_by].unique(), key=lambda x: (isinstance(x, str), x))

    if figsize is None:
        figsize = (max(8, len(groups) * 1.5), 6)

    fig, ax = plt.subplots(figsize=figsize)

    data = [filtered[filtered[group_by] == g][metric_col].dropna().values for g in groups]

    # Filter out empty groups
    valid = [(d, g) for d, g in zip(data, groups) if len(d) > 0]
    if not valid:
        raise ValueError(f"No valid data for violin plot grouped by '{group_by}'.")

    data, groups = zip(*valid)

    parts = ax.violinplot(data, positions=range(len(groups)), showmeans=True, showmedians=True)

    # Color the violins
    for pc in parts.get("bodies", []):
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
    ax.set_xlabel(group_by)
    ax.set_ylabel(metric_col)

    if title is None:
        title = f"{metric_col} distribution by {group_by}"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_train_vs_test(
    df: pd.DataFrame,
    metric: str = "mean_score",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of training score vs test score for each configuration.

    Points above the diagonal indicate overfitting. This is a key
    diagnostic for understanding the generalization landscape.

    Args:
        df: Results DataFrame (must have mean_train_score and mean_score).
        metric: Metric name (for axis labels).
        title: Plot title.
        figsize: Figure size.
        output_path: Save path.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(
        df, required_columns=["mean_train_score", "mean_score"], metric=metric
    )

    train_scores = filtered["mean_train_score"].values
    test_scores = filtered["mean_score"].values

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(test_scores, train_scores, alpha=0.5, color="steelblue", s=30, edgecolors="white", linewidth=0.5)

    # Draw diagonal (perfect generalization line)
    all_vals = np.concatenate([train_scores, test_scores])
    lims = [all_vals.min() - 0.01, all_vals.max() + 0.01]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect generalization")

    ax.set_xlabel(f"Test {metric}")
    ax.set_ylabel(f"Train {metric}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    if title is None:
        title = f"Train vs Test Score ({len(filtered)} configurations)"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
