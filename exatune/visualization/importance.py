"""Parameter importance visualization.

Bar charts showing how much each hyperparameter affects performance,
and interaction heatmaps showing pairwise parameter dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def _compute_variance_importance(
    df: pd.DataFrame, metric_col: str, param_columns: List[str]
) -> Dict[str, float]:
    """Compute variance-based importance for visualization."""
    total_var = df[metric_col].var()
    if total_var == 0:
        return {p: 0.0 for p in param_columns}

    importance = {}
    for param in param_columns:
        group_means = df.groupby(param)[metric_col].mean()
        between_var = group_means.var() * len(group_means) / len(df)
        importance[param] = float(between_var / total_var) if total_var > 0 else 0.0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def _compute_correlation_importance(
    df: pd.DataFrame, metric_col: str, param_columns: List[str]
) -> Dict[str, float]:
    """Compute correlation-based importance for visualization."""
    importance = {}
    for param in param_columns:
        try:
            series = pd.to_numeric(df[param], errors="coerce")
            valid = series.notna()
            if valid.sum() > 2:
                corr = series[valid].corr(df.loc[valid, metric_col], method="spearman")
                importance[param] = abs(float(corr)) if pd.notna(corr) else 0.0
            else:
                importance[param] = 0.0
        except (TypeError, ValueError):
            importance[param] = 0.0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def _compute_range_importance(
    df: pd.DataFrame, metric_col: str, param_columns: List[str]
) -> Dict[str, float]:
    """Compute range-based importance for visualization."""
    total_range = df[metric_col].max() - df[metric_col].min()
    if total_range == 0:
        return {p: 0.0 for p in param_columns}

    importance = {}
    for param in param_columns:
        group_means = df.groupby(param)[metric_col].mean()
        param_range = group_means.max() - group_means.min()
        importance[param] = float(param_range / total_range)

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def plot_importance(
    df: pd.DataFrame,
    metric: str = "mean_score",
    method: str = "variance",
    params: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot hyperparameter importance as a horizontal bar chart.

    Args:
        df: Results DataFrame.
        metric: Metric to analyze.
        method: Importance method ('variance', 'correlation', 'range').
        params: Specific parameters to include (None = all).
        title: Plot title.
        figsize: Figure size.
        output_path: Save path.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, metric=metric)
    metric_col = get_metric_column(filtered, metric)

    if params is None:
        params = infer_hyperparameter_columns(filtered)

    if not params:
        raise ValueError("No hyperparameter columns found.")

    compute_funcs = {
        "variance": _compute_variance_importance,
        "correlation": _compute_correlation_importance,
        "range": _compute_range_importance,
    }

    if method not in compute_funcs:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(compute_funcs.keys())}")

    importance = compute_funcs[method](filtered, metric_col, params)

    fig, ax = get_figure_and_axes(ax, figsize)

    # Sort by importance (ascending for horizontal bars, so largest is on top)
    names = list(importance.keys())
    values = list(importance.values())

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Importance ({method})")

    if title is None:
        title = f"Hyperparameter Importance ({method})"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_interaction_heatmap(
    df: pd.DataFrame,
    metric: str = "mean_score",
    params: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a heatmap of pairwise parameter interaction strengths.

    Interaction strength is the additional variance explained by the pair
    beyond the sum of their individual variances.

    Args:
        df: Results DataFrame.
        metric: Metric to analyze.
        params: Parameters to include.
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

    if len(params) < 2:
        raise ValueError("Need at least 2 hyperparameters for interaction analysis.")

    total_var = filtered[metric_col].var()
    n_params = len(params)
    interaction_matrix = np.zeros((n_params, n_params))

    # Compute individual variances
    individual_vars = {}
    for param in params:
        group_means = filtered.groupby(param)[metric_col].mean()
        individual_vars[param] = group_means.var() * len(group_means) / len(filtered)

    # Compute pairwise joint variances
    for i in range(n_params):
        for j in range(i + 1, n_params):
            p1, p2 = params[i], params[j]
            joint_means = filtered.groupby([p1, p2])[metric_col].mean()
            joint_var = joint_means.var() * len(joint_means) / len(filtered)
            interaction = max(0, joint_var - individual_vars[p1] - individual_vars[p2])
            if total_var > 0:
                interaction_matrix[i, j] = interaction / total_var
                interaction_matrix[j, i] = interaction / total_var

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(interaction_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(params)

    # Annotate
    for i in range(n_params):
        for j in range(n_params):
            val = interaction_matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Interaction Strength")

    if title is None:
        title = "Pairwise Hyperparameter Interactions"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
