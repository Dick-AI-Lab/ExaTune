"""Landscape visualization: 2D heatmaps, 3D surface plots, and contour plots.

These plots show performance metric values over two hyperparameter dimensions,
providing direct insight into the shape of the hyperparameter landscape.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_default_colormap,
    get_figure_and_axes,
    get_metric_column,
    pivot_for_heatmap,
    save_figure,
    validate_results_dataframe,
)


def plot_heatmap(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot a 2D heatmap of metric values over two hyperparameters.

    When more than two hyperparameters exist, non-displayed parameters
    are aggregated using agg_func.

    Args:
        df: Results DataFrame from collect_results().
        x_param: Hyperparameter for x-axis.
        y_param: Hyperparameter for y-axis.
        metric: Metric column name or shorthand.
        agg_func: Aggregation for extra dimensions ('mean', 'max', 'min', 'std').
        title: Plot title (auto-generated if None).
        cmap: Matplotlib colormap name.
        annotate: Whether to show values in cells.
        figsize: Figure size in inches.
        output_path: Optional file path to save figure.
        ax: Optional existing Axes for multi-panel layouts.

    Returns:
        matplotlib Figure object.
    """
    filtered = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)

    pivot = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    if cmap is None:
        cmap = get_default_colormap()

    fig, ax = get_figure_and_axes(ax, figsize)

    im = ax.imshow(
        pivot.values,
        cmap=cmap,
        aspect="auto",
        origin="lower",
    )

    # Set tick labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    # Annotate cells
    if annotate and pivot.size <= 200:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    text_color = "white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label=metric_col)

    if title is None:
        title = f"{metric_col} by {x_param} vs {y_param} ({agg_func})"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_surface(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 9),
    elev: float = 30.0,
    azim: float = -60.0,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a 3D surface of metric values over two hyperparameters.

    Args:
        df: Results DataFrame.
        x_param: Hyperparameter for x-axis.
        y_param: Hyperparameter for y-axis.
        metric: Metric to plot on z-axis.
        agg_func: Aggregation function for extra dimensions.
        title: Plot title.
        cmap: Colormap.
        figsize: Figure size.
        elev: Camera elevation angle.
        azim: Camera azimuth angle.
        output_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)

    pivot = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    if cmap is None:
        cmap = get_default_colormap()

    fig, ax = get_figure_and_axes(figsize=figsize, projection="3d")

    # Create meshgrid from pivot table indices
    x_vals = np.arange(len(pivot.columns))
    y_vals = np.arange(len(pivot.index))
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pivot.values.astype(float)

    # Fill NaN with interpolation or nearest for surface rendering
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        from scipy.interpolate import griddata

        valid = ~mask
        points = np.array(np.where(valid)).T
        values = Z[valid]
        fill_points = np.array(np.where(mask)).T
        if len(fill_points) > 0 and len(points) > 0:
            filled = griddata(points, values, fill_points, method="nearest")
            Z[mask] = filled

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, edgecolor="none")

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, fontsize=7)
    ax.set_yticks(y_vals)
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=7)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(metric_col)
    ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, ax=ax, shrink=0.5, label=metric_col)

    if title is None:
        title = f"{metric_col} landscape: {x_param} vs {y_param}"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_contour(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    levels: int = 15,
    filled: bool = True,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot contour lines of metric over two hyperparameters.

    Contour plots reveal ridges, plateaus, and gradient directions
    in the performance landscape.

    Args:
        df: Results DataFrame.
        x_param: x-axis parameter.
        y_param: y-axis parameter.
        metric: Metric name.
        agg_func: Aggregation function.
        levels: Number of contour levels.
        filled: Use filled contours (contourf) vs line contours (contour).
        title: Plot title.
        cmap: Colormap.
        figsize: Figure size.
        output_path: Save path.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    filtered = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)

    pivot = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    if cmap is None:
        cmap = get_default_colormap()

    fig, ax = get_figure_and_axes(ax, figsize)

    x_vals = np.arange(len(pivot.columns))
    y_vals = np.arange(len(pivot.index))
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pivot.values.astype(float)

    plot_func = ax.contourf if filled else ax.contour
    cs = plot_func(X, Y, Z, levels=levels, cmap=cmap)

    if not filled:
        ax.clabel(cs, inline=True, fontsize=8)

    fig.colorbar(cs, ax=ax, label=metric_col)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(y_vals)
    ax.set_yticklabels([str(v) for v in pivot.index])

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    if title is None:
        title = f"{metric_col} contours: {x_param} vs {y_param}"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
