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

# Default hyperparameter values to mark on plots
_DEFAULTS = {
    "learning_rate": 0.3,
    "n_estimators": 100,
    "max_depth": 6,
}


def _find_default_index(labels: list, param: str) -> Optional[float]:
    """Return the axis index (float) corresponding to the default value for a
    parameter, or None if the parameter has no registered default or the
    default value is not present among the plotted labels."""
    default_val = _DEFAULTS.get(param)
    if default_val is None:
        return None
    str_labels = [str(v) for v in labels]
    target = str(default_val)
    # Try exact string match first
    if target in str_labels:
        return float(str_labels.index(target))
    # Try numeric comparison in case of float formatting differences
    try:
        numeric_labels = [float(v) for v in str_labels]
        numeric_target = float(default_val)
        for i, v in enumerate(numeric_labels):
            if abs(v - numeric_target) < 1e-9:
                return float(i)
    except (ValueError, TypeError):
        pass
    return None


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

    A star marker (★) is drawn at the cell with the highest metric value.
    A diamond marker (◆) is drawn at the cell corresponding to the default
    hyperparameters (learning_rate=0.3, n_estimators=100, max_depth=6),
    if those parameters are among the axes being plotted.

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

    # --- Peak marker (gold star) ---
    values = pivot.values.astype(float)
    if not np.all(np.isnan(values)):
        peak_idx = np.unravel_index(np.nanargmax(values), values.shape)
        peak_row, peak_col = peak_idx  # row=y-axis, col=x-axis
        ax.plot(
            peak_col, peak_row,
            marker=".", markersize=30, color="black",
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"Peak ({pivot.index[peak_row]}, {pivot.columns[peak_col]})",
            zorder=5,
        )

    # --- Default parameters marker (red diamond) ---
    default_x = _find_default_index(list(pivot.columns), x_param)
    default_y = _find_default_index(list(pivot.index), y_param)
    if default_x is not None and default_y is not None:
        ax.plot(
            default_x, default_y,
            marker=".", markersize=30, color="red",
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"Defaults ({_DEFAULTS.get(x_param)}, {_DEFAULTS.get(y_param)})",
            zorder=5,
        )

    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

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

    A gold star is plotted above the peak metric cell and a red diamond above
    the default hyperparameter cell (learning_rate=0.3, n_estimators=100,
    max_depth=6), where applicable.

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

    # Keep a clean copy of the original (pre-NaN-fill) values for marker placement
    z_orig = pivot.values.astype(float)
    z_range = np.nanmax(z_orig) - np.nanmin(z_orig)
    # Raise markers well above the surface so they are never obscured by it
    z_offset = z_range * 0.12

    # --- Peak marker (gold star) ---
    # nanargmax on z_orig guarantees we mark the true best metric cell,
    # not an artifact of the NaN-fill interpolation used for rendering.
    if not np.all(np.isnan(z_orig)):
        peak_idx = np.unravel_index(np.nanargmax(z_orig), z_orig.shape)
        peak_row, peak_col = peak_idx  # row → y-axis index, col → x-axis index
        peak_z = z_orig[peak_row, peak_col]
        ax.scatter(
            [peak_col], [peak_row], [peak_z + z_offset],
            marker=".", s=5, color="black",
            edgecolors="black", linewidths=1.0,
            label=f"Peak  {x_param}={pivot.columns[peak_col]}, {y_param}={pivot.index[peak_row]}",
            zorder=10,
            depthshade=False,
        )
        # Vertical drop-line from marker down to the surface so it is easy to
        # read off which column / row the peak belongs to
        ax.plot(
            [peak_col, peak_col], [peak_row, peak_row], [peak_z, peak_z + z_offset],
            color="black", linewidth=1.5, linestyle="--", zorder=9,
        )

    # --- Default parameters marker (red diamond) ---
    default_x = _find_default_index(list(pivot.columns), x_param)
    default_y = _find_default_index(list(pivot.index), y_param)
    if default_x is not None and default_y is not None:
        default_x_i = int(round(default_x))
        default_y_i = int(round(default_y))
        default_z = z_orig[default_y_i, default_x_i]
        if not np.isnan(default_z):
            ax.scatter(
                [default_x], [default_y], [default_z + z_offset],
                marker=".", s=5, color="black",
                edgecolors="black", linewidths=1.0,
                label=f"Defaults  {x_param}={_DEFAULTS.get(x_param)}, {y_param}={_DEFAULTS.get(y_param)}",
                zorder=10,
                depthshade=False,
            )
            ax.plot(
                [default_x, default_x], [default_y, default_y], [default_z, default_z + z_offset],
                color="red", linewidth=1.5, linestyle="--", zorder=9,
            )

    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

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

    A black star is drawn at the peak metric location and a red diamond at the
    default hyperparameter location (learning_rate=0.3, n_estimators=100,
    max_depth=6), where applicable.

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

    # --- Peak marker (gold star) ---
    values = pivot.values.astype(float)
    if not np.all(np.isnan(values)):
        peak_idx = np.unravel_index(np.nanargmax(values), values.shape)
        peak_row, peak_col = peak_idx
        ax.plot(
            peak_col, peak_row,
            marker=".", markersize=30, color="black",
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"Peak ({pivot.index[peak_row]}, {pivot.columns[peak_col]})",
            zorder=5,
        )

    # --- Default parameters marker (red diamond) ---
    default_x = _find_default_index(list(pivot.columns), x_param)
    default_y = _find_default_index(list(pivot.index), y_param)
    if default_x is not None and default_y is not None:
        ax.plot(
            default_x, default_y,
            marker=".", markersize=30, color="red",
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"Defaults ({_DEFAULTS.get(x_param)}, {_DEFAULTS.get(y_param)})",
            zorder=5,
        )

    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    if title is None:
        title = f"{metric_col} contours: {x_param} vs {y_param}"
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig