"""Landscape visualizations: heatmaps, 2-D/3-D surfaces, and contour plots.

Shows how performance metrics vary across the hyperparameter space.

Marker conventions (consistent across all plot types)
------------------------------------------------------
  Peak    → #D63384  (magenta/pink)  — highest-performing grid cell
  Default → #111D4F  (dark navy)     — sklearn/XGBoost library defaults

Heatmap
  Both markers are **square box outlines** drawn around the cell.

3-D static surface (matplotlib)
  Both markers are **filled circles** floating above the surface, each
  connected to its base by a dashed vertical drop-line.

3-D interactive surface (Plotly)  →  landscape_3d.py
  Same circle + drop-line style, matching the matplotlib version.

Model defaults are resolved automatically from the experiment YAML via
:mod:`exatune.visualization._defaults`.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_metric_column,
    pivot_for_heatmap,
    save_figure,
    validate_results_dataframe,
)
from exatune.visualization._defaults import (
    get_defaults,
    get_defaults_for_model,
    find_default_index,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PEAK_COLOR    = "#D63384"   # magenta/pink  — best cell
_DEFAULT_COLOR = "#111D4F"   # dark navy     — library default


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_defaults(config_or_defaults) -> Dict[str, Any]:
    """Accept either a config object/dict *or* an already-resolved defaults dict."""
    if config_or_defaults is None:
        return {}
    if isinstance(config_or_defaults, dict):
        # Could be a raw defaults dict OR a raw YAML dict — distinguish by
        # checking for a 'model' key (YAML) vs plain param keys.
        if "model" in config_or_defaults or "experiment" in config_or_defaults:
            return get_defaults(config_or_defaults)
        return config_or_defaults   # already a defaults dict
    # ExaTuneConfig object
    return get_defaults(config_or_defaults)


def _find_cell(pivot: pd.DataFrame, x_param: str, y_param: str,
               defaults: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Return (col_idx, row_idx) of the default cell, or (None, None)."""
    col_i = find_default_index(list(pivot.columns), x_param, defaults)
    row_i = find_default_index(list(pivot.index),   y_param, defaults)
    return col_i, row_i


def _draw_square_marker(ax: matplotlib.axes.Axes, col: int, row: int,
                        color: str, linewidth: float = 2.5) -> None:
    """Draw a bold square outline around heatmap cell (col, row)."""
    rect = mpatches.FancyBboxPatch(
        (col - 0.47, row - 0.47),
        0.94, 0.94,
        boxstyle="square,pad=0",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    annotate: bool = True,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    # Model defaults — pass an ExaTuneConfig, raw YAML dict, or pre-built
    # defaults dict.  If None the default marker is simply omitted.
    config=None,
) -> matplotlib.figure.Figure:
    """Plot a 2-D heatmap of *metric* over *x_param* x *y_param*.

    Peak cell    → bold #D63384 **square outline** + value label
    Default cell → bold #111D4F **square outline** + value label

    Args:
        df:          Results DataFrame.
        x_param:     Hyperparameter for the x-axis.
        y_param:     Hyperparameter for the y-axis.
        metric:      Metric column to visualise.
        agg_func:    Aggregation per cell ('mean', 'max', …).
        annotate:    Draw square markers when True.
        colormap:    Matplotlib colormap name.
        title:       Auto-generated when None.
        figsize:     Figure size in inches.
        output_path: Optional save path.
        ax:          Optional existing axes to draw on.
        config:      ExaTuneConfig, raw YAML dict, or defaults dict used to
                     locate the default marker.  None → no default marker.

    Returns:
        matplotlib Figure.
    """
    filtered   = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)
    pivot      = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    Z        = pivot.values.astype(float)
    n_rows, n_cols = Z.shape
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]

    defaults = _resolve_defaults(config)

    # --- figure / axes ---
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(Z, aspect="auto", origin="upper",
                   cmap=colormap, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_col, fontsize=11)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)

    if annotate:
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        z_mid = (z_min + z_max) / 2.0

        # Value label on every cell
        for row in range(n_rows):
            for col in range(n_cols):
                val = Z[row, col]
                if np.isnan(val):
                    continue
                text_color = "white" if val < z_mid else "#1a1a1a"
                ax.text(col, row, f"{val:.3f}",
                        ha="center", va="center",
                        color=text_color, fontsize=7, zorder=3)

        legend_handles = []

        # --- Peak marker (square outline) ---
        peak_row = peak_col = None
        if not np.all(np.isnan(Z)):
            peak_row, peak_col = np.unravel_index(np.nanargmax(Z), Z.shape)
            _draw_square_marker(ax, peak_col, peak_row, _PEAK_COLOR)
            legend_handles.append(
                mpatches.Patch(edgecolor=_PEAK_COLOR, facecolor="none",
                               linewidth=2.5, label="Peak")
            )

        # --- Default marker (square outline) ---
        col_i, row_i = _find_cell(pivot, x_param, y_param, defaults)
        if col_i is not None and row_i is not None:
            default_val = Z[row_i, col_i]
            if not np.isnan(default_val):
                _draw_square_marker(ax, col_i, row_i, _DEFAULT_COLOR)
                label = "Default"
                # Check whether default IS the peak
                if peak_row is not None and row_i == peak_row and col_i == peak_col:
                    label = "Default (= Peak)"
                legend_handles.append(
                    mpatches.Patch(edgecolor=_DEFAULT_COLOR, facecolor="none",
                                   linewidth=2.5, label=label)
                )

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.18, 0.18),
                bbox_transform=ax.transAxes,
                framealpha=0.9,
                fontsize=9,
                borderaxespad=0,
            )

    if title is None:
        title = f"{metric_col} heatmap: {x_param} vs {y_param}"
    ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# plot_surface  (static matplotlib 3-D)
# ---------------------------------------------------------------------------

def plot_surface(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[Union[str, Path]] = None,
    config=None,
    draw_markers: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a static 3-D surface of *metric* over *x_param* x *y_param*.

    Peak marker    → #D63384 filled circle + dashed vertical drop-line.
    Default marker → #111D4F filled circle + dashed vertical drop-line.
    Legend labels both clearly.

    Args:
        df:          Results DataFrame.
        x_param:     Hyperparameter for the x-axis.
        y_param:     Hyperparameter for the y-axis.
        metric:      Metric column for the z-axis.
        agg_func:    Aggregation per cell.
        colormap:    Matplotlib colormap name.
        title:       Auto-generated when None.
        figsize:     Figure size in inches.
        output_path: Save path.
        config:      ExaTuneConfig / YAML dict / defaults dict for the
                     default marker.  None → no default marker.

    Returns:
        matplotlib Figure.
    """
    filtered   = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)
    pivot      = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    Z        = pivot.values.astype(float)
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)
    X, Y     = np.meshgrid(range(len(x_labels)), range(len(y_labels)))

    defaults = _resolve_defaults(config)

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric_col)

    z_range  = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.12

    legend_items = []   # (handle_artist, label)

    if draw_markers:
        # --- Peak marker: filled circle + dashed drop-line ---
        peak_row = peak_col = None
        if not np.all(np.isnan(Z)):
            peak_row, peak_col = np.unravel_index(np.nanargmax(Z), Z.shape)
            peak_z = Z[peak_row, peak_col]
            marker_z = peak_z + z_offset

            sc_peak = ax.scatter([peak_col], [peak_row], [marker_z],
                                 color=_PEAK_COLOR, s=100, zorder=6,
                                 edgecolors="white", linewidths=0.8)
            ax.plot([peak_col, peak_col], [peak_row, peak_row],
                    [peak_z, marker_z],
                    color=_PEAK_COLOR, linewidth=2, linestyle="--")
            legend_items.append((sc_peak, "Peak"))

        # --- Default marker: filled circle + dashed drop-line ---
        col_i, row_i = _find_cell(pivot, x_param, y_param, defaults)
        if col_i is not None and row_i is not None:
            default_z = Z[row_i, col_i]
            if not np.isnan(default_z):
                marker_z_d = default_z + z_offset
                sc_def = ax.scatter([col_i], [row_i], [marker_z_d],
                                    color=_DEFAULT_COLOR, s=100, zorder=6,
                                    edgecolors="white", linewidths=0.8)
                ax.plot([col_i, col_i], [row_i, row_i],
                        [default_z, marker_z_d],
                        color=_DEFAULT_COLOR, linewidth=2, linestyle="--")
                label = "Default"
                if peak_row is not None and row_i == peak_row and col_i == peak_col:
                    label = "Default (= Peak)"
                legend_items.append((sc_def, label))

        if legend_items:
            handles, labels = zip(*legend_items)
            ax.legend(handles, labels, fontsize=9, loc="upper left")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param, fontsize=10)
    ax.set_ylabel(y_param, fontsize=10)
    ax.set_zlabel(metric_col, fontsize=10)

    if title is None:
        title = f"{metric_col} surface: {x_param} vs {y_param}"
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# plot_contour
# ---------------------------------------------------------------------------

def plot_contour(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    levels: int = 15,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[Union[str, Path]] = None,
    config=None,
) -> matplotlib.figure.Figure:
    """Plot a filled contour map of *metric* over *x_param* x *y_param*.

    Peak marker    → #D63384 star.
    Default marker → #111D4F diamond.

    Args:
        df:          Results DataFrame.
        x_param:     Hyperparameter for the x-axis.
        y_param:     Hyperparameter for the y-axis.
        metric:      Metric column.
        agg_func:    Aggregation per cell.
        levels:      Number of contour levels.
        colormap:    Matplotlib colormap name.
        title:       Auto-generated when None.
        figsize:     Figure size in inches.
        output_path: Save path.
        config:      ExaTuneConfig / YAML dict / defaults dict.

    Returns:
        matplotlib Figure.
    """
    filtered   = validate_results_dataframe(df, param_columns=[x_param, y_param], metric=metric)
    metric_col = get_metric_column(filtered, metric)
    pivot      = pivot_for_heatmap(filtered, x_param, y_param, metric_col, agg_func)

    Z        = pivot.values.astype(float)
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)
    X, Y     = np.meshgrid(range(len(x_labels)), range(len(y_labels)))

    defaults = _resolve_defaults(config)

    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(X, Y, Z, levels=levels, cmap=colormap)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label=metric_col)

    legend_handles = []

    # Peak marker
    peak_row = peak_col = None
    if not np.all(np.isnan(Z)):
        peak_row, peak_col = np.unravel_index(np.nanargmax(Z), Z.shape)
        sc = ax.scatter([peak_col], [peak_row], color=_PEAK_COLOR, s=150,
                        marker="*", zorder=5,
                        edgecolors="white", linewidths=0.5)
        legend_handles.append(
            matplotlib.lines.Line2D([], [], color=_PEAK_COLOR, marker="*",
                                    linestyle="None", markersize=12, label="Peak")
        )

    # Default marker
    col_i, row_i = _find_cell(pivot, x_param, y_param, defaults)
    if col_i is not None and row_i is not None:
        label = "Default"
        if peak_row is not None and row_i == peak_row and col_i == peak_col:
            label = "Default (= Peak)"
        ax.scatter([col_i], [row_i], color=_DEFAULT_COLOR, s=120,
                   marker="D", zorder=5,
                   edgecolors="white", linewidths=0.5)
        legend_handles.append(
            matplotlib.lines.Line2D([], [], color=_DEFAULT_COLOR, marker="D",
                                    linestyle="None", markersize=9, label=label)
        )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    if title is None:
        title = f"{metric_col} contour: {x_param} vs {y_param}"
    ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig