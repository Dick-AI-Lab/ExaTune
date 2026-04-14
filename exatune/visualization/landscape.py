"""Landscape visualizations: heatmaps, 2D/3D surfaces, and contour plots.

Shows how performance metrics vary across the hyperparameter space.
Supports both static matplotlib output and interactive Plotly surfaces.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from exatune.visualization._utils import (
    get_metric_column,
    pivot_for_heatmap,
    save_figure,
    validate_results_dataframe,
)

# ---------------------------------------------------------------------------
# Brand colours for peak / default markers
# ---------------------------------------------------------------------------
_PEAK_COLOR    = "#D63384"   # pink/magenta — best-performing cell
_DEFAULT_COLOR = "#111D4F"   # dark navy    — default hyperparameter cell

# Known default hyperparameter values
_DEFAULTS = {
    "learning_rate": 0.3,
    "n_estimators":  100,
    "max_depth":     6,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_default_cell(pivot: pd.DataFrame, x_param: str, y_param: str):
    """Return (col_index, row_index) of the default cell, or (None, None)."""

    def _find(labels, param):
        default_val = _DEFAULTS.get(param)
        if default_val is None:
            return None
        str_labels = [str(v) for v in labels]
        target = str(default_val)
        if target in str_labels:
            return str_labels.index(target)
        try:
            numeric = [float(v) for v in str_labels]
            for i, v in enumerate(numeric):
                if abs(v - float(default_val)) < 1e-9:
                    return i
        except (ValueError, TypeError):
            pass
        return None

    col_i = _find(list(pivot.columns), x_param)
    row_i = _find(list(pivot.index), y_param)
    return col_i, row_i


def _draw_square_marker(
    ax: matplotlib.axes.Axes,
    col: int,
    row: int,
    color: str,
) -> None:
    """Draw a bold square outline only on heatmap cell (col, row).

    imshow places cell (col, row) centred at x=col, y=row (origin='upper').
    The per-cell annotation loop handles all value text.
    """
    rect = mpatches.FancyBboxPatch(
        (col - 0.47, row - 0.47),
        0.94, 0.94,
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor=color,
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Public API
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
) -> matplotlib.figure.Figure:
    """Plot a 2-D heatmap of *metric* over *x_param* x *y_param*.

    Peak cell    -> bold #D63384 square outline + metric value label
    Default cell -> bold #111D4F square outline + metric value label

    Args:
        df:           Results DataFrame.
        x_param:      Hyperparameter for the x-axis.
        y_param:      Hyperparameter for the y-axis.
        metric:       Metric column to visualise (default 'mean_score').
        agg_func:     Aggregation applied per cell ('mean', 'max', etc.).
        annotate:     If True, draw the square markers on peak & default cells.
        colormap:     Matplotlib colormap name.
        title:        Plot title; auto-generated when None.
        figsize:      Figure size in inches (width, height).
        output_path:  If given, save the figure here.
        ax:           Optional existing axes to draw on.

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

    # --- figure / axes ---
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="upper",
        cmap=colormap,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_col, fontsize=11)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)

    if annotate:
        # --- Value label on every cell ---
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        z_mid = (z_min + z_max) / 2.0
        for row in range(n_rows):
            for col in range(n_cols):
                val = Z[row, col]
                if np.isnan(val):
                    continue
                # White text on dark cells, dark text on bright cells
                text_color = "white" if val < z_mid else "#1a1a1a"
                ax.text(
                    col, row,
                    f"{val:.3f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=7,
                    zorder=3,
                )

        # --- Peak marker (square outline + bold value overwrites base label) ---
        if not np.all(np.isnan(Z)):
            pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
            _draw_square_marker(ax, pc, pr, _PEAK_COLOR)

        # --- Default marker ---
        col_i, row_i = _find_default_cell(pivot, x_param, y_param)
        if col_i is not None and row_i is not None:
            default_val = Z[row_i, col_i]
            if not np.isnan(default_val):
                _draw_square_marker(ax, col_i, row_i, _DEFAULT_COLOR)

        # --- Legend — placed outside the axes, below the colorbar ---
        legend_handles = [
            mpatches.Patch(edgecolor=_PEAK_COLOR,    facecolor="none",
                           linewidth=2.5, label="Peak"),
            mpatches.Patch(edgecolor=_DEFAULT_COLOR, facecolor="none",
                           linewidth=2.5, label="Default"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.18, 0.18),   # right of axes, below colorbar
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
) -> matplotlib.figure.Figure:
    """Plot a static 3-D surface of *metric* over *x_param* x *y_param*.

    Peak marker: #D63384 dot + dashed drop-line.
    Default marker: #111D4F dot + dashed drop-line.

    Args:
        df:           Results DataFrame.
        x_param:      Hyperparameter for the x-axis.
        y_param:      Hyperparameter for the y-axis.
        metric:       Metric column for the z-axis.
        agg_func:     Aggregation per cell.
        colormap:     Matplotlib colormap name.
        title:        Plot title.
        figsize:      Figure size in inches.
        output_path:  Save path.

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

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric_col)

    z_range  = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.12

    # Peak marker
    if not np.all(np.isnan(Z)):
        pr, pc  = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z  = Z[pr, pc]
        ax.scatter([pc], [pr], [peak_z + z_offset],
                   color=_PEAK_COLOR, s=80, zorder=5, label="Peak")
        ax.plot([pc, pc], [pr, pr], [peak_z, peak_z + z_offset],
                color=_PEAK_COLOR, linewidth=2, linestyle="--")

    # Default marker
    col_i, row_i = _find_default_cell(pivot, x_param, y_param)
    if col_i is not None and row_i is not None:
        default_z = Z[row_i, col_i]
        if not np.isnan(default_z):
            ax.scatter([col_i], [row_i], [default_z + z_offset],
                       color=_DEFAULT_COLOR, s=80, zorder=5, label="Default")
            ax.plot([col_i, col_i], [row_i, row_i], [default_z, default_z + z_offset],
                    color=_DEFAULT_COLOR, linewidth=2, linestyle="--")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param, fontsize=10)
    ax.set_ylabel(y_param, fontsize=10)
    ax.set_zlabel(metric_col, fontsize=10)
    ax.legend(fontsize=9)

    if title is None:
        title = f"{metric_col} surface: {x_param} vs {y_param}"
    ax.set_title(title, fontsize=13)

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
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a filled contour map of *metric* over *x_param* x *y_param*.

    Peak marker: #D63384 star.
    Default marker: #111D4F diamond.

    Args:
        df:           Results DataFrame.
        x_param:      Hyperparameter for the x-axis.
        y_param:      Hyperparameter for the y-axis.
        metric:       Metric column.
        agg_func:     Aggregation per cell.
        levels:       Number of contour levels.
        colormap:     Matplotlib colormap name.
        title:        Plot title.
        figsize:      Figure size in inches.
        output_path:  Save path.

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

    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(X, Y, Z, levels=levels, cmap=colormap)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label=metric_col)

    # Peak marker
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter([pc], [pr], color=_PEAK_COLOR, s=150, marker="*",
                   zorder=5, label="Peak", edgecolors="white", linewidths=0.5)

    # Default marker
    col_i, row_i = _find_default_cell(pivot, x_param, y_param)
    if col_i is not None and row_i is not None:
        ax.scatter([col_i], [row_i], color=_DEFAULT_COLOR, s=120, marker="D",
                   zorder=5, label="Default", edgecolors="white", linewidths=0.5)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    if title is None:
        title = f"{metric_col} contour: {x_param} vs {y_param}"
    ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig