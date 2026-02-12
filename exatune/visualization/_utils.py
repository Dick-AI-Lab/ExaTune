"""Shared visualization utilities for ExaTune.

Provides helper functions used across all visualization modules including
DataFrame validation, hyperparameter column inference, metric resolution,
pivot operations, and figure saving.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Known metadata columns that are NOT hyperparameters
_METADATA_COLUMNS = {
    "job_id",
    "config_hash",
    "timestamp",
    "success",
    "mean_score",
    "std_score",
    "mean_train_score",
    "std_train_score",
    "fit_time_mean",
    "fit_time_std",
    "score_time_mean",
    "score_time_std",
    "rank",
    "best_score",
    "error_message",
    "error_type",
}

# Suffixes that indicate derived metric columns
_METRIC_SUFFIXES = ("_mean", "_std", "_min", "_max")


def check_optional_backend(name: str) -> bool:
    """Check if an optional visualization backend is available.

    Args:
        name: Backend name ('seaborn', 'plotly', 'bokeh').

    Returns:
        True if the backend is importable.
    """
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def validate_results_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    param_columns: Optional[List[str]] = None,
    metric: str = "mean_score",
    filter_success: bool = True,
) -> pd.DataFrame:
    """Validate and filter a results DataFrame for visualization.

    Args:
        df: Results DataFrame from collect_results().
        required_columns: Columns that must be present.
        param_columns: Hyperparameter column names to verify.
        metric: Metric column that must be present and non-null.
        filter_success: Whether to filter to only successful rows.

    Returns:
        Filtered DataFrame ready for visualization.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Results DataFrame is empty.")

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Check param columns exist
    if param_columns:
        missing = set(param_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing hyperparameter columns: {missing}")

    # Resolve metric column
    metric_col = get_metric_column(df, metric)

    # Filter to successful runs
    filtered = df.copy()
    if filter_success and "success" in filtered.columns:
        filtered = filtered[filtered["success"] == True]  # noqa: E712

    # Drop rows with NaN in metric column
    filtered = filtered.dropna(subset=[metric_col])

    if filtered.empty:
        raise ValueError(
            f"No valid results after filtering (metric='{metric_col}', success=True)."
        )

    return filtered


def infer_hyperparameter_columns(df: pd.DataFrame) -> List[str]:
    """Infer which columns are hyperparameters vs metadata.

    Excludes known metadata columns and columns with metric-like suffixes.

    Args:
        df: Results DataFrame.

    Returns:
        List of hyperparameter column names.
    """
    hp_cols = []
    for col in df.columns:
        if col in _METADATA_COLUMNS:
            continue
        if any(col.endswith(suffix) for suffix in _METRIC_SUFFIXES):
            continue
        hp_cols.append(col)
    return hp_cols


def get_metric_column(df: pd.DataFrame, metric: str) -> str:
    """Resolve a metric name to its DataFrame column name.

    Handles common aliases and metric name patterns:
    - Direct column name match (e.g., 'mean_score')
    - Shorthand to *_mean suffix (e.g., 'f1_macro' -> 'f1_macro_mean')
    - Common aliases (e.g., 'accuracy' -> 'mean_score')

    Args:
        df: Results DataFrame.
        metric: User-specified metric name.

    Returns:
        Actual column name in the DataFrame.

    Raises:
        ValueError: If metric column cannot be resolved.
    """
    # Direct match
    if metric in df.columns:
        return metric

    # Try with _mean suffix
    suffixed = f"{metric}_mean"
    if suffixed in df.columns:
        return suffixed

    # Common aliases
    aliases = {
        "accuracy": "mean_score",
        "score": "mean_score",
        "train_score": "mean_train_score",
    }
    if metric in aliases and aliases[metric] in df.columns:
        return aliases[metric]

    raise ValueError(
        f"Cannot resolve metric '{metric}' to a column. "
        f"Available columns: {sorted(df.columns.tolist())}"
    )


def save_figure(
    fig: matplotlib.figure.Figure,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> Optional[Path]:
    """Save a matplotlib figure to file, auto-detecting format from extension.

    Args:
        fig: matplotlib Figure object.
        output_path: Where to save (supports .png, .pdf, .svg). None to skip.
        dpi: Resolution for raster formats.
        bbox_inches: Bounding box setting.

    Returns:
        Path where figure was saved, or None if not saved.
    """
    if output_path is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches=bbox_inches)
    return output_path


def get_default_colormap(metric_direction: str = "higher_better") -> str:
    """Return a sensible default colormap name.

    Args:
        metric_direction: 'higher_better' or 'lower_better'.

    Returns:
        Colormap name string.
    """
    if metric_direction == "lower_better":
        return "viridis_r"
    return "viridis"


def pivot_for_heatmap(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric_col: str,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Pivot results DataFrame into a 2D grid for heatmap rendering.

    When there are >2 hyperparameters, this aggregates over the
    non-displayed parameters using agg_func.

    Args:
        df: Results DataFrame (should already be validated/filtered).
        x_param: Column for x-axis.
        y_param: Column for y-axis.
        metric_col: Column for the cell values.
        agg_func: Aggregation function ('mean', 'max', 'min', 'std').

    Returns:
        Pivoted DataFrame with y_param as index, x_param as columns.

    Raises:
        ValueError: If x_param or y_param not in DataFrame.
    """
    for param in (x_param, y_param):
        if param not in df.columns:
            raise ValueError(f"Parameter '{param}' not found in DataFrame columns.")

    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")

    pivot = df.pivot_table(
        values=metric_col,
        index=y_param,
        columns=x_param,
        aggfunc=agg_func,
    )

    # Sort index and columns for consistent ordering
    try:
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot[sorted(pivot.columns, key=lambda x: (isinstance(x, str), x))]
    except TypeError:
        pass  # Mixed types, keep original order

    return pivot


def get_figure_and_axes(
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    projection: Optional[str] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Get or create a figure and axes pair.

    Args:
        ax: Optional existing axes to use.
        figsize: Figure size if creating new.
        projection: Axes projection (e.g., '3d').

    Returns:
        Tuple of (figure, axes).
    """
    if ax is not None:
        return ax.get_figure(), ax

    if projection:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
