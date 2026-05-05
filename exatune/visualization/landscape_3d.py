"""Interactive 3-D landscape visualization using Plotly.

Replaces the static matplotlib plot_surface with a fully rotatable Plotly
surface.  The public API mirrors the original so it can be used as a
drop-in replacement.

Marker conventions
------------------
  Peak    → #D63384 (magenta)  filled circle + dashed vertical drop-line
  Default → #111D4F (dark navy) filled circle + dashed vertical drop-line

Model defaults are resolved automatically from the experiment YAML (or any
ExaTuneConfig object) via :mod:`exatune.visualization._defaults`.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from exatune.visualization._defaults import (
    get_defaults,
    find_default_index,
)


# ---------------------------------------------------------------------------
# Colour palette — same as landscape.py
# ---------------------------------------------------------------------------
_PEAK_COLOR    = "#D63384"   # magenta/pink
_DEFAULT_COLOR = "#111D4F"   # dark navy


# ---------------------------------------------------------------------------
# Internal helpers (self-contained so this module has no landscape.py dep)
# ---------------------------------------------------------------------------

def _resolve_defaults(config_or_defaults) -> Dict[str, Any]:
    if config_or_defaults is None:
        return {}
    if isinstance(config_or_defaults, dict):
        if "model" in config_or_defaults or "experiment" in config_or_defaults:
            return get_defaults(config_or_defaults)
        return config_or_defaults
    return get_defaults(config_or_defaults)


def _validate_results_dataframe(
    df: pd.DataFrame,
    param_columns: list,
    metric: str,
) -> pd.DataFrame:
    required = set(param_columns) | {metric}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    return df.dropna(subset=list(required)).copy()


def _pivot_for_surface(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric_col: str,
    agg_func: str,
) -> pd.DataFrame:
    agg = df.groupby([y_param, x_param])[metric_col].agg(agg_func).reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric_col)
    try:
        pivot = pivot.sort_index(axis=0, key=lambda s: s.astype(float))
        pivot = pivot.sort_index(axis=1, key=lambda s: s.astype(float))
    except (TypeError, ValueError):
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def _find_default_cell(pivot, x_param, y_param, defaults):
    col_i = find_default_index(list(pivot.columns), x_param, defaults)
    row_i = find_default_index(list(pivot.index),   y_param, defaults)
    return col_i, row_i


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_surface_plotly(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "mean_score",
    agg_func: str = "mean",
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    figsize: Tuple[int, int] = (900, 700),
    output_path: Optional[Union[str, Path]] = None,
    # Model defaults
    config=None,
    peak_params: "Optional[Dict[str, Any]]" = None,
) -> go.Figure:
    """Plot an interactive, rotatable 3-D surface of metric values.

    A **#D63384 (magenta) filled circle** marks the peak cell and a
    **#111D4F (dark navy) filled circle** marks the default hyperparameter
    cell.  Each marker floats above the surface connected by a dashed
    vertical drop-line.  The Plotly legend labels both markers with their
    colour and role (Peak / Default).

    Args:
        df:          Results DataFrame with columns for *x_param*, *y_param*,
                     and *metric*.
        x_param:     Hyperparameter for the x-axis.
        y_param:     Hyperparameter for the y-axis.
        metric:      Column name of the performance metric (z-axis).
        agg_func:    Aggregation per cell ('mean', 'max', 'min', 'std').
        title:       Auto-generated when None.
        colorscale:  Any Plotly named colorscale.
        figsize:     (width, height) in pixels for the HTML figure.
        output_path: Optional path to save an interactive HTML file.
        config:      ExaTuneConfig object, raw YAML dict, or pre-resolved
                     defaults dict used to locate the default marker.
                     None → no default marker shown.

    Returns:
        plotly.graph_objects.Figure — call .show() to open in a browser.
    """
    # ------------------------------------------------------------------ data
    filtered  = _validate_results_dataframe(df, [x_param, y_param], metric)
    pivot     = _pivot_for_surface(filtered, x_param, y_param, metric, agg_func)
    defaults  = _resolve_defaults(config)

    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]
    x_vals   = np.arange(len(x_labels))
    y_vals   = np.arange(len(y_labels))
    Z        = pivot.values.astype(float)

    # Fill isolated NaN cells via nearest-neighbour interpolation
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        try:
            from scipy.interpolate import griddata
            valid    = ~mask
            pts      = np.array(np.where(valid)).T
            vals     = Z[valid]
            fill_pts = np.array(np.where(mask)).T
            if len(fill_pts) and len(pts):
                Z[mask] = griddata(pts, vals, fill_pts, method="nearest")
        except ImportError:
            pass  # scipy not available — leave NaN as-is

    # ---------------------------------------------------------------- traces
    traces: list = []

    # 1. Main surface
    hover_template = (
        f"{x_param}: %{{x}}<br>"
        f"{y_param}: %{{y}}<br>"
        f"{metric}: %{{z:.4f}}<extra></extra>"
    )
    surface = go.Surface(
        x=x_vals, y=y_vals, z=Z,
        colorscale=colorscale,
        opacity=0.92,
        colorbar=dict(title=dict(text=metric, side="right")),
        hovertemplate=hover_template,
        name="Metric surface",
        showlegend=False,
    )
    traces.append(surface)

    z_orig   = pivot.values.astype(float)
    z_range  = np.nanmax(z_orig) - np.nanmin(z_orig) if not np.all(np.isnan(z_orig)) else 0.0
    z_offset = z_range * 0.12

    # 2. Peak marker — #D63384 circle + dashed drop-line
    # If peak_params is explicitly {} caller wants no peak marker.
    # If peak_params is None, auto-detect via argmax.
    peak_row = peak_col = None
    _suppress_peak = isinstance(peak_params, dict) and len(peak_params) == 0
    if not _suppress_peak and not np.all(np.isnan(z_orig)):
        if peak_params:
            # Locate peak from caller-supplied param values
            def _find(labels, val):
                for i, v in enumerate(labels):
                    try:
                        if abs(float(v) - float(val)) < 1e-9:
                            return i
                    except (TypeError, ValueError):
                        if str(v) == str(val):
                            return i
                return None
            pc = _find(x_labels, peak_params.get(x_param))
            pr = _find(y_labels, peak_params.get(y_param))
            if pc is not None and pr is not None:
                peak_col, peak_row = pc, pr
        else:
            # Auto-detect: argmax of the pivot
            peak_row, peak_col = np.unravel_index(np.nanargmax(z_orig), z_orig.shape)

    if peak_row is not None and peak_col is not None:
        peak_z   = z_orig[peak_row, peak_col]
        marker_z = peak_z + z_offset

        traces.append(go.Scatter3d(
            x=[peak_col], y=[peak_row], z=[marker_z],
            mode="markers",
            marker=dict(size=10, color=_PEAK_COLOR, symbol="circle",
                        line=dict(color="white", width=1)),
            name=(
                f"Peak — {x_param}={x_labels[peak_col]}, "
                f"{y_param}={y_labels[peak_row]}"
            ),
            hovertemplate=(
                f"<b>Peak</b><br>"
                f"{x_param}: {x_labels[peak_col]}<br>"
                f"{y_param}: {y_labels[peak_row]}<br>"
                f"{metric}: {peak_z:.4f}"
                "<extra></extra>"
            ),
            legendgroup="peak",
            showlegend=True,
        ))
        # Dashed drop-line (no legend entry)
        traces.append(go.Scatter3d(
            x=[peak_col, peak_col], y=[peak_row, peak_row],
            z=[peak_z, marker_z],
            mode="lines",
            line=dict(color=_PEAK_COLOR, width=3, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="peak",
        ))

    # 3. Default marker — #111D4F circle + dashed drop-line
    col_i, row_i = _find_default_cell(pivot, x_param, y_param, defaults)
    if col_i is not None and row_i is not None:
        default_z = z_orig[row_i, col_i]
        if not np.isnan(default_z):
            marker_z_d = default_z + z_offset

            # Label: note if default == peak
            is_same = (peak_row is not None
                       and row_i == peak_row and col_i == peak_col)
            label = (
                f"Default (= Peak) — {x_param}={x_labels[col_i]}, "
                f"{y_param}={y_labels[row_i]}"
                if is_same else
                f"Default — {x_param}={x_labels[col_i]}, "
                f"{y_param}={y_labels[row_i]}"
            )

            traces.append(go.Scatter3d(
                x=[col_i], y=[row_i], z=[marker_z_d],
                mode="markers",
                marker=dict(size=10, color=_DEFAULT_COLOR, symbol="circle",
                            line=dict(color="white", width=1)),
                name=label,
                hovertemplate=(
                    f"<b>Default</b><br>"
                    f"{x_param}: {x_labels[col_i]}<br>"
                    f"{y_param}: {y_labels[row_i]}<br>"
                    f"{metric}: {default_z:.4f}"
                    "<extra></extra>"
                ),
                legendgroup="default",
                showlegend=True,
            ))
            # Dashed drop-line
            traces.append(go.Scatter3d(
                x=[col_i, col_i], y=[row_i, row_i],
                z=[default_z, marker_z_d],
                mode="lines",
                line=dict(color=_DEFAULT_COLOR, width=3, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="default",
            ))

    # --------------------------------------------------------------- layout
    if title is None:
        title = f"{metric} landscape: {x_param} vs {y_param}"

    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        width=figsize[0],
        height=figsize[1],
        scene=dict(
            xaxis=dict(
                title=dict(text=x_param),
                tickmode="array",
                tickvals=list(x_vals),
                ticktext=x_labels,
            ),
            yaxis=dict(
                title=dict(text=y_param),
                tickmode="array",
                tickvals=list(y_vals),
                ticktext=y_labels,
            ),
            zaxis=dict(title=dict(text=metric)),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
            aspectmode="auto",
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig = go.Figure(data=traces, layout=layout)

    # -------------------------------------------------------- optional save
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() == ".html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path))

    return fig


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    lrs = [0.01, 0.05, 0.1, 0.3, 0.5]
    depths = [3, 4, 5, 6, 8, 10]
    rows = []
    for lr in lrs:
        for d in depths:
            for _ in range(3):
                score = (
                    0.85
                    - 0.5 * (lr - 0.1) ** 2
                    - 0.003 * (d - 5) ** 2
                    + rng.normal(0, 0.005)
                )
                rows.append({"learning_rate": lr, "max_depth": d, "mean_score": score})
    demo_df = pd.DataFrame(rows)

    # Simulating XGBoost defaults (learning_rate=0.3, max_depth=6)
    xgb_defaults = {"learning_rate": 0.3, "max_depth": 6}

    fig = plot_surface_plotly(
        demo_df,
        x_param="learning_rate",
        y_param="max_depth",
        metric="mean_score",
        title="Demo: mean_score landscape (rotate me!)",
        output_path="demo_landscape.html",
        config=xgb_defaults,
    )
    print("Saved demo_landscape.html — open in a browser to rotate.")
    fig.show()