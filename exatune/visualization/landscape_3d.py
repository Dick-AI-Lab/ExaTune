"""Interactive 3D landscape visualization using Plotly.

Replaces the static matplotlib plot_surface with a fully rotatable Plotly
surface plot.  The public API mirrors the original plot_surface() so it can
be used as a drop-in replacement.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
# Remove or keep the old one if still used elsewhere:


# ---------------------------------------------------------------------------
# Default hyperparameter values to mark on plots (same as original)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "learning_rate": 0.3,
    "n_estimators": 100,
    "max_depth": 6,
}


# ---------------------------------------------------------------------------
# Minimal local helpers (replace exatune internal imports)
# ---------------------------------------------------------------------------

def _validate_results_dataframe(
    df: pd.DataFrame,
    param_columns: list[str],
    metric: str,
) -> pd.DataFrame:
    """Raise if required columns are missing, else return a copy."""
    required = set(param_columns) | {metric}
    missing = required - set(df.columns)
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
    """Return a pivot table with y_param as index and x_param as columns."""
    agg = df.groupby([y_param, x_param])[metric_col].agg(agg_func).reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric_col)
    # Sort axes so numeric values appear in order
    try:
        pivot = pivot.sort_index(axis=0, key=lambda s: s.astype(float))
        pivot = pivot.sort_index(axis=1, key=lambda s: s.astype(float))
    except (TypeError, ValueError):
        pivot = pivot.sort_index(axis=0)
        pivot = pivot.sort_index(axis=1)
    return pivot


def _find_default_index(labels: list, param: str) -> Optional[float]:
    """Return the axis index (float) for the registered default of *param*."""
    default_val = _DEFAULTS.get(param)
    if default_val is None:
        return None
    str_labels = [str(v) for v in labels]
    target = str(default_val)
    if target in str_labels:
        return float(str_labels.index(target))
    try:
        numeric_labels = [float(v) for v in str_labels]
        for i, v in enumerate(numeric_labels):
            if abs(v - float(default_val)) < 1e-9:
                return float(i)
    except (ValueError, TypeError):
        pass
    return None


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
) -> go.Figure:
    """Plot an interactive, rotatable 3D surface of metric values over two
    hyperparameters using Plotly.

    A black dot marker is placed above the peak metric cell and a red dot
    marker above the default hyperparameter cell (learning_rate=0.3,
    n_estimators=100, max_depth=6), where applicable.

    Args:
        df:           Results DataFrame with at least columns for *x_param*,
                      *y_param*, and *metric*.
        x_param:      Hyperparameter for the x-axis.
        y_param:      Hyperparameter for the y-axis.
        metric:       Column name of the performance metric (z-axis).
        agg_func:     How to aggregate when more than two hyperparameters
                      exist ('mean', 'max', 'min', 'std').
        title:        Plot title; auto-generated when None.
        colorscale:   Any Plotly named colorscale (e.g. 'Viridis', 'Plasma').
        figsize:      (width, height) in pixels for the HTML figure.
        output_path:  Optional path to save an interactive HTML file.

    Returns:
        plotly.graph_objects.Figure — call .show() to open in a browser.
    """
    # ------------------------------------------------------------------ data
    filtered = _validate_results_dataframe(df, [x_param, y_param], metric)
    pivot = _pivot_for_surface(filtered, x_param, y_param, metric, agg_func)

    x_labels = [str(v) for v in pivot.columns]   # tick labels along x
    y_labels = [str(v) for v in pivot.index]      # tick labels along y
    x_vals = np.arange(len(x_labels))
    y_vals = np.arange(len(y_labels))
    Z = pivot.values.astype(float)

    # Fill NaN for surface rendering (nearest-neighbour)
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        from scipy.interpolate import griddata
        valid = ~mask
        pts = np.array(np.where(valid)).T
        vals = Z[valid]
        fill_pts = np.array(np.where(mask)).T
        if len(fill_pts) and len(pts):
            Z[mask] = griddata(pts, vals, fill_pts, method="nearest")

    # ---------------------------------------------------------------- traces
    traces: list[go.BaseTraceType] = []

    # 1. Main surface
    hover_template = (
        f"{x_param}: %{{x}}<br>"
        f"{y_param}: %{{y}}<br>"
        f"{metric}: %{{z:.4f}}<extra></extra>"
    )
    surface = go.Surface(
        x=x_vals,
        y=y_vals,
        z=Z,
        colorscale=colorscale,
        opacity=0.92,
        colorbar=dict(title=dict(text=metric, side="right")),
        hovertemplate=hover_template,
        name="Metric surface",
    )
    traces.append(surface)

    z_orig = pivot.values.astype(float)
    z_range = np.nanmax(z_orig) - np.nanmin(z_orig) if not np.all(np.isnan(z_orig)) else 0.0
    z_offset = z_range * 0.12

    # 2. Peak marker (black dot + dashed drop-line)
    if not np.all(np.isnan(z_orig)):
        pr, pc = np.unravel_index(np.nanargmax(z_orig), z_orig.shape)
        peak_z = z_orig[pr, pc]
        marker_z = peak_z + z_offset

        traces.append(go.Scatter3d(
            x=[pc], y=[pr], z=[marker_z],
            mode="markers+text",
            marker=dict(size=10, color="#D63384", symbol="circle"),
            text=[f"Peak<br>{x_param}={x_labels[pc]}<br>{y_param}={y_labels[pr]}"],
            textposition="top center",
            textfont=dict(size=11, color="#D63384"),
            name=f"Peak  {x_param}={x_labels[pc]}, {y_param}={y_labels[pr]}",
            hovertemplate=(
                f"<b>Peak</b><br>{x_param}: {x_labels[pc]}<br>"
                f"{y_param}: {y_labels[pr]}<br>{metric}: {peak_z:.4f}"
                "<extra></extra>"
            ),
        ))
        # Drop-line
        traces.append(go.Scatter3d(
            x=[pc, pc], y=[pr, pr], z=[peak_z, marker_z],
            mode="lines",
            line=dict(color="#D63384", width=3, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 3. Default parameters marker (red dot + dashed drop-line)
    default_x = _find_default_index(list(pivot.columns), x_param)
    default_y = _find_default_index(list(pivot.index), y_param)
    if default_x is not None and default_y is not None:
        dx_i = int(round(default_x))
        dy_i = int(round(default_y))
        default_z = z_orig[dy_i, dx_i]
        if not np.isnan(default_z):
            marker_z_d = default_z + z_offset
            traces.append(go.Scatter3d(
                x=[default_x], y=[default_y], z=[marker_z_d],
                mode="markers+text",
                marker=dict(size=10, color="#111D4F", symbol="circle"),
                text=[f"Defaults<br>{x_param}={_DEFAULTS.get(x_param)}<br>{y_param}={_DEFAULTS.get(y_param)}"],
                textposition="top center",
                textfont=dict(size=11, color="#111D4F"),
                name=f"Defaults  {x_param}={_DEFAULTS.get(x_param)}, {y_param}={_DEFAULTS.get(y_param)}",
                hovertemplate=(
                    f"<b>Defaults</b><br>{x_param}: {_DEFAULTS.get(x_param)}<br>"
                    f"{y_param}: {_DEFAULTS.get(y_param)}<br>{metric}: {default_z:.4f}"
                    "<extra></extra>"
                ),
            ))
            traces.append(go.Scatter3d(
                x=[default_x, default_x], y=[default_y, default_y],
                z=[default_z, marker_z_d],
                mode="lines",
                line=dict(color="#111D4F", width=3, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
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
            camera=dict(
                eye=dict(x=1.5, y=-1.8, z=1.2),   # initial viewpoint
            ),
            aspectmode="auto",
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
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
# Quick demo — runs when the module is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a small synthetic results DataFrame to prove the plot works
    rng = np.random.default_rng(42)
    lrs = [0.01, 0.05, 0.1, 0.3, 0.5]
    depths = [3, 4, 5, 6, 8, 10]
    rows = []
    for lr in lrs:
        for d in depths:
            for _ in range(3):           # 3 seeds per combo
                score = (
                    0.85
                    - 0.5 * (lr - 0.1) ** 2
                    - 0.003 * (d - 5) ** 2
                    + rng.normal(0, 0.005)
                )
                rows.append({"learning_rate": lr, "max_depth": d, "mean_score": score})
    demo_df = pd.DataFrame(rows)

    fig = plot_surface_plotly(
        demo_df,
        x_param="learning_rate",
        y_param="max_depth",
        metric="mean_score",
        title="Demo: mean_score landscape (rotate me!)",
        output_path="demo_landscape.html",
    )
    print("Saved demo_landscape.html — open it in a browser to rotate the surface.")
    fig.show()