#!/usr/bin/env python3
"""
ExaTune Search Comparison Visualizer
======================================
Overlays results from grid search, random search, and/or Bayesian search on
shared 3D landscape surfaces and heatmaps so you can visually compare where
each search method sampled and how well it performed.

Each search type gets its own marker color and symbol on every plot.
The underlying surface/heatmap is built from the UNION of all loaded results,
giving a richer landscape estimate than any single search alone.

Usage
-----
    # Compare random vs Bayesian search
    python visualize_search_comparison.py \\
        --random   ./results/random_search/iris/results.parquet \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml \\
        --x-param  n_estimators --y-param max_depth

    # All three search types
    python visualize_search_comparison.py \\
        --grid     ./results/benchmarks/iris_rf/results.parquet \\
        --random   ./results/random_search/iris/results.parquet \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml

    # Only Bayesian, all param pairs auto-generated
    python visualize_search_comparison.py \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml --all-pairs

    # Custom metric and output dir
    python visualize_search_comparison.py \\
        --random   ./results/random_search/iris/results.parquet \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml \\
        --metric   f1_macro_mean --output-dir ./comparison_plots
"""

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Search type registry — colors, markers, labels
# ---------------------------------------------------------------------------

SEARCH_STYLES: Dict[str, dict] = {
    "grid": {
        "label":        "Grid Search",
        "color":        "#2196F3",   # blue  — grid IS the background, no heatmap boxes
        "marker_3d":    "square",
        "plotly_color": "#2196F3",
        "alpha":        0.75,
        "size_3d":      6,
    },
    "random": {
        "label":        "Random Search",
        "color":        "#FF9800",   # orange
        "marker_3d":    "diamond",
        "plotly_color": "#FF9800",
        "alpha":        0.80,
        "size_3d":      6,
        "box_lw":       2.0,         # normal box linewidth on heatmap
        "box_lw_best":  4.0,         # extra-bold for best cell
    },
    "bayesian": {
        "label":        "Bayesian Search",
        "color":        "#E91E63",   # pink/magenta
        "marker_3d":    "circle",
        "plotly_color": "#E91E63",
        "alpha":        0.85,
        "size_3d":      7,
        "box_lw":       2.0,
        "box_lw_best":  4.0,
    },
}

# Color used when random AND bayesian both hit the same cell
_OVERLAP_COLOR = "#9C27B0"   # purple — signals both searches agree

# Overall peak box on heatmap (applies regardless of search type)
_PEAK_COLOR    = "#1B5E20"   # dark green
_PEAK_LW       = 3.5         # linewidth for overall-peak box


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_hp_names(config: dict) -> List[str]:
    return list(config.get("hyperparameters", {}).keys())


def get_primary_metric(config: dict) -> str:
    """Returns 'mean_score' — the unified column name used by all ExaTune runners."""
    return "mean_score"


# ---------------------------------------------------------------------------
# Parquet loading and normalisation
# ---------------------------------------------------------------------------

def load_and_tag(path: Path, search_type: str) -> pd.DataFrame:
    """Load a results parquet, keep only successful rows, tag with search_type."""
    df = pd.read_parquet(path)

    # Normalise column names (strip json_normalize prefixes if present)
    df.columns = [
        c.replace("hyperparameters.", "").replace("additional_metrics.", "")
        for c in df.columns
    ]
    df = df.drop(columns=["cv_scores", "train_scores"], errors="ignore")

    # Keep only successful rows
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    df["_search_type"] = search_type
    return df


def load_all(paths: Dict[str, Optional[Path]]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load all provided parquet files.

    Returns:
        combined  : single DataFrame with _search_type column
        per_type  : dict mapping search_type -> DataFrame (only provided types)
    """
    per_type: Dict[str, pd.DataFrame] = {}
    for stype, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            print(f"  WARNING: {stype} parquet not found: {path} — skipping")
            continue
        df = load_and_tag(path, stype)
        print(f"  Loaded {len(df):>5} rows from {stype:<10} → {path}")
        per_type[stype] = df

    if not per_type:
        print("ERROR: No valid result files were loaded.")
        sys.exit(1)

    combined = pd.concat(per_type.values(), ignore_index=True)
    return combined, per_type


# ---------------------------------------------------------------------------
# Pivot / aggregation helpers
# ---------------------------------------------------------------------------

def build_pivot(df: pd.DataFrame, x_param: str, y_param: str, metric: str) -> pd.DataFrame:
    """Aggregate metric over (x_param, y_param) from the combined DataFrame."""
    needed = [x_param, y_param, metric]
    sub = df.dropna(subset=needed)
    agg = sub.groupby([y_param, x_param])[metric].mean().reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric)

    # Sort axes
    def _sort_index(idx):
        try:
            return idx.astype(float)
        except (TypeError, ValueError):
            return idx

    pivot = pivot.sort_index(axis=0, key=_sort_index)
    pivot = pivot.sort_index(axis=1, key=_sort_index)
    return pivot


def get_best_per_search(
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    metric: str,
) -> Dict[str, pd.Series]:
    """Return the best-scoring row per search type."""
    bests = {}
    for stype, df in per_type.items():
        needed = [x_param, y_param, metric]
        sub = df.dropna(subset=needed)
        if sub.empty:
            continue
        bests[stype] = sub.loc[sub[metric].idxmax()]
    return bests


def pivot_coords(pivot: pd.DataFrame, x_val, y_val) -> Tuple[Optional[int], Optional[int]]:
    """Return (col_idx, row_idx) in pivot for a given (x_val, y_val), or (None, None)."""
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]

    def _find(labels, val):
        sv = str(val)
        if sv in labels:
            return labels.index(sv)
        try:
            numeric = [float(v) for v in labels]
            fv = float(val)
            for i, v in enumerate(numeric):
                if abs(v - fv) < 1e-9:
                    return i
        except (ValueError, TypeError):
            pass
        return None

    return _find(x_labels, x_val), _find(y_labels, y_val)


# ---------------------------------------------------------------------------
# 2D Heatmap with overlaid search markers
# ---------------------------------------------------------------------------

def _draw_box(ax, col: int, row: int, color: str, lw: float, zorder: int = 6) -> None:
    """Draw a square outline on heatmap cell (col, row)."""
    import matplotlib.patches as mp
    rect = mp.FancyBboxPatch(
        (col - 0.47, row - 0.47), 0.94, 0.94,
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        zorder=zorder,
    )
    ax.add_patch(rect)


def _draw_split_box(ax, col: int, row: int, color_top_left: str, color_bot_right: str,
                    lw: float, zorder: int = 8) -> None:
    """
    Draw a cell with two half-borders showing two search colors simultaneously.
    Top+left edges in color_top_left (random/orange).
    Bottom+right edges in color_bot_right (bayesian/pink).
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    x0, y0 = col - 0.47, row - 0.47
    x1, y1 = col + 0.47, row + 0.47

    # Top edge + left edge
    path_tl = MPath(
        [(x0, y1), (x1, y1), (x0, y1), (x0, y0)],
        [MPath.MOVETO, MPath.LINETO, MPath.MOVETO, MPath.LINETO],
    )
    ax.add_patch(PathPatch(path_tl, edgecolor=color_top_left, facecolor="none",
                           linewidth=lw, zorder=zorder))

    # Bottom edge + right edge
    path_br = MPath(
        [(x0, y0), (x1, y0), (x1, y0), (x1, y1)],
        [MPath.MOVETO, MPath.LINETO, MPath.MOVETO, MPath.LINETO],
    )
    ax.add_patch(PathPatch(path_br, edgecolor=color_bot_right, facecolor="none",
                           linewidth=lw, zorder=zorder))


def plot_comparison_heatmap(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_pivot(combined, x_param, y_param, metric)
    Z = pivot.values.astype(float)
    n_rows, n_cols = Z.shape
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 0.9 + 2), max(7, n_rows * 0.75 + 2)))

    im = ax.imshow(Z, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric, fontsize=11)

    # Cell value annotations
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    z_mid = (z_min + z_max) / 2.0
    for row in range(n_rows):
        for col in range(n_cols):
            val = Z[row, col]
            if np.isnan(val):
                continue
            text_color = "white" if val < z_mid else "#1a1a1a"
            ax.text(col, row, f"{val:.3f}", ha="center", va="center",
                    color=text_color, fontsize=7, zorder=3)

    # Overall peak — bold green box (drawn first, underneath per-search boxes)
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        _draw_box(ax, pc, pr, _PEAK_COLOR, lw=_PEAK_LW, zorder=5)

    # ----------------------------------------------------------------
    # Per-search boxes on heatmap
    #
    # Rules:
    #   - grid search  → no boxes (it IS the background surface)
    #   - random       → orange box; extra-bold if it is that search's best cell
    #   - bayesian     → pink box;   extra-bold if it is that search's best cell
    #   - cell hit by BOTH random AND bayesian → purple box (overlap color),
    #                    extra-bold if it is the best for either search
    # ----------------------------------------------------------------

    # Collect all cells visited per non-grid search type
    visited: Dict[str, set] = {}   # stype -> set of (col_i, row_i)
    bests = get_best_per_search(per_type, x_param, y_param, metric)

    for stype, df in per_type.items():
        if stype == "grid":
            continue   # grid is the background — no markers
        cells = set()
        sub = df.dropna(subset=[x_param, y_param, metric])
        for _, row_data in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row_data[x_param], row_data[y_param])
            if col_i is not None and row_i is not None:
                cells.add((col_i, row_i))
        visited[stype] = cells

    # Determine which cells are visited by multiple non-grid searches
    non_grid_types = [s for s in per_type if s != "grid"]
    all_visited_cells: Dict[tuple, List[str]] = {}  # (col,row) -> list of stypes
    for stype, cells in visited.items():
        for cell in cells:
            all_visited_cells.setdefault(cell, []).append(stype)

    # Draw boxes — overlap cells get purple, single-search cells get their color
    drawn_legend: Dict[str, bool] = {}  # track what's already in legend

    for (col_i, row_i), stypes_here in all_visited_cells.items():
        # Determine best cells per search for bold linewidth
        is_best = {
            stype: (
                stype in bests and
                pivot_coords(pivot, bests[stype][x_param], bests[stype][y_param]) == (col_i, row_i)
            )
            for stype in stypes_here
        }
        any_best = any(is_best.values())

        if len(stypes_here) >= 2:
            # Overlap — split box: random color on top+left, bayesian on bottom+right
            lw = SEARCH_STYLES[stypes_here[0]]["box_lw_best"] if any_best                  else SEARCH_STYLES[stypes_here[0]]["box_lw"]
            c_tl = SEARCH_STYLES["random"]["color"]   if "random"   in stypes_here                    else SEARCH_STYLES[stypes_here[0]]["color"]
            c_br = SEARCH_STYLES["bayesian"]["color"] if "bayesian" in stypes_here                    else SEARCH_STYLES[stypes_here[-1]]["color"]
            _draw_split_box(ax, col_i, row_i, c_tl, c_br, lw=lw, zorder=8)
            if "overlap" not in drawn_legend:
                drawn_legend["overlap"] = True
        else:
            stype = stypes_here[0]
            style = SEARCH_STYLES[stype]
            lw = style["box_lw_best"] if is_best[stype] else style["box_lw"]
            _draw_box(ax, col_i, row_i, style["color"], lw=lw, zorder=7)
            if stype not in drawn_legend:
                drawn_legend[stype] = True

    # Legend
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)

    if title is None:
        search_names = " vs ".join(SEARCH_STYLES[s]["label"] for s in per_type)
        title = f"{metric} heatmap: {x_param} × {y_param}\n{search_names}"
    ax.set_title(title, fontsize=12, pad=10)

    # Build legend handles
    legend_handles = [
        mpatches.Patch(edgecolor=_PEAK_COLOR, facecolor="none",
                       linewidth=_PEAK_LW, label="Overall Peak"),
    ]
    for stype in non_grid_types:
        if stype in drawn_legend:
            style = SEARCH_STYLES[stype]
            legend_handles.append(
                mpatches.Patch(edgecolor=style["color"], facecolor="none",
                               linewidth=style["box_lw"],
                               label=f"{style['label']} (visited)")
            )
            legend_handles.append(
                mpatches.Patch(edgecolor=style["color"], facecolor="none",
                               linewidth=style["box_lw_best"],
                               label=f"{style['label']} best")
            )
    if "overlap" in drawn_legend:
        # Show both colors in legend for the split-border overlap cell
        legend_handles.append(
            mpatches.Patch(edgecolor=SEARCH_STYLES["random"]["color"], facecolor="none",
                           linewidth=2.5, label="Both visited (orange=random side)")
        )
        legend_handles.append(
            mpatches.Patch(edgecolor=SEARCH_STYLES["bayesian"]["color"], facecolor="none",
                           linewidth=2.5, label="Both visited (pink=Bayesian side)")
        )
    if "grid" in per_type:
        legend_handles.append(
            mpatches.Patch(facecolor=SEARCH_STYLES["grid"]["color"],
                           alpha=0.4, label="Grid Search (surface)")
        )

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.22, 1.0), bbox_transform=ax.transAxes,
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Static 3D surface with overlaid search markers (matplotlib)
# ---------------------------------------------------------------------------

def plot_comparison_surface_static(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_pivot(combined, x_param, y_param, metric)
    Z = pivot.values.astype(float)
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)
    X, Y = np.meshgrid(range(len(x_labels)), range(len(y_labels)))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.75, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, label=metric)

    z_range = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.10

    # Overall peak
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z = Z[pr, pc]
        ax.scatter([pc], [pr], [peak_z + z_offset * 1.5],
                   color=_PEAK_COLOR, s=120, marker=_PEAK_MARKER,
                   zorder=10, label="Overall Peak")
        ax.plot([pc, pc], [pr, pr], [peak_z, peak_z + z_offset * 1.5],
                color=_PEAK_COLOR, linewidth=1.5, linestyle="--")

    # Per-search best + all sampled points
    # Grid search is the background surface itself — skip its points and best marker
    bests = get_best_per_search(per_type, x_param, y_param, metric)
    for stype, df in per_type.items():
        if stype == "grid":
            continue   # grid IS the surface — no overlaid markers needed

        style = SEARCH_STYLES[stype]

        # All evaluated points — projected onto surface
        sub = df.dropna(subset=[x_param, y_param, metric])
        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is None or row_i is None:
                continue
            cell_z = Z[row_i, col_i]
            if np.isnan(cell_z):
                continue
            ax.scatter([col_i], [row_i], [cell_z + z_offset * 0.3],
                       color=style["color"], s=12, alpha=0.30, zorder=5)

        # Best point for this search type
        if stype in bests:
            best_row = bests[stype]
            col_i, row_i = pivot_coords(pivot, best_row[x_param], best_row[y_param])
            if col_i is not None and row_i is not None:
                bz = Z[row_i, col_i]
                if not np.isnan(bz):
                    ax.scatter([col_i], [row_i], [bz + z_offset],
                               color=style["color"], s=90, marker="o", zorder=9,
                               edgecolors="white", linewidths=0.8,
                               label=f"{style['label']} best ({best_row[metric]:.4f})")
                    ax.plot([col_i, col_i], [row_i, row_i], [bz, bz + z_offset],
                            color=style["color"], linewidth=1.5, linestyle="--")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param, fontsize=10)
    ax.set_ylabel(y_param, fontsize=10)
    ax.set_zlabel(metric, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    if title is None:
        search_names = " vs ".join(SEARCH_STYLES[s]["label"] for s in per_type)
        title = f"{metric} surface: {x_param} × {y_param}\n{search_names}"
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved static 3D surface → {output_path}")


# ---------------------------------------------------------------------------
# Interactive Plotly 3D surface with overlaid search markers
# ---------------------------------------------------------------------------

def plot_comparison_surface_plotly(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_pivot(combined, x_param, y_param, metric)
    Z = pivot.values.astype(float)
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]
    x_vals = np.arange(len(x_labels))
    y_vals = np.arange(len(y_labels))

    # Fill NaN for surface rendering
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        try:
            from scipy.interpolate import griddata
            valid = ~mask
            pts = np.array(np.where(valid)).T
            vals = Z[valid]
            fill_pts = np.array(np.where(mask)).T
            if len(fill_pts) and len(pts):
                Z[mask] = griddata(pts, vals, fill_pts, method="nearest")
        except ImportError:
            pass

    traces = []

    # Base surface
    hover_tpl = (
        f"{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>"
    )
    traces.append(go.Surface(
        x=x_vals, y=y_vals, z=Z,
        colorscale="Viridis",
        opacity=0.78,
        colorbar=dict(title=dict(text=metric, side="right")),
        hovertemplate=hover_tpl,
        name="Landscape (combined)",
        showlegend=True,
    ))

    z_orig = pivot.values.astype(float)
    z_range = np.nanmax(z_orig) - np.nanmin(z_orig) if not np.all(np.isnan(z_orig)) else 0.0
    z_offset = z_range * 0.10

    # Overall peak
    if not np.all(np.isnan(z_orig)):
        pr, pc = np.unravel_index(np.nanargmax(z_orig), z_orig.shape)
        peak_z = z_orig[pr, pc]
        traces.append(go.Scatter3d(
            x=[pc], y=[pr], z=[peak_z + z_offset * 1.6],
            mode="markers+text",
            marker=dict(size=12, color=_PEAK_COLOR, symbol="diamond"),
            text=[f"Peak<br>{x_labels[pc]} / {y_labels[pr]}"],
            textposition="top center",
            textfont=dict(size=10, color=_PEAK_COLOR),
            name=f"Overall Peak ({peak_z:.4f})",
            hovertemplate=(
                f"<b>Overall Peak</b><br>{x_param}: {x_labels[pc]}<br>"
                f"{y_param}: {y_labels[pr]}<br>{metric}: {peak_z:.4f}<extra></extra>"
            ),
        ))
        traces.append(go.Scatter3d(
            x=[pc, pc], y=[pr, pr], z=[peak_z, peak_z + z_offset * 1.6],
            mode="lines", line=dict(color=_PEAK_COLOR, width=3, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # Per-search-type: all sampled points + best marker
    # Grid search is the background surface itself — skip its points and best marker
    bests = get_best_per_search(per_type, x_param, y_param, metric)
    for stype, df in per_type.items():
        if stype == "grid":
            continue   # grid IS the surface — no overlaid markers needed

        style = SEARCH_STYLES[stype]

        # All evaluated points
        sub = df.dropna(subset=[x_param, y_param, metric])
        pt_x, pt_y, pt_z, pt_hover = [], [], [], []
        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is None or row_i is None:
                continue
            cell_z = z_orig[row_i, col_i]
            if np.isnan(cell_z):
                continue
            pt_x.append(col_i)
            pt_y.append(row_i)
            pt_z.append(cell_z + z_offset * 0.25)
            pt_hover.append(
                f"{x_param}: {row[x_param]}<br>{y_param}: {row[y_param]}<br>"
                f"{metric}: {row[metric]:.4f}"
            )

        if pt_x:
            traces.append(go.Scatter3d(
                x=pt_x, y=pt_y, z=pt_z,
                mode="markers",
                marker=dict(
                    size=style["size_3d"] * 0.55,
                    color=style["plotly_color"],
                    symbol=style["marker_3d"],
                    opacity=0.40,
                    line=dict(width=0),
                ),
                name=f"{style['label']} (all)",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=pt_hover,
                legendgroup=stype,
            ))

        # Best point
        if stype in bests:
            best_row = bests[stype]
            col_i, row_i = pivot_coords(pivot, best_row[x_param], best_row[y_param])
            if col_i is not None and row_i is not None:
                bz = z_orig[row_i, col_i]
                if not np.isnan(bz):
                    traces.append(go.Scatter3d(
                        x=[col_i], y=[row_i], z=[bz + z_offset],
                        mode="markers+text",
                        marker=dict(
                            size=style["size_3d"] + 3,
                            color=style["plotly_color"],
                            symbol=style["marker_3d"],
                            line=dict(color="white", width=1),
                        ),
                        text=[f"{style['label']}<br>best: {best_row[metric]:.4f}"],
                        textposition="top center",
                        textfont=dict(size=10, color=style["plotly_color"]),
                        name=f"{style['label']} best ({best_row[metric]:.4f})",
                        legendgroup=stype,
                        hovertemplate=(
                            f"<b>{style['label']} Best</b><br>"
                            f"{x_param}: {best_row[x_param]}<br>"
                            f"{y_param}: {best_row[y_param]}<br>"
                            f"{metric}: {best_row[metric]:.4f}<extra></extra>"
                        ),
                    ))
                    traces.append(go.Scatter3d(
                        x=[col_i, col_i], y=[row_i, row_i], z=[bz, bz + z_offset],
                        mode="lines",
                        line=dict(color=style["plotly_color"], width=3, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                        legendgroup=stype,
                    ))

    if title is None:
        search_names = " vs ".join(SEARCH_STYLES[s]["label"] for s in per_type)
        title = f"{metric}: {x_param} × {y_param} — {search_names}"

    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        width=1000, height=750,
        scene=dict(
            xaxis=dict(title=x_param, tickmode="array",
                       tickvals=list(x_vals), ticktext=x_labels),
            yaxis=dict(title=y_param, tickmode="array",
                       tickvals=list(y_vals), ticktext=y_labels),
            zaxis=dict(title=metric),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
            aspectmode="auto",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgrey", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=55),
    )

    fig = go.Figure(data=traces, layout=layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"    Saved interactive 3D surface → {output_path}")


# ---------------------------------------------------------------------------
# Score distribution comparison (bonus: violin/box per search type)
# ---------------------------------------------------------------------------

def plot_comparison_distribution(
    per_type: Dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_type = []
    labels = []
    colors = []
    for stype, df in per_type.items():
        sub = df[metric].dropna()
        if sub.empty:
            continue
        data_by_type.append(sub.values)
        labels.append(SEARCH_STYLES[stype]["label"])
        colors.append(SEARCH_STYLES[stype]["color"])

    parts = ax.violinplot(data_by_type, positions=range(len(data_by_type)),
                          showmedians=True, showextrema=True)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)

    for part_name in ("cmedians", "cmaxes", "cmins", "cbars"):
        parts[part_name].set_color("black")
        parts[part_name].set_linewidth(1.2)

    # Overlay individual points with jitter
    for i, (vals, color) in enumerate(zip(data_by_type, colors)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.35, s=10, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} distribution by search type", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Summary stats as text
    for i, (vals, stype) in enumerate(zip(data_by_type, per_type.keys())):
        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07,
                f"n={len(vals)}\nbest={vals.max():.4f}",
                ha="center", fontsize=8, color=SEARCH_STYLES[stype]["color"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved distribution plot → {output_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(args) -> None:
    print("=" * 70)
    print("  ExaTune Search Comparison Visualizer")
    print("=" * 70)

    # Determine which searches were provided
    paths = {
        "grid":     args.grid,
        "random":   args.random,
        "bayesian": args.bayesian,
    }
    active = {k: v for k, v in paths.items() if v is not None}
    if not active:
        print("ERROR: Provide at least one of --grid, --random, or --bayesian.")
        sys.exit(1)

    print(f"\n  Search types requested: {', '.join(active.keys())}")

    # Load config for param names
    param_names: List[str] = []
    if args.config:
        config = load_yaml(Path(args.config))
        param_names = get_hp_names(config)
        print(f"  Hyperparameters from YAML: {', '.join(param_names)}")
    else:
        print("  No --config provided; param names will be inferred from data.")

    # Load data
    print("\n  Loading result files...")
    combined, per_type = load_all({k: Path(v) for k, v in active.items()})

    # Infer param names from data if no config
    if not param_names:
        exclude = {
            "job_id", "config_hash", "timestamp", "success",
            "mean_score", "std_score", "mean_train_score",
            "fit_time_mean", "score_time_mean", "error_message", "_search_type",
        }
        param_names = [c for c in combined.columns if c not in exclude
                       and not c.endswith("_mean") and not c.endswith("_std")]
        print(f"  Inferred hyperparameters: {', '.join(param_names)}")

    metric = args.metric
    print(f"  Metric: {metric}")
    print(f"  Output dir: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine param pairs to plot
    if args.x_param and args.y_param:
        pairs = [(args.x_param, args.y_param)]
    elif args.all_pairs:
        pairs = list(itertools.combinations(param_names, 2))
        print(f"  Plotting all {len(pairs)} parameter pairs (--all-pairs)")
    else:
        # Default: first two params
        if len(param_names) < 2:
            print("ERROR: Need at least 2 hyperparameters to plot. "
                  "Use --x-param and --y-param to specify.")
            sys.exit(1)
        pairs = [(param_names[0], param_names[1])]
        print(f"  Defaulting to first two params: {param_names[0]} × {param_names[1]}")
        print("  (Use --x-param / --y-param to specify, or --all-pairs for all combinations)")

    print(f"\n  Generating plots for {len(pairs)} parameter pair(s)...\n")

    for x_param, y_param in pairs:
        # Check params exist in data
        missing = [p for p in [x_param, y_param] if p not in combined.columns]
        if missing:
            print(f"  SKIP {x_param} × {y_param}: columns not found: {missing}")
            continue

        pair_label = f"{x_param}_x_{y_param}"
        print(f"  ── {x_param} × {y_param} ──")

        # Heatmap
        if not args.no_heatmap:
            try:
                plot_comparison_heatmap(
                    combined, per_type, x_param, y_param, metric,
                    output_dir / f"heatmap_{pair_label}.png",
                )
            except Exception as e:
                print(f"    WARNING: heatmap failed: {e}")

        # Static 3D surface
        if not args.no_surface:
            try:
                plot_comparison_surface_static(
                    combined, per_type, x_param, y_param, metric,
                    output_dir / f"surface_{pair_label}.png",
                )
            except Exception as e:
                print(f"    WARNING: static surface failed: {e}")

        # Interactive Plotly surface
        if not args.no_plotly:
            try:
                plot_comparison_surface_plotly(
                    combined, per_type, x_param, y_param, metric,
                    output_dir / f"surface_{pair_label}.html",
                )
            except Exception as e:
                print(f"    WARNING: Plotly surface failed: {e}")

    # Score distribution
    if not args.no_distribution:
        try:
            plot_comparison_distribution(
                per_type, metric,
                output_dir / "score_distribution.png",
            )
        except Exception as e:
            print(f"    WARNING: distribution plot failed: {e}")

    # Summary table
    print("\n  ── Summary ──")
    print(f"  {'Search':<18} {'N':>5} {'Best':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for stype, df in per_type.items():
        scores = df[metric].dropna()
        if scores.empty:
            continue
        print(f"  {SEARCH_STYLES[stype]['label']:<18} {len(scores):>5} "
              f"{scores.max():>8.4f} {scores.mean():>8.4f} {scores.std():>8.4f}")

    print(f"\n  All plots saved to: {output_dir}/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune: compare grid / random / Bayesian search on shared landscape plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random vs Bayesian, auto-detect first two params
  python visualize_search_comparison.py \\
      --random   ./results/random_search/iris/results.parquet \\
      --bayesian ./results/bayesian_search/iris/results.parquet \\
      --config   iris_classification.yaml

  # All three searches, specific param pair
  python visualize_search_comparison.py \\
      --grid     ./results/benchmarks/iris_rf/results.parquet \\
      --random   ./results/random_search/iris/results.parquet \\
      --bayesian ./results/bayesian_search/iris/results.parquet \\
      --config   iris_classification.yaml \\
      --x-param  n_estimators --y-param max_depth

  # All param pairs, custom metric
  python visualize_search_comparison.py \\
      --random   ./results/random_search/iris/results.parquet \\
      --bayesian ./results/bayesian_search/iris/results.parquet \\
      --config   iris_classification.yaml \\
      --metric   f1_macro_mean --all-pairs --output-dir ./comparison

  # Skip Plotly (faster), only heatmaps
  python visualize_search_comparison.py \\
      --random   ./results/random_search/iris/results.parquet \\
      --bayesian ./results/bayesian_search/iris/results.parquet \\
      --config   iris_classification.yaml \\
      --no-plotly --no-surface

Plot types produced (per param pair):
  heatmap_<x>_x_<y>.png       — 2D heatmap with per-search best markers
  surface_<x>_x_<y>.png       — static 3D matplotlib surface
  surface_<x>_x_<y>.html      — interactive rotatable Plotly surface
  score_distribution.png      — violin plot comparing score distributions
        """,
    )

    # Search type inputs
    search_group = parser.add_argument_group("Search result inputs (provide at least one)")
    search_group.add_argument("--grid",     type=Path, default=None,
                              metavar="PARQUET",
                              help="Grid search results.parquet")
    search_group.add_argument("--random",   type=Path, default=None,
                              metavar="PARQUET",
                              help="Random search results.parquet")
    search_group.add_argument("--bayesian", type=Path, default=None,
                              metavar="PARQUET",
                              help="Bayesian search results.parquet")

    # Config
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="ExaTune YAML config (used to read hyperparameter names)")

    # Param selection
    pair_group = parser.add_argument_group("Parameter pair selection")
    pair_group.add_argument("--x-param", type=str, default=None,
                            help="Hyperparameter for X axis")
    pair_group.add_argument("--y-param", type=str, default=None,
                            help="Hyperparameter for Y axis")
    pair_group.add_argument("--all-pairs", action="store_true",
                            help="Generate plots for ALL hyperparameter pairs")

    # Metric and output
    parser.add_argument("--metric", "-m", type=str, default="mean_score",
                        help="Metric column to visualize (default: mean_score)")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="./comparison_plots",
                        help="Output directory (default: ./comparison_plots)")

    # Plot type toggles
    toggle_group = parser.add_argument_group("Plot type toggles")
    toggle_group.add_argument("--no-heatmap",      action="store_true",
                              help="Skip heatmap plots")
    toggle_group.add_argument("--no-surface",      action="store_true",
                              help="Skip static 3D surface plots")
    toggle_group.add_argument("--no-plotly",       action="store_true",
                              help="Skip interactive Plotly HTML surfaces")
    toggle_group.add_argument("--no-distribution", action="store_true",
                              help="Skip score distribution violin plot")

    args = parser.parse_args()

    # Validate x/y param pairing
    if (args.x_param is None) != (args.y_param is None):
        parser.error("--x-param and --y-param must both be specified together.")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()