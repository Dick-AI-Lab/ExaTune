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
        "color":        "black",   # blue  — grid IS the background, no heatmap boxes
        "marker_3d":    "square",
        "plotly_color": "black",
        "alpha":        0.75,
        "size_3d":      6,
    },
    "random": {
        "label":        "Random Search",
        "color":        "black",   # orange
        "marker_3d":    "diamond",
        "plotly_color": "black",
        "alpha":        0.80,
        "size_3d":      6,
    },
    "bayesian": {
        "label":        "Bayesian Search",
        "color":        "black",   # pink/magenta
        "marker_3d":    "circle",
        "plotly_color": "black",
        "alpha":        0.85,
        "size_3d":      7,
    },
}

# Overall peak box on heatmap (applies regardless of search type)
_PEAK_COLOR = "#D63384"   # bright green — distinct from both search colors
_PEAK_LW    = 3.0

# Glyph sizing
_GLYPH_SIZE_NORMAL = 55   # scatter marker size (pts^2)
_GLYPH_SIZE_BEST   = 130  # larger for the best-found cell
_STAR_SIZE         = 60   # extra star marker for best cells


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
# 2D Heatmap with overlaid glyph markers
# ---------------------------------------------------------------------------

def _draw_peak_box(ax, col: int, row: int, color: str, lw: float, zorder: int = 5) -> None:
    """Draw a square outline on heatmap cell (col, row) — used only for overall peak."""
    rect = mpatches.FancyBboxPatch(
        (col - 0.47, row - 0.47), 0.94, 0.94,
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        zorder=zorder,
    )
    ax.add_patch(rect)


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

    # Cell value annotations — centred, numbers stay fully visible
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    z_mid = (z_min + z_max) / 2.0
    for row in range(n_rows):
        for col in range(n_cols):
            val = Z[row, col]
            if np.isnan(val):
                continue
            text_color = "white" if val < z_mid else "#1a1a1a"
            ax.text(
                col, row + 0.18,          # slightly below centre — glyphs sit top-right
                f"{val:.3f}",
                ha="center", va="center",
                color=text_color, fontsize=7, zorder=3,
            )

    # ── Overall peak — bright green border box (no glyph, just an outline) ──
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        _draw_peak_box(ax, pc, pr, _PEAK_COLOR, lw=_PEAK_LW, zorder=5)

    # ── Collect visited cells and best cells per non-grid search type ──
    non_grid_types = [s for s in per_type if s != "grid"]
    visited: Dict[str, set] = {}
    for stype in non_grid_types:
        cells = set()
        sub = per_type[stype].dropna(subset=[x_param, y_param, metric])
        for _, row_data in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row_data[x_param], row_data[y_param])
            if col_i is not None and row_i is not None:
                cells.add((col_i, row_i))
        visited[stype] = cells

    bests = get_best_per_search(per_type, x_param, y_param, metric)
    best_cells: Dict[str, Optional[Tuple[int, int]]] = {}
    for stype, best_row in bests.items():
        if stype == "grid":
            continue
        col_i, row_i = pivot_coords(pivot, best_row[x_param], best_row[y_param])
        best_cells[stype] = (col_i, row_i) if col_i is not None else None

    # ── Glyph layout per cell ──
    #
    # Glyphs sit in the TOP-RIGHT corner of each cell so the centred number
    # (nudged slightly downward) remains fully readable.
    #
    # Horizontal offsets when both searches visit the same cell:
    #   random   → (col + 0.28, row - 0.28)
    #   bayesian → (col + 0.28, row - 0.28) but stacked vertically if both present
    #   → when BOTH present: random at top-right, bayesian just below it
    #
    # Glyph legend:
    #   random   → hollow circle  (facecolor=none, edgecolor=orange)
    #   bayesian → filled circle  (facecolor=pink)
    #   best     → same glyph but larger + a small ★ outline below it

    GLYPH_X_OFF  =  0.29   # distance from cell centre → right
    GLYPH_Y_OFF  = -0.29   # distance from cell centre → top (negative = up in imshow)
    GLYPH_Y_SEP  =  0.16   # vertical gap between stacked glyphs

    glyph_scatter_args = []   # list of dicts fed to ax.scatter at the end

    all_cells = set()
    for cells in visited.values():
        all_cells.update(cells)

    for (col_i, row_i) in all_cells:
        types_here = [s for s in non_grid_types if (col_i, row_i) in visited[s]]

        # Vertical positions — if only one search, centred at GLYPH offset;
        # if two, stack them (random above, bayesian below)
        if len(types_here) == 1:
            y_positions = {types_here[0]: row_i + GLYPH_Y_OFF}
        else:
            # random gets the topmost slot, bayesian below
            order = [s for s in ["random", "bayesian"] if s in types_here]
            y_positions = {}
            for k, stype in enumerate(order):
                y_positions[stype] = row_i + GLYPH_Y_OFF + k * GLYPH_Y_SEP

        x_glyph = col_i + GLYPH_X_OFF

        for stype in types_here:
            y_glyph = y_positions[stype]
            is_best  = best_cells.get(stype) == (col_i, row_i)
            color    = SEARCH_STYLES[stype]["color"]
            size     = _GLYPH_SIZE_BEST if is_best else _GLYPH_SIZE_NORMAL

            if stype == "random":
                # Hollow circle: no fill, coloured edge
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph,
                    s=size,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2.0 if is_best else 1.5,
                    zorder=8,
                    marker="o",
                ))
            elif stype == "bayesian":
                # Filled circle
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph,
                    s=size,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=8,
                    marker="o",
                ))

            # Extra star below glyph to call out best-found cell
            if is_best:
                star_y = y_glyph + 0.15   # just below the glyph (rows increase downward)
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=star_y,
                    s=_STAR_SIZE,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=9,
                    marker="*",
                ))

    # Draw all glyphs in one pass
    for kwargs in glyph_scatter_args:
        ax.scatter(**kwargs)

    # ── Axis labels & ticks ──
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

    # ── Legend ──
    legend_handles = [
        mpatches.Patch(edgecolor=_PEAK_COLOR, facecolor="none",
                       linewidth=_PEAK_LW, label="Overall Peak"),
    ]
    for stype in non_grid_types:
        style = SEARCH_STYLES[stype]
        color = style["color"]

        if stype == "random":
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                            facecolors="none", edgecolors=color, linewidths=1.5,
                            marker="o", label=f"{style['label']} (visited)")
            )
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_BEST,
                            facecolors="none", edgecolors=color, linewidths=2.0,
                            marker="o", label=f"{style['label']} best ★")
            )
        elif stype == "bayesian":
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                            facecolors=color, edgecolors="white", linewidths=0.8,
                            marker="o", label=f"{style['label']} (visited)")
            )
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_BEST,
                            facecolors=color, edgecolors="white", linewidths=0.8,
                            marker="o", label=f"{style['label']} best ★")
            )

    if "grid" in per_type:
        legend_handles.append(
            mpatches.Patch(facecolor=SEARCH_STYLES["grid"]["color"],
                           alpha=0.4, label="Grid Search (surface)")
        )

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.18, 1.0), bbox_transform=ax.transAxes,
              fontsize=9, framealpha=0.9, scatterpoints=1)

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

    # ---------------------------
    # Overall peak
    # ---------------------------
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z = Z[pr, pc]

        ax.scatter([pc], [pr], [peak_z + z_offset],
                   color=_PEAK_COLOR, s=120,
                   zorder=10, label="Overall Peak")

        ax.plot([pc, pc], [pr, pr], [peak_z, peak_z + z_offset],
                color=_PEAK_COLOR, linestyle="--")

    # ---------------------------
    # Random search → hollow circles
    # ---------------------------
    for _, row in per_type.get("random", pd.DataFrame()).iterrows():
        col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
        if col_i is None or row_i is None:
            continue

        z_val = Z[row_i, col_i]
        if np.isnan(z_val):
            continue

        ax.scatter(
            [col_i], [row_i], [z_val + z_offset * 0.2],
            facecolors="none",
            edgecolors="black",
            s=25,
            linewidths=1.2,
            alpha=0.6,
        )

    # ---------------------------
    # Bayesian → just points (no spaghetti)
    # ---------------------------
    for _, row in per_type.get("bayesian", pd.DataFrame()).iterrows():
        col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
        if col_i is None or row_i is None:
            continue

        z_val = Z[row_i, col_i]
        if np.isnan(z_val):
            continue

        ax.scatter(
            [col_i], [row_i], [z_val + z_offset * 0.2],
            color="black",
            s=10,
        )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(metric)

    ax.legend()

    if title is None:
        title = f"{metric} surface: {x_param} × {y_param}"
    ax.set_title(title)

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
            continue

        sub = df.dropna(subset=[x_param, y_param, metric])

        x_pts, y_pts, z_pts = [], [], []

        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is None or row_i is None:
                continue

            z_val = z_orig[row_i, col_i]
            if np.isnan(z_val):
                continue

            x_pts.append(col_i)
            y_pts.append(row_i)
            z_pts.append(z_val + z_offset * 0.2)

        # -------------------------------
        # RANDOM → light hollow circles
        # -------------------------------
        if stype == "random":
            traces.append(go.Scatter3d(
                x=x_pts,
                y=y_pts,
                z=z_pts,
                mode="markers",
                marker=dict(
                    size=5,
                    color="rgba(0,0,0,0)",  # no fill
                    symbol="circle",
                    line=dict(color="rgba(0,0,0,0.3)", width=1)
                ),
                opacity=0.6,
                name="Random Search",
            ))

        # -------------------------------
        # BAYESIAN → solid black dots
        # -------------------------------
        elif stype == "bayesian":
            traces.append(go.Scatter3d(
                x=x_pts,
                y=y_pts,
                z=z_pts,
                mode="markers",
                marker=dict(
                    size=4,
                    color="black"
                ),
                name="Bayesian Search",
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
  heatmap_<x>_x_<y>.png       — 2D heatmap with per-search glyph markers
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
    toggle_group.add_range_argument = None
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