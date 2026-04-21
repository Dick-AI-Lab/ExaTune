#!/usr/bin/env python3
"""
ExaTune Search Comparison Visualizer
======================================
Overlays results from grid search, random search, and/or Bayesian search on
shared 3D landscape surfaces and heatmaps so you can visually compare where
each search method sampled and how well it performed.

Changes vs previous version:
  - Heatmap best markers:
      Random best   → black dotted square outline
      Bayesian best → black solid square outline
  - Legend now includes coverage % (cells visited / total cells) per search type
  - 3D surfaces (static + Plotly) now show best markers for overall peak,
    best random, and best Bayesian — each with a distinct marker and drop line

Usage
-----
    python visualize_search_comparison.py \\
        --random   ./results/random_search/iris/results.parquet \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml \\
        --x-param  n_estimators --y-param max_depth

    python visualize_search_comparison.py \\
        --grid     ./results/benchmarks/iris_rf/results.parquet \\
        --random   ./results/random_search/iris/results.parquet \\
        --bayesian ./results/bayesian_search/iris/results.parquet \\
        --config   iris_classification.yaml --all-pairs

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
        "color":        "#2196F3",
        "marker_3d":    "square",
        "plotly_color": "#2196F3",
        "alpha":        0.75,
        "size_3d":      6,
    },
    "random": {
        "label":        "Random Search",
        "color":        "black",
        "marker_3d":    "diamond",
        "plotly_color": "black",
        "alpha":        0.80,
        "size_3d":      6,
    },
    "bayesian": {
        "label":        "Bayesian Search",
        "color":        "black",
        "marker_3d":    "circle",
        "plotly_color": "black",
        "alpha":        0.85,
        "size_3d":      7,
    },
}

# Overall peak outline on heatmap
_PEAK_COLOR = "#D63384"
_PEAK_LW    = 3.0

# Best-marker colors on 3D surfaces
_BEST_RANDOM_COLOR   = "black"   # orange
_BEST_BAYESIAN_COLOR = "black"   # pink
_BEST_OVERALL_COLOR  = "#00E676"   # bright green

# Glyph sizing (heatmap scatter markers)
_GLYPH_SIZE_NORMAL = 55
_GLYPH_SIZE_BEST   = 130
_STAR_SIZE         = 60


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_hp_names(config: dict) -> List[str]:
    return list(config.get("hyperparameters", {}).keys())


# ---------------------------------------------------------------------------
# Parquet loading and normalisation
# ---------------------------------------------------------------------------

def load_and_tag(path: Path, search_type: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalise column names:
    #   hyperparameters.X        → X
    #   additional_metrics.X.mean → X_mean
    #   additional_metrics.X.std  → X_std
    #   additional_metrics.X     → X   (fallback)
    def _norm(c: str) -> str:
        if c.startswith("hyperparameters."):
            return c[len("hyperparameters."):]
        if c.startswith("additional_metrics."):
            rest = c[len("additional_metrics."):]   # e.g. "f1_macro.mean"
            if rest.endswith(".mean"):
                return rest[:-5].replace(".", "_") + "_mean"
            if rest.endswith(".std"):
                return rest[:-4].replace(".", "_") + "_std"
            return rest.replace(".", "_")
        return c

    df.columns = [_norm(c) for c in df.columns]
    df = df.drop(columns=["cv_scores", "train_scores"], errors="ignore")
    if "success" in df.columns:
        df = df[df["success"] == True].copy()
    df["_search_type"] = search_type
    return df


def load_all(
    paths: Dict[str, Optional[Path]]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
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

def build_pivot(
    df: pd.DataFrame, x_param: str, y_param: str, metric: str
) -> pd.DataFrame:
    needed = [x_param, y_param, metric]
    sub = df.dropna(subset=needed)
    agg = sub.groupby([y_param, x_param])[metric].mean().reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric)

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
    bests = {}
    for stype, df in per_type.items():
        sub = df.dropna(subset=[x_param, y_param, metric])
        if sub.empty:
            continue
        bests[stype] = sub.loc[sub[metric].idxmax()]
    return bests


def pivot_coords(
    pivot: pd.DataFrame, x_val, y_val
) -> Tuple[Optional[int], Optional[int]]:
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
# Coverage calculation
# ---------------------------------------------------------------------------

def compute_coverage(
    per_type: Dict[str, pd.DataFrame],
    pivot: pd.DataFrame,
    x_param: str,
    y_param: str,
) -> Dict[str, float]:
    """
    Return fraction of pivot cells visited by each search type.
    Grid search covers the full plane by definition → 100%.
    """
    total_cells = pivot.shape[0] * pivot.shape[1]
    coverage = {}
    for stype, df in per_type.items():
        if stype == "grid":
            coverage[stype] = 1.0
            continue
        sub = df.dropna(subset=[x_param, y_param])
        visited = set()
        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is not None and row_i is not None:
                visited.add((col_i, row_i))
        coverage[stype] = len(visited) / total_cells if total_cells > 0 else 0.0
    return coverage


# ---------------------------------------------------------------------------
# Heatmap best-cell square drawing helpers
# ---------------------------------------------------------------------------

def _draw_best_square_random(ax, col: int, row: int, zorder: int = 7) -> None:
    """Black dotted square outline — marks best random search cell."""
    rect = mpatches.FancyBboxPatch(
        (col - 0.44, row - 0.44), 0.88, 0.88,
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor="black",
        facecolor="none",
        linestyle=(0, (4, 3)),   # dotted
        zorder=zorder,
    )
    ax.add_patch(rect)


def _draw_best_square_bayesian(ax, col: int, row: int, zorder: int = 7) -> None:
    """Black solid square outline — marks best Bayesian search cell."""
    rect = mpatches.FancyBboxPatch(
        (col - 0.44, row - 0.44), 0.88, 0.88,
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor="black",
        facecolor="none",
        linestyle="solid",
        zorder=zorder,
    )
    ax.add_patch(rect)


def _draw_peak_box(ax, col: int, row: int, color: str, lw: float, zorder: int = 5) -> None:
    """Colored outline for the overall peak cell."""
    rect = mpatches.FancyBboxPatch(
        (col - 0.47, row - 0.47), 0.94, 0.94,
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        zorder=zorder,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# 2D Heatmap
# ---------------------------------------------------------------------------

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

    coverage = compute_coverage(per_type, pivot, x_param, y_param)

    fig, ax = plt.subplots(
        figsize=(max(9, n_cols * 0.9 + 2), max(7, n_rows * 0.75 + 2))
    )

    im = ax.imshow(Z, aspect="auto", origin="upper",
                   cmap="viridis", interpolation="nearest")
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
            ax.text(col, row + 0.18, f"{val:.3f}",
                    ha="center", va="center",
                    color=text_color, fontsize=7, zorder=3)

    # Overall peak — colored border
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        _draw_peak_box(ax, pc, pr, _PEAK_COLOR, lw=_PEAK_LW, zorder=5)

    # Per-search best squares on heatmap cells
    non_grid_types = [s for s in per_type if s != "grid"]
    bests = get_best_per_search(per_type, x_param, y_param, metric)

    # Overall peak cell (set above)
    overall_peak_cell = (pc, pr) if not np.all(np.isnan(Z)) else None

    best_cells: Dict[str, Optional[Tuple[int, int]]] = {}
    found_overall: Dict[str, bool] = {}
    for stype in non_grid_types:
        if stype not in bests:
            best_cells[stype] = None
            found_overall[stype] = False
            continue
        best_row = bests[stype]
        col_i, row_i = pivot_coords(pivot, best_row[x_param], best_row[y_param])
        best_cells[stype] = (col_i, row_i) if col_i is not None else None
        found_overall[stype] = (overall_peak_cell == (col_i, row_i))

    # Draw best-cell squares only when the search did NOT find the overall peak
    if best_cells.get("random") is not None and not found_overall.get("random"):
        c, r = best_cells["random"]
        _draw_best_square_random(ax, c, r, zorder=7)

    if best_cells.get("bayesian") is not None and not found_overall.get("bayesian"):
        c, r = best_cells["bayesian"]
        _draw_best_square_bayesian(ax, c, r, zorder=7)

    # Glyph markers (visited cells)
    visited: Dict[str, set] = {}
    for stype in non_grid_types:
        cells = set()
        sub = per_type[stype].dropna(subset=[x_param, y_param, metric])
        for _, row_data in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row_data[x_param], row_data[y_param])
            if col_i is not None and row_i is not None:
                cells.add((col_i, row_i))
        visited[stype] = cells

    GLYPH_X_OFF = 0.29
    GLYPH_Y_OFF = -0.29
    GLYPH_Y_SEP = 0.16

    all_cells: set = set()
    for cells in visited.values():
        all_cells.update(cells)

    glyph_scatter_args = []
    for (col_i, row_i) in all_cells:
        types_here = [s for s in non_grid_types if (col_i, row_i) in visited[s]]

        if len(types_here) == 1:
            y_positions = {types_here[0]: row_i + GLYPH_Y_OFF}
        else:
            order = [s for s in ["random", "bayesian"] if s in types_here]
            y_positions = {s: row_i + GLYPH_Y_OFF + k * GLYPH_Y_SEP
                           for k, s in enumerate(order)}

        x_glyph = col_i + GLYPH_X_OFF

        for stype in types_here:
            y_glyph = y_positions[stype]
            color = SEARCH_STYLES[stype]["color"]

            if stype == "random":
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph, s=_GLYPH_SIZE_NORMAL,
                    facecolors="none", edgecolors=color,
                    linewidths=1.5, zorder=8, marker="o",
                ))
            elif stype == "bayesian":
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph, s=_GLYPH_SIZE_NORMAL,
                    facecolors=color, edgecolors="white",
                    linewidths=0.8, zorder=8, marker="o",
                ))

    for kwargs in glyph_scatter_args:
        ax.scatter(**kwargs)

    # Axis labels
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
        cov_pct = coverage.get(stype, 0.0) * 100

        # Visited glyph
        if stype == "random":
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                            facecolors="none", edgecolors=color, linewidths=1.5,
                            marker="o",
                            label=f"{style['label']} (visited, coverage={cov_pct:.1f}%)")
            )
        elif stype == "bayesian":
            legend_handles.append(
                plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                            facecolors=color, edgecolors="white", linewidths=0.8,
                            marker="o",
                            label=f"{style['label']} (visited, coverage={cov_pct:.1f}%)")
            )

    # Best-cell square legend entries — or "found overall best" note if applicable
    if "random" in non_grid_types:
        if found_overall.get("random"):
            legend_handles.append(
                mpatches.Patch(facecolor="none", edgecolor="black", linewidth=0,
                               label=f"Random ✓ found overall best")
            )
        else:
            legend_handles.append(
                mpatches.FancyBboxPatch(
                    (0, 0), 1, 1,
                    boxstyle="square,pad=0",
                    linewidth=2.5, edgecolor="black", facecolor="none",
                    linestyle=(0, (4, 3)),
                    label="Random best cell",
                )
            )
    if "bayesian" in non_grid_types:
        if found_overall.get("bayesian"):
            legend_handles.append(
                mpatches.Patch(facecolor="none", edgecolor="black", linewidth=0,
                               label=f"Bayesian ✓ found overall best")
            )
        else:
            legend_handles.append(
                mpatches.FancyBboxPatch(
                    (0, 0), 1, 1,
                    boxstyle="square,pad=0",
                    linewidth=2.5, edgecolor="black", facecolor="none",
                    linestyle="solid",
                    label="Bayesian best cell",
                )
            )

    # Grid search is the landscape surface itself — no legend entry needed

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.18, 1.0), bbox_transform=ax.transAxes,
              fontsize=9, framealpha=0.9, scatterpoints=1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Static 3D surface — best markers for overall, random, Bayesian
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

    coverage = compute_coverage(per_type, pivot, x_param, y_param)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.75, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, label=metric)

    z_range = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.12

    # ── Overall peak ──
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z = Z[pr, pc]
        ax.scatter([pc], [pr], [peak_z + z_offset],
                   color=_BEST_OVERALL_COLOR, s=140, marker="*",
                   zorder=10, label=f"Overall Peak ({peak_z:.4f})")
        ax.plot([pc, pc], [pr, pr], [peak_z, peak_z + z_offset],
                color=_BEST_OVERALL_COLOR, linestyle="--", linewidth=1.5)

    bests = get_best_per_search(per_type, x_param, y_param, metric)

    # Determine overall peak cell for comparison
    overall_peak_cell = None
    if not np.all(np.isnan(Z)):
        overall_peak_cell = (pc, pr)  # (col, row) set above

    # ── Best Random ──
    if "random" in bests:
        br = bests["random"]
        col_i, row_i = pivot_coords(pivot, br[x_param], br[y_param])
        if col_i is not None and row_i is not None:
            z_val = Z[row_i, col_i]
            if not np.isnan(z_val):
                cov_pct = coverage.get("random", 0.0) * 100
                found_overall = (overall_peak_cell == (col_i, row_i))
                if found_overall:
                    # Just a legend entry — no extra marker needed
                    ax.scatter([], [], [], color=_BEST_RANDOM_COLOR, s=0,
                               label=f"Random ✓ found overall best ({z_val:.4f}, cov={cov_pct:.1f}%)")
                else:
                    ax.scatter([col_i], [row_i], [z_val + z_offset],
                               color=_BEST_RANDOM_COLOR, s=120, marker="^",
                               zorder=10,
                               label=f"Random best ({z_val:.4f}, cov={cov_pct:.1f}%)")
                    ax.plot([col_i, col_i], [row_i, row_i], [z_val, z_val + z_offset],
                            color=_BEST_RANDOM_COLOR, linestyle="--", linewidth=1.2)

    # ── Best Bayesian ──
    if "bayesian" in bests:
        bb = bests["bayesian"]
        col_i, row_i = pivot_coords(pivot, bb[x_param], bb[y_param])
        if col_i is not None and row_i is not None:
            z_val = Z[row_i, col_i]
            if not np.isnan(z_val):
                cov_pct = coverage.get("bayesian", 0.0) * 100
                found_overall = (overall_peak_cell == (col_i, row_i))
                if found_overall:
                    ax.scatter([], [], [], color=_BEST_BAYESIAN_COLOR, s=0,
                               label=f"Bayesian ✓ found overall best ({z_val:.4f}, cov={cov_pct:.1f}%)")
                else:
                    ax.scatter([col_i], [row_i], [z_val + z_offset],
                               color=_BEST_BAYESIAN_COLOR, s=120, marker="D",
                               zorder=10,
                               label=f"Bayesian best ({z_val:.4f}, cov={cov_pct:.1f}%)")
                    ax.plot([col_i, col_i], [row_i, row_i], [z_val, z_val + z_offset],
                            color=_BEST_BAYESIAN_COLOR, linestyle="--", linewidth=1.2)

    # ── All sampled points (random + Bayesian only; grid IS the surface) ──
    for stype, df in per_type.items():
        if stype == "grid":
            continue  # grid search is the landscape surface — no dots plotted
        sub = df.dropna(subset=[x_param, y_param, metric])
        color = SEARCH_STYLES[stype]["color"]
        x_pts, y_pts, z_pts = [], [], []

        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is None or row_i is None:
                continue
            z_val = Z[row_i, col_i]
            if np.isnan(z_val):
                continue
            x_pts.append(col_i)
            y_pts.append(row_i)
            z_pts.append(z_val + z_offset * 0.15)

        if not x_pts:
            continue

        if stype == "random":
            ax.scatter(x_pts, y_pts, z_pts,
                       facecolors="none", edgecolors=color,
                       s=22, linewidths=1.1, alpha=0.5)
        elif stype == "bayesian":
            ax.scatter(x_pts, y_pts, z_pts,
                       color=color, s=12, alpha=0.55)
        # grid search is the landscape surface — no dots plotted

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(metric)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    if title is None:
        title = f"{metric} surface: {x_param} × {y_param}"
    ax.set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved static 3D surface → {output_path}")


# ---------------------------------------------------------------------------
# Interactive Plotly 3D surface — best markers for overall, random, Bayesian
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

    coverage = compute_coverage(per_type, pivot, x_param, y_param)

    # Fill NaN for surface rendering
    Z_surf = Z.copy()
    mask = np.isnan(Z_surf)
    if mask.any() and not mask.all():
        try:
            from scipy.interpolate import griddata
            valid = ~mask
            pts = np.array(np.where(valid)).T
            vals = Z_surf[valid]
            fill_pts = np.array(np.where(mask)).T
            if len(fill_pts) and len(pts):
                Z_surf[mask] = griddata(pts, vals, fill_pts, method="nearest")
        except ImportError:
            pass

    traces = []

    # Base surface
    traces.append(go.Surface(
        x=x_vals, y=y_vals, z=Z_surf,
        colorscale="Viridis",
        opacity=0.78,
        colorbar=dict(title=dict(text=metric, side="right")),
        hovertemplate=(
            f"{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>"
            f"{metric}: %{{z:.4f}}<extra></extra>"
        ),
        name="Landscape (combined)",
        showlegend=True,
    ))

    z_orig = pivot.values.astype(float)
    z_range = (np.nanmax(z_orig) - np.nanmin(z_orig)
               if not np.all(np.isnan(z_orig)) else 0.0)
    z_offset = z_range * 0.12

    # ── Overall peak ──
    if not np.all(np.isnan(z_orig)):
        pr, pc = np.unravel_index(np.nanargmax(z_orig), z_orig.shape)
        peak_z = z_orig[pr, pc]
        traces.append(go.Scatter3d(
            x=[pc], y=[pr], z=[peak_z + z_offset * 1.6],
            mode="markers+text",
            marker=dict(size=14, color=_BEST_OVERALL_COLOR, symbol="diamond"),
            text=[f"Overall Peak<br>{x_labels[pc]}/{y_labels[pr]}<br>{peak_z:.4f}"],
            textposition="top center",
            textfont=dict(size=10, color=_BEST_OVERALL_COLOR),
            name=f"Overall Peak ({peak_z:.4f})",
            hovertemplate=(
                f"<b>Overall Peak</b><br>{x_param}: {x_labels[pc]}<br>"
                f"{y_param}: {y_labels[pr]}<br>{metric}: {peak_z:.4f}<extra></extra>"
            ),
        ))
        traces.append(go.Scatter3d(
            x=[pc, pc], y=[pr, pr], z=[peak_z, peak_z + z_offset * 1.6],
            mode="lines",
            line=dict(color=_BEST_OVERALL_COLOR, width=3, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    bests = get_best_per_search(per_type, x_param, y_param, metric)

    # Overall peak cell for comparison
    overall_peak_cell = (pc, pr) if not np.all(np.isnan(z_orig)) else None

    # ── Best Random ──
    if "random" in bests:
        br = bests["random"]
        col_i, row_i = pivot_coords(pivot, br[x_param], br[y_param])
        if col_i is not None and row_i is not None:
            z_val = z_orig[row_i, col_i]
            if not np.isnan(z_val):
                cov_pct = coverage.get("random", 0.0) * 100
                found_overall = (overall_peak_cell == (col_i, row_i))
                if found_overall:
                    # Legend-only entry — no marker needed, already shown by overall peak
                    traces.append(go.Scatter3d(
                        x=[], y=[], z=[],
                        mode="markers",
                        marker=dict(size=8, color=_BEST_RANDOM_COLOR, symbol="square"),
                        name=f"Random ✓ found overall best ({z_val:.4f}, cov={cov_pct:.1f}%)",
                    ))
                else:
                    traces.append(go.Scatter3d(
                        x=[col_i], y=[row_i], z=[z_val + z_offset * 1.3],
                        mode="markers+text",
                        marker=dict(size=11, color=_BEST_RANDOM_COLOR, symbol="square"),
                        text=[f"Random Best<br>{z_val:.4f}"],
                        textposition="top center",
                        textfont=dict(size=9, color=_BEST_RANDOM_COLOR),
                        name=f"Random best ({z_val:.4f}, cov={cov_pct:.1f}%)",
                        hovertemplate=(
                            f"<b>Random Best</b><br>{x_param}: {x_labels[col_i]}<br>"
                            f"{y_param}: {y_labels[row_i]}<br>"
                            f"{metric}: {z_val:.4f}<extra></extra>"
                        ),
                    ))
                    traces.append(go.Scatter3d(
                        x=[col_i, col_i], y=[row_i, row_i],
                        z=[z_val, z_val + z_offset * 1.3],
                        mode="lines",
                        line=dict(color=_BEST_RANDOM_COLOR, width=2, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))

    # ── Best Bayesian ──
    if "bayesian" in bests:
        bb = bests["bayesian"]
        col_i, row_i = pivot_coords(pivot, bb[x_param], bb[y_param])
        if col_i is not None and row_i is not None:
            z_val = z_orig[row_i, col_i]
            if not np.isnan(z_val):
                cov_pct = coverage.get("bayesian", 0.0) * 100
                found_overall = (overall_peak_cell == (col_i, row_i))
                if found_overall:
                    traces.append(go.Scatter3d(
                        x=[], y=[], z=[],
                        mode="markers",
                        marker=dict(size=8, color=_BEST_BAYESIAN_COLOR, symbol="circle"),
                        name=f"Bayesian ✓ found overall best ({z_val:.4f}, cov={cov_pct:.1f}%)",
                    ))
                else:
                    traces.append(go.Scatter3d(
                        x=[col_i], y=[row_i], z=[z_val + z_offset * 1.3],
                        mode="markers+text",
                        marker=dict(size=11, color=_BEST_BAYESIAN_COLOR, symbol="circle"),
                        text=[f"Bayesian Best<br>{z_val:.4f}"],
                        textposition="top center",
                        textfont=dict(size=9, color=_BEST_BAYESIAN_COLOR),
                        name=f"Bayesian best ({z_val:.4f}, cov={cov_pct:.1f}%)",
                        hovertemplate=(
                            f"<b>Bayesian Best</b><br>{x_param}: {x_labels[col_i]}<br>"
                            f"{y_param}: {y_labels[row_i]}<br>"
                            f"{metric}: {z_val:.4f}<extra></extra>"
                        ),
                    ))
                    traces.append(go.Scatter3d(
                        x=[col_i, col_i], y=[row_i, row_i],
                        z=[z_val, z_val + z_offset * 1.3],
                        mode="lines",
                        line=dict(color=_BEST_BAYESIAN_COLOR, width=2, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))

    # ── All sampled points (random + Bayesian only; grid IS the surface) ──
    for stype, df in per_type.items():
        if stype == "grid":
            continue  # grid search is the landscape — no dots plotted
        sub = df.dropna(subset=[x_param, y_param, metric])
        color = SEARCH_STYLES[stype]["plotly_color"]
        cov_pct = coverage.get(stype, 0.0) * 100

        x_pts, y_pts, z_pts, hover_texts = [], [], [], []
        for _, row in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row[x_param], row[y_param])
            if col_i is None or row_i is None:
                continue
            z_val = z_orig[row_i, col_i]
            if np.isnan(z_val):
                continue
            x_pts.append(col_i)
            y_pts.append(row_i)
            z_pts.append(z_val + z_offset * 0.12)
            hover_texts.append(
                f"{x_param}: {row[x_param]}<br>"
                f"{y_param}: {row[y_param]}<br>"
                f"{metric}: {row[metric]:.4f}"
            )

        if not x_pts:
            continue

        if stype == "random":
            marker_kwargs = dict(size=4, color="rgba(0,0,0,0)",
                                 symbol="circle",
                                 line=dict(color=color, width=1.5))
        elif stype == "bayesian":
            marker_kwargs = dict(size=4, color=color, symbol="circle")
        else:
            marker_kwargs = dict(size=3, color=color, symbol="square", opacity=0.4)

        traces.append(go.Scatter3d(
            x=x_pts, y=y_pts, z=z_pts,
            mode="markers",
            marker=marker_kwargs,
            name=f"{SEARCH_STYLES[stype]['label']} (cov={cov_pct:.1f}%)",
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            opacity=0.6,
        ))

    if title is None:
        search_names = " vs ".join(SEARCH_STYLES[s]["label"] for s in per_type)
        title = f"{metric}: {x_param} × {y_param} — {search_names}"

    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        width=1050, height=780,
        scene=dict(
            xaxis=dict(title=x_param, tickmode="array",
                       tickvals=list(x_vals), ticktext=x_labels),
            yaxis=dict(title=y_param, tickmode="array",
                       tickvals=list(y_vals), ticktext=y_labels),
            zaxis=dict(title=metric),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
            aspectmode="auto",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgrey", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=0, r=0, b=0, t=55),
    )

    fig = go.Figure(data=traces, layout=layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"    Saved interactive 3D surface → {output_path}")


# ---------------------------------------------------------------------------
# Score distribution comparison
# ---------------------------------------------------------------------------

def plot_comparison_distribution(
    per_type: Dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_type, labels, colors = [], [], []
    for stype, df in per_type.items():
        sub = df[metric].dropna()
        if sub.empty:
            continue
        data_by_type.append(sub.values)
        labels.append(SEARCH_STYLES[stype]["label"])
        colors.append(SEARCH_STYLES[stype]["color"])

    if not data_by_type:
        return

    parts = ax.violinplot(data_by_type, positions=range(len(data_by_type)),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
    for part_name in ("cmedians", "cmaxes", "cmins", "cbars"):
        parts[part_name].set_color("black")
        parts[part_name].set_linewidth(1.2)

    for i, (vals, color) in enumerate(zip(data_by_type, colors)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.35, s=10, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} distribution by search type", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for i, (vals, stype) in enumerate(zip(data_by_type, per_type.keys())):
        ax.text(i, ylim[0] - span * 0.07,
                f"n={len(vals)}\nbest={vals.max():.4f}",
                ha="center", fontsize=8,
                color=SEARCH_STYLES[stype]["color"])

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

    param_names: List[str] = []
    if args.config:
        config = load_yaml(Path(args.config))
        param_names = get_hp_names(config)
        print(f"  Hyperparameters from YAML: {', '.join(param_names)}")
    else:
        print("  No --config provided; param names will be inferred from data.")

    print("\n  Loading result files...")
    combined, per_type = load_all({k: Path(v) for k, v in active.items()})

    if not param_names:
        exclude = {
            "job_id", "config_hash", "timestamp", "success",
            "mean_score", "std_score", "mean_train_score",
            "fit_time_mean", "score_time_mean", "error_message",
            "_search_type", "kg_co2", "energy_kwh",
            "emissions_kg_co2", "energy_consumed_kwh",
        }
        param_names = [c for c in combined.columns
                       if c not in exclude
                       and not c.endswith("_mean") and not c.endswith("_std")]
        print(f"  Inferred hyperparameters: {', '.join(param_names)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {args.output_dir}")

    # ── Default params for --default mode ──
    DEFAULT_PARAMS = ["n_estimators", "learning_rate", "max_depth"]
    DEFAULT_PAIRS  = [
        ("n_estimators", "learning_rate"),
        ("n_estimators", "max_depth"),
        ("learning_rate", "max_depth"),
    ]

    # ── Resolve parameter pairs ──
    if args.default:
        # Only keep pairs where both params actually exist in the data
        pairs = [(x, y) for x, y in DEFAULT_PAIRS
                 if x in combined.columns and y in combined.columns]
        missing_params = [p for p in DEFAULT_PARAMS if p not in combined.columns]
        if missing_params:
            print(f"  NOTE: Default params not found in data (skipped): {missing_params}")
        print(f"  Default mode: {len(pairs)} param pair(s): "
              + ", ".join(f"{x}×{y}" for x, y in pairs))
    elif args.x_param and args.y_param:
        pairs = [(args.x_param, args.y_param)]
    elif args.all_pairs:
        pairs = list(itertools.combinations(param_names, 2))
        print(f"  Plotting all {len(pairs)} parameter pairs (--all-pairs)")
    else:
        if len(param_names) < 2:
            print("ERROR: Need at least 2 hyperparameters. "
                  "Use --x-param and --y-param, --all-pairs, or --default.")
            sys.exit(1)
        pairs = [(param_names[0], param_names[1])]
        print(f"  Defaulting to first two params: {param_names[0]} × {param_names[1]}")

    # ── Resolve metrics to plot ──
    # In default mode: all score/metric columns present in data.
    # Otherwise: just args.metric (single metric as before).
    non_metric_cols = {
        "job_id", "config_hash", "timestamp", "success",
        "std_score", "mean_train_score", "fit_time_mean", "score_time_mean",
        "error_message", "_search_type", "kg_co2", "energy_kwh",
        "emissions_kg_co2", "energy_consumed_kwh",
    }
    if args.default or args.all_metrics:
        # Collect mean_score plus any *_mean columns that look like metric aggregates
        metrics = ["mean_score"] + sorted([
            c for c in combined.columns
            if c.endswith("_mean")
            and c not in non_metric_cols
            and combined[c].notna().any()
        ])
        # Deduplicate while preserving order
        seen_m: set = set()
        metrics = [m for m in metrics if not (m in seen_m or seen_m.add(m))]
        print(f"  Metrics ({len(metrics)}): {', '.join(metrics)}")
    else:
        metrics = [args.metric]
        print(f"  Metric: {args.metric}")

    print(f"\n  Generating plots: {len(pairs)} pair(s) × {len(metrics)} metric(s)"
          f" = {len(pairs) * len(metrics)} plot set(s)...\n")

    for metric in metrics:
        if metric not in combined.columns:
            print(f"  SKIP metric '{metric}': column not found")
            continue

        metric_dir = output_dir / metric if len(metrics) > 1 else output_dir
        metric_dir.mkdir(parents=True, exist_ok=True)

        print(f"  ══ Metric: {metric} ══")

        for x_param, y_param in pairs:
            missing = [p for p in [x_param, y_param] if p not in combined.columns]
            if missing:
                print(f"    SKIP {x_param} × {y_param}: columns not found: {missing}")
                continue

            pair_label = f"{x_param}_x_{y_param}"
            print(f"  ── {x_param} × {y_param} ──")

            if not args.no_heatmap:
                try:
                    plot_comparison_heatmap(
                        combined, per_type, x_param, y_param, metric,
                        metric_dir / f"heatmap_{pair_label}.png",
                    )
                except Exception as e:
                    print(f"    WARNING: heatmap failed: {e}")

            if not args.no_surface:
                try:
                    plot_comparison_surface_static(
                        combined, per_type, x_param, y_param, metric,
                        metric_dir / f"surface_{pair_label}.png",
                    )
                except Exception as e:
                    print(f"    WARNING: static surface failed: {e}")

            if not args.no_plotly:
                try:
                    plot_comparison_surface_plotly(
                        combined, per_type, x_param, y_param, metric,
                        metric_dir / f"surface_{pair_label}.html",
                    )
                except Exception as e:
                    print(f"    WARNING: Plotly surface failed: {e}")

        if not args.no_distribution:
            try:
                plot_comparison_distribution(
                    per_type, metric,
                    metric_dir / "score_distribution.png",
                )
            except Exception as e:
                print(f"    WARNING: distribution plot failed: {e}")

    # ── Summary (using primary metric) ──
    primary_metric = metrics[0]
    print(f"\n  ── Summary (metric: {primary_metric}) ──")
    print(f"  {'Search':<18} {'N':>5} {'Best':>8} {'Mean':>8} {'Std':>8} {'Coverage':>10}")
    print(f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    if pairs:
        x0, y0 = pairs[0]
        if x0 in combined.columns and y0 in combined.columns:
            piv0 = build_pivot(combined, x0, y0, primary_metric)
            cov0 = compute_coverage(per_type, piv0, x0, y0)
        else:
            cov0 = {}
    else:
        cov0 = {}

    for stype, df in per_type.items():
        scores = df[primary_metric].dropna() if primary_metric in df.columns else pd.Series()
        if scores.empty:
            continue
        cov_str = f"{cov0.get(stype, 0)*100:.1f}%" if stype in cov0 else "N/A"
        print(f"  {SEARCH_STYLES[stype]['label']:<18} {len(scores):>5} "
              f"{scores.max():>8.4f} {scores.mean():>8.4f} {scores.std():>8.4f} "
              f"{cov_str:>10}")

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
  # Default mode: n_estimators/learning_rate/max_depth pairs, all metrics
  python visualize_search_comparison.py \\
      --grid     ./results/grid/results.parquet \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --default --output-dir ./comparison

  # All metrics, specific param pair
  python visualize_search_comparison.py \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml \\
      --all-metrics --x-param n_estimators --y-param max_depth

  # Single metric, all pairs
  python visualize_search_comparison.py \\
      --grid     ./results/grid/results.parquet \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml \\
      --metric   f1_macro_mean --all-pairs --no-plotly

Output structure with multiple metrics (--default or --all-metrics):
  comparison_plots/
    mean_score/
      heatmap_n_estimators_x_max_depth.png
      surface_n_estimators_x_max_depth.html  ...
    f1_macro_mean/
      heatmap_n_estimators_x_max_depth.png  ...
    score_distribution.png  (per metric)

Plot types produced (per param pair × metric):
  heatmap_<x>_x_<y>.png   — 2D heatmap; dotted=random best, solid=Bayesian best
  surface_<x>_x_<y>.png   — static 3D surface with best markers + drop lines
  surface_<x>_x_<y>.html  — interactive Plotly surface with best markers
  score_distribution.png  — violin plot
        """,
    )

    search_group = parser.add_argument_group("Search result inputs (provide at least one)")
    search_group.add_argument("--grid",     type=Path, default=None, metavar="PARQUET")
    search_group.add_argument("--random",   type=Path, default=None, metavar="PARQUET")
    search_group.add_argument("--bayesian", type=Path, default=None, metavar="PARQUET")

    parser.add_argument("--config", "-c", type=Path, default=None)

    pair_group = parser.add_argument_group("Parameter pair selection")
    pair_group.add_argument("--x-param",    type=str, default=None)
    pair_group.add_argument("--y-param",    type=str, default=None)
    pair_group.add_argument("--all-pairs",  action="store_true")
    pair_group.add_argument(
        "--default", action="store_true",
        help="Default mode: plot n_estimators×learning_rate, n_estimators×max_depth, "
             "learning_rate×max_depth across ALL available metrics (overrides "
             "--x-param, --y-param, --all-pairs, and --metric)",
    )

    parser.add_argument("--metric", "-m", type=str, default="mean_score",
                        help="Metric column to visualize (default: mean_score). "
                             "Ignored when --default or --all-metrics is set.")
    parser.add_argument(
        "--all-metrics", action="store_true",
        help="Generate plots for every metric column found in the data "
             "(mean_score + all *_mean columns). Output is grouped into "
             "per-metric subdirectories.",
    )
    parser.add_argument("--output-dir", "-o", type=str, default="./comparison_plots")

    parser.add_argument("--no-heatmap",      action="store_true")
    parser.add_argument("--no-surface",      action="store_true")
    parser.add_argument("--no-plotly",       action="store_true")
    parser.add_argument("--no-distribution", action="store_true")

    args = parser.parse_args()

    if not args.default and (args.x_param is None) != (args.y_param is None):
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