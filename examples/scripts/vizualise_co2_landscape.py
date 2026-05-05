#!/usr/bin/env python3
"""
ExaTune CO2 Emissions Landscape Visualizer
============================================
Renders hyperparameter landscape plots where the Z-axis is kg_co2 (carbon
emissions per evaluation) instead of model accuracy.  Accepts up to three
Parquet files (grid / random / Bayesian), overlays search-type markers, and
shows total CO2 per search type in the legend.

Produces per parameter-pair:
    co2_heatmap_<x>_x_<y>.png        — 2-D heatmap (mean CO2 per cell)
    co2_surface_<x>_x_<y>.png        — static matplotlib 3-D surface
    co2_surface_<x>_x_<y>.html       — interactive Plotly HTML surface
    co2_distribution.png             — violin plot of per-eval CO2 by search type

Usage
-----
    # Random + Bayesian, auto-detect first two params
    python visualize_co2_landscape.py \\
        --random   ./results/random/results.parquet \\
        --bayesian ./results/bayesian/results.parquet \\
        --config   iris_classification.yaml

    # All three searches, specific param pair
    python visualize_co2_landscape.py \\
        --grid     ./results/grid/results.parquet \\
        --random   ./results/random/results.parquet \\
        --bayesian ./results/bayesian/results.parquet \\
        --config   iris_classification.yaml \\
        --x-param  n_estimators --y-param max_depth

    # All param pairs
    python visualize_co2_landscape.py \\
        --random   ./results/random/results.parquet \\
        --bayesian ./results/bayesian/results.parquet \\
        --config   iris_classification.yaml --all-pairs

    # Skip interactive HTML (faster)
    python visualize_co2_landscape.py \\
        --random   ./results/random/results.parquet \\
        --bayesian ./results/bayesian/results.parquet \\
        --config   iris_classification.yaml --no-plotly
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
# Constants
# ---------------------------------------------------------------------------

CO2_METRIC = "kg_co2"          # column in Parquet
ENERGY_METRIC = "energy_kwh"   # secondary column

SEARCH_STYLES: Dict[str, dict] = {
    "grid": {
        "label":        "Grid Search",
        "color":        "black",
        "marker_3d":    "square",
        "plotly_color": "black",
        "alpha":        0.75,
        "size_3d":      6,
    },
    "random": {
        "label":        "Random Search",
        "color":        "black",
        "marker_3d":    "circle",
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

_PEAK_COLOR = "#00E676"   # bright green — highest CO2 cell
_PEAK_LW    = 3.0

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
# Data loading
# ---------------------------------------------------------------------------

def load_and_tag(path: Path, search_type: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [
        c.replace("hyperparameters.", "").replace("additional_metrics.", "")
        for c in df.columns
    ]
    df = df.drop(columns=["cv_scores", "train_scores"], errors="ignore")
    if "success" in df.columns:
        df = df[df["success"] == True].copy()
    df["_search_type"] = search_type

    # Normalize CO2/energy column name variants
    if "emissions_kg_co2" in df.columns and "kg_co2" not in df.columns:
        df = df.rename(columns={"emissions_kg_co2": "kg_co2"})
    if "energy_consumed_kwh" in df.columns and "energy_kwh" not in df.columns:
        df = df.rename(columns={"energy_consumed_kwh": "energy_kwh"})

    # Ensure columns always exist so combined df has consistent schema
    for col in [CO2_METRIC, ENERGY_METRIC]:
        if col not in df.columns:
            df[col] = np.nan

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

        if CO2_METRIC not in df.columns or df[CO2_METRIC].isna().all():
            print(f"  WARNING: '{CO2_METRIC}' column missing or all-NaN in {stype} file.")
            print(f"           Ensure codecarbon was installed when the search ran.")
        else:
            n_valid = df[CO2_METRIC].notna().sum()
            total   = df[CO2_METRIC].sum(skipna=True)
            print(f"  Loaded {len(df):>5} rows from {stype:<10}  "
                  f"CO2 valid={n_valid}  total={total*1000:.4f} g CO₂eq  → {path}")

        per_type[stype] = df

    if not per_type:
        print("ERROR: No valid result files were loaded.")
        sys.exit(1)

    combined = pd.concat(per_type.values(), ignore_index=True)
    return combined, per_type


# ---------------------------------------------------------------------------
# CO2 totals per search type (for legend labels)
# ---------------------------------------------------------------------------

def co2_totals(per_type: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    totals = {}
    for stype, df in per_type.items():
        if CO2_METRIC in df.columns:
            totals[stype] = float(df[CO2_METRIC].sum(skipna=True))
        else:
            totals[stype] = 0.0
    return totals


def _legend_label(stype: str, total_kg: float) -> str:
    """Format legend label with total CO2."""
    base = SEARCH_STYLES[stype]["label"]
    if total_kg == 0.0:
        return f"{base} (CO₂: N/A)"
    if total_kg < 0.001:
        return f"{base} (total: {total_kg*1e6:.2f} µg CO₂eq)"
    if total_kg < 1.0:
        return f"{base} (total: {total_kg*1000:.4f} g CO₂eq)"
    return f"{base} (total: {total_kg:.4f} kg CO₂eq)"


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def build_co2_pivot(
    df: pd.DataFrame, x_param: str, y_param: str
) -> pd.DataFrame:
    needed = [x_param, y_param, CO2_METRIC]
    sub = df.dropna(subset=needed)
    agg = sub.groupby([y_param, x_param])[CO2_METRIC].mean().reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=CO2_METRIC)

    def _sort_index(idx):
        try:
            return idx.astype(float)
        except (TypeError, ValueError):
            return idx

    pivot = pivot.sort_index(axis=0, key=_sort_index)
    pivot = pivot.sort_index(axis=1, key=_sort_index)
    return pivot


def pivot_coords(pivot: pd.DataFrame, x_val, y_val):
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


def get_best_co2_per_search(
    per_type: Dict[str, pd.DataFrame], x_param: str, y_param: str
) -> Dict[str, pd.Series]:
    """Return row with LOWEST CO2 per search type (best = least emissions)."""
    bests = {}
    for stype, df in per_type.items():
        sub = df.dropna(subset=[x_param, y_param, CO2_METRIC])
        if sub.empty:
            continue
        bests[stype] = sub.loc[sub[CO2_METRIC].idxmin()]
    return bests


# ---------------------------------------------------------------------------
# 2D Heatmap
# ---------------------------------------------------------------------------

def plot_co2_heatmap(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_co2_pivot(combined, x_param, y_param)
    Z = pivot.values.astype(float)
    n_rows, n_cols = Z.shape
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]

    totals = co2_totals(per_type)

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 0.9 + 3), max(7, n_rows * 0.75 + 3)))

    im = ax.imshow(Z, aspect="auto", origin="upper",
                   cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean CO₂ per evaluation (kg CO₂eq)", fontsize=10)

    # Cell annotations
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    z_mid = (z_min + z_max) / 2.0
    for row in range(n_rows):
        for col in range(n_cols):
            val = Z[row, col]
            if np.isnan(val):
                continue
            text_color = "white" if val > z_mid else "#1a1a1a"
            if val < 1e-6:
                label = f"{val*1e9:.1f}ng"
            elif val < 1e-3:
                label = f"{val*1e6:.2f}µg"
            else:
                label = f"{val*1000:.3f}g"
            ax.text(col, row + 0.18, label,
                    ha="center", va="center",
                    color=text_color, fontsize=7, zorder=3)

    # Peak (highest CO2) — bright green border
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        rect = mpatches.FancyBboxPatch(
            (pc - 0.47, pr - 0.47), 0.94, 0.94,
            boxstyle="square,pad=0", linewidth=_PEAK_LW,
            edgecolor=_PEAK_COLOR, facecolor="none", zorder=5,
        )
        ax.add_patch(rect)

    # Glyphs — random and bayesian only (grid is the landscape)
    non_grid_types = [s for s in per_type if s != "grid"]
    GLYPH_X_OFF = 0.29
    GLYPH_Y_OFF = -0.29
    GLYPH_Y_SEP = 0.16

    visited: Dict[str, set] = {}
    for stype in non_grid_types:
        cells = set()
        sub = per_type[stype].dropna(subset=[x_param, y_param, CO2_METRIC])
        for _, row_data in sub.iterrows():
            col_i, row_i = pivot_coords(pivot, row_data[x_param], row_data[y_param])
            if col_i is not None and row_i is not None:
                cells.add((col_i, row_i))
        visited[stype] = cells

    bests = get_best_co2_per_search(per_type, x_param, y_param)
    best_cells: Dict[str, Optional[Tuple]] = {}
    for stype, best_row in bests.items():
        if stype == "grid":
            continue
        col_i, row_i = pivot_coords(pivot, best_row[x_param], best_row[y_param])
        best_cells[stype] = (col_i, row_i) if col_i is not None else None

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
            is_best = best_cells.get(stype) == (col_i, row_i)
            color   = SEARCH_STYLES[stype]["color"]
            size    = _GLYPH_SIZE_BEST if is_best else _GLYPH_SIZE_NORMAL

            if stype == "random":
                # Circle with border only
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph, s=size,
                    facecolors="none", edgecolors=color,
                    linewidths=2.0 if is_best else 1.5, zorder=8, marker="o",
                ))
            elif stype == "bayesian":
                # Black filled dot
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph, s=size,
                    facecolors="black", edgecolors="black",
                    linewidths=0.8, zorder=8, marker="o",
                ))

            if is_best:
                glyph_scatter_args.append(dict(
                    x=x_glyph, y=y_glyph + 0.15, s=_STAR_SIZE,
                    facecolors=color, edgecolors="white",
                    linewidths=0.6, zorder=9, marker="*",
                ))

    for kwargs in glyph_scatter_args:
        ax.scatter(**kwargs)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)

    if title is None:
        title = f"CO₂ emissions landscape: {x_param} × {y_param}"
    ax.set_title(title, fontsize=12, pad=10)

    # Legend
    legend_handles = [
        mpatches.Patch(edgecolor=_PEAK_COLOR, facecolor="none",
                       linewidth=_PEAK_LW, label="Highest CO₂ cell"),
    ]
    for stype in non_grid_types:
        lbl = _legend_label(stype, totals.get(stype, 0.0))
        if stype == "random":
            legend_handles.append(plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                facecolors="none", edgecolors="black", linewidths=1.5,
                marker="o", label=lbl))
        elif stype == "bayesian":
            legend_handles.append(plt.scatter([], [], s=_GLYPH_SIZE_NORMAL,
                facecolors="black", edgecolors="black", linewidths=0.8,
                marker="o", label=lbl))

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.18, 1.0), bbox_transform=ax.transAxes,
              fontsize=8.5, framealpha=0.9, scatterpoints=1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved CO2 heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Static matplotlib 3D surface
# ---------------------------------------------------------------------------

def plot_co2_surface_static(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_co2_pivot(combined, x_param, y_param)
    Z = pivot.values.astype(float)
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)
    X, Y = np.meshgrid(range(len(x_labels)), range(len(y_labels)))

    totals = co2_totals(per_type)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="YlOrRd", alpha=0.78, edgecolor="none")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10)
    cbar.set_label("kg CO₂eq", fontsize=10)

    z_range = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.12

    # Peak marker (highest CO2 point)
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z = Z[pr, pc]
        ax.scatter([pc], [pr], [peak_z + z_offset],
                   color=_PEAK_COLOR, s=120, zorder=10, label="Highest CO₂")
        ax.plot([pc, pc], [pr, pr], [peak_z, peak_z + z_offset],
                color=_PEAK_COLOR, linestyle="--")

    # Search type markers — skip grid (it is the landscape)
    for stype, df in per_type.items():
        if stype == "grid":
            continue
        sub = df.dropna(subset=[x_param, y_param, CO2_METRIC])
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
            z_pts.append(z_val + z_offset * 0.2)

        if not x_pts:
            continue

        lbl = _legend_label(stype, totals.get(stype, 0.0))
        if stype == "random":
            # Circle with border only
            ax.scatter(x_pts, y_pts, z_pts,
                       facecolors="none", edgecolors=color,
                       s=28, linewidths=1.3, alpha=0.85, label=lbl)
        elif stype == "bayesian":
            # Black filled dot
            ax.scatter(x_pts, y_pts, z_pts,
                       color="black", s=18, alpha=0.90, label=lbl)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(v) for v in x_labels], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(v) for v in y_labels], fontsize=8)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel("kg CO₂eq")

    if title is None:
        title = f"CO₂ emissions: {x_param} × {y_param}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved static CO2 surface → {output_path}")


# ---------------------------------------------------------------------------
# Interactive Plotly 3D surface
# ---------------------------------------------------------------------------

def plot_co2_surface_plotly(
    combined: pd.DataFrame,
    per_type: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    pivot = build_co2_pivot(combined, x_param, y_param)
    Z = pivot.values.astype(float)
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]
    x_vals = np.arange(len(x_labels))
    y_vals = np.arange(len(y_labels))

    totals = co2_totals(per_type)

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

    # Surface
    traces.append(go.Surface(
        x=x_vals, y=y_vals, z=Z_surf,
        colorscale="YlOrRd",
        opacity=0.80,
        colorbar=dict(title=dict(text="kg CO₂eq", side="right")),
        hovertemplate=(
            f"{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>"
            f"CO₂: %{{z:.6f}} kg<extra></extra>"
        ),
        name="CO₂ landscape",
        showlegend=True,
    ))

    z_range = np.nanmax(Z) - np.nanmin(Z) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.12

    # Peak marker
    if not np.all(np.isnan(Z)):
        pr, pc = np.unravel_index(np.nanargmax(Z), Z.shape)
        peak_z = Z[pr, pc]
        traces.append(go.Scatter3d(
            x=[pc], y=[pr], z=[peak_z + z_offset * 1.6],
            mode="markers+text",
            marker=dict(size=12, color=_PEAK_COLOR, symbol="diamond"),
            text=[f"Peak CO₂<br>{x_labels[pc]} / {y_labels[pr]}"],
            textposition="top center",
            textfont=dict(size=10, color=_PEAK_COLOR),
            name=f"Highest CO₂ ({peak_z*1000:.4f} g)",
            hovertemplate=(
                f"<b>Highest CO₂ cell</b><br>{x_param}: {x_labels[pc]}<br>"
                f"{y_param}: {y_labels[pr]}<br>CO₂: {peak_z:.6f} kg<extra></extra>"
            ),
        ))
        traces.append(go.Scatter3d(
            x=[pc, pc], y=[pr, pr], z=[peak_z, peak_z + z_offset * 1.6],
            mode="lines", line=dict(color=_PEAK_COLOR, width=3, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # Per-search markers — skip grid (it is the landscape)
    for stype, df in per_type.items():
        if stype == "grid":
            continue
        sub = df.dropna(subset=[x_param, y_param, CO2_METRIC])
        color = SEARCH_STYLES[stype]["plotly_color"]
        total_kg = totals.get(stype, 0.0)
        legend_lbl = _legend_label(stype, total_kg)

        x_pts, y_pts, z_pts, hover_texts = [], [], [], []
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
            hover_texts.append(
                f"{x_param}: {row[x_param]}<br>"
                f"{y_param}: {row[y_param]}<br>"
                f"CO₂: {row[CO2_METRIC]*1000:.4f} g CO₂eq"
            )

        if not x_pts:
            continue

        if stype == "random":
            # Circle with border only
            marker_kwargs = dict(
                size=5, color="rgba(0,0,0,0)",
                symbol="circle",
                line=dict(color=color, width=2),
            )
        elif stype == "bayesian":
            # Black filled dot
            marker_kwargs = dict(size=5, color="black", symbol="circle")

        traces.append(go.Scatter3d(
            x=x_pts, y=y_pts, z=z_pts,
            mode="markers",
            marker=marker_kwargs,
            name=legend_lbl,
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        ))

    if title is None:
        title = f"CO₂ Emissions Landscape: {x_param} × {y_param}"

    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        width=1050, height=780,
        scene=dict(
            xaxis=dict(title=x_param, tickmode="array",
                       tickvals=list(x_vals), ticktext=x_labels),
            yaxis=dict(title=y_param, tickmode="array",
                       tickvals=list(y_vals), ticktext=y_labels),
            zaxis=dict(title="kg CO₂eq"),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
            aspectmode="auto",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="lightgrey", borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, b=0, t=55),
    )

    fig = go.Figure(data=traces, layout=layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"    Saved interactive CO2 surface → {output_path}")


# ---------------------------------------------------------------------------
# CO2 distribution violin plot
# ---------------------------------------------------------------------------

def plot_co2_distribution(
    per_type: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    totals = co2_totals(per_type)

    data_by_type, labels, colors, stypes = [], [], [], []
    for stype, df in per_type.items():
        if CO2_METRIC not in df.columns:
            continue
        sub = df[CO2_METRIC].dropna()
        if sub.empty:
            continue
        data_by_type.append(sub.values * 1e6)   # convert to µg for readability
        labels.append(_legend_label(stype, totals.get(stype, 0.0)))
        colors.append(SEARCH_STYLES[stype]["color"])
        stypes.append(stype)

    if not data_by_type:
        print("    WARNING: No CO2 data available for distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(data_by_type) * 3), 5))
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
    ax.set_xticklabels(labels, fontsize=9, wrap=True)
    ax.set_ylabel("CO₂ per evaluation (µg CO₂eq)", fontsize=11)
    ax.set_title("CO₂ emissions distribution by search type", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for i, (vals, stype) in enumerate(zip(data_by_type, stypes)):
        ax.text(i, ylim[0] - span * 0.09,
                f"n={len(vals)}\nmin={vals.min():.2f}µg",
                ha="center", fontsize=8,
                color=SEARCH_STYLES[stype]["color"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved CO2 distribution → {output_path}")


# ---------------------------------------------------------------------------
# Infer HP names from DataFrame
# ---------------------------------------------------------------------------

def infer_hp_names(combined: pd.DataFrame) -> List[str]:
    exclude = {
        "job_id", "config_hash", "timestamp", "success",
        "mean_score", "std_score", "mean_train_score",
        "fit_time_mean", "score_time_mean", "error_message",
        "_search_type", CO2_METRIC, ENERGY_METRIC,
    }
    return [c for c in combined.columns
            if c not in exclude
            and not c.endswith("_mean") and not c.endswith("_std")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    print("=" * 70)
    print("  ExaTune CO₂ Emissions Landscape Visualizer")
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

    print(f"\n  Search types: {', '.join(active.keys())}")

    param_names: List[str] = []
    if args.config:
        config = load_yaml(Path(args.config))
        param_names = get_hp_names(config)
        print(f"  Hyperparameters: {', '.join(param_names)}")

    print("\n  Loading result files...")
    combined, per_type = load_all({k: Path(v) for k, v in active.items()})

    if not param_names:
        param_names = infer_hp_names(combined)
        print(f"  Inferred hyperparameters: {', '.join(param_names)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {output_dir}\n")

    # Param pairs
    if args.x_param and args.y_param:
        pairs = [(args.x_param, args.y_param)]
    elif args.all_pairs:
        pairs = list(itertools.combinations(param_names, 2))
        print(f"  Plotting all {len(pairs)} parameter pairs")
    else:
        if len(param_names) < 2:
            print("ERROR: Need at least 2 hyperparameters. "
                  "Use --x-param / --y-param or --all-pairs.")
            sys.exit(1)
        pairs = [(param_names[0], param_names[1])]
        print(f"  Defaulting to: {param_names[0]} × {param_names[1]}")
        print("  (Use --all-pairs for all combinations)")

    print(f"  Generating plots for {len(pairs)} pair(s)...\n")

    for x_param, y_param in pairs:
        missing = [p for p in [x_param, y_param] if p not in combined.columns]
        if missing:
            print(f"  SKIP {x_param} × {y_param}: columns not found: {missing}")
            continue

        pair_label = f"{x_param}_x_{y_param}"
        print(f"  ── {x_param} × {y_param} ──")

        if not args.no_heatmap:
            try:
                plot_co2_heatmap(combined, per_type, x_param, y_param,
                                 output_dir / f"co2_heatmap_{pair_label}.png")
            except Exception as e:
                print(f"    WARNING: heatmap failed: {e}")

        if not args.no_surface:
            try:
                plot_co2_surface_static(combined, per_type, x_param, y_param,
                                        output_dir / f"co2_surface_{pair_label}.png")
            except Exception as e:
                print(f"    WARNING: static surface failed: {e}")

        if not args.no_plotly:
            try:
                plot_co2_surface_plotly(combined, per_type, x_param, y_param,
                                        output_dir / f"co2_surface_{pair_label}.html")
            except Exception as e:
                print(f"    WARNING: Plotly surface failed: {e}")

    if not args.no_distribution:
        try:
            plot_co2_distribution(per_type, output_dir / "co2_distribution.png")
        except Exception as e:
            print(f"    WARNING: distribution plot failed: {e}")

    # Summary
    totals = co2_totals(per_type)
    print("\n  ── CO₂ Summary ──")
    print(f"  {'Search':<18} {'N':>5} {'Total CO₂':>14} {'Mean/eval':>14} {'Min/eval':>14}")
    print(f"  {'-'*18} {'-'*5} {'-'*14} {'-'*14} {'-'*14}")
    for stype, df in per_type.items():
        if CO2_METRIC not in df.columns:
            continue
        scores = df[CO2_METRIC].dropna()
        if scores.empty:
            continue
        print(f"  {SEARCH_STYLES[stype]['label']:<18} {len(scores):>5} "
              f"{totals[stype]*1000:>13.4f}g "
              f"{scores.mean()*1e6:>13.2f}µg "
              f"{scores.min()*1e6:>13.2f}µg")

    print(f"\n  All plots saved to: {output_dir}/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune CO₂ landscape visualizer — Z-axis = carbon emissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random + Bayesian, auto params
  python visualize_co2_landscape.py \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml

  # All three, specific param pair
  python visualize_co2_landscape.py \\
      --grid     ./results/grid/results.parquet \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml \\
      --x-param  n_estimators --y-param max_depth

  # All pairs
  python visualize_co2_landscape.py \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml --all-pairs

  # Skip Plotly (faster)
  python visualize_co2_landscape.py \\
      --random   ./results/random/results.parquet \\
      --bayesian ./results/bayesian/results.parquet \\
      --config   iris_classification.yaml --no-plotly
        """,
    )
    parser.add_argument("--grid",     type=Path, default=None, metavar="PARQUET")
    parser.add_argument("--random",   type=Path, default=None, metavar="PARQUET")
    parser.add_argument("--bayesian", type=Path, default=None, metavar="PARQUET")
    parser.add_argument("--config", "-c", type=Path, default=None)

    pair_group = parser.add_argument_group("Parameter pair selection")
    pair_group.add_argument("--x-param", type=str, default=None)
    pair_group.add_argument("--y-param",  type=str, default=None)
    pair_group.add_argument("--all-pairs", action="store_true")

    parser.add_argument("--output-dir", "-o", type=str, default="./co2_plots")
    parser.add_argument("--no-heatmap",      action="store_true")
    parser.add_argument("--no-surface",      action="store_true")
    parser.add_argument("--no-plotly",       action="store_true")
    parser.add_argument("--no-distribution", action="store_true")

    args = parser.parse_args()
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