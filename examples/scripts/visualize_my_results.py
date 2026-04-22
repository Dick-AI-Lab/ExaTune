#!/usr/bin/env python3
"""ExaTune visualization — all metrics x all hyperparameter pairs.

Reads hyperparameter names and metrics directly from the experiment YAML.
If no --config is provided, auto-detects hyperparameters and metrics from
the Parquet file itself — works for ANY model.

Axis ordering: for every (x, y) pair the parameter with MORE unique values
is always placed on the y-axis so plots are easier to read.

When --model is supplied, two markers are overlaid on every heatmap,
violin, and slice plot:

  ■  dark navy square / circle  — sklearn / XGBoost factory defaults
  ■  pink square / circle       — best configuration found in the grid search

Supported --model values (case-insensitive):
    xgboost, randomforest, ridgeregression, gradientboosting, svm, knn
    Aliases: xgb, rf, ridge, svc, gb, kneighborsclassifier, …

Output structure:
    output_dir/
      f1_macro/
        histogram.png
        violin_<param>.png
        <x>_x_<y>/          ← always fewer-values x more-values
          heatmap.png
          surface.png
          surface.html
      ...
      train_vs_test.png
      slice_<param>.png
      ...

Usage:
    python visualize_my_results.py \\
        --results path/to/results.parquet \\
        --config  path/to/experiment.yaml \\
        --model   gradientboosting \\
        --output-dir ./my_plots
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Column classification — used for auto-detection when no --config given
# ---------------------------------------------------------------------------

NON_PARAM_COLS = {
    "job_id", "config_hash", "timestamp", "success",
    "mean_score", "std_score", "mean_train_score",
    "fit_time_mean", "score_time_mean",
    "cv_scores", "train_scores",
    "emissions_kg_co2", "energy_consumed_kwh",
    "additional_metrics",
}

METRIC_COL_SUFFIXES = ("_mean", ".mean", "mean_score")

KNOWN_METRIC_PREFIXES = (
    "f1_", "precision_", "recall_", "roc_auc", "accuracy",
    "r2", "mse", "mae", "rmse", "log_loss",
    "mean_score", "mean_train_score",
)

# ---------------------------------------------------------------------------
# Marker styles
# ---------------------------------------------------------------------------

_DEFAULT_COLOR = "#1a1a2e"   # dark navy
_PEAK_COLOR    = "#FF69B4"   # hot pink

DEFAULT_LINE_COLOR = _DEFAULT_COLOR
PEAK_LINE_COLOR    = _PEAK_COLOR

# ---------------------------------------------------------------------------
# Model default hyperparameters
# ---------------------------------------------------------------------------

MODEL_DEFAULTS = {
    "xgboost": {
        "n_estimators":     100,
        "max_depth":        6,
        "learning_rate":    0.3,
        "subsample":        1.0,
        "min_child_weight": 1,
        "gamma":            0.0,
        "colsample_bytree": 1.0,
    },
    "randomforest": {
        "n_estimators":      100,
        "max_depth":         None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "max_features":      "sqrt",
    },
    "ridgeregression": {
        "alpha":         1.0,
        "tol":           1e-4,
        "max_iter":      None,
        "fit_intercept": True,
        "solver":        "auto",
    },
    "gradientboosting": {
        "n_estimators":  100,
        "max_depth":     3,
        "learning_rate": 0.1,
        "subsample":     1.0,
    },
    "svm": {
        "C":     1.0,
        "gamma": "scale",
    },
    "knn": {
        "n_neighbours": 5,
        "weights":      "uniform",
        "p":            2,
        "algorithm":    "auto",
    },
}

_ALIASES = {
    "xgb":                        "xgboost",
    "xgboostclassifier":          "xgboost",
    "xgboostregressor":           "xgboost",
    "rf":                         "randomforest",
    "randomforestclassifier":     "randomforest",
    "randomforestregressor":      "randomforest",
    "ridge":                      "ridgeregression",
    "ridgeclassifier":            "ridgeregression",
    "gradientboostingregressor":  "gradientboosting",
    "gradientboostingclassifier": "gradientboosting",
    "gb":                         "gradientboosting",
    "svc":                        "svm",
    "svr":                        "svm",
    "kneighborsclassifier":       "knn",
    "kneighborsregressor":        "knn",
}


def get_model_defaults(model_name: str) -> dict:
    key = model_name.lower().replace(" ", "").replace("_", "")
    key = _ALIASES.get(key, key)
    if key not in MODEL_DEFAULTS:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(set(list(MODEL_DEFAULTS) + list(_ALIASES)))}"
        )
    return dict(MODEL_DEFAULTS[key])


# ---------------------------------------------------------------------------
# Axis ordering — more unique values → y axis
# ---------------------------------------------------------------------------

def orient_params(df: pd.DataFrame, param_a: str, param_b: str) -> tuple:
    n_a = df[param_a].nunique(dropna=False)
    n_b = df[param_b].nunique(dropna=False)
    if n_b >= n_a:
        return param_a, param_b
    else:
        return param_b, param_a


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_hyperparameter_names(config: dict) -> list:
    return list(config.get("hyperparameters", {}).keys())


def extract_metric_pairs(config: dict) -> list:
    evaluation = config.get("evaluation", {})
    scoring    = evaluation.get("scoring", "accuracy")
    additional = evaluation.get("additional_metrics") or []
    pairs = []
    for m in additional:
        pairs.append((m, f"{m}_mean", f"{m}.mean"))
    pairs.append((scoring, "mean_score", "mean_score"))
    return pairs


def resolve_column(df: pd.DataFrame, underscore_col: str, dot_col: str):
    if underscore_col in df.columns:
        return underscore_col
    if dot_col in df.columns:
        return dot_col
    return None


# ---------------------------------------------------------------------------
# Auto-detection (no --config path)
# ---------------------------------------------------------------------------

def autodetect_params(df: pd.DataFrame) -> list:
    out = []
    for col in df.columns:
        cl = col.lower()
        if cl in NON_PARAM_COLS:
            continue
        if any(cl.endswith(s) for s in METRIC_COL_SUFFIXES):
            continue
        if any(cl.startswith(p) for p in KNOWN_METRIC_PREFIXES):
            continue
        if df[col].isna().all():
            continue
        out.append(col)
    return out


def autodetect_metrics(df: pd.DataFrame) -> list:
    seen, out = set(), []
    for col in df.columns:
        cl = col.lower()
        if not (any(cl.endswith(s) for s in METRIC_COL_SUFFIXES) or
                any(cl.startswith(p) for p in KNOWN_METRIC_PREFIXES)):
            continue
        if df[col].isna().all() or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        folder = col.replace(".mean", "").replace("_mean", "").replace("additional_metrics.", "")
        if folder in seen:
            continue
        seen.add(folder)
        out.append((folder, col))
    primary = [(f, c) for f, c in out if c == "mean_score"]
    rest    = [(f, c) for f, c in out if c != "mean_score"]
    return rest + primary


# ---------------------------------------------------------------------------
# Default snapping and peak detection
# ---------------------------------------------------------------------------

def snap_defaults(df: pd.DataFrame, defaults: dict, param_names: list) -> dict:
    result = {}
    for param in param_names:
        if param not in defaults or param not in df.columns:
            continue
        dv     = defaults[param]
        unique = df[param].dropna().unique()
        if dv is None:
            if df[param].isna().any():
                result[param] = None
        elif isinstance(dv, (bool, str)):
            if dv in unique:
                result[param] = dv
        else:
            candidates = [v for v in unique
                          if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if candidates:
                result[param] = min(candidates,
                                    key=lambda v: abs(float(v) - float(dv)))
    return result


def get_peak(df: pd.DataFrame, metric_col: str, param_names: list) -> dict:
    if metric_col not in df.columns or df[metric_col].isna().all():
        return {}
    row = df.loc[df[metric_col].idxmax()]
    return {p: row[p] for p in param_names if p in df.columns}


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _cat_pos(tick_texts: list, target) -> int | None:
    for i, t in enumerate(tick_texts):
        try:
            if float(t) == float(target):
                return i
        except (TypeError, ValueError):
            if str(t) == str(target):
                return i
    return None


def _first_ax() -> plt.Axes:
    axes = plt.gcf().get_axes()
    return axes[0] if axes else plt.gca()


def _draw_heatmap_square(ax, col: int, row: int, color: str,
                          linestyle: str = "solid", lw: float = 2.5,
                          zorder: int = 8) -> None:
    rect = mpatches.FancyBboxPatch(
        (col - 0.46, row - 0.46), 0.92, 0.92,
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(rect)


def _add_violin_overlays(ax, param, default_params, peak_params):
    y_ticks = [t.get_text() for t in ax.get_yticklabels()]
    for params, color, label in [
        (default_params, DEFAULT_LINE_COLOR, "Default"),
        (peak_params,    PEAK_LINE_COLOR,    "Peak"),
    ]:
        pos = _cat_pos(y_ticks, params.get(param))
        if pos is not None:
            ax.axhline(pos, color=color, linewidth=2.2,
                       linestyle="--", alpha=0.9, label=label, zorder=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.75)


def _add_slice_overlays(ax, param, default_params, peak_params):
    for params, color, label in [
        (default_params, DEFAULT_LINE_COLOR, "Default"),
        (peak_params,    PEAK_LINE_COLOR,    "Peak"),
    ]:
        val = params.get(param)
        if val is None:
            continue
        try:
            ax.axvline(float(val), color=color, linewidth=2.2,
                       linestyle="--", alpha=0.9, label=label, zorder=10)
        except (TypeError, ValueError):
            pass
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=7, framealpha=0.75)


# ---------------------------------------------------------------------------
# Wrapped plot calls
# ---------------------------------------------------------------------------

def wrapped_heatmap(plot_fn, df, x_param, y_param, metric,
                    output_path, default_params, peak_params):
    needed = [x_param, y_param, metric]
    sub = df.dropna(subset=needed)
    if sub.empty:
        plot_fn(df, x_param=x_param, y_param=y_param, metric=metric,
                output_path=output_path)
        return

    agg   = sub.groupby([y_param, x_param])[metric].mean().reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric)

    def _try_sort(idx):
        try:
            return idx.astype(float)
        except (TypeError, ValueError):
            return idx

    pivot = pivot.sort_index(axis=0, key=_try_sort)
    pivot = pivot.sort_index(axis=1, key=_try_sort)

    Z        = pivot.values.astype(float)
    n_rows, n_cols = Z.shape
    x_labels = [str(v) for v in pivot.columns]
    y_labels = [str(v) for v in pivot.index]

    cell_px   = 0.52
    fig_w     = max(8,  n_cols * cell_px + 3.5)
    fig_h     = max(5,  n_rows * cell_px + 1.8)
    fig, ax   = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="equal", origin="upper",
                   cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric, fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    z_mid = (z_min + z_max) / 2.0
    ann_fs = max(4, min(8, int(90 / max(n_cols, n_rows))))
    for row in range(n_rows):
        for col in range(n_cols):
            val = Z[row, col]
            if np.isnan(val):
                continue
            txt_color = "white" if val < z_mid else "#111111"
            ax.text(col, row, f"{val:.3f}",
                    ha="center", va="center",
                    color=txt_color, fontsize=ann_fs, zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(x_param, fontsize=10, labelpad=6)
    ax.set_ylabel(y_param, fontsize=10, labelpad=6)
    ax.set_title(f"{metric} heatmap: {x_param} vs {y_param}", fontsize=11, pad=8)

    def _grid_col(val):
        for i, lbl in enumerate(x_labels):
            try:
                if abs(float(lbl) - float(val)) < 1e-9:
                    return i
            except (TypeError, ValueError):
                if str(lbl) == str(val):
                    return i
        return None

    def _grid_row(val):
        for i, lbl in enumerate(y_labels):
            try:
                if abs(float(lbl) - float(val)) < 1e-9:
                    return i
            except (TypeError, ValueError):
                if str(lbl) == str(val):
                    return i
        return None

    legend_handles = []

    # Default square
    d_col = _grid_col(default_params.get(x_param))
    d_row = _grid_row(default_params.get(y_param))
    if d_col is not None and d_row is not None:
        _draw_heatmap_square(ax, d_col, d_row, color=_DEFAULT_COLOR,
                             linestyle="solid", lw=2.8, zorder=8)
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor=_DEFAULT_COLOR,
                       linewidth=2.5, label="Default")
    )

    # Peak square
    p_col = _grid_col(peak_params.get(x_param))
    p_row = _grid_row(peak_params.get(y_param))
    if p_col is not None and p_row is not None:
        _draw_heatmap_square(ax, p_col, p_row, color=_PEAK_COLOR,
                             linestyle="solid", lw=2.8, zorder=9)
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor=_PEAK_COLOR,
                       linewidth=2.5, label="Peak")
    )

    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              bbox_transform=ax.transAxes,
              fontsize=9, framealpha=0.95, borderpad=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def wrapped_violin(plot_fn, df, group_by, metric,
                   output_path, default_params, peak_params):
    plot_fn(df, group_by=group_by, metric=metric, output_path=None)
    _add_violin_overlays(_first_ax(), group_by, default_params, peak_params)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def wrapped_slice(plot_fn, df, param, output_path,
                  default_params, peak_params, **kwargs):
    plot_fn(df, param=param, output_path=None, **kwargs)
    _add_slice_overlays(_first_ax(), param, default_params, peak_params)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def wrapped_surface(plot_fn, df, x_param, y_param, metric,
                    output_path, default_params, peak_params):
    """Draw static 3D surface with landscape.py's internal markers suppressed,
    then add our own controlled Default + Peak circle markers."""

    # draw_markers=False stops landscape.py drawing its own peak/default
    plot_fn(df, x_param=x_param, y_param=y_param, metric=metric,
            output_path=None, draw_markers=False)

    fig = plt.gcf()
    axes3d = [a for a in fig.get_axes() if hasattr(a, "get_zlim")]
    if not axes3d:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return
    ax = axes3d[0]

    needed = [x_param, y_param, metric]
    sub = df.dropna(subset=needed)
    agg = sub.groupby([y_param, x_param])[metric].mean().reset_index()
    pivot = agg.pivot(index=y_param, columns=x_param, values=metric)
    try:
        pivot = pivot.sort_index(axis=0, key=lambda i: i.astype(float))
        pivot = pivot.sort_index(axis=1, key=lambda i: i.astype(float))
    except (TypeError, ValueError):
        pass
    Z = pivot.values.astype(float)
    x_labels = list(pivot.columns)
    y_labels  = list(pivot.index)
    z_range = (np.nanmax(Z) - np.nanmin(Z)) if not np.all(np.isnan(Z)) else 0.0
    z_offset = z_range * 0.10

    legend_handles = []

    def _find_idx(labels, val):
        for i, v in enumerate(labels):
            try:
                if abs(float(v) - float(val)) < 1e-9:
                    return i
            except (TypeError, ValueError):
                if str(v) == str(val):
                    return i
        return None

    for params, color, label in [
        (default_params, _DEFAULT_COLOR, "Default"),
        (peak_params,    _PEAK_COLOR,    "Peak"),
    ]:
        xi = _find_idx(x_labels, params.get(x_param))
        yi = _find_idx(y_labels,  params.get(y_param))
        if xi is not None and yi is not None:
            z_val = Z[yi, xi]
            if not np.isnan(z_val):
                ax.scatter([xi], [yi], [z_val + z_offset],
                           color=color, s=120, marker="o",
                           edgecolors="white", linewidths=0.8, zorder=12)
                ax.plot([xi, xi], [yi, yi], [z_val, z_val + z_offset],
                        color=color, linestyle="--", linewidth=1.5)
                legend_handles.append(
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color, markeredgecolor="white",
                               markersize=9, label=label)
                )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# safe_plot
# ---------------------------------------------------------------------------

def safe_plot(label: str, func):
    try:
        print(f"    {label}...")
        func()
        plt.close("all")
    except Exception as e:
        print(f"      ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune visualizer — all metrics x all param pairs"
    )
    parser.add_argument("--results",    required=True,
                        help="Path to results.parquet")
    parser.add_argument("--config",     default=None,
                        help="Path to experiment YAML config")
    parser.add_argument("--model",      default=None,
                        help=(
                            "Model name for default-param overlay. One of: "
                            "xgboost, randomforest, ridgeregression, "
                            "gradientboosting, svm, knn "
                            "(aliases: xgb, rf, svc, gb, ridge, …). "
                            "Omit to skip the overlay entirely."
                        ))
    parser.add_argument("--output-dir", default="./visualization_output",
                        help="Root output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ config
    if args.config:
        config         = load_yaml_config(Path(args.config))
        param_names    = extract_hyperparameter_names(config)
        metric_triples = extract_metric_pairs(config)
        print(f"Config : {args.config}")
    else:
        print("No --config provided — auto-detecting params and metrics from Parquet.")
        param_names    = None
        metric_triples = None

    # ------------------------------------------------------------------ data
    print(f"\nLoading: {args.results}")
    df = pd.read_parquet(args.results)

    df.columns = [
        c.replace("hyperparameters.", "")
         .replace("additional_metrics.", "")
        for c in df.columns
    ]
    df = df.drop(columns=["cv_scores", "train_scores"], errors="ignore")

    if "success" in df.columns:
        n_before = len(df)
        df = df[df["success"] == True].copy()
        dropped = n_before - len(df)
        if dropped:
            print(f"  Dropped {dropped} failed rows")

    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}\n")

    if param_names is None:
        param_names = autodetect_params(df)
    if metric_triples is None:
        available = autodetect_metrics(df)
    else:
        available = []
        for folder_name, underscore_col, dot_col in metric_triples:
            col = resolve_column(df, underscore_col, dot_col)
            if col and df[col].notna().any():
                available.append((folder_name, col))
            else:
                print(f"  SKIP '{folder_name}' — neither '{underscore_col}' nor "
                      f"'{dot_col}' found / all-NaN")

    param_names = [p for p in param_names if p in df.columns]

    if not param_names:
        print("ERROR: no hyperparameter columns found. Exiting.")
        sys.exit(1)
    if not available:
        print("ERROR: no valid metric columns found. Exiting.")
        sys.exit(1)

    raw_pairs   = list(itertools.combinations(param_names, 2))
    param_pairs = [orient_params(df, a, b) for a, b in raw_pairs]
    primary_col = available[-1][1]

    print(f"  Hyperparameters : {param_names}")
    print(f"  Param pairs     : {len(param_pairs)}  (more-unique-values param → y-axis)")
    print(f"  Metrics         : {[f for f, _ in available]}")
    print(f"  Primary metric  : {primary_col}\n")

    # ------------------------------------------------------------------ defaults + peak
    default_params: dict = {}
    if args.model:
        try:
            raw            = get_model_defaults(args.model)
            default_params = snap_defaults(df, raw, param_names)
            print(f"  Model defaults ({args.model})  →  grid-snapped: {default_params}")
        except KeyError as e:
            print(f"  WARNING: {e}  — default overlay disabled")

    peak_by_metric = {mc: get_peak(df, mc, param_names) for _, mc in available}
    print(f"  Primary peak    : {peak_by_metric.get(primary_col, {})}\n")

    use_overlays = bool(default_params or any(peak_by_metric.values()))

    # ------------------------------------------------------------------ imports
    from exatune.visualization import (
        plot_all_slices,
        plot_conditional_slice,
        plot_contour,
        plot_dashboard,
        plot_heatmap,
        plot_importance,
        plot_interaction_heatmap,
        plot_parallel_coordinates,
        plot_score_histogram,
        plot_score_violin,
        plot_slice,
        plot_surface,
        plot_train_vs_test,
    )
    from exatune.visualization.landscape_3d import plot_surface_plotly

    # ================================================================ per-metric plots
    print("=" * 60)
    print("  Per-metric plots")
    print("=" * 60)

    for folder_name, metric_col in available:
        metric_dir = output_dir / folder_name
        metric_dir.mkdir(exist_ok=True)
        peak = peak_by_metric.get(metric_col, {})
        print(f"\n  [{folder_name}]  col={metric_col}")

        # Histogram
        safe_plot(
            "histogram",
            lambda m=metric_col, d=metric_dir: plot_score_histogram(
                df, metric=m, output_path=d / "histogram.png"
            ),
        )

        # Violin — one per hyperparameter
        for param in param_names:
            if use_overlays:
                safe_plot(
                    f"violin by {param}",
                    lambda m=metric_col, d=metric_dir, p=param, pk=peak: wrapped_violin(
                        plot_score_violin, df, group_by=p, metric=m,
                        output_path=d / f"violin_{p}.png",
                        default_params=default_params, peak_params=pk,
                    ),
                )
            else:
                safe_plot(
                    f"violin by {param}",
                    lambda m=metric_col, d=metric_dir, p=param: plot_score_violin(
                        df, group_by=p, metric=m,
                        output_path=d / f"violin_{p}.png"
                    ),
                )

        # Per param-pair: heatmap + surfaces
        for x_param, y_param in param_pairs:
            pair_dir = metric_dir / f"{x_param}_x_{y_param}"
            pair_dir.mkdir(exist_ok=True)

            # Heatmap
            if use_overlays:
                safe_plot(
                    f"heatmap  {x_param} x {y_param}",
                    lambda m=metric_col, d=pair_dir, x=x_param, y=y_param, pk=peak: (
                        wrapped_heatmap(
                            plot_heatmap, df, x_param=x, y_param=y, metric=m,
                            output_path=d / "heatmap.png",
                            default_params=default_params, peak_params=pk,
                        )
                    ),
                )
            else:
                safe_plot(
                    f"heatmap  {x_param} x {y_param}",
                    lambda m=metric_col, d=pair_dir, x=x_param, y=y_param: wrapped_heatmap(
                        plot_heatmap, df, x_param=x, y_param=y, metric=m,
                        output_path=d / "heatmap.png",
                        default_params={}, peak_params={},
                    ),
                )

            # Static 3D surface — draw_markers=False suppresses landscape.py's
            # internal markers; wrapped_surface adds its own single set
            if use_overlays:
                safe_plot(
                    f"surface  {x_param} x {y_param}",
                    lambda m=metric_col, d=pair_dir, x=x_param, y=y_param, pk=peak: (
                        wrapped_surface(
                            plot_surface, df, x_param=x, y_param=y, metric=m,
                            output_path=d / "surface.png",
                            default_params=default_params, peak_params=pk,
                        )
                    ),
                )
            else:
                safe_plot(
                    f"surface  {x_param} x {y_param}",
                    lambda m=metric_col, d=pair_dir, x=x_param, y=y_param: plot_surface(
                        df, x_param=x, y_param=y, metric=m,
                        output_path=d / "surface.png"
                    ),
                )

            # Plotly interactive surface — pass snapped defaults + per-metric peak
            safe_plot(
                f"surface (plotly)  {x_param} x {y_param}",
                lambda m=metric_col, d=pair_dir, x=x_param, y=y_param, pk=peak: plot_surface_plotly(
                    df, x_param=x, y_param=y, metric=m,
                    output_path=d / "surface.html",
                    config=default_params,   # already a snapped dict — no re-resolution needed
                    peak_params=pk,          # per-metric peak values
                ),
            )

    # ================================================================ structural plots
    print(f"\n{'=' * 60}")
    print(f"  Structural plots  (primary: {primary_col})")
    print("=" * 60)

    primary_peak = peak_by_metric.get(primary_col, {})

    safe_plot("train vs test",
              lambda: plot_train_vs_test(df, output_path=output_dir / "train_vs_test.png"))

    for param in param_names:
        if use_overlays:
            safe_plot(
                f"slice: {param}",
                lambda p=param: wrapped_slice(
                    plot_slice, df, param=p,
                    output_path=output_dir / f"slice_{p}.png",
                    default_params=default_params, peak_params=primary_peak,
                    show_individual=True,
                ),
            )
        else:
            safe_plot(f"slice: {param}",
                      lambda p=param: plot_slice(
                          df, param=p, show_individual=True,
                          output_path=output_dir / f"slice_{p}.png"
                      ))

    safe_plot("all slices",
              lambda: plot_all_slices(df, output_path=output_dir / "all_slices.png"))

    if len(param_names) >= 2:
        safe_plot(
            f"conditional slice {param_names[0]} | {param_names[1]}",
            lambda: plot_conditional_slice(
                df, param=param_names[0], condition_param=param_names[1],
                output_path=output_dir / (
                    f"conditional_slice_{param_names[0]}_by_{param_names[1]}.png"
                ),
            ),
        )

    for x_param, y_param in param_pairs:
        safe_plot(
            f"contour {x_param} x {y_param}",
            lambda x=x_param, y=y_param: plot_contour(
                df, x_param=x, y_param=y,
                output_path=output_dir / f"contour_{x}_x_{y}.png"
            ),
        )

    safe_plot("importance (variance)",
              lambda: plot_importance(df, method="variance",
                                      output_path=output_dir / "importance_variance.png"))
    safe_plot("importance (correlation)",
              lambda: plot_importance(df, method="correlation",
                                      output_path=output_dir / "importance_correlation.png"))
    safe_plot("interaction heatmap",
              lambda: plot_interaction_heatmap(df, output_path=output_dir / "interactions.png"))
    safe_plot("parallel coordinates",
              lambda: plot_parallel_coordinates(df, highlight_best=10,
                                                output_path=output_dir / "parallel.png"))
    safe_plot("dashboard",
              lambda: plot_dashboard(df, output_path=output_dir / "dashboard.png"))

    # ------------------------------------------------------------------ summary
    print(f"\n{'=' * 60}")
    print(f"  Done — {output_dir}/")
    print(f"{'=' * 60}")
    print(f"\n  Metric folders ({len(available)}):")
    for folder, _ in available:
        print(f"    {folder}/")
        print(f"      histogram.png  +  {len(param_names)} violin plots")
        print(f"      {len(param_pairs)} param-pair subfolders "
              f"(heatmap.png, surface.png, surface.html each)")
    print(f"\n  Structural plots at root: slices, contours, importance, "
          f"interactions, dashboard")
    if use_overlays:
        print(f"\n  Overlays:")
        print(f"    ■  dark navy  = factory defaults  (--model {args.model or 'not set'})")
        print(f"    ■  pink       = peak (best grid result per metric)")


if __name__ == "__main__":
    main()